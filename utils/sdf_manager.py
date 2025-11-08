"""SDF 管理器

提供统一的 SDF / 距离场缓存管理：
- 计算网格哈希 (顶点+面)
- 基于参数构建缓存键
- 读写 npz 压缩文件
- 返回 (distance_field, field_bounds, field_shape, cache_hit)

后续可扩展：LRU、统计、版本迁移。
"""
from __future__ import annotations
import os
import json
import hashlib
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Callable
import numpy as np
import trimesh

@dataclass
class SDFParams:
    resolution: float = 0.002
    padding_ratio: float = 0.05
    version: int = 1  # 结构或算法变动时+1
    # 内存安全上限：允许覆盖（点的总数上限，用于自适应提高步长以限制内存）
    max_points: Optional[int] = None
    # 计算方法："signed" 使用逐点 trimesh.proximity.signed_distance；"fast" 使用体素+EDT+内外判定
    method: str = "signed"

class SDFManager:
    def __init__(self, cache_dir: Optional[str] = None, enabled: bool = True):
        env_dir = os.environ.get("DGM_SDF_CACHE")
        if cache_dir is None:
            cache_dir = env_dir if env_dir else ".cache/sdf"
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ----------------- 公共主入口 -----------------
    def compute_or_load(self, mesh: trimesh.Trimesh, params: SDFParams,
                        mesh_source: Optional[str] = None,
                        abort_fn: Optional[Callable[[], bool]] = None) -> Tuple[np.ndarray, np.ndarray, Tuple[int,int,int], bool]:
        """计算或加载 SDF.

        返回: (distance_field, field_bounds(2,3), field_shape(tuple), cache_hit)
        """
        mesh_hash = self._mesh_hash(mesh)
        cache_key = self._build_cache_key(mesh_hash, params)
        if self.enabled and self._try_load(cache_key):
            data = self._try_load(cache_key)
            if data is not None:
                df, bounds, shape = data
                return df, bounds, shape, True
        # 生成
        df, bounds, shape = self._generate_sdf(mesh, params, abort_fn=abort_fn)
        if self.enabled:
            self._save(cache_key, df, bounds, shape)
        return df, bounds, shape, False

    # ----------------- 内部: 生成 -----------------
    def _generate_sdf(self, mesh: trimesh.Trimesh, params: SDFParams,
                      abort_fn: Optional[Callable[[], bool]] = None):
        """调度 SDF 生成，根据 params.method 选择具体实现。"""
        method = getattr(params, 'method', 'signed')
        if method == 'fast':
            return self._generate_fast_sdf(mesh, params, abort_fn)
        # 默认使用 signed 方法
        return self._generate_signed_sdf(mesh, params, abort_fn)

    def _generate_signed_sdf(self, mesh: trimesh.Trimesh, params: SDFParams,
                             abort_fn: Optional[Callable[[], bool]] = None):
        """精确 SDF 实现：逐点 signed distance 采样 + 分块进度显示。

        特性：
        - 使用 trimesh.proximity.signed_distance 提供准确的穿透/外部分离距离
        - 三维瓦片分块，支持进度显示与中断
        - 自动内存限制与自适应瓦片数量调整
        缺点：首次生成耗时较长，适合需要高精度的场景。
        """
        bounds = mesh.bounds
        if not isinstance(bounds, np.ndarray):
            bounds = np.array(bounds)
        if bounds.shape == (6,):
            bounds = bounds.reshape(2,3)
        size = bounds[1] - bounds[0]
        max_size = size.max()
        padding = max_size * params.padding_ratio
        grid_bounds = np.array([bounds[0]-padding, bounds[1]+padding])
        grid_size = grid_bounds[1] - grid_bounds[0]
    # 计算每个维度的单元数
        num_cells = np.maximum(5, np.ceil(grid_size / params.resolution).astype(int))
        if np.isscalar(num_cells):
            num_cells = np.array([num_cells, num_cells, num_cells])
        # 内存安全：如果总点数过大，则自适应提高步长（降低分辨率）
        max_points = params.max_points
        if max_points is None:
            # 环境变量覆盖，默认 8e6 点（约 8M）
            try:
                max_points = int(float(os.environ.get('DGM_SDF_MAX_POINTS', '8000000')))
            except Exception:
                max_points = 8000000
        total_points_initial = int(num_cells[0] * num_cells[1] * num_cells[2])
        if total_points_initial > max_points and max_points > 0:
            scale = (total_points_initial / max_points) ** (1.0/3.0)
            num_cells = np.maximum(5, np.floor(num_cells / scale).astype(int))
        field_shape = tuple(int(x) for x in num_cells)
        total_points = int(field_shape[0] * field_shape[1] * field_shape[2])
        est_mem_bytes = total_points * 4
        est_mem_mb = est_mem_bytes / (1024*1024)
        print(
            "SDFManager: 开始生成SDF "
            f"resolution={params.resolution} padding_ratio={params.padding_ratio} "
            f"bounds_min={bounds[0].tolist()} bounds_max={bounds[1].tolist()} "
            f"grid_min={(grid_bounds[0]).tolist()} grid_max={(grid_bounds[1]).tolist()} "
            f"field_shape={field_shape} total_points={total_points} (~{est_mem_mb:.1f}MB float32)"
        )
        t_start = time.time()
        # 构建可复用的 ProximityQuery（内部会复用加速结构，显著减少重复开销）
        try:
            pq = trimesh.proximity.ProximityQuery(mesh)
            # 尝试记录所用 ray 引擎类型（若可用）
            try:
                engine_name = type(getattr(mesh, 'ray', None)).__name__
            except Exception:
                engine_name = 'unknown'
            print(f"SDFManager(Signed): 使用 ProximityQuery 加速 (ray_engine={engine_name})")
        except Exception as _e:
            pq = None
            print(f"SDFManager(Signed): ProximityQuery 初始化失败，回退直接函数。err={_e}")
        # 预分配输出并三维分块计算，避免单批点数过大且支持快速中断
        xs = np.linspace(grid_bounds[0,0], grid_bounds[1,0], field_shape[0], dtype=np.float32)
        ys = np.linspace(grid_bounds[0,1], grid_bounds[1,1], field_shape[1], dtype=np.float32)
        zs = np.linspace(grid_bounds[0,2], grid_bounds[1,2], field_shape[2], dtype=np.float32)
        distance_field = np.empty(field_shape, dtype=np.float32)
        # 每批最多点数（可通过环境变量覆盖）
        try:
            max_batch_points = int(float(os.environ.get('DGM_SDF_MAX_BATCH', '1000000')))
        except Exception:
            max_batch_points = 1000000

        Nx, Ny, Nz = field_shape
        # 粗略选择瓦片尺寸（尝试接近立方根），确保不超过各维大小
        root = int(np.cbrt(max(1, max_batch_points)))
        tx = max(1, min(Nx, root))
        ty = max(1, min(Ny, max(1, max_batch_points // max(1, tx))))
        tz = max(1, min(Nz, max(1, max_batch_points // max(1, tx * ty))))
        # 如果仍然超上限（整数整除导致），向下调整 tz/ty/tx
        while tx * ty * tz > max_batch_points and tz > 1:
            tz -= 1
        while tx * ty * tz > max_batch_points and ty > 1:
            ty -= 1
        while tx * ty * tz > max_batch_points and tx > 1:
            tx -= 1

        nx_tiles = (Nx + tx - 1) // tx
        ny_tiles = (Ny + ty - 1) // ty
        nz_tiles = (Nz + tz - 1) // tz
        total_tiles = nx_tiles * ny_tiles * nz_tiles
        # 确保有足够的瓦片以便进度可见（默认至少 10 个瓦片）；可通过环境变量覆盖
        try:
            min_tiles = int(os.environ.get('DGM_SDF_MIN_TILES', '10'))
        except Exception:
            min_tiles = 10
        if min_tiles < 1:
            min_tiles = 1
        if total_tiles < min_tiles:
            # 目标瓦片数放大倍数
            scale_tiles = int(np.ceil(min_tiles / max(1, total_tiles)))
            # 尝试等比分割 xyz 维度
            factor = int(np.ceil(scale_tiles ** (1.0/3.0)))
            tx = max(1, min(tx, Nx // max(1, factor)))
            ty = max(1, min(ty, Ny // max(1, factor)))
            tz = max(1, min(tz, Nz // max(1, factor)))
            if tx < 1: tx = 1
            if ty < 1: ty = 1
            if tz < 1: tz = 1
            nx_tiles = (Nx + tx - 1) // tx
            ny_tiles = (Ny + ty - 1) // ty
            nz_tiles = (Nz + tz - 1) // tz
            total_tiles = nx_tiles * ny_tiles * nz_tiles

        debug = os.environ.get('DGM_SDF_DEBUG', '0') == '1'
        if debug:
            print(f"SDFManager: 瓦片计算 distance field, shape={field_shape}, tile=({tx},{ty},{tz}), tiles={total_tiles}")
        try:
            progress_interval = float(os.environ.get('DGM_SDF_PROGRESS_INTERVAL', '0.1'))
            if progress_interval <= 0:
                progress_interval = 0.1
        except Exception:
            progress_interval = 0.1
        next_progress = progress_interval
        tile_counter = 0

        for xi in range(0, Nx, tx):
            xj = min(Nx, xi + tx)
            xseg = xs[xi:xj]
            for yi in range(0, Ny, ty):
                yj = min(Ny, yi + ty)
                yseg = ys[yi:yj]
                for zi in range(0, Nz, tz):
                    if abort_fn and abort_fn():
                        raise RuntimeError('SDF generation aborted')
                    zj = min(Nz, zi + tz)
                    zseg = zs[zi:zj]
                    X, Y, Z = np.meshgrid(xseg, yseg, zseg, indexing='ij')
                    query_points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
                    if pq is not None:
                        d = pq.signed_distance(query_points)
                    else:
                        d = trimesh.proximity.signed_distance(mesh, query_points)
                    d = d.astype(np.float32, copy=False)
                    distance_field[xi:xj, yi:yj, zi:zj] = d.reshape((xj - xi, yj - yi, zj - zi))
                    tile_counter += 1
                    progress = tile_counter / total_tiles
                    if progress >= next_progress or tile_counter == total_tiles:
                        elapsed = time.time() - t_start
                        print(f"SDFManager: 进度 {progress*100:.1f}% ({tile_counter}/{total_tiles} tiles, {elapsed:.1f}s)")
                        while next_progress <= progress:
                            next_progress += progress_interval
        elapsed_total = time.time() - t_start
        print(f"SDFManager(Signed): 生成SDF完成 用时 {elapsed_total:.2f}s shape={field_shape} 点数={total_points}")
        return distance_field, bounds, field_shape

    def _generate_fast_sdf(self, mesh: trimesh.Trimesh, params: SDFParams,
                           abort_fn: Optional[Callable[[], bool]] = None):
        """快速 SDF 实现：体素占据 + 欧氏距离变换(EDT) + 符号判定。

        特性：
        - 极快的生成速度（相对 signed 方法）
        - 使用 occupancy + EDT 获得近似 signed 距离
        - 适合交互、迭代优化的默认选择
        局限：
        - 精度取决于分辨率与 mesh.contains 的鲁棒性
        - 边界处可能较为粗糙
        """
        from scipy import ndimage
        bounds = mesh.bounds
        if not isinstance(bounds, np.ndarray):
            bounds = np.array(bounds)
        if bounds.shape == (6,):
            bounds = bounds.reshape(2,3)
        size = bounds[1] - bounds[0]
        max_size = size.max()
        padding = max_size * params.padding_ratio
        grid_bounds = np.array([bounds[0]-padding, bounds[1]+padding])
        grid_size = grid_bounds[1] - grid_bounds[0]
        num_cells = np.maximum(5, np.ceil(grid_size / params.resolution).astype(int))
        if np.isscalar(num_cells):
            num_cells = np.array([num_cells, num_cells, num_cells])
        field_shape = tuple(int(x) for x in num_cells)
        total_points = int(field_shape[0] * field_shape[1] * field_shape[2])
        print(
            "SDFManager(Fast): 开始生成快速SDF "
            f"resolution={params.resolution} field_shape={field_shape} total_points={total_points}"
        )
        xs = np.linspace(grid_bounds[0,0], grid_bounds[1,0], field_shape[0])
        ys = np.linspace(grid_bounds[0,1], grid_bounds[1,1], field_shape[1])
        zs = np.linspace(grid_bounds[0,2], grid_bounds[1,2], field_shape[2])
        occupancy = np.zeros(field_shape, dtype=bool)
        try:
            max_batch_slices = int(float(os.environ.get('DGM_FAST_SDF_MAX_SLICES', '8')))
        except Exception:
            max_batch_slices = 8
        t0 = time.time()
        for zi in range(0, field_shape[2], max_batch_slices):
            zj = min(field_shape[2], zi + max_batch_slices)
            zseg = zs[zi:zj]
            X,Y,Z = np.meshgrid(xs, ys, zseg, indexing='ij')
            centers = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
            inside = mesh.contains(centers)
            occupancy[:, :, zi:zj] = inside.reshape((field_shape[0], field_shape[1], zj - zi))
            if abort_fn and abort_fn():
                raise RuntimeError('Fast SDF aborted')
        t_occ = time.time() - t0
        outside_dist = ndimage.distance_transform_edt(~occupancy) * params.resolution
        inside_dist = ndimage.distance_transform_edt(occupancy) * params.resolution
        sdf = inside_dist.astype(np.float32)
        sdf[~occupancy] = -outside_dist[~occupancy].astype(np.float32)
        t_total = time.time() - t0
        print(f"SDFManager(Fast): 体素化耗时 {t_occ:.2f}s 总耗时 {t_total:.2f}s 距离范围=[{sdf.min():.4f}, {sdf.max():.4f}]")
        return sdf, bounds, field_shape

    # ----------------- 内部: 缓存 -----------------
    def _mesh_hash(self, mesh: trimesh.Trimesh) -> str:
        v_bytes = mesh.vertices.astype(np.float32).tobytes()
        f_bytes = mesh.faces.astype(np.int32).tobytes()
        return hashlib.sha256(v_bytes + f_bytes).hexdigest()

    def _build_cache_key(self, mesh_hash: str, params: SDFParams) -> str:
        key_obj: Dict[str, Any] = {
            'mesh_hash': mesh_hash,
            'resolution': float(params.resolution),
            'padding_ratio': float(params.padding_ratio),
            'version': int(params.version),
            'method': getattr(params, 'method', 'signed'),
        }
        key_json = json.dumps(key_obj, sort_keys=True)
        return hashlib.sha256(key_json.encode('utf-8')).hexdigest()

    def _cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f'sdf_{cache_key}.npz'

    def _try_load(self, cache_key: str):
        path = self._cache_path(cache_key)
        if not path.exists():
            return None
        try:
            data = np.load(path)
            df = data['distance_field']
            bounds = data['field_bounds']
            shape = tuple(data['field_shape'].tolist())
            if df.shape != shape:
                return None
            return df, bounds, shape
        except Exception:
            return None

    def _save(self, cache_key: str, df: np.ndarray, bounds: np.ndarray, shape: Tuple[int,int,int]):
        path = self._cache_path(cache_key)
        try:
            np.savez_compressed(path,
                distance_field=df.astype(np.float32),
                field_bounds=np.asarray(bounds, dtype=np.float32),
                field_shape=np.array(shape, dtype=np.int32))
        except Exception as e:
            print(f'SDFManager: 缓存写入失败: {e}')
