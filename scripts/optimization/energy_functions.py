# optimization/energy_functions.py (migrated)

"""
能量函数模块：定义各种优化目标

所有能量函数都是可微分的，可以用 JAX 自动求导
"""

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
from typing import List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
import trimesh
import trimesh.proximity
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import hashlib  # 保留旧接口的向后兼容（若外部还在调用旧方法）
import json
from pathlib import Path
import os
from utils.sdf_manager import SDFManager, SDFParams
from utils.constants import SDF_DEFAULT_METHOD, SDF_SUPPORTED_METHODS, SDF_DEFAULT_RESOLUTION_M


class EnergyFunction(ABC):
    """能量函数基类"""
    
    @abstractmethod
    def compute(self, state, robot, **kwargs) -> jnp.ndarray:
        """
        计算能量值
        
        Args:
            state: OptimizerState 对象
            robot: pyroki.Robot 对象
            **kwargs: 额外参数
            
        Returns:
            scalar energy value
        """
        pass
    
    def get_weight(self) -> float:
        """返回此能量项的权重"""
        return 1.0


class AnchorPointEnergy(EnergyFunction):
    """
    锚点能量：最小化手部锚点和物体锚点之间的距离
    
    E = Σ ||T_world_link @ p_local - p_obj||^2
    """
    
    def __init__(self, weight: float = 1.0):
        self.weight = weight
    
    def compute(self, state, robot, anchor_pairs: List[Dict], **kwargs) -> jnp.ndarray:
        """
        计算锚点能量
        
        Args:
            anchor_pairs: 锚点对列表，每个元素包含:
                {
                    'hand_point_local': [x, y, z],  # 手部局部坐标
                    'hand_link_name': str,           # link 名称
                    'hand_link_idx': int,            # link 索引
                    'obj_point': [x, y, z]           # 物体世界坐标
                }
        """
        if not anchor_pairs:
            return jnp.array(0.0)
        
        total_energy = 0.0
        
        # 1. 运行 FK 获取所有 link 位姿
        link_poses_rel_root = robot.forward_kinematics(state.joint_values)
        T_world_base = state.to_base_pose_se3()
        
        # 2. 对每个锚点对计算能量
        for anchor in anchor_pairs:
            link_idx = anchor['hand_link_idx']
            p_local = jnp.array(anchor['hand_point_local'])  # (3,)
            p_obj = jnp.array(anchor['obj_point'])            # (3,)
            
            # 获取 link 在世界坐标系中的位姿
            T_root_link = jaxlie.SE3(link_poses_rel_root[link_idx])
            T_world_link = T_world_base @ T_root_link
            
            # 将局部坐标转换为世界坐标
            p_local_homo = jnp.append(p_local, 1.0)  # (4,)
            p_world = (T_world_link.as_matrix() @ p_local_homo)[:3]
            
            # 计算距离的平方
            distance_sq = jnp.sum((p_world - p_obj) ** 2)
            total_energy += distance_sq
        
        return total_energy * self.weight
    
    def get_weight(self) -> float:
        return self.weight


class PenetrationAvoidanceEnergy(EnergyFunction):
    """
    穿透避免能量：防止手部穿透到物体内部
    
    使用预计算的距离场实现精确的穿透检测
    """
    
    def __init__(self, weight: float = 1.0, margin: float = 0.001, 
                 hand_key_points: Dict[str, np.ndarray] = None,
                 resolution: Optional[float] = None,  # 若为 None，使用全局默认常量
                 cache_enabled: bool = True,
                 cache_dir: Optional[str] = None,
                 max_cells_per_dim: Optional[int] = None,  # 已弃用，保留参数不再生效
                 padding_ratio: float = 0.05,
                 sdf_method: Optional[str] = None):
        """
        Args:
            weight: 能量权重
            margin: 安全距离（米），距离小于此值时开始施加惩罚
            hand_key_points: 手部关键点字典 {link_name: points_array}
            resolution: 距离场分辨率（米）
            cache_enabled: 是否启用SDF缓存
            cache_dir: 缓存目录，默认使用项目根目录下 .cache/sdf
            max_cells_per_dim: 单维度最大网格单元数（用于限制内存）
            padding_ratio: 边界框向外扩展的比例
        """
        self.weight = weight
        self.margin = margin
        self.hand_key_points = hand_key_points
        self.resolution = resolution if (resolution is not None) else SDF_DEFAULT_RESOLUTION_M
        self.cache_enabled = cache_enabled
        self.padding_ratio = padding_ratio
        # 新的 SDF 管理器封装
        self._sdf_manager = SDFManager(cache_dir=cache_dir, enabled=cache_enabled)
        # SDF 计算方法（'fast' 或 'signed'），优先使用传入参数，否则使用全局默认
        if sdf_method is None:
            sdf_method = SDF_DEFAULT_METHOD
        if sdf_method not in SDF_SUPPORTED_METHODS:
            print(f"PenetrationAvoidanceEnergy: 未知的 sdf_method={sdf_method}，回退到默认 {SDF_DEFAULT_METHOD}")
            sdf_method = SDF_DEFAULT_METHOD
        self.sdf_method = sdf_method

        # 距离场数据（运行态）
        self.distance_field: Optional[np.ndarray] = None
        self.field_bounds: Optional[np.ndarray] = None
        self.field_shape: Optional[Tuple[int,int,int]] = None
        self._last_cache_hit: Optional[bool] = None
    
    def set_key_points(self, key_points: Dict[str, np.ndarray]):
        """
        设置手部关键点
        
        Args:
            key_points: {link_name: points_array} 字典
        """
        self.hand_key_points = key_points
    
    # 旧私有缓存方法移除，改用 SDFManager

    def precompute_distance_field(self, mesh, mesh_source: Optional[str] = None, abort_fn: Optional[callable] = None):
        """
        预计算物体的距离场
        
        Args:
            mesh: trimesh.Trimesh 或 pyvista.PolyData 对象
            mesh_source: 原始网格来源（文件路径或标识字符串），用于提示与调试
        """
        if mesh is None:
            return
        
        # 转换为trimesh格式
        if isinstance(mesh, trimesh.Trimesh):
            # 已经是trimesh格式
            tri_mesh = mesh
        else:
            # 转换为trimesh
            import pyvista as pv
            if isinstance(mesh, pv.PolyData):
                # 从pyvista转换为trimesh
                vertices = mesh.points
                faces = mesh.faces.reshape(-1, 4)[:, 1:]  # 移除面的顶点数
                tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            else:
                print("PenetrationAvoidanceEnergy: 不支持的网格格式")
                return
        
        # 使用 SDFManager 计算或加载，根据 self.sdf_method 选择具体实现
        params = SDFParams(resolution=self.resolution, padding_ratio=self.padding_ratio, version=1, method=self.sdf_method)
        distance_field, bounds, shape, cache_hit = self._sdf_manager.compute_or_load(
            tri_mesh, params, mesh_source=mesh_source, abort_fn=abort_fn
        )
        self.distance_field = distance_field
        self.field_bounds = bounds
        self.field_shape = shape
        self._last_cache_hit = cache_hit
        if cache_hit:
            print(f"PenetrationAvoidanceEnergy: 命中缓存（形状={shape}）。")
            return
        if mesh_source:
            print(f"PenetrationAvoidanceEnergy: 新生成SDF method={self.sdf_method} (源: {mesh_source}) 形状={shape}")
        else:
            print(f"PenetrationAvoidanceEnergy: 新生成SDF method={self.sdf_method} 形状={shape}")

        # 以下日志与统计保留（无需重复计算生成逻辑，这里直接输出范围）
        bounds = tri_mesh.bounds
        # 确保bounds是numpy数组
        if not isinstance(bounds, np.ndarray):
            bounds = np.array(bounds)
        
        # 如果bounds是扁平化的 (6,) 数组，重新构造为 (2, 3)
        if bounds.shape == (6,):
            bounds = bounds.reshape(2, 3)
        
        try:
            signed_distances = self.distance_field.ravel()
            print(f"PenetrationAvoidanceEnergy: 距离场范围: [{signed_distances.min():.4f}, {signed_distances.max():.4f}]")
        except Exception:
            pass
    
    def _query_distance_field(self, points: jnp.ndarray) -> jnp.ndarray:
        """
        查询距离场中的距离值
        
        Args:
            points: (N, 3) 世界坐标系中的点
            
        Returns:
            (N,) 带符号距离数组
        """
        if self.distance_field is None or self.field_bounds is None:
            # 如果没有距离场，返回0（无惩罚）
            return jnp.zeros(points.shape[0])
        
        # 将世界坐标转换为网格索引
        bounds_min = jnp.array(self.field_bounds[0])
        bounds_size = jnp.array(self.field_bounds[1] - self.field_bounds[0])
        grid_shape = jnp.array(self.field_shape)
        
        # 归一化坐标 [0, 1]
        normalized_coords = (points - bounds_min) / bounds_size
        
        # 转换为网格索引
        grid_coords = normalized_coords * (grid_shape - 1)
        
        # 三线性插值
        distances = self._trilinear_interpolate(grid_coords)
        
        return distances
    
    def _trilinear_interpolate(self, coords: jnp.ndarray) -> jnp.ndarray:
        """
        三线性插值查询距离场
        
        Args:
            coords: (N, 3) 网格坐标
            
        Returns:
            (N,) 插值后的距离值
        """
        # 获取整数和分数部分
        coords_floor = jnp.floor(coords).astype(jnp.int32)
        coords_frac = coords - coords_floor
        
        # 确保坐标在有效范围内
        grid_shape = jnp.array(self.field_shape)
        coords_floor = jnp.clip(coords_floor, 0, grid_shape - 2)
        
        # 获取8个角点的值
        c000 = self._get_field_value(coords_floor[:, 0], coords_floor[:, 1], coords_floor[:, 2])
        c001 = self._get_field_value(coords_floor[:, 0], coords_floor[:, 1], coords_floor[:, 2] + 1)
        c010 = self._get_field_value(coords_floor[:, 0], coords_floor[:, 1] + 1, coords_floor[:, 2])
        c011 = self._get_field_value(coords_floor[:, 0], coords_floor[:, 1] + 1, coords_floor[:, 2] + 1)
        c100 = self._get_field_value(coords_floor[:, 0] + 1, coords_floor[:, 1], coords_floor[:, 2])
        c101 = self._get_field_value(coords_floor[:, 0] + 1, coords_floor[:, 1], coords_floor[:, 2] + 1)
        c110 = self._get_field_value(coords_floor[:, 0] + 1, coords_floor[:, 1] + 1, coords_floor[:, 2])
        c111 = self._get_field_value(coords_floor[:, 0] + 1, coords_floor[:, 1] + 1, coords_floor[:, 2] + 1)
        
        # 三线性插值
        x, y, z = coords_frac[:, 0], coords_frac[:, 1], coords_frac[:, 2]
        
        c00 = c000 * (1 - x) + c100 * x
        c01 = c001 * (1 - x) + c101 * x
        c10 = c010 * (1 - x) + c110 * x
        c11 = c011 * (1 - x) + c111 * x
        
        c0 = c00 * (1 - y) + c10 * y
        c1 = c01 * (1 - y) + c11 * y
        
        return c0 * (1 - z) + c1 * z
    
    def _get_field_value(self, x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        """
        获取距离场中的值（向量化）
        
        Args:
            x, y, z: 整数坐标数组
            
        Returns:
            对应的距离值数组
        """
        # 将numpy数组转换为JAX数组
        field = jnp.array(self.distance_field)
        
        # 使用高级索引获取值
        return field[x, y, z]
    
    def compute(self, state, robot, object_mesh=None, **kwargs) -> jnp.ndarray:
        """
        计算穿透避免能量
        
        Args:
            object_mesh: 物体网格 (用于预计算距离场，如果还没有的话)
        """
        if self.hand_key_points is None:
            return jnp.array(0.0)
        
        # 如果还没有距离场，尝试预计算
        if self.distance_field is None and object_mesh is not None:
            self.precompute_distance_field(object_mesh)
        
        # 如果仍然没有距离场，返回0
        if self.distance_field is None:
            return jnp.array(0.0)
        
        total_energy = 0.0
        
        # 1. 运行 FK 获取所有 link 位姿
        link_poses_rel_root = robot.forward_kinematics(state.joint_values)
        T_world_base = state.to_base_pose_se3()
        
        # 收集所有关键点
        all_points_world = []
        
        # 2. 对每个link的关键点计算距离
        for link_name, key_points in self.hand_key_points.items():
            # 查找 link 索引
            try:
                link_idx = robot.links.names.index(link_name)
            except ValueError:
                # 如果找不到指定的 link，跳过
                continue
            
            # 获取 link 在世界坐标系中的位姿
            T_root_link = jaxlie.SE3(link_poses_rel_root[link_idx])
            T_world_link = T_world_base @ T_root_link
            T_world_link_mat = T_world_link.as_matrix()
            
            # 转换关键点到世界坐标系
            for p_local in key_points:
                p_local = jnp.array(p_local)
                
                # 将局部坐标转换为世界坐标
                p_local_homo = jnp.append(p_local, 1.0)
                p_world = (T_world_link_mat @ p_local_homo)[:3]
                all_points_world.append(p_world)
        
        if not all_points_world:
            return jnp.array(0.0)
        
        # 批量查询距离
        points_array = jnp.stack(all_points_world)
        distances = self._query_distance_field(points_array)
        
        # 只惩罚穿模点（距离 > 0的点）：穿模距离的总和
        # 距离 > 0 表示在物体内部，值为穿透深度
        penetration_depths = jnp.maximum(0.0, distances)
        
        # 直接使用穿透距离的总和作为能量（不使用二次惩罚）
        total_energy = jnp.sum(penetration_depths)
        
        return total_energy * self.weight
    
    def get_weight(self) -> float:
        return self.weight


class CollisionAvoidanceEnergy(EnergyFunction):
    """
    碰撞避免能量：惩罚手部与物体的碰撞
    
    使用简单的距离场方法（可以扩展为 SDF）
    """
    
    def __init__(self, weight: float = 0.1, margin: float = 0.01):
        """
        Args:
            weight: 能量权重
            margin: 安全距离（米）
        """
        self.weight = weight
        self.margin = margin
    
    def compute(self, state, robot, **kwargs) -> jnp.ndarray:
        """
        计算碰撞避免能量
        
        这是一个占位符实现，需要实际的碰撞检测
        """
        # TODO: 实现真实的碰撞检测
        # 可以使用 pyroki.collision 模块
        return jnp.array(0.0)
    
    def get_weight(self) -> float:
        return self.weight


class SelfCollisionAvoidanceEnergy(EnergyFunction):
    """
    自碰撞避免能量：防止手部各link之间的碰撞

    使用球体集合来近似每个link的形状，计算球体间距离
    """

    def __init__(self, weight: float = 0.5, margin: float = 0.005,
                 link_spheres: Dict[str, List[Tuple[np.ndarray, float]]] = None):
        """
        Args:
            weight: 能量权重
            margin: 安全距离（米）
            link_spheres: {link_name: [(center, radius), ...]} 球体数据
        """
        self.weight = weight
        self.margin = margin
        self.link_spheres = link_spheres or {}

    def set_link_spheres(self, link_spheres: Dict[str, List[Tuple[np.ndarray, float]]]):
        """
        设置link球体数据

        Args:
            link_spheres: {link_name: [(center, radius), ...]}
        """
        self.link_spheres = link_spheres

    def compute(self, state, robot, **kwargs) -> jnp.ndarray:
        """
        计算自碰撞避免能量

        计算所有link球体之间的距离，当距离小于安全阈值时施加惩罚
        """
        if not self.link_spheres:
            return jnp.array(0.0)

        total_energy = 0.0

        # 1. 运行FK获取所有link位姿
        link_poses_rel_root = robot.forward_kinematics(state.joint_values)
        T_world_base = state.to_base_pose_se3()

        # 2. 收集所有球体在世界坐标系中的位置
        world_spheres = []  # [(center_world, radius, link_name), ...]

        for link_name, spheres in self.link_spheres.items():
            try:
                link_idx = robot.links.names.index(link_name)
            except ValueError:
                continue  # 跳过不存在的link

            # 获取link在世界坐标系中的位姿
            T_root_link = jaxlie.SE3(link_poses_rel_root[link_idx])
            T_world_link = T_world_base @ T_root_link
            T_world_link_mat = T_world_link.as_matrix()

            # 转换球体中心到世界坐标系
            for center_local, radius in spheres:
                center_local_homo = jnp.append(jnp.array(center_local), 1.0)
                center_world = (T_world_link_mat @ center_local_homo)[:3]
                world_spheres.append((center_world, radius, link_name))

        # 3. 计算球体间距离并施加惩罚
        num_spheres = len(world_spheres)
        for i in range(num_spheres):
            for j in range(i + 1, num_spheres):
                center1, radius1, link1 = world_spheres[i]
                center2, radius2, link2 = world_spheres[j]

                # 跳过同一link内的球体（除非link有多个分离的部分）
                if link1 == link2:
                    continue

                # 计算球心距离
                distance = jnp.linalg.norm(center1 - center2)

                # 计算最小安全距离
                min_distance = radius1 + radius2 + self.margin

                # 惩罚函数：当距离小于阈值时施加二次惩罚
                if distance < min_distance:
                    violation = min_distance - distance
                    # 使用二次惩罚函数
                    energy_penalty = violation ** 2
                    total_energy += energy_penalty

        return jnp.array(total_energy * self.weight)

    def get_weight(self) -> float:
        return self.weight


class ManipulabilityEnergy(EnergyFunction):
    """
    可操作性能量：鼓励远离奇异位形
    
    E = -log(det(J @ J^T))
    其中 J 是 Jacobian 矩阵
    """
    
    def __init__(self, weight: float = 0.01):
        self.weight = weight
    
    def compute(self, state, robot, **kwargs) -> jnp.ndarray:
        """
        计算可操作性能量
        
        注意：pyroki 可能没有直接提供 Jacobian，
        需要使用 JAX 自动微分计算
        """
        # TODO: 实现 Jacobian 计算
        # 可以使用 jax.jacfwd 或 jax.jacrev
        return jnp.array(0.0)
    
    def get_weight(self) -> float:
        return self.weight


class JointLimitEnergy(EnergyFunction):
    """
    关节限位能量：软约束，防止关节超出限位
    
    E = Σ max(0, q - q_max)^2 + max(0, q_min - q)^2
    """
    
    def __init__(self, weight: float = 1.0, margin: float = 0.1):
        """
        Args:
            weight: 能量权重
            margin: 从限位的安全边距（弧度）
        """
        self.weight = weight
        self.margin = margin
    
    def compute(self, state, robot, **kwargs) -> jnp.ndarray:
        """计算关节限位能量"""
        q = state.joint_values
        q_min = robot.joints.lower_limits + self.margin
        q_max = robot.joints.upper_limits - self.margin
        
        # 超出上限的惩罚
        upper_violation = jnp.maximum(0.0, q - q_max)
        # 超出下限的惩罚
        lower_violation = jnp.maximum(0.0, q_min - q)
        
        energy = jnp.sum(upper_violation ** 2) + jnp.sum(lower_violation ** 2)
        
        return energy * self.weight
    
    def get_weight(self) -> float:
        return self.weight


class CompositeEnergy(EnergyFunction):
    """
    组合能量：多个能量项的加权和
    
    E_total = Σ w_i * E_i
    """
    
    def __init__(self, energy_functions: List[EnergyFunction]):
        """
        Args:
            energy_functions: 能量函数列表
        """
        self.energy_functions = energy_functions
    
    def compute(self, state, robot, **kwargs) -> jnp.ndarray:
        """计算所有能量的总和"""
        total = jnp.array(0.0)
        
        for energy_fn in self.energy_functions:
            total += energy_fn.compute(state, robot, **kwargs)
        
        return total
    
    def compute_detailed(self, state, robot, **kwargs) -> Dict[str, float]:
        """
        计算每个能量项的值（用于调试）
        
        Returns:
            {energy_name: value}
        """
        energies = {}
        
        for energy_fn in self.energy_functions:
            name = energy_fn.__class__.__name__
            value = float(energy_fn.compute(state, robot, **kwargs))
            energies[name] = value
        
        return energies
