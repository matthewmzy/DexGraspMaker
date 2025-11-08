#!/usr/bin/env python3
"""测试 SDF 缓存机制。

目标：
1. 首次生成距离场写入缓存。
2. 第二次相同参数命中缓存且结果一致。

使用 test_assets/objects/Mug.obj 作为测试物体。
"""
import os
import time
import numpy as np
import pytest
import trimesh

from scripts.optimization.energy_functions import PenetrationAvoidanceEnergy

@pytest.mark.order(1)
def test_sdf_cache_roundtrip(tmp_path):
    """验证两次预计算：第二次直接命中缓存且数据一致。"""
    # 物体路径
    mug_path = os.path.join(os.path.dirname(__file__), '..', 'test_assets', 'objects', 'Mug.obj')
    mug_path = os.path.abspath(mug_path)
    assert os.path.exists(mug_path), f"测试物体不存在: {mug_path}"

    # 加载并缩放到米
    mesh = trimesh.load(mug_path)
    mesh.apply_scale(0.001)  # mm -> m

    # 使用临时目录作为缓存，避免污染真实缓存
    cache_dir = tmp_path / 'sdf_cache'

    # 第一次：构建能量实例并生成
    energy1 = PenetrationAvoidanceEnergy(
        weight=1.0,
        margin=0.0,
        resolution=0.005,  # 5mm 分辨率
        cache_enabled=True,
        cache_dir=str(cache_dir),
        max_cells_per_dim=20,
        padding_ratio=0.05,
    )
    t0 = time.time()
    energy1.precompute_distance_field(mesh, mesh_source=mug_path)
    t1 = time.time()
    first_duration = t1 - t0

    assert energy1.distance_field is not None, "第一次预计算失败，distance_field 为空"
    assert energy1.field_bounds is not None, "第一次预计算失败，field_bounds 为空"
    assert energy1.field_shape is not None, "第一次预计算失败，field_shape 为空"

    # 第二次：新实例应命中缓存
    energy2 = PenetrationAvoidanceEnergy(
        weight=1.0,
        margin=0.0,
        resolution=0.005,
        cache_enabled=True,
        cache_dir=str(cache_dir),
        max_cells_per_dim=20,
        padding_ratio=0.05,
    )
    t2 = time.time()
    energy2.precompute_distance_field(mesh, mesh_source=mug_path)
    t3 = time.time()
    second_duration = t3 - t2

    assert energy2.distance_field is not None, "第二次预计算/加载失败"

    # 数据一致性检查
    assert energy1.field_shape == energy2.field_shape, "缓存加载后的 shape 不一致"
    assert np.allclose(energy1.field_bounds, energy2.field_bounds), "缓存加载后的 bounds 不一致"
    assert np.allclose(energy1.distance_field, energy2.distance_field), "缓存加载后的 distance_field 不一致"

    # 性能检查（第二次应显著更快），设置一个宽松阈值
    # 如果第一轮非常快，则不强制
    if first_duration > 0.05:  # 仅当第一次超过50ms才做对比
        assert second_duration < first_duration * 0.5, (
            f"缓存命中未显著加速: first={first_duration:.4f}s second={second_duration:.4f}s"
        )

    print(f"SDF缓存测试: 首次 {first_duration:.4f}s, 二次 {second_duration:.4f}s (shape={energy1.field_shape})")

if __name__ == '__main__':
    pytest.main([__file__])
