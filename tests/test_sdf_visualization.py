#!/usr/bin/env python3
"""
SDF可视化测试脚本

测试PenetrationAvoidanceEnergy的距离场预计算和可视化
使用test_assets/objects/Mug.obj作为测试物体
"""

import numpy as np
import pytest
pytestmark = pytest.mark.skip(reason="Visualization/interactive SDF test; run manually")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import trimesh
import pyvista as pv
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.optimization.energy_functions import PenetrationAvoidanceEnergy

def visualize_sdf_slices(energy, mesh, num_slices=5):
    """
    可视化距离场的2D切片

    Args:
        energy: PenetrationAvoidanceEnergy实例
        mesh: trimesh.Trimesh对象
        num_slices: 切片数量
    """
    if energy.distance_field is None:
        print("错误：距离场未预计算")
        return

    field = energy.distance_field
    bounds = energy.field_bounds

    # 创建图形
    fig, axes = plt.subplots(2, num_slices, figsize=(15, 6))
    fig.suptitle('SDF Distance Field Visualization - Mug.obj', fontsize=16)

    # X方向切片
    for i in range(num_slices):
        slice_idx = i * (field.shape[0] - 1) // (num_slices - 1)
        slice_data = field[slice_idx, :, :]

        # 计算实际坐标
        y_coords = np.linspace(bounds[0, 1], bounds[1, 1], field.shape[1])
        z_coords = np.linspace(bounds[0, 2], bounds[1, 2], field.shape[2])
        Y, Z = np.meshgrid(y_coords, z_coords)

        # 上排：距离场
        im1 = axes[0, i].contourf(Y, Z, slice_data.T, levels=20, cmap='RdYlBu_r')
        axes[0, i].set_title(f'X-slice {slice_idx}/{field.shape[0]-1}')
        axes[0, i].set_xlabel('Y')
        axes[0, i].set_ylabel('Z')
        plt.colorbar(im1, ax=axes[0, i])

        # 下排：二值化（内部/外部）
        binary_slice = (slice_data > 0).astype(float)
        im2 = axes[1, i].contourf(Y, Z, binary_slice.T, levels=[-0.5, 0.5, 1.5], colors=['blue', 'red'])
        axes[1, i].set_title(f'Binary (X-slice {slice_idx})')
        axes[1, i].set_xlabel('Y')
        axes[1, i].set_ylabel('Z')

    plt.tight_layout()
    plt.show()

def visualize_mesh_and_sdf(mesh, energy, num_points=1000):
    """
    3D可视化网格和距离场采样点

    Args:
        mesh: trimesh.Trimesh对象
        energy: PenetrationAvoidanceEnergy实例
        num_points: 采样点数量
    """
    if energy.distance_field is None:
        print("错误：距离场未预计算")
        return

    # 创建pyvista plotter
    plotter = pv.Plotter(shape=(1, 2))

    # 左侧：原始网格
    plotter.subplot(0, 0)
    pv_mesh = pv.wrap(mesh)
    plotter.add_mesh(pv_mesh, color='lightblue', opacity=0.7)
    plotter.add_text("Original Mesh", font_size=12)

    # 右侧：距离场可视化
    plotter.subplot(0, 1)

    # 生成采样点
    bounds = energy.field_bounds
    x = np.random.uniform(bounds[0, 0], bounds[1, 0], num_points)
    y = np.random.uniform(bounds[0, 1], bounds[1, 1], num_points)
    z = np.random.uniform(bounds[0, 2], bounds[1, 2], num_points)
    sample_points = np.column_stack([x, y, z])

    # 查询距离
    import jax.numpy as jnp
    distances = energy._query_distance_field(jnp.array(sample_points))

    # 创建点云
    point_cloud = pv.PolyData(sample_points)

    # 根据距离着色：红色=内部，蓝色=外部
    colors = np.zeros((num_points, 3))
    colors[distances > 0] = [1, 0, 0]  # 红色：内部
    colors[distances <= 0] = [0, 0, 1]  # 蓝色：外部

    plotter.add_points(point_cloud, scalars=colors, rgb=True, point_size=3)
    plotter.add_text("SDF Sampling (Red=Inside, Blue=Outside)", font_size=12)

    plotter.show()

def test_sdf_accuracy(mesh, energy, num_test_points=100):
    """
    测试SDF精度的定量评估

    Args:
        mesh: trimesh.Trimesh对象
        energy: PenetrationAvoidanceEnergy实例
        num_test_points: 测试点数量
    """
    print("=== SDF精度测试 ===")

    # 生成测试点
    bounds = mesh.bounds
    margin = 0.02  # 在边界框外2cm范围内测试

    test_bounds = np.array([
        bounds[0] - margin,
        bounds[1] + margin
    ])

    np.random.seed(42)  # 固定随机种子
    test_points = np.random.uniform(
        test_bounds[0],
        test_bounds[1],
        (num_test_points, 3)
    )

    # 使用trimesh计算真实距离
    true_distances = trimesh.proximity.signed_distance(mesh, test_points)

    # 使用我们的SDF查询距离
    import jax.numpy as jnp
    predicted_distances = energy._query_distance_field(jnp.array(test_points))

    # 计算误差
    errors = predicted_distances - true_distances
    abs_errors = np.abs(errors)

    print(f"测试点数量: {num_test_points}")
    print(f"平均误差: {errors.mean():.4f}")
    print(f"最大绝对误差: {abs_errors.max():.4f}")
    print(f"95%分位数误差: {np.percentile(abs_errors, 95):.4f}")
    print(f"标准差: {errors.std():.4f}")
    print(f"RMSE: {np.sqrt(np.mean(errors**2)):.4f}")

    # 分析不同区域的误差
    inside_mask = true_distances > 0
    outside_mask = true_distances <= 0

    if np.any(inside_mask):
        inside_errors = abs_errors[inside_mask]
        print(f"内部点误差 - 平均: {inside_errors.mean():.4f}, 最大: {inside_errors.max():.4f}")

    if np.any(outside_mask):
        outside_errors = abs_errors[outside_mask]
        print(f"外部点误差 - 平均: {outside_errors.mean():.4f}, 最大: {outside_errors.max():.4f}")
    # 可视化误差分布
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.hist(errors, bins=50, alpha=0.7)
    plt.xlabel('Error (Predicted - True)')
    plt.ylabel('Count')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.scatter(true_distances, predicted_distances, alpha=0.6, s=10)
    plt.plot([-0.05, 0.05], [-0.05, 0.05], 'r--', linewidth=2)
    plt.xlabel('True Distance')
    plt.ylabel('Predicted Distance')
    plt.title('Predicted vs True')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')

    plt.subplot(1, 3, 3)
    plt.scatter(true_distances, abs_errors, alpha=0.6, s=10)
    plt.xlabel('True Distance')
    plt.ylabel('Absolute Error')
    plt.title('Error vs True Distance')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/ubuntu/Documents/DexGraspMaker/sdf_analysis.png', dpi=150, bbox_inches='tight')
    print("误差分析图已保存到: sdf_analysis.png")
    plt.show()
    """主函数"""
    print("=== SDF可视化测试 - Mug.obj ===")

    # 加载测试物体
    mug_path = "/home/ubuntu/Documents/DexGraspMaker/test_assets/objects/Mug.obj"
    print(f"加载网格: {mug_path}")

    try:
        mesh = trimesh.load(mug_path)
        # 缩放从毫米到米
        mesh.apply_scale(0.001)
        print(f"网格加载成功: {len(mesh.vertices)}顶点, {len(mesh.faces)}面")
        print(f"缩放后边界框: {mesh.bounds}")
    except Exception as e:
        print(f"加载网格失败: {e}")
        return

def main():
    """主函数"""
    print("=== SDF可视化测试 - Mug.obj ===")

    # 加载测试物体
    mug_path = "/home/ubuntu/Documents/DexGraspMaker/test_assets/objects/Mug.obj"
    print(f"加载网格: {mug_path}")

    try:
        mesh = trimesh.load(mug_path)
        # 缩放从毫米到米
        mesh.apply_scale(0.001)
        print(f"网格加载成功: {len(mesh.vertices)}顶点, {len(mesh.faces)}面")
        print(f"缩放后边界框: {mesh.bounds}")
    except Exception as e:
        print(f"加载网格失败: {e}")
        return

    # 创建穿透避免能量
    energy = PenetrationAvoidanceEnergy(
        weight=1.0,
        margin=0.001,  # 1mm
        resolution=0.005  # 5mm分辨率
    )

    # 预计算距离场
    print("\n预计算距离场...")
    energy.precompute_distance_field(mesh)

    if energy.distance_field is None:
        print("距离场预计算失败")
        return

    print(f"距离场形状: {energy.field_shape}")
    print(f"距离范围: [{energy.distance_field.min():.4f}, {energy.distance_field.max():.4f}]")

    # 测试精度
    test_sdf_accuracy(mesh, energy, num_test_points=1000)

    # 可视化切片
    print("\n生成SDF切片可视化...")
    visualize_sdf_slices(energy, mesh, num_slices=5)

    # 3D可视化
    print("\n生成3D可视化...")
    visualize_mesh_and_sdf(mesh, energy, num_points=5000)

if __name__ == "__main__":
    main()