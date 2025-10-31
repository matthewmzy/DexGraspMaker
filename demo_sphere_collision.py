#!/usr/bin/env python3
"""
演示球体拟合算法和自碰撞检测能量项
"""

import sys
import os
sys.path.append('interactive_gripper_tool')

from sphere_fitting import SphereFitting, generate_link_spheres, load_link_spheres
import pyvista as pv
import numpy as np


def demo_sphere_fitting():
    """演示球体拟合算法"""
    print("=== 球体拟合算法演示 ===\n")

    # 创建测试网格
    sphere_mesh = pv.Sphere(radius=0.05, center=(0, 0, 0))
    box_mesh = pv.Box(bounds=(-0.03, 0.03, -0.02, 0.02, -0.04, 0.04))

    test_meshes = [
        ("球体网格", sphere_mesh),
        ("盒子网格", box_mesh)
    ]

    for name, mesh in test_meshes:
        print(f"处理 {name}:")
        print(f"  顶点数: {len(mesh.points)}")
        print(f"  体积: {mesh.volume * 1e6:.1f} cm³")

        # 测试不同方法
        methods = ['minimum', 'hierarchical', 'sample_based']
        for method in methods:
            num_spheres = 3 if method != 'minimum' else 1
            spheres = SphereFitting.fit_mesh_to_spheres(mesh, method, num_spheres=num_spheres, sample_ratio=0.5)
            print(f"  {method}: {len(spheres)} 个球体")
            for i, (center, radius) in enumerate(spheres):
                print(".3f")

        print()


def demo_collision_energy():
    """演示自碰撞检测能量计算"""
    print("=== 自碰撞检测能量演示 ===\n")

    # 创建两个相交的球体
    sphere1_center = np.array([0, 0, 0])
    sphere1_radius = 0.05

    sphere2_center = np.array([0.03, 0, 0])  # 轻微重叠
    sphere2_radius = 0.05

    # 计算距离
    distance = np.linalg.norm(sphere1_center - sphere2_center)
    min_distance = sphere1_radius + sphere2_radius
    margin = 0.005

    print(f"球体1: 中心{sphere1_center}, 半径{sphere1_radius}")
    print(f"球体2: 中心{sphere2_center}, 半径{sphere2_radius}")
    print(f"球心距离: {distance:.4f}")
    print(f"最小安全距离: {min_distance:.4f}")
    print(f"安全裕度: {margin}")

    # 计算惩罚
    if distance < min_distance + margin:
        violation = (min_distance + margin) - distance
        energy_penalty = violation ** 2
        print(f"碰撞违规: {violation:.4f}")
        print(f"能量惩罚: {energy_penalty:.6f}")
    else:
        print("无碰撞")

    print()


def demo_link_spheres():
    """演示link球体生成"""
    print("=== Link球体生成演示 ===\n")

    # 这里可以加载实际的URDF文件，但为了演示，我们创建模拟数据
    mock_links = {
        'forearm': pv.Cylinder(radius=0.03, height=0.2, center=(0, 0, 0.1)),
        'palm': pv.Box(bounds=(-0.02, 0.02, -0.01, 0.01, -0.03, 0.03)),
        'finger': pv.Sphere(radius=0.015, center=(0, 0, 0))
    }

    link_spheres = {}
    for link_name, mesh in mock_links.items():
        volume = mesh.volume * 1e6  # cm³
        num_spheres = 2 if volume < 50 else 4

        spheres = SphereFitting.fit_mesh_to_spheres(mesh, 'hierarchical', num_spheres)
        link_spheres[link_name] = spheres

        print(f"{link_name}: 体积={volume:.1f}cm³ -> {len(spheres)}个球体")

    print(f"\n总共生成 {sum(len(s) for s in link_spheres.values())} 个球体")


if __name__ == '__main__':
    demo_sphere_fitting()
    demo_collision_energy()
    demo_link_spheres()

    print("=== 总结 ===")
    print("1. 实现了三种球体拟合算法：最小包围球、层次分解、基于采样")
    print("2. 自碰撞检测使用球体间距离计算，施加二次惩罚")
    print("3. 基于link体积自适应确定球体数量")
    print("4. 球体数据自动保存/加载，与关键点类似")