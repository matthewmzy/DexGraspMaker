#!/usr/bin/env python3
"""
测试新的bounding box球体填充算法
"""

import sys
import os
sys.path.append('interactive_gripper_tool')

import numpy as np
import pyvista as pv
from sphere_fitting import SphereFitting
import trimesh

def test_bounding_box_sphere_packing():
    """测试新的bounding box球体填充算法"""

    print("测试新的bounding box球体填充算法")

    # 创建测试meshes
    # 1. 一个简单的盒子
    box_mesh = pv.Box(bounds=(-0.05, 0.05, -0.02, 0.02, -0.03, 0.03))

    # 2. 一个球体
    sphere_mesh = pv.Sphere(radius=0.025, center=(0.02, 0, 0))

    # 3. 一个圆柱体
    cylinder_mesh = pv.Cylinder(radius=0.015, height=0.08, center=(-0.02, 0, 0))

    test_meshes = [box_mesh, sphere_mesh, cylinder_mesh]

    print(f"测试meshes数量: {len(test_meshes)}")
    for i, mesh in enumerate(test_meshes):
        bounds = mesh.bounds
        size = (bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
        print(f"  Mesh {i+1}: 尺寸 {size}")

    # 应用新算法
    spheres = SphereFitting.bounding_box_sphere_packing(test_meshes)

    print(f"\n生成的球体数量: {len(spheres)}")
    for i, (center, radius) in enumerate(spheres):
        print(".3f")

    # 可视化
    plotter = pv.Plotter()
    plotter.set_background('white')

    # 添加原始meshes
    colors = ['red', 'green', 'blue']
    for i, mesh in enumerate(test_meshes):
        plotter.add_mesh(mesh, color=colors[i], opacity=0.7, label=f'Mesh {i+1}')

    # 添加球体
    for i, (center, radius) in enumerate(spheres):
        # 创建球体mesh并添加到plotter
        sphere_geom = pv.Sphere(radius=radius, center=center)
        plotter.add_mesh(sphere_geom, color='orange', opacity=0.5)
        plotter.add_point_labels([center], [f'S{i+1}'], font_size=12)

    # 计算并显示组合bounding box
    combined_bounds = None
    for mesh in test_meshes:
        bounds = mesh.bounds
        if combined_bounds is None:
            combined_bounds = bounds
        else:
            combined_bounds = (
                min(combined_bounds[0], bounds[0]),
                max(combined_bounds[1], bounds[1]),
                min(combined_bounds[2], bounds[2]),
                max(combined_bounds[3], bounds[3]),
                min(combined_bounds[4], bounds[4]),
                max(combined_bounds[5], bounds[5])
            )

    # 添加bounding box wireframe
    bbox_mesh = pv.Box(bounds=combined_bounds)
    plotter.add_mesh(bbox_mesh, style='wireframe', color='black', line_width=2, label='Combined BBox')

    plotter.add_legend()
    plotter.view_isometric()
    plotter.show()

    return spheres

if __name__ == '__main__':
    test_bounding_box_sphere_packing()