#!/usr/bin/env python3
"""
打印Shadow手各个link的体积
"""

import numpy as np
import pyvista as pv
import trimesh
import yourdfpy
import os
from typing import Dict


def load_hand_meshes_and_volumes(urdf_path: str) -> Dict[str, Dict]:
    """
    加载URDF并计算每个link的体积

    Args:
        urdf_path: URDF文件路径

    Returns:
        {link_name: {'mesh': mesh, 'volume': volume}} 字典
    """
    print(f"加载URDF: {urdf_path}")

    # 使用yourdfpy加载URDF
    urdf_obj = yourdfpy.URDF.load(urdf_path)

    # 使用pyroki创建robot来获取link名称
    import pyroki as pk
    robot = pk.Robot.from_urdf(urdf_obj)

    # 提取Link Meshes (用于 PyVista 可视化)
    trimesh_scene = urdf_obj.scene
    if trimesh_scene is None:
        raise ValueError("yourdfpy 未能加载场景")

    link_data = {}

    scene_graph = trimesh_scene.graph
    transform_graph = scene_graph.transforms
    all_node_data = scene_graph.transforms.node_data
    geometry_dict = trimesh_scene.geometry

    for link_name in robot.links.names:
        if link_name not in transform_graph.nodes:
            print(f"警告: Pyroki link '{link_name}' 不在 trimesh 场景图中。")
            continue

        link_meshes_trimesh = []
        # 查找作为此 link_name 子节点的所有 "visual" 节点
        child_nodes = [to_node for from_node, to_node in transform_graph.edge_data if from_node == link_name]
        for child_node_name in child_nodes:
            if child_node_name in all_node_data and "geometry" in all_node_data[child_node_name]:

                geom_key = all_node_data[child_node_name]["geometry"]
                if geom_key not in geometry_dict:
                    print(f"警告: 找不到 '{geom_key}' 的几何体。")
                    continue

                trimesh_geom = geometry_dict[geom_key]

                transform_matrix = scene_graph.get(child_node_name, link_name)[0]

                if hasattr(trimesh_geom, 'to_mesh'):
                    # 如果是 Box, Sphere, Cylinder，调用 .to_mesh()
                    trimesh_mesh = trimesh_geom.to_mesh()
                else:
                    trimesh_mesh = trimesh_geom.copy()

                trimesh_mesh.apply_transform(transform_matrix)
                link_meshes_trimesh.append(trimesh_mesh)

        if not link_meshes_trimesh:
            continue # 此 link 没有可视化 meshes

        # 将此 link 的所有 visual meshes 合并为一个
        if len(link_meshes_trimesh) > 1:
            combined_mesh = trimesh.util.concatenate(link_meshes_trimesh)
        else:
            combined_mesh = link_meshes_trimesh[0]

        # 转换为 PyVista 并计算体积
        pv_mesh = pv.wrap(combined_mesh)

        try:
            volume = pv_mesh.volume
        except Exception as e:
            print(f"计算link {link_name} 体积失败: {e}")
            volume = 0.0

        link_data[link_name] = {
            'mesh': pv_mesh,
            'volume': volume,
            'vertex_count': pv_mesh.n_points
        }

    print(f"处理了 {len(link_data)} 个link")
    return link_data


def main():
    # 测试用的URDF路径
    urdf_path = "/home/ubuntu/Documents/DexGraspMaker/test_assets/shadow/shadow_hand_right.urdf"

    if not os.path.exists(urdf_path):
        print(f"URDF文件不存在: {urdf_path}")
        return

    # 加载meshes和体积
    link_data = load_hand_meshes_and_volumes(urdf_path)

    # 打印体积信息
    print("\n" + "="*60)
    print("Shadow手各个link的体积统计")
    print("="*60)

    total_volume = 0.0
    total_vertices = 0

    # 按体积排序输出
    sorted_links = sorted(link_data.items(), key=lambda x: x[1]['volume'], reverse=True)

    print(f"{'Link Name':<15} {'Volume (cm³)':<15} {'Vertices':<10}")
    print("-" * 60)

    for link_name, data in sorted_links:
        volume = data['volume'] * 1_000_000  # 转换为立方厘米
        vertices = data['vertex_count']
        total_volume += data['volume']  # 保持原始单位用于总计
        total_vertices += vertices

        print(f"{link_name:<15} {volume:<15.2f} {vertices:<10}")

    print("-" * 60)
    total_volume_cm3 = total_volume * 1_000_000
    print(f"{'Total':<15} {total_volume_cm3:<15.2f} {total_vertices:<10}")
    print("="*60)

    # 找出体积为0的links
    zero_volume_links = [name for name, data in link_data.items() if data['volume'] <= 1e-8]
    if zero_volume_links:
        print(f"\n体积为0的links ({len(zero_volume_links)}个):")
        for link_name in zero_volume_links:
            print(f"  - {link_name}")
    else:
        print("\n没有体积为0的links")


if __name__ == "__main__":
    main()