#!/usr/bin/env python3
"""
Shadow手FK可视化脚本

这个脚本用于可视化Shadow手的正向运动学(FK)结果，包括mesh、关键点和拟合球体。

功能：
- 加载Shadow手URDF
- 计算正向运动学得到全局姿态
- 可视化所有link的mesh（使用open3d）
- 显示关键点（红色不透明）
- 显示拟合球体（橙色不透明）
- 支持关节角度调整

依赖：
- numpy, trimesh, yourdfpy, jax, jaxlie, pyroki, open3d

使用方法：
python visualize_hand_fk.py
python visualize_hand_fk.py --joint rh_FFJ1 0.5 --joint rh_FFJ2 0.3
"""

import sys
import os
import argparse
import json
from typing import Dict, List, Tuple

# 添加路径：项目根与第三方库
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)
sys.path.append(os.path.join(repo_root, 'thirdparty/pyroki/src'))

import numpy as np
import trimesh
import yourdfpy
import jax.numpy as jnp
import jaxlie
from scripts.sphere_fitting import load_link_spheres
import pyroki as pk
import open3d as o3d


def load_hand_with_fk(urdf_path: str, joint_values: Dict[str, float] = None) -> Tuple[Dict[str, trimesh.Trimesh], Dict[str, np.ndarray]]:
    """
    加载Shadow手，计算FK，应用全局变换到meshes

    Args:
        urdf_path: URDF文件路径
        joint_values: 关节值字典 {joint_name: value}

    Returns:
        (meshes_dict, poses_dict): meshes和全局位姿
    """
    print(f"加载URDF: {urdf_path}")

    # 1. 加载URDF和创建pyroki机器人
    urdf_obj = yourdfpy.URDF.load(urdf_path)
    robot = pk.Robot.from_urdf(urdf_obj)

    # 2. 设置关节值（默认零位姿）
    if joint_values is None:
        joint_values = {}
    actuated_names = urdf_obj.actuated_joint_names
    for joint_name in actuated_names:
        if joint_name not in joint_values:
            joint_values[joint_name] = 0.0

    # 3. 计算FK
    cfg_array = jnp.array([joint_values[name] for name in actuated_names])
    link_poses_rel_root = robot.forward_kinematics(cfg_array)

    # 4. 加载meshes并应用全局变换
    trimesh_scene = urdf_obj.scene
    if trimesh_scene is None:
        raise ValueError("无法加载URDF场景")

    meshes_dict = {}
    poses_dict = {}

    scene_graph = trimesh_scene.graph
    transform_graph = scene_graph.transforms
    all_node_data = scene_graph.transforms.node_data
    geometry_dict = trimesh_scene.geometry

    for link_name in scene_graph.nodes:
        if link_name == 'world':
            continue
        if link_name not in transform_graph.nodes:
            continue

        # 获取link的全局位姿
        if link_name in robot.links.names:
            link_idx = robot.links.names.index(link_name)
            T_world_link = jaxlie.SE3(link_poses_rel_root[link_idx])
            global_pose = np.array(T_world_link.as_matrix())
        else:
            # 对于不在pyroki中的links，使用单位矩阵
            global_pose = np.eye(4)

        poses_dict[link_name] = global_pose

        # 加载link的meshes
        link_meshes = []
        child_nodes = [to_node for from_node, to_node in transform_graph.edge_data if from_node == link_name]

        for child_node_name in child_nodes:
            if child_node_name in all_node_data and "geometry" in all_node_data[child_node_name]:
                geom_key = all_node_data[child_node_name]["geometry"]
                if geom_key not in geometry_dict:
                    continue

                trimesh_geom = geometry_dict[geom_key]
                local_transform = scene_graph.get(child_node_name, link_name)[0]

                if hasattr(trimesh_geom, 'to_mesh'):
                    trimesh_mesh = trimesh_geom.to_mesh()
                else:
                    trimesh_mesh = trimesh_geom.copy()

                # 应用局部变换 + 全局FK变换
                full_transform = global_pose @ local_transform
                trimesh_mesh.apply_transform(full_transform)
                link_meshes.append(trimesh_mesh)

        if not link_meshes:
            continue

        # 合并meshes
        if len(link_meshes) > 1:
            combined_mesh = trimesh.util.concatenate(link_meshes)
        else:
            combined_mesh = link_meshes[0]

        meshes_dict[link_name] = combined_mesh

    print(f"加载了 {len(meshes_dict)} 个link meshes，计算了 {len(poses_dict)} 个全局位姿")
    return meshes_dict, poses_dict


def load_keypoints(keypoints_path: str) -> Dict[str, np.ndarray]:
    """
    加载关键点数据

    Args:
        keypoints_path: 关键点文件路径

    Returns:
        keypoints_dict: {link_name: keypoints_array}
    """
    with open(keypoints_path, 'r') as f:
        data = json.load(f)

    keypoints_dict = {}
    for link_name, keypoints_list in data.items():
        keypoints_dict[link_name] = np.array(keypoints_list)

    print(f"加载了 {len(keypoints_dict)} 个link的关键点")
    return keypoints_dict


def visualize_hand(meshes_dict: Dict[str, trimesh.Trimesh],
                  keypoints_dict: Dict[str, np.ndarray],
                  spheres_dict: Dict[str, List[Tuple[np.ndarray, float]]],
                  poses_dict: Dict[str, np.ndarray] = None,
                  title: str = "Shadow Hand FK Visualization"):
    """
    可视化整个手、关键点和球体

    Args:
        meshes_dict: {link_name: mesh}
        keypoints_dict: {link_name: keypoints_array}
        spheres_dict: {link_name: [(center, radius), ...]}
        poses_dict: {link_name: 4x4 pose matrix} (可选)
        title: 图表标题
    """
    print("创建3D可视化...")

    # 创建open3d可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1200, height=800)

    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([0.9, 0.9, 0.9])  # 浅灰色背景
    render_option.light_on = True

    # 添加meshes - 所有都用半透明浅蓝色
    for link_name, mesh in meshes_dict.items():
        # 转换为open3d mesh
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

        # 设置半透明浅蓝色
        o3d_mesh.compute_vertex_normals()
        o3d_mesh.paint_uniform_color([0.7, 0.8, 1.0])  # 浅蓝色

        # 添加到可视化器
        vis.add_geometry(o3d_mesh)

    # 添加关键点 - 红色不透明，需要FK变换
    for link_name, keypoints in keypoints_dict.items():
        if link_name in poses_dict:
            link_pose = poses_dict[link_name]
            for keypoint in keypoints:
                # 将关键点从link局部坐标系变换到世界坐标系
                keypoint_homogeneous = np.append(keypoint, 1.0)  # 添加齐次坐标
                keypoint_world = (link_pose @ keypoint_homogeneous)[:3]  # 变换并取前3维
                
                # 创建小球体表示关键点
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.003)
                sphere.translate(keypoint_world)
                sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
                vis.add_geometry(sphere)

    # 添加碰撞球体 - 橙色不透明，需要FK变换
    for link_name, spheres in spheres_dict.items():
        if link_name in poses_dict:
            link_pose = poses_dict[link_name]
            for center, radius in spheres:
                # 将球心从link局部坐标系变换到世界坐标系
                center_homogeneous = np.append(center, 1.0)  # 添加齐次坐标
                center_world = (link_pose @ center_homogeneous)[:3]  # 变换并取前3维
                
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
                sphere.translate(center_world)
                sphere.paint_uniform_color([1.0, 0.65, 0.0])  # 橙色
                vis.add_geometry(sphere)

    # 设置相机位置
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0.0, 0.0, -1.0])
    ctr.set_lookat([0.0, 0.0, 0.0])
    ctr.set_up([0.0, 1.0, 0.0])

    # 显示可视化窗口
    print("显示3D可视化窗口...")
    vis.run()
    vis.destroy_window()
def main():
    parser = argparse.ArgumentParser(description='Shadow手FK可视化')
    parser.add_argument('--urdf', default='test_assets/shadow/shadow_hand_right.urdf',
                       help='URDF文件路径')
    parser.add_argument('--keypoints', default='test_assets/shadow/shadow_hand_right_keypoints.json',
                       help='关键点文件路径')
    parser.add_argument('--spheres', default='test_assets/shadow/shadow_hand_right_spheres.json',
                       help='球体文件路径')
    parser.add_argument('--joint', action='append', nargs=2, metavar=('name', 'value'),
                       help='设置关节角度 (可以多次使用)')

    args = parser.parse_args()

    # 解析关节角度
    joint_values = {}
    if args.joint:
        for joint_name, joint_value in args.joint:
            joint_values[joint_name] = float(joint_value)

    print("=== Shadow手FK可视化 ===")
    print(f"URDF: {args.urdf}")
    print(f"关键点: {args.keypoints}")
    print(f"球体: {args.spheres}")
    if joint_values:
        print(f"关节角度: {joint_values}")
    else:
        print("关节角度: 默认(全零)")

    try:
        # 1. 加载手和计算FK
        meshes_dict, poses_dict = load_hand_with_fk(args.urdf, joint_values)

        # 2. 加载关键点
        keypoints_dict = load_keypoints(args.keypoints)

        # 3. 加载球体
        spheres_dict = load_link_spheres(args.spheres)

        print("\n数据统计:")
        print(f"  Links: {len(meshes_dict)}")
        print(f"  总关键点数: {sum(len(kp) for kp in keypoints_dict.values())}")
        print(f"  总球体数: {sum(len(spheres) for spheres in spheres_dict.values())}")

        # 4. 可视化
        title = "Shadow Hand FK Visualization"
        if joint_values:
            joint_str = ", ".join([f"{k}={v:.2f}" for k, v in joint_values.items()])
            title += f" (Joints: {joint_str})"

        visualize_hand(meshes_dict, keypoints_dict, spheres_dict, poses_dict, title)

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())