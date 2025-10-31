#!/usr/bin/env python3
"""
可视化Shadow手FK和球体拟合效果
加载URDF，计算每个link的全局坐标，搭建整个手，然后添加球体
"""

import sys
import os
sys.path.append('interactive_gripper_tool')

import numpy as np
import pyvista as pv
import yourdfpy
import trimesh
import jax.numpy as jnp
import jaxlie
from typing import Dict, List, Tuple
from sphere_fitting import generate_link_spheres
import pyroki as pk


def load_shadow_hand_with_fk(urdf_path: str, joint_values: Dict[str, float] = None) -> Tuple[Dict[str, pv.PolyData], Dict[str, np.ndarray]]:
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

        # 转换为PyVista
        pv_mesh = pv.wrap(combined_mesh)
        meshes_dict[link_name] = pv_mesh

    print(f"加载了 {len(meshes_dict)} 个link meshes，计算了 {len(poses_dict)} 个全局位姿")
    return meshes_dict, poses_dict


def visualize_hand_with_spheres(meshes_dict: Dict[str, pv.PolyData],
                               poses_dict: Dict[str, np.ndarray],
                               spheres_dict: Dict[str, List[Tuple[np.ndarray, float]]],
                               save_image: bool = False):
    """
    可视化整个手和球体拟合

    Args:
        meshes_dict: {link_name: mesh}
        poses_dict: {link_name: 4x4 pose matrix}
        spheres_dict: {link_name: [(center, radius), ...]}
        save_image: 是否保存图像
    """
    # 创建plotter
    plotter = pv.Plotter(off_screen=save_image, window_size=(1200, 800))
    plotter.set_background('white')

    # 设置相机
    plotter.camera_position = [(0.5, 0.5, 0.5), (0, 0, 0), (0, 0, 1)]

    # 颜色映射
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink',
              'lightcyan', 'lavender', 'honeydew', 'aliceblue', 'beige']

    # 添加meshes
    for i, (link_name, mesh) in enumerate(meshes_dict.items()):
        color = colors[i % len(colors)]
        plotter.add_mesh(mesh, color=color, opacity=0.5, label=f'{link_name} (mesh)')  # 手半透明

    # 添加球体
    sphere_count = 0
    for link_name, spheres in spheres_dict.items():
        for center, radius in spheres:
            # 创建球体mesh
            sphere_mesh = pv.Sphere(radius=radius, center=center)
            plotter.add_mesh(sphere_mesh, color='red', opacity=1.0, label=f'{link_name} spheres' if sphere_count == 0 else None)  # 球完全不透明
            sphere_count += 1

    # 添加坐标轴
    plotter.add_axes()

    # 添加图例
    plotter.add_legend()

    if save_image:
        plotter.screenshot('shadow_hand_with_spheres.png')
        print("图像已保存为 shadow_hand_with_spheres.png")
    else:
        plotter.show()


def main():
    """主函数"""
    # Shadow手URDF路径
    urdf_path = "test_assets/shadow/shadow_hand_right.urdf"

    if not os.path.exists(urdf_path):
        print(f"错误: 找不到URDF文件 {urdf_path}")
        return

    try:
        # 1. 加载URDF
        print("加载URDF...")
        urdf_obj = yourdfpy.URDF.load(urdf_path)

        # 2. 加载手并计算FK
        print("加载Shadow手并计算FK...")
        meshes_dict, poses_dict = load_shadow_hand_with_fk(urdf_path)

        # 3. 生成球体拟合
        print("生成球体拟合...")
        spheres_dict = generate_link_spheres(meshes_dict, method='bounding_box_packing', urdf_obj=urdf_obj)

        # 3. 统计信息
        total_spheres = sum(len(spheres) for spheres in spheres_dict.values())
        print(f"\n统计信息:")
        print(f"  Links: {len(meshes_dict)}")
        print(f"  总球体数: {total_spheres}")

        for link_name, spheres in spheres_dict.items():
            volume = meshes_dict[link_name].volume * 1e6  # cm³
            print(f"  {link_name}: 体积={volume:.1f}cm³, {len(spheres)}个球体")

        # 4. 可视化
        print("\n开始可视化...")
        visualize_hand_with_spheres(meshes_dict, poses_dict, spheres_dict, save_image=False)

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()