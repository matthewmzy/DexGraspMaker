#!/usr/bin/env python3
"""
可视化球体拟合效果：显示Shadow手各link的原始mesh和拟合球体
"""

import sys
import os
sys.path.append('interactive_gripper_tool')

import numpy as np
import pyvista as pv
from sphere_fitting import load_link_spheres
import yourdfpy
import trimesh


def load_shadow_hand_meshes(urdf_path: str) -> dict:
    """
    加载Shadow手的link meshes

    Args:
        urdf_path: URDF文件路径

    Returns:
        {link_name: pyvista_mesh}
    """
    print(f"加载URDF: {urdf_path}")

    # 使用yourdfpy加载URDF
    urdf_obj = yourdfpy.URDF.load(urdf_path)
    trimesh_scene = urdf_obj.scene

    if trimesh_scene is None:
        raise ValueError("无法加载URDF场景")

    hand_links_mesh_dict = {}

    scene_graph = trimesh_scene.graph
    transform_graph = scene_graph.transforms
    all_node_data = scene_graph.transforms.node_data
    geometry_dict = trimesh_scene.geometry

    for link_name in scene_graph.nodes:
        if link_name == 'world':  # 跳过world节点
            continue
        if link_name not in transform_graph.nodes:
            continue

        link_meshes = []
        child_nodes = [to_node for from_node, to_node in transform_graph.edge_data if from_node == link_name]

        for child_node_name in child_nodes:
            if child_node_name in all_node_data and "geometry" in all_node_data[child_node_name]:
                geom_key = all_node_data[child_node_name]["geometry"]
                if geom_key not in geometry_dict:
                    continue

                trimesh_geom = geometry_dict[geom_key]
                transform_matrix = scene_graph.get(child_node_name, link_name)[0]

                if hasattr(trimesh_geom, 'to_mesh'):
                    trimesh_mesh = trimesh_geom.to_mesh()
                else:
                    trimesh_mesh = trimesh_geom.copy()

                trimesh_mesh.apply_transform(transform_matrix)
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
        hand_links_mesh_dict[link_name] = pv_mesh

    print(f"加载了 {len(hand_links_mesh_dict)} 个link meshes")
    return hand_links_mesh_dict


def visualize_link_spheres(link_name: str, mesh: pv.PolyData,
                          spheres: list, spheres_file: str, save_image: bool = True):
    """
    可视化单个link的mesh和球体拟合结果

    Args:
        link_name: link名称
        mesh: PyVista mesh
        spheres: [(center, radius), ...]
        spheres_file: 球体文件路径
        save_image: 是否保存图像而不是显示窗口
    """
    # 创建plotter
    plotter = pv.Plotter(off_screen=save_image, window_size=(1200, 800))
    plotter.set_background('white')

    # 添加原始mesh（半透明）
    plotter.add_mesh(mesh, color='lightblue', opacity=0.7, label='Original Mesh')

    # 添加拟合球体
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray']
    for i, (center, radius) in enumerate(spheres):
        color = colors[i % len(colors)]
        sphere_mesh = pv.Sphere(radius=radius, center=center)
        plotter.add_mesh(sphere_mesh, color=color, opacity=0.5,
                        label=f'Sphere {i+1} (r={radius:.3f})')

        # 添加球心点
        plotter.add_points(np.array([center]), color=color, point_size=10,
                          label=f'Center {i+1}')

    # 设置相机和视图
    plotter.view_isometric()
    plotter.add_legend()

    # 添加标题
    volume = mesh.volume * 1e6  # cm³
    title = f"Link: {link_name}\nVolume: {volume:.1f} cm³, Spheres: {len(spheres)}\nFile: {os.path.basename(spheres_file)}"
    plotter.add_text(title, position='upper_left', font_size=12)

    # 显示坐标轴
    plotter.add_axes()

    if save_image:
        # 保存图像
        image_file = f"sphere_fitting_{link_name}.png"
        plotter.screenshot(image_file)
        print(f"  保存到: {image_file}")
    else:
        # 显示
        plotter.show(title=f"Sphere Fitting: {link_name}")


def visualize_all_links(hand_meshes: dict, link_spheres: dict, spheres_file: str, save_image: bool = True):
    """
    可视化所有links的球体拟合结果（概览）
    """
    plotter = pv.Plotter(off_screen=save_image, window_size=(1400, 1000))
    plotter.set_background('white')

    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']

    total_spheres = 0

    for link_idx, (link_name, mesh) in enumerate(hand_meshes.items()):
        if link_name not in link_spheres:
            continue

        spheres = link_spheres[link_name]
        total_spheres += len(spheres)

        # 为每个link使用不同颜色
        link_color = colors[link_idx % len(colors)]

        # 添加mesh（非常透明）
        plotter.add_mesh(mesh, color=link_color, opacity=0.1, label=link_name)

        # 添加球体
        for sphere_idx, (center, radius) in enumerate(spheres):
            sphere_mesh = pv.Sphere(radius=radius, center=center)
            plotter.add_mesh(sphere_mesh, color=link_color, opacity=0.3)

            # 添加球心点
            plotter.add_points(np.array([center]), color=link_color, point_size=5)

    # 设置视图
    plotter.view_isometric()
    plotter.add_axes()

    # 添加统计信息
    title = f"Shadow Hand Sphere Fitting Overview\n{len(hand_meshes)} links, {total_spheres} spheres total\nFile: {os.path.basename(spheres_file)}"
    plotter.add_text(title, position='upper_left', font_size=12)

    if save_image:
        # 保存图像
        image_file = "shadow_hand_sphere_fitting.png"
        plotter.screenshot(image_file)
        print(f"可视化结果已保存到: {image_file}")
    else:
        plotter.show(title="Shadow Hand Sphere Fitting Overview")


def main():
    """主函数"""
    # 文件路径
    urdf_path = "test_assets/shadow/shadow_hand_right.urdf"
    spheres_file = "test_assets/shadow/shadow_hand_right_spheres.json"

    if not os.path.exists(urdf_path):
        print(f"错误: URDF文件不存在: {urdf_path}")
        return

    if not os.path.exists(spheres_file):
        print(f"错误: 球体文件不存在: {spheres_file}")
        print("请先运行应用程序生成球体数据")
        return

    try:
        # 加载数据
        print("加载Shadow手meshes...")
        hand_meshes = load_shadow_hand_meshes(urdf_path)

        print("加载球体数据...")
        link_spheres = load_link_spheres(spheres_file)

        # 统计信息
        total_spheres = sum(len(spheres) for spheres in link_spheres.values())
        print(f"\n统计信息:")
        print(f"- Links: {len(hand_meshes)}")
        print(f"- Spheres: {total_spheres}")
        print(f"- Average spheres per link: {total_spheres / len(link_spheres):.1f}")

        # 显示概览
        print("\n显示概览...")
        visualize_all_links(hand_meshes, link_spheres, spheres_file, save_image=True)

        # 生成所有link的详细图像
        print("\n生成所有link的详细图像...")
        generated_count = 0
        for link_name in hand_meshes.keys():
            if link_name in link_spheres:
                try:
                    mesh = hand_meshes[link_name]
                    spheres = link_spheres[link_name]
                    volume = mesh.volume * 1e6

                    print(f"生成 {link_name}: 体积={volume:.1f}cm³, 球体={len(spheres)}")
                    visualize_link_spheres(link_name, mesh, spheres, spheres_file, save_image=True)
                    generated_count += 1
                except Exception as e:
                    print(f"  跳过 {link_name}: {e}")
            else:
                print(f"  跳过 {link_name}: 没有球体数据")

        print(f"成功生成 {generated_count} 个link的可视化")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()