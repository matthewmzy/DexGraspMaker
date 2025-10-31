#!/usr/bin/env python3
"""
测试脚本：可视化手部关键点采样

使用FPS（Farthest Point Sampling）算法在每个link上采样关键点
"""

import numpy as np
import pyvista as pv
import trimesh
import yourdfpy
import os
import json
from typing import Dict, List, Tuple


def farthest_point_sampling(points: np.ndarray, num_samples: int) -> np.ndarray:
    """
    使用FPS算法从点云中采样指定数量的点

    Args:
        points: (N, 3) 点云数组
        num_samples: 要采样的点数

    Returns:
        (num_samples, 3) 采样点数组
    """
    if len(points) <= num_samples:
        return points

    # 初始化
    selected_indices = []

    # 随机选择第一个点
    first_idx = np.random.randint(len(points))
    selected_indices.append(first_idx)

    # 计算所有点到已选点的距离
    distances = np.full(len(points), np.inf)

    for _ in range(num_samples - 1):
        # 更新距离：每个点到最近已选点的距离
        current_selected = points[selected_indices]
        diff = points[:, np.newaxis] - current_selected[np.newaxis, :]
        dist_to_selected = np.min(np.sum(diff ** 2, axis=2), axis=1)
        distances = np.minimum(distances, dist_to_selected)

        # 选择距离最远的点
        farthest_idx = np.argmax(distances)
        selected_indices.append(farthest_idx)

    return points[selected_indices]


def sample_keypoints_from_link_mesh(mesh: pv.PolyData, num_samples: int = 8) -> np.ndarray:
    """
    从link网格采样关键点

    Args:
        mesh: pyvista.PolyData网格
        num_samples: 采样点数

    Returns:
        (num_samples, 3) 关键点数组（局部坐标）
    """
    if mesh.n_points == 0:
        return np.array([])

    # 获取网格顶点
    points = mesh.points

    # 使用FPS采样
    sampled_points = farthest_point_sampling(points, num_samples)

    return sampled_points


def load_hand_meshes(urdf_path: str) -> Dict[str, pv.PolyData]:
    """
    加载URDF并提取link meshes（使用与data_manager相同的逻辑）

    Args:
        urdf_path: URDF文件路径

    Returns:
        {link_name: mesh} 字典
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

    link_meshes = {}

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

        # 转换为 PyVista 并存储
        pv_mesh = pv.wrap(combined_mesh)
        link_meshes[link_name] = pv_mesh

    print(f"提取了 {len(link_meshes)} 个link meshes")
    return link_meshes


def filter_links_by_volume(link_meshes: Dict[str, pv.PolyData]) -> Dict[str, pv.PolyData]:
    """
    过滤出体积大于0的links

    Args:
        link_meshes: 所有link meshes

    Returns:
        体积大于0的link meshes
    """
    filtered_meshes = {}

    for link_name, mesh in link_meshes.items():
        try:
            # 计算体积
            volume = mesh.volume
            if volume > 1e-8:  # 体积大于很小的阈值
                filtered_meshes[link_name] = mesh
                print(".6f")
            else:
                print(".6f")
        except Exception as e:
            print(f"计算link {link_name} 体积失败: {e}")

    print(f"过滤后剩余 {len(filtered_meshes)} 个有体积的links")
    return filtered_meshes


def generate_keypoints(link_meshes: Dict[str, pv.PolyData], num_samples: int = 8) -> Dict[str, np.ndarray]:
    """
    为每个link生成关键点（根据体积阶梯式采样）

    Args:
        link_meshes: link meshes字典
        num_samples: 默认采样点数（当不使用体积阶梯时）

    Returns:
        {link_name: keypoints} 关键点字典
    """
    keypoints = {}

    for link_name, mesh in link_meshes.items():
        # 根据体积决定采样点数
        try:
            volume_cm3 = mesh.volume * 1_000_000  # 转换为立方厘米

            if volume_cm3 < 10:
                samples_for_this_link = 6
            elif volume_cm3 < 100:
                samples_for_this_link = 8
            elif volume_cm3 < 1000:
                samples_for_this_link = 16
            else:  # volume >= 1000
                samples_for_this_link = 32

        except Exception as e:
            print(f"计算link {link_name} 体积失败，使用默认采样数: {e}")
            samples_for_this_link = num_samples

        sampled_points = sample_keypoints_from_link_mesh(mesh, samples_for_this_link)
        keypoints[link_name] = sampled_points
        print(f"Link {link_name}: 体积={volume_cm3:.2f}cm³, 采样了 {len(sampled_points)} 个关键点")

    return keypoints


def save_keypoints(keypoints: Dict[str, np.ndarray], output_path: str):
    """
    保存关键点到JSON文件

    Args:
        keypoints: 关键点字典
        output_path: 输出文件路径
    """
    # 转换为可序列化的格式
    serializable_keypoints = {}
    for link_name, points in keypoints.items():
        serializable_keypoints[link_name] = points.tolist()

    with open(output_path, 'w') as f:
        json.dump(serializable_keypoints, f, indent=2)

    print(f"关键点已保存到: {output_path}")


def load_keypoints(input_path: str) -> Dict[str, np.ndarray]:
    """
    从JSON文件加载关键点

    Args:
        input_path: 输入文件路径

    Returns:
        关键点字典
    """
    with open(input_path, 'r') as f:
        data = json.load(f)

    keypoints = {}
    for link_name, points_list in data.items():
        keypoints[link_name] = np.array(points_list)

    return keypoints


def visualize_keypoints(link_meshes: Dict[str, pv.PolyData], keypoints: Dict[str, np.ndarray]):
    """
    可视化关键点

    Args:
        link_meshes: link meshes
        keypoints: 关键点
    """
    print("开始可视化...")
    print(f"共有 {len(link_meshes)} 个meshes和 {len(keypoints)} 个关键点集合")

    # 打印一些统计信息
    total_keypoints = 0
    for link_name, points in keypoints.items():
        print(f"Link {link_name}: {len(points)} 个关键点")
        total_keypoints += len(points)

    print(f"总共采样了 {total_keypoints} 个关键点")

    # 检查关键点分布
    print("\n关键点分布检查:")
    for link_name, points in list(keypoints.items())[:3]:  # 只检查前3个
        print(f"Link {link_name}:")
        print(f"  范围 X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
        print(f"  范围 Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
        print(f"  范围 Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")

    # 创建plotter
    plotter = pv.Plotter(off_screen=True)  # 无GUI模式
    plotter.set_background('white')

    # 添加每个link的mesh和关键点
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    color_idx = 0

    for link_name, mesh in list(link_meshes.items())[:5]:  # 只显示前5个link
        color = colors[color_idx % len(colors)]
        color_idx += 1

        # 添加mesh（半透明）
        plotter.add_mesh(mesh, color=color, opacity=0.3, label=f'{link_name}_mesh')

        # 添加关键点
        if link_name in keypoints:
            points = keypoints[link_name]
            if len(points) > 0:
                # 创建点云
                point_cloud = pv.PolyData(points)
                plotter.add_mesh(point_cloud, color='red', point_size=8,
                               render_points_as_spheres=True, label=f'{link_name}_keypoints')

    # 保存截图而不是显示GUI
    screenshot_path = "/home/ubuntu/Documents/DexGraspMaker/keypoint_sampling_visualization.png"
    plotter.screenshot(screenshot_path)
    print(f"可视化截图已保存到: {screenshot_path}")

    plotter.close()


def main():
    # 测试用的URDF路径
    urdf_path = "/home/ubuntu/Documents/DexGraspMaker/test_assets/shadow/shadow_hand_right.urdf"

    if not os.path.exists(urdf_path):
        print(f"URDF文件不存在: {urdf_path}")
        return

    # 关键点文件路径
    keypoints_path = urdf_path.replace('.urdf', '_keypoints.json')

    # 1. 加载hand meshes
    link_meshes = load_hand_meshes(urdf_path)

    # 2. 过滤有体积的links
    volume_meshes = filter_links_by_volume(link_meshes)

    # 3. 生成关键点（根据体积阶梯式采样）
    if os.path.exists(keypoints_path):
        print(f"加载现有关键点文件: {keypoints_path}")
        keypoints = load_keypoints(keypoints_path)
    else:
        print("生成新的关键点（根据体积阶梯式采样）...")
        keypoints = generate_keypoints(volume_meshes)
        save_keypoints(keypoints, keypoints_path)

    # 4. 可视化
    print("开始可视化...")
    visualize_keypoints(volume_meshes, keypoints)


if __name__ == "__main__":
    main()