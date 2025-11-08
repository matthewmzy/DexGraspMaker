# sphere_fitting.py (migrated)
"""
球体拟合算法集合，用于将3D网格近似为球体集合

用于机器人碰撞检测和自碰撞避免
"""

import numpy as np
import pyvista as pv
from typing import List, Tuple, Dict
import json
import os


class SphereFitting:
    """
    球体拟合算法集合
    """

    @staticmethod
    def minimum_bounding_sphere(points: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        计算点集的最小包围球 (Ritter's algorithm)

        Args:
            points: (N, 3) 点集

        Returns:
            (center, radius): 球心和半径
        """
        if len(points) == 0:
            return np.array([0, 0, 0]), 0.0

        if len(points) == 1:
            return points[0], 0.0

        # 初始化：取前两个点作为直径
        center = (points[0] + points[1]) / 2
        radius = np.linalg.norm(points[0] - points[1]) / 2

        # Ritter's algorithm
        for i in range(2, len(points)):
            dist = np.linalg.norm(points[i] - center)
            if dist > radius:
                # 点在球外，需要扩展球体
                center, radius = SphereFitting._expand_sphere(center, radius, points[i])

        return center, radius

    @staticmethod
    def _expand_sphere(center: np.ndarray, radius: float, point: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        扩展球体以包含新点
        """
        dist = np.linalg.norm(point - center)
        if dist <= radius:
            return center, radius

        # 计算新球体
        direction = (point - center) / dist
        new_center = center + direction * (dist - radius) / 2
        new_radius = (dist + radius) / 2

        return new_center, new_radius

    @staticmethod
    def hierarchical_sphere_decomposition(mesh: pv.PolyData, max_spheres: int = 8,
                                        min_volume_ratio: float = 0.1) -> List[Tuple[np.ndarray, float]]:
        """
        层次球体分解：递归地将网格分解为球体集合（确保球体在mesh内部）

        使用简化的方法：基于mesh内部点的采样

        Args:
            mesh: PyVista网格
            max_spheres: 最大球体数量
            min_volume_ratio: 最小体积比例阈值

        Returns:
            spheres: [(center, radius), ...]
        """
        spheres = []

        # 使用mesh的点作为球心候选
        points = mesh.points

        # 选择候选点数量
        num_candidates = min(max_spheres, max(1, len(points) // 20))
        if num_candidates < 1:
            num_candidates = 1

        # 随机选择候选点
        np.random.seed(42)
        candidate_indices = np.random.choice(len(points), num_candidates, replace=False)
        candidate_centers = points[candidate_indices]

        for center in candidate_centers:
            # 为每个候选中心计算最大内接球半径
            radius = SphereFitting._compute_max_inner_radius(mesh, center)
            if radius > 0.001:  # 只保留有意义的球体
                spheres.append((center, radius))

        # 如果没有生成足够的球体，使用mesh中心
        if not spheres:
            center = np.array(mesh.center)
            radius = min(mesh.bounds[1] - mesh.bounds[0],
                        mesh.bounds[3] - mesh.bounds[2],
                        mesh.bounds[5] - mesh.bounds[4]) * 0.1
            spheres.append((center, radius))

        return spheres[:max_spheres]  # 限制数量

    @staticmethod
    def _contract_sphere_to_mesh(mesh: pv.PolyData, center: np.ndarray, max_radius: float,
                                num_samples: int = 50) -> float:
        """
        收缩球体半径以确保球体完全在mesh内部

        Args:
            mesh: PyVista网格
            center: 球心
            max_radius: 最大允许半径
            num_samples: 采样点数量

        Returns:
            收缩后的半径
        """
        # 在球面上采样点
        phi = np.random.uniform(0, 2*np.pi, num_samples)
        theta = np.random.uniform(0, np.pi, num_samples)

        x = center[0] + max_radius * np.sin(theta) * np.cos(phi)
        y = center[1] + max_radius * np.sin(theta) * np.sin(phi)
        z = center[2] + max_radius * np.cos(theta)

        surface_points = np.column_stack([x, y, z])

        # 计算每个表面点到mesh的最小距离
        distances = []
        for point in surface_points:
            # 计算点到mesh的最小距离
            dist = SphereFitting._point_to_mesh_distance(point, mesh)
            distances.append(dist)

        distances = np.array(distances)

        # 找到最小距离（负值表示在mesh内部，正值表示在外部）
        min_distance = np.min(distances)

        if min_distance >= 0:
            # 球体完全在mesh外部，需要缩小
            contracted_radius = max_radius * 0.5  # 保守缩小
        else:
            # 球体部分在mesh内部，收缩到安全距离
            safe_distance = -min_distance * 0.8  # 留一些安全裕度
            contracted_radius = min(max_radius + safe_distance, max_radius)

        return max(contracted_radius, max_radius * 0.1)  # 确保最小半径

    @staticmethod
    def _point_to_mesh_distance(point: np.ndarray, mesh: pv.PolyData) -> float:
        """
        计算点到mesh的最小距离

        Returns:
            距离（正值表示在mesh外部，负值表示在mesh内部）
        """
        # 使用PyVista的距离计算
        try:
            # 计算点到mesh的距离
            distances = []
            for cell in mesh.cell_centers().points:
                dist = np.linalg.norm(point - cell)
                distances.append(dist)

            # 找到最小距离
            min_dist = min(distances)

            # 简单的方法：检查点是否在mesh边界框内
            bounds = mesh.bounds
            in_bounds = (bounds[0] <= point[0] <= bounds[1] and
                        bounds[2] <= point[1] <= bounds[3] and
                        bounds[4] <= point[2] <= bounds[5])

            if in_bounds:
                return -min_dist  # 在内部
            else:
                return min_dist   # 在外部

        except:
            # 后备方法：使用边界框距离
            bounds = mesh.bounds
            center = np.array([
                (bounds[0] + bounds[1]) / 2,
                (bounds[2] + bounds[3]) / 2,
                (bounds[4] + bounds[5]) / 2
            ])

            half_size = np.array([
                (bounds[1] - bounds[0]) / 2,
                (bounds[3] - bounds[2]) / 2,
                (bounds[5] - bounds[4]) / 2
            ])

            diff = point - center
            dist_to_box = np.maximum(np.abs(diff) - half_size, 0)
            distance = np.linalg.norm(dist_to_box)

            # 如果在边界框内，返回负距离
            if np.all(np.abs(diff) <= half_size):
                return -distance
            else:
                return distance

    @staticmethod
    def _recursive_decompose_conservative(mesh: pv.PolyData, spheres: List[Tuple[np.ndarray, float]],
                                        max_spheres: int) -> List[Tuple[np.ndarray, float]]:
        """
        保守的递归分解，确保所有球体都在mesh内部
        """
        if len(spheres) >= max_spheres:
            return spheres

        # 找到最大的球体进行分解
        largest_idx = np.argmax([r for _, r in spheres])
        center, radius = spheres[largest_idx]

        # 计算分割方向（使用mesh的主方向）
        points = mesh.points
        if len(points) > 10:
            # 使用PCA找到主方向
            centered = points - np.mean(points, axis=0)
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            main_axis = eigenvectors[:, -1]  # 最大特征向量
        else:
            main_axis = np.array([1, 0, 0])  # 默认X轴

        # 沿主方向分割，但保持较小的重叠
        offset = main_axis * radius * 0.3  # 较小的重叠

        new_center1 = center - offset
        new_center2 = center + offset

        # 为新球体计算合适的半径
        new_radius1 = SphereFitting._contract_sphere_to_mesh(mesh, new_center1, radius)
        new_radius2 = SphereFitting._contract_sphere_to_mesh(mesh, new_center2, radius)

        # 只有当新球体有足够大小时才添加
        min_radius_threshold = radius * 0.2

        # 替换原来的球体
        spheres[largest_idx] = (new_center1, new_radius1)

        if new_radius2 > min_radius_threshold:
            spheres.append((new_center2, new_radius2))

        return spheres

    @staticmethod
    def sample_based_fitting(mesh: pv.PolyData, num_spheres: int = 4,
                           sample_ratio: float = 0.1) -> List[Tuple[np.ndarray, float]]:
        """
        基于表面采样的球体拟合（确保球体在mesh内部）

        Args:
            mesh: PyVista网格
            num_spheres: 球体数量
            sample_ratio: 采样比例

        Returns:
            spheres: [(center, radius), ...]
        """
        spheres = []

        # 从网格表面采样点
        num_samples = max(10, int(len(mesh.points) * sample_ratio))
        num_samples = min(num_samples, len(mesh.points))  # 确保不超过可用点数

        if num_samples < num_spheres:
            num_samples = max(num_spheres, len(mesh.points))

        sampled_points = mesh.points[np.random.choice(len(mesh.points), num_samples, replace=num_samples > len(mesh.points))]

        # 简化的方法：随机选择种子点
        np.random.seed(42)  # 确保可重现
        indices = np.random.choice(len(sampled_points), num_spheres, replace=False)
        seed_points = sampled_points[indices]

        # 为每个种子点创建球体
        for seed in seed_points:
            # 计算种子点附近的局部区域
            distances = np.linalg.norm(sampled_points - seed, axis=1)
            # 取最近的点来确定局部区域大小
            n_local = max(5, len(sampled_points) // (num_spheres * 5))
            local_indices = np.argsort(distances)[:n_local]
            local_points = sampled_points[local_indices]

            # 计算局部包围球
            local_center, local_radius = SphereFitting.minimum_bounding_sphere(local_points)

            # 收缩到mesh内部
            contracted_radius = SphereFitting._compute_max_inner_radius(mesh, local_center)

            if contracted_radius > 0.001:
                spheres.append((local_center, contracted_radius))

        # 如果没有生成足够的球体，使用mesh中心
        if not spheres:
            center = np.array(mesh.center)
            radius = min(mesh.bounds[1] - mesh.bounds[0],
                        mesh.bounds[3] - mesh.bounds[2],
                        mesh.bounds[5] - mesh.bounds[4]) * 0.1
            spheres.append((center, radius))

        return spheres[:num_spheres]  # 限制数量

    @staticmethod
    def bounding_box_sphere_packing(meshes) -> List[Tuple[np.ndarray, float]]:
        """
        基于bounding box的球体填充算法：用于关节自碰撞避免能量项

        算法步骤：
        1. 合并所有mesh的bounding box得到组合bounding box
        2. 球半径 = 最短边长度 / 2
        3. 沿最长边方向居中放置尽可能多的球体

        Args:
            meshes: link的所有mesh列表，或单个PyVista mesh

        Returns:
            spheres: [(center, radius), ...]
        """
        # 处理单个mesh的情况
        if isinstance(meshes, pv.PolyData):
            meshes = [meshes]

        if not meshes:
            return []

        # 步骤1：计算组合bounding box
        combined_bounds = None
        for mesh in meshes:
            bounds = mesh.bounds
            if combined_bounds is None:
                combined_bounds = bounds
            else:
                combined_bounds = (
                    min(combined_bounds[0], bounds[0]),  # x_min
                    max(combined_bounds[1], bounds[1]),  # x_max
                    min(combined_bounds[2], bounds[2]),  # y_min
                    max(combined_bounds[3], bounds[3]),  # y_max
                    min(combined_bounds[4], bounds[4]),  # z_min
                    max(combined_bounds[5], bounds[5])   # z_max
                )

        # 计算各维度长度
        x_length = combined_bounds[1] - combined_bounds[0]
        y_length = combined_bounds[3] - combined_bounds[2]
        z_length = combined_bounds[5] - combined_bounds[4]

        lengths = [x_length, y_length, z_length]
        dimensions = ['x', 'y', 'z']

        # 找到最短边和最长边
        min_length = min(lengths)
        max_length = max(lengths)
        min_idx = lengths.index(min_length)
        max_idx = lengths.index(max_length)

        # 步骤2：球半径 = 最短边 / 2
        radius = min_length / 2.0

        # 步骤3：沿最长边方向居中放置球体
        # 计算最多能放多少个球（考虑球直径和边界）
        sphere_diameter = 2 * radius
        max_spheres_in_line = max(1, int(max_length / sphere_diameter))

        # 确保球体不会超出边界：调整数量
        while max_spheres_in_line > 1:
            total_length_needed = (max_spheres_in_line - 1) * sphere_diameter + sphere_diameter  # 包括两端球的直径
            if total_length_needed <= max_length:
                break
            max_spheres_in_line -= 1

        # 如果只有一个球体，放中心
        if max_spheres_in_line == 1:
            center_x = (combined_bounds[0] + combined_bounds[1]) / 2
            center_y = (combined_bounds[2] + combined_bounds[3]) / 2
            center_z = (combined_bounds[4] + combined_bounds[5]) / 2
            return [(np.array([center_x, center_y, center_z]), radius)]

        # 计算球体间距，确保居中对称分布
        # 对于N个球，需要N-1个间隙，但要确保总体居中
        total_length_needed = max_spheres_in_line * sphere_diameter
        remaining_space = max_length - total_length_needed
        spacing = remaining_space / (max_spheres_in_line + 1)  # 包括两端的边距

        spheres = []
        for i in range(max_spheres_in_line):
            # 计算当前位置（从左到右，居中分布）
            pos_along_axis = combined_bounds[max_idx * 2] + spacing + sphere_diameter/2 + i * (sphere_diameter + spacing)

            # 根据最长边方向设置坐标
            if max_idx == 0:  # x轴最长
                center = np.array([
                    pos_along_axis,
                    (combined_bounds[2] + combined_bounds[3]) / 2,  # y中心
                    (combined_bounds[4] + combined_bounds[5]) / 2   # z中心
                ])
            elif max_idx == 1:  # y轴最长
                center = np.array([
                    (combined_bounds[0] + combined_bounds[1]) / 2,  # x中心
                    pos_along_axis,
                    (combined_bounds[4] + combined_bounds[5]) / 2   # z中心
                ])
            else:  # z轴最长
                center = np.array([
                    (combined_bounds[0] + combined_bounds[1]) / 2,  # x中心
                    (combined_bounds[2] + combined_bounds[3]) / 2,  # y中心
                    pos_along_axis
                ])

            spheres.append((center, radius))

        return spheres

    @staticmethod
    def _compute_max_inner_radius(mesh: pv.PolyData, center: np.ndarray,
                                 num_directions: int = 20) -> float:
        """
        计算以center为球心，在mesh内部的最大球半径

        Args:
            mesh: PyVista网格
            center: 球心
            num_directions: 采样方向数量

        Returns:
            最大内接球半径
        """
        # 在不同方向上采样
        directions = []
        for i in range(num_directions):
            # 均匀采样球面
            z = 2 * np.random.random() - 1
            phi = 2 * np.pi * np.random.random()
            x = np.sqrt(1 - z*z) * np.cos(phi)
            y = np.sqrt(1 - z*z) * np.sin(phi)
            directions.append(np.array([x, y, z]))

        directions = np.array(directions)

        # 计算每个方向上的最大半径
        max_radii = []

        for direction in directions:
            # 沿射线寻找mesh边界
            max_radius = SphereFitting._find_mesh_boundary(mesh, center, direction)
            max_radii.append(max_radius)

        # 返回最小值（限制半径的因素）
        if max_radii:
            return min(max_radii) * 0.8  # 留一些安全裕度
        else:
            return 0.01  # 默认小半径

    @staticmethod
    def _find_mesh_boundary(mesh: pv.PolyData, start_point: np.ndarray,
                           direction: np.ndarray, max_distance: float = 1.0) -> float:
        """
        沿给定方向从起点找到mesh边界的距离

        Args:
            mesh: PyVista网格
            start_point: 起始点
            direction: 方向向量
            max_distance: 最大搜索距离

        Returns:
            到边界的距离
        """
        # 简化的方法：使用固定步长采样
        num_steps = 50
        step_size = max_distance / num_steps

        for i in range(num_steps):
            distance = i * step_size
            test_point = start_point + direction * distance

            # 检查点是否在mesh边界框内
            bounds = mesh.bounds
            in_bounds = (bounds[0] <= test_point[0] <= bounds[1] and
                        bounds[2] <= test_point[1] <= bounds[3] and
                        bounds[4] <= test_point[2] <= bounds[5])

            if not in_bounds:
                # 超出边界框，返回之前的距离
                return max(0.001, (i-1) * step_size)

        return max_distance * 0.5  # 默认值


def generate_link_spheres(hand_links_mesh_dict, method: str = 'bounding_box_packing',
                         urdf_obj=None, **kwargs) -> Dict[str, List[Tuple[np.ndarray, float]]]:
    """
    为手部所有link生成球体近似，并进行树形碰撞消除

    Args:
        hand_links_mesh_dict: {link_name: mesh} 或 {link_name: [mesh1, mesh2, ...]}
                             支持单个mesh或mesh列表
        method: 拟合方法 ('bounding_box_packing' 为新的算法)
        urdf_obj: yourdfpy.URDF对象，用于获取link树结构进行碰撞消除
        **kwargs: 拟合参数

    Returns:
        link_spheres: {link_name: [(center, radius), ...]}
    """
    link_spheres = {}

    for link_name, meshes in hand_links_mesh_dict.items():
        # 处理单个mesh的情况（向后兼容）
        if isinstance(meshes, pv.PolyData):
            meshes = [meshes]

        if not meshes:
            continue

        if method == 'bounding_box_packing':
            spheres = SphereFitting.bounding_box_sphere_packing(meshes)
        else:
            # 后备方法：合并所有mesh为一个，然后使用旧方法
            if len(meshes) == 1:
                combined_mesh = meshes[0]
            else:
                combined_mesh = meshes[0].copy()
                for mesh in meshes[1:]:
                    combined_mesh = combined_mesh.merge(mesh)

            # 根据link体积决定球体数量（旧逻辑）
            volume = combined_mesh.volume * 1e6  # 转换为cm³

            if volume <= 0.1 or combined_mesh.n_points < 10:
                center = np.array(combined_mesh.center)
                spheres = [(center, 0.005)]  # 5mm半径
            else:
                if volume < 10:
                    num_spheres = 2
                elif volume < 100:
                    num_spheres = 3
                elif volume < 1000:
                    num_spheres = 4
                else:
                    num_spheres = 6

                spheres = SphereFitting.fit_mesh_to_spheres(combined_mesh, method, num_spheres, **kwargs)

        link_spheres[link_name] = spheres

        # 计算总体积用于日志
        total_volume = sum(mesh.volume for mesh in meshes) * 1e6
        print(f"Link '{link_name}': 体积={total_volume:.1f}cm³, {len(meshes)}个mesh, 生成{len(spheres)}个球体")

    # 如果提供了URDF，进行树形碰撞消除
    if urdf_obj is not None:
        link_spheres = _eliminate_tree_collisions(link_spheres, urdf_obj)

    return link_spheres


def save_link_spheres(link_spheres: Dict[str, List[Tuple[np.ndarray, float]]],
                     file_path: str) -> None:
    """
    保存link球体数据到JSON文件
    """
    data = {}
    for link_name, spheres in link_spheres.items():
        data[link_name] = [
            {'center': center.tolist(), 'radius': float(radius)}
            for center, radius in spheres
        ]

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_link_spheres(file_path: str) -> Dict[str, List[Tuple[np.ndarray, float]]]:
    """
    从JSON文件加载link球体数据
    """
    with open(file_path, 'r') as f:
        data = json.load(f)

    link_spheres = {}
    for link_name, spheres_data in data.items():
        spheres = [
            (np.array(sphere['center']), sphere['radius'])
            for sphere in spheres_data
        ]
        link_spheres[link_name] = spheres

    return link_spheres


def _spheres_collide(sphere1: Tuple[np.ndarray, float], sphere2: Tuple[np.ndarray, float]) -> bool:
    """
    检查两个球体是否碰撞

    Args:
        sphere1: (center1, radius1)
        sphere2: (center2, radius2)

    Returns:
        True if spheres collide
    """
    center1, radius1 = sphere1
    center2, radius2 = sphere2
    distance = np.linalg.norm(center1 - center2)
    return distance < (radius1 + radius2)


def _build_link_tree(urdf_obj) -> Dict[str, List[str]]:
    """
    构建link的父子关系树

    Args:
        urdf_obj: yourdfpy.URDF对象

    Returns:
        {parent_link: [child_links]}
    """
    tree = {}
    for joint_name, joint in urdf_obj.joint_map.items():
        parent = joint.parent
        child = joint.child

        if parent not in tree:
            tree[parent] = []
        tree[parent].append(child)

    return tree


def _eliminate_tree_collisions(link_spheres: Dict[str, List[Tuple[np.ndarray, float]]],
                              urdf_obj) -> Dict[str, List[Tuple[np.ndarray, float]]]:
    """
    从根节点开始进行树形遍历，消除父子节点间的球体碰撞

    Args:
        link_spheres: {link_name: [(center, radius), ...]}
        urdf_obj: yourdfpy.URDF对象

    Returns:
        更新后的link_spheres
    """
    # 构建link树
    tree = _build_link_tree(urdf_obj)

    # 找到根节点（没有父节点的节点）
    all_links = set(link_spheres.keys())
    child_links = set()
    for children in tree.values():
        child_links.update(children)

    root_candidates = all_links - child_links

    if not root_candidates:
        # 如果没有找到根节点，尝试查找'forearm'或'world'
        for candidate in ['forearm', 'world']:
            if candidate in all_links:
                root_candidates = {candidate}
                break

    if not root_candidates:
        print("警告: 找不到根节点，跳过树形碰撞消除")
        return link_spheres

    root = list(root_candidates)[0]
    print(f"树形碰撞消除: 从根节点 '{root}' 开始遍历")

    # 深度优先遍历
    def dfs_traverse(current_link: str):
        if current_link not in link_spheres:
            return

        # 获取当前节点的子节点
        children = tree.get(current_link, [])

        # 检查当前节点与每个子节点之间的碰撞
        for child_link in children:
            if child_link not in link_spheres:
                continue

            parent_spheres = link_spheres[current_link]
            child_spheres = link_spheres[child_link]

            # 找到所有碰撞的子节点球体
            colliding_indices = []
            for i, child_sphere in enumerate(child_spheres):
                for parent_sphere in parent_spheres:
                    if _spheres_collide(parent_sphere, child_sphere):
                        colliding_indices.append(i)
                        break  # 一个球体只需检查一次碰撞

            # 删除碰撞的球体
            if colliding_indices:
                print(f"  移除 {child_link} 的 {len(colliding_indices)} 个碰撞球体")
                # 从后往前删除以保持索引有效
                for idx in sorted(colliding_indices, reverse=True):
                    del child_spheres[idx]

            # 递归处理子节点
            dfs_traverse(child_link)

    # 开始遍历
    dfs_traverse(root)

    # 统计最终结果
    total_spheres_after = sum(len(spheres) for spheres in link_spheres.values())
    print(f"树形碰撞消除完成: 最终球体总数 {total_spheres_after}")

    return link_spheres
