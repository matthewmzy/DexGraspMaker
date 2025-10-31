# optimization/energy_functions.py

"""
能量函数模块：定义各种优化目标

所有能量函数都是可微分的，可以用 JAX 自动求导
"""

import jax
import jax.numpy as jnp
import jaxlie
import numpy as np
from typing import List, Dict, Tuple
from abc import ABC, abstractmethod


class EnergyFunction(ABC):
    """能量函数基类"""
    
    @abstractmethod
    def compute(self, state, robot, **kwargs) -> jnp.ndarray:
        """
        计算能量值
        
        Args:
            state: OptimizerState 对象
            robot: pyroki.Robot 对象
            **kwargs: 额外参数
            
        Returns:
            scalar energy value
        """
        pass
    
    def get_weight(self) -> float:
        """返回此能量项的权重"""
        return 1.0


class AnchorPointEnergy(EnergyFunction):
    """
    锚点能量：最小化手部锚点和物体锚点之间的距离
    
    E = Σ ||T_world_link @ p_local - p_obj||^2
    """
    
    def __init__(self, weight: float = 1.0):
        self.weight = weight
    
    def compute(self, state, robot, anchor_pairs: List[Dict], **kwargs) -> jnp.ndarray:
        """
        计算锚点能量
        
        Args:
            anchor_pairs: 锚点对列表，每个元素包含:
                {
                    'hand_point_local': [x, y, z],  # 手部局部坐标
                    'hand_link_name': str,           # link 名称
                    'hand_link_idx': int,            # link 索引
                    'obj_point': [x, y, z]           # 物体世界坐标
                }
        """
        if not anchor_pairs:
            return jnp.array(0.0)
        
        total_energy = 0.0
        
        # 1. 运行 FK 获取所有 link 位姿
        link_poses_rel_root = robot.forward_kinematics(state.joint_values)
        T_world_base = state.to_base_pose_se3()
        
        # 2. 对每个锚点对计算能量
        for anchor in anchor_pairs:
            link_idx = anchor['hand_link_idx']
            p_local = jnp.array(anchor['hand_point_local'])  # (3,)
            p_obj = jnp.array(anchor['obj_point'])            # (3,)
            
            # 获取 link 在世界坐标系中的位姿
            T_root_link = jaxlie.SE3(link_poses_rel_root[link_idx])
            T_world_link = T_world_base @ T_root_link
            
            # 将局部坐标转换为世界坐标
            p_local_homo = jnp.append(p_local, 1.0)  # (4,)
            p_world = (T_world_link.as_matrix() @ p_local_homo)[:3]
            
            # 计算距离的平方
            distance_sq = jnp.sum((p_world - p_obj) ** 2)
            total_energy += distance_sq
        
        return total_energy * self.weight
    
    def get_weight(self) -> float:
        return self.weight


class PenetrationAvoidanceEnergy(EnergyFunction):
    """
    穿透避免能量：防止手部穿透到物体内部
    
    计算手部关键点到物体网格的最小距离，当距离为负时施加惩罚
    """
    
    def __init__(self, weight: float = 1.0, margin: float = 0.005, 
                 hand_key_points: Dict[str, np.ndarray] = None):
        """
        Args:
            weight: 能量权重
            margin: 安全距离（米），距离小于此值时开始施加惩罚
            hand_key_points: 手部关键点字典 {link_name: points_array}
                如果为None，则需要通过set_key_points()方法设置
        """
        self.weight = weight
        self.margin = margin
        self.hand_key_points = hand_key_points
    
    def set_key_points(self, key_points: Dict[str, np.ndarray]):
        """
        设置手部关键点
        
        Args:
            key_points: {link_name: points_array} 字典
        """
        self.hand_key_points = key_points
    
    def compute(self, state, robot, object_mesh=None, **kwargs) -> jnp.ndarray:
        """
        计算穿透避免能量
        
        Args:
            object_mesh: 物体网格 (pyvista.PolyData 或 trimesh.Trimesh)
        """
        if object_mesh is None or self.hand_key_points is None:
            return jnp.array(0.0)
        
        total_energy = 0.0
        
        # 1. 运行 FK 获取所有 link 位姿
        link_poses_rel_root = robot.forward_kinematics(state.joint_values)
        T_world_base = state.to_base_pose_se3()
        
        # 2. 对每个link的关键点计算到物体的距离
        for link_name, key_points in self.hand_key_points.items():
            # 查找 link 索引
            try:
                link_idx = robot.links.names.index(link_name)
            except ValueError:
                # 如果找不到指定的 link，跳过
                continue
            
            # 获取 link 在世界坐标系中的位姿
            T_root_link = jaxlie.SE3(link_poses_rel_root[link_idx])
            T_world_link = T_world_base @ T_root_link
            
            # 对每个关键点计算距离
            for p_local in key_points:
                p_local = jnp.array(p_local)
                
                # 将局部坐标转换为世界坐标
                p_local_homo = jnp.append(p_local, 1.0)
                p_world = (T_world_link.as_matrix() @ p_local_homo)[:3]
                
                # 计算到物体网格的最小距离
                distance = self._compute_point_to_mesh_distance(p_world, object_mesh)
                
                # 施加惩罚：当距离小于 margin 时，惩罚强度随穿透深度增加
                if distance < self.margin:
                    penetration_depth = self.margin - distance
                    # 使用二次惩罚函数
                    penalty = penetration_depth ** 2
                    total_energy += penalty
        
        return total_energy * self.weight
    
    def _compute_point_to_mesh_distance(self, point: jnp.ndarray, mesh) -> float:
        """
        计算点到网格的最小距离
        
        Args:
            point: 世界坐标系中的点 [x, y, z] (JAX array)
            mesh: pyvista.PolyData 或 trimesh.Trimesh 对象
            
        Returns:
            最小距离（正值表示分离，负值表示穿透）
        """
        # 为了保持JAX兼容性，我们使用边界框距离
        # 这是一个近似，但对于穿透避免已经足够
        
        if hasattr(mesh, 'bounds'):
            # trimesh格式
            bounds = jnp.array(mesh.bounds).reshape(3, 2)
        elif hasattr(mesh, 'GetBounds'):
            # pyvista格式
            bounds = jnp.array(mesh.GetBounds()).reshape(3, 2)
        else:
            # 默认值，如果无法获取边界框
            bounds = jnp.array([[-0.1, 0.1], [-0.1, 0.1], [-0.1, 0.1]])
        
        # 计算边界框中心和半尺寸
        center = (bounds[:, 0] + bounds[:, 1]) / 2
        half_size = (bounds[:, 1] - bounds[:, 0]) / 2
        
        # 计算点到边界框的距离
        diff = point - center
        dist_to_box = jnp.maximum(jnp.abs(diff) - half_size, 0)
        distance = jnp.linalg.norm(dist_to_box)
        
        return distance
    
    def get_weight(self) -> float:
        return self.weight


class CollisionAvoidanceEnergy(EnergyFunction):
    """
    碰撞避免能量：惩罚手部与物体的碰撞
    
    使用简单的距离场方法（可以扩展为 SDF）
    """
    
    def __init__(self, weight: float = 0.1, margin: float = 0.01):
        """
        Args:
            weight: 能量权重
            margin: 安全距离（米）
        """
        self.weight = weight
        self.margin = margin
    
    def compute(self, state, robot, **kwargs) -> jnp.ndarray:
        """
        计算碰撞避免能量
        
        这是一个占位符实现，需要实际的碰撞检测
        """
        # TODO: 实现真实的碰撞检测
        # 可以使用 pyroki.collision 模块
        return jnp.array(0.0)
    
    def get_weight(self) -> float:
        return self.weight


class SelfCollisionAvoidanceEnergy(EnergyFunction):
    """
    自碰撞避免能量：防止手部各link之间的碰撞

    使用球体集合来近似每个link的形状，计算球体间距离
    """

    def __init__(self, weight: float = 0.5, margin: float = 0.005,
                 link_spheres: Dict[str, List[Tuple[np.ndarray, float]]] = None):
        """
        Args:
            weight: 能量权重
            margin: 安全距离（米）
            link_spheres: {link_name: [(center, radius), ...]} 球体数据
        """
        self.weight = weight
        self.margin = margin
        self.link_spheres = link_spheres or {}

    def set_link_spheres(self, link_spheres: Dict[str, List[Tuple[np.ndarray, float]]]):
        """
        设置link球体数据

        Args:
            link_spheres: {link_name: [(center, radius), ...]}
        """
        self.link_spheres = link_spheres

    def compute(self, state, robot, **kwargs) -> jnp.ndarray:
        """
        计算自碰撞避免能量

        计算所有link球体之间的距离，当距离小于安全阈值时施加惩罚
        """
        if not self.link_spheres:
            return jnp.array(0.0)

        total_energy = 0.0

        # 1. 运行FK获取所有link位姿
        link_poses_rel_root = robot.forward_kinematics(state.joint_values)
        T_world_base = state.to_base_pose_se3()

        # 2. 收集所有球体在世界坐标系中的位置
        world_spheres = []  # [(center_world, radius, link_name), ...]

        for link_name, spheres in self.link_spheres.items():
            try:
                link_idx = robot.links.names.index(link_name)
            except ValueError:
                continue  # 跳过不存在的link

            # 获取link在世界坐标系中的位姿
            T_root_link = jaxlie.SE3(link_poses_rel_root[link_idx])
            T_world_link = T_world_base @ T_root_link
            T_world_link_mat = T_world_link.as_matrix()

            # 转换球体中心到世界坐标系
            for center_local, radius in spheres:
                center_local_homo = jnp.append(jnp.array(center_local), 1.0)
                center_world = (T_world_link_mat @ center_local_homo)[:3]
                world_spheres.append((center_world, radius, link_name))

        # 3. 计算球体间距离并施加惩罚
        num_spheres = len(world_spheres)
        for i in range(num_spheres):
            for j in range(i + 1, num_spheres):
                center1, radius1, link1 = world_spheres[i]
                center2, radius2, link2 = world_spheres[j]

                # 跳过同一link内的球体（除非link有多个分离的部分）
                if link1 == link2:
                    continue

                # 计算球心距离
                distance = jnp.linalg.norm(center1 - center2)

                # 计算最小安全距离
                min_distance = radius1 + radius2 + self.margin

                # 惩罚函数：当距离小于阈值时施加二次惩罚
                if distance < min_distance:
                    violation = min_distance - distance
                    # 使用二次惩罚函数
                    energy_penalty = violation ** 2
                    total_energy += energy_penalty

        return jnp.array(total_energy * self.weight)

    def get_weight(self) -> float:
        return self.weight


class ManipulabilityEnergy(EnergyFunction):
    """
    可操作性能量：鼓励远离奇异位形
    
    E = -log(det(J @ J^T))
    其中 J 是 Jacobian 矩阵
    """
    
    def __init__(self, weight: float = 0.01):
        self.weight = weight
    
    def compute(self, state, robot, **kwargs) -> jnp.ndarray:
        """
        计算可操作性能量
        
        注意：pyroki 可能没有直接提供 Jacobian，
        需要使用 JAX 自动微分计算
        """
        # TODO: 实现 Jacobian 计算
        # 可以使用 jax.jacfwd 或 jax.jacrev
        return jnp.array(0.0)
    
    def get_weight(self) -> float:
        return self.weight


class JointLimitEnergy(EnergyFunction):
    """
    关节限位能量：软约束，防止关节超出限位
    
    E = Σ max(0, q - q_max)^2 + max(0, q_min - q)^2
    """
    
    def __init__(self, weight: float = 1.0, margin: float = 0.1):
        """
        Args:
            weight: 能量权重
            margin: 从限位的安全边距（弧度）
        """
        self.weight = weight
        self.margin = margin
    
    def compute(self, state, robot, **kwargs) -> jnp.ndarray:
        """计算关节限位能量"""
        q = state.joint_values
        q_min = robot.joints.lower_limits + self.margin
        q_max = robot.joints.upper_limits - self.margin
        
        # 超出上限的惩罚
        upper_violation = jnp.maximum(0.0, q - q_max)
        # 超出下限的惩罚
        lower_violation = jnp.maximum(0.0, q_min - q)
        
        energy = jnp.sum(upper_violation ** 2) + jnp.sum(lower_violation ** 2)
        
        return energy * self.weight
    
    def get_weight(self) -> float:
        return self.weight


class CompositeEnergy(EnergyFunction):
    """
    组合能量：多个能量项的加权和
    
    E_total = Σ w_i * E_i
    """
    
    def __init__(self, energy_functions: List[EnergyFunction]):
        """
        Args:
            energy_functions: 能量函数列表
        """
        self.energy_functions = energy_functions
    
    def compute(self, state, robot, **kwargs) -> jnp.ndarray:
        """计算所有能量的总和"""
        total = jnp.array(0.0)
        
        for energy_fn in self.energy_functions:
            total += energy_fn.compute(state, robot, **kwargs)
        
        return total
    
    def compute_detailed(self, state, robot, **kwargs) -> Dict[str, float]:
        """
        计算每个能量项的值（用于调试）
        
        Returns:
            {energy_name: value}
        """
        energies = {}
        
        for energy_fn in self.energy_functions:
            name = energy_fn.__class__.__name__
            value = float(energy_fn.compute(state, robot, **kwargs))
            energies[name] = value
        
        return energies
