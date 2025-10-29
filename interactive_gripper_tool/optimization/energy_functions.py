# optimization/energy_functions.py

"""
能量函数模块：定义各种优化目标

所有能量函数都是可微分的，可以用 JAX 自动求导
"""

import jax
import jax.numpy as jnp
import jaxlie
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
