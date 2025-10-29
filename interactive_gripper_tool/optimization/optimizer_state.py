# optimization/optimizer_state.py

"""
优化器状态：存储优化变量和参数
"""

import jax.numpy as jnp
import jaxlie
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class OptimizerState:
    """
    优化状态容器
    
    存储所有需要优化的变量：
    - base_pose: 机械手基座位姿 (SE3)
    - joint_values: 关节角度 (数组)
    """
    
    # 基座位姿 (7D: wxyz quaternion + xyz translation)
    base_pose_wxyz_xyz: jnp.ndarray  # shape: (7,)
    
    # 关节角度 (N_joints,)
    joint_values: jnp.ndarray  # shape: (N_joints,)
    
    # 关节名称到索引的映射 (用于调试)
    joint_names: List[str]
    
    @classmethod
    def from_numpy(cls, base_pose_4x4, joint_dict, joint_names):
        """
        从 NumPy 数组和字典创建优化状态
        
        Args:
            base_pose_4x4: 4x4 变换矩阵 (NumPy)
            joint_dict: {joint_name: value} 字典
            joint_names: 关节名称列表（有序）
        """
        # 转换基座位姿为 SE3
        base_se3 = jaxlie.SE3.from_matrix(base_pose_4x4)
        
        # 转换关节字典为数组
        joint_array = jnp.array([joint_dict[name] for name in joint_names])
        
        return cls(
            base_pose_wxyz_xyz=base_se3.wxyz_xyz,
            joint_values=joint_array,
            joint_names=joint_names
        )
    
    def to_base_pose_matrix(self):
        """转换基座位姿为 4x4 矩阵"""
        return jaxlie.SE3(self.base_pose_wxyz_xyz).as_matrix()
    
    def to_base_pose_se3(self):
        """转换基座位姿为 SE3 对象"""
        return jaxlie.SE3(self.base_pose_wxyz_xyz)
    
    def to_joint_dict(self) -> Dict[str, float]:
        """转换关节数组为字典"""
        return {name: float(val) for name, val in zip(self.joint_names, self.joint_values)}
    
    def get_optimization_vector(self):
        """
        获取优化向量 (用于优化器)
        
        Returns:
            concatenated vector: [base_wxyz(4), base_xyz(3), joints(N)]
        """
        return jnp.concatenate([self.base_pose_wxyz_xyz, self.joint_values])
    
    @classmethod
    def from_optimization_vector(cls, vec, joint_names):
        """
        从优化向量重建状态
        
        Args:
            vec: [base_wxyz(4), base_xyz(3), joints(N)]
            joint_names: 关节名称列表
        """
        base_pose_wxyz_xyz = vec[:7]
        joint_values = vec[7:]
        
        return cls(
            base_pose_wxyz_xyz=base_pose_wxyz_xyz,
            joint_values=joint_values,
            joint_names=joint_names
        )
