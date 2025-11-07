# optimization/optimizer_state.py

"""
优化器状态：存储优化变量和参数
"""

import jax.numpy as jnp
import jaxlie
from typing import Dict, List
from dataclasses import dataclass

DEFAULT_SCALE_FACTORS = {
    'rotation': 0.1,      # 旋转四元数
    'translation': 1.0,   # 平移
    'joints': 1.0         # 关节角度保持原尺度
}


@dataclass
class OptimizerState:
    """
    优化状态容器
    
    存储所有需要优化的变量：
    - base_pose: 机械手基座位姿 (SE3)
    - joint_values: 关节角度 (数组)
    
    支持参数缩放以平衡不同量纲的梯度
    """
    
    # 基座位姿 (7D: wxyz quaternion + xyz translation)
    base_pose_wxyz_xyz: jnp.ndarray  # shape: (7,)
    
    # 关节角度 (N_joints,)
    joint_values: jnp.ndarray  # shape: (N_joints,)
    
    # 关节名称到索引的映射 (用于调试)
    joint_names: List[str]
    
    # 参数缩放因子
    scale_factors: Dict[str, float]
    
    @classmethod
    def from_numpy(cls, base_pose_4x4, joint_dict, joint_names, 
                   scale_factors: Dict[str, float] = None):
        """
        从 NumPy 数组和字典创建优化状态
        
        Args:
            base_pose_4x4: 4x4 变换矩阵 (NumPy)
            joint_dict: {joint_name: value} 字典
            joint_names: 关节名称列表（有序）
            scale_factors: 参数缩放因子 {'rotation': float, 'translation': float, 'joints': float}
        """
        # 转换基座位姿为 SE3
        base_se3 = jaxlie.SE3.from_matrix(base_pose_4x4)
        
        # 转换关节字典为数组
        joint_array = jnp.array([joint_dict[name] for name in joint_names])
        
        # 设置默认缩放因子
        if scale_factors is None:
            scale_factors = DEFAULT_SCALE_FACTORS
        
        return cls(
            base_pose_wxyz_xyz=base_se3.wxyz_xyz,
            joint_values=joint_array,
            joint_names=joint_names,
            scale_factors=scale_factors
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
        
        应用缩放因子以平衡不同参数的梯度：
        - rotation: 缩放四元数部分
        - translation: 缩放平移部分  
        - joints: 缩放关节角度
        
        Returns:
            concatenated vector: [base_wxyz_scaled(4), base_xyz_scaled(3), joints_scaled(N)]
        """
        # 应用缩放因子
        scaled_wxyz = self.base_pose_wxyz_xyz[:4] * self.scale_factors['rotation']
        scaled_xyz = self.base_pose_wxyz_xyz[4:] * self.scale_factors['translation']
        scaled_joints = self.joint_values * self.scale_factors['joints']
        
        scaled_pose = jnp.concatenate([scaled_wxyz, scaled_xyz])
        
        return jnp.concatenate([scaled_pose, scaled_joints])
    
    @classmethod
    def from_optimization_vector(cls, vec, joint_names, scale_factors: Dict[str, float] = None):
        """
        从优化向量重建状态
        
        Args:
            vec: 缩放后的优化向量 [base_wxyz_scaled(4), base_xyz_scaled(3), joints_scaled(N)]
            joint_names: 关节名称列表
            scale_factors: 参数缩放因子 (与from_numpy中的相同)
        """
        if scale_factors is None:
            scale_factors = DEFAULT_SCALE_FACTORS
        
        # 应用逆缩放因子
        scaled_wxyz = vec[:4] / scale_factors['rotation']
        scaled_xyz = vec[4:7] / scale_factors['translation']
        scaled_joints = vec[7:] / scale_factors['joints']
        
        base_pose_wxyz_xyz = jnp.concatenate([scaled_wxyz, scaled_xyz])
        joint_values = scaled_joints
        
        return cls(
            base_pose_wxyz_xyz=base_pose_wxyz_xyz,
            joint_values=joint_values,
            joint_names=joint_names,
            scale_factors=scale_factors
        )
