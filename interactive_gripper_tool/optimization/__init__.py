# optimization/__init__.py

"""
优化模块：提供梯度优化和能量函数
"""

from .optimizer_state import OptimizerState
from .energy_functions import (
    EnergyFunction,
    AnchorPointEnergy,
    CollisionAvoidanceEnergy,
    ManipulabilityEnergy,
    JointLimitEnergy,
    CompositeEnergy
)
from .optimizers import (
    Optimizer,
    create_adam,
    create_adamw,
    create_lion,
    create_with_schedule,
    # 向后兼容别名
    AdamOptimizer,
    GradientDescentOptimizer,
    MomentumOptimizer,
)

__all__ = [
    # 状态
    'OptimizerState',
    
    # 能量函数
    'EnergyFunction',
    'AnchorPointEnergy',
    'CollisionAvoidanceEnergy',
    'ManipulabilityEnergy',
    'JointLimitEnergy',
    'CompositeEnergy',
    
    # 优化器创建函数（推荐）
    'Optimizer',
    'create_adam',
    'create_adamw',
    'create_lion',
    'create_with_schedule',
    
    # 向后兼容别名
    'AdamOptimizer',
    'GradientDescentOptimizer',
    'MomentumOptimizer',
]
