"""
优化器模块：使用 Optax 高性能优化器

Optax 是 JAX 官方优化器库，经过高度优化，性能优于 PyTorch
"""

import jax
import jax.numpy as jnp
from typing import Callable, Tuple, Optional
from .optimizer_state import OptimizerState

try:
    import optax
except ImportError:
    raise ImportError(
        "Optax 是必需的依赖。请安装: pip install optax\n"
        "Optax 提供高性能优化器，性能优于 PyTorch Adam"
    )


class Optimizer:
    """
    统一的优化器接口
    
    使用 Optax 进行高性能优化
    
    性能优势：
    - JIT 编译优化
    - 向量化操作
    - 数值稳定的实现
    - GPU/TPU 加速支持
    """
    
    def __init__(self, optax_optimizer):
        """
        Args:
            optax_optimizer: optax 优化器实例
        """
        self.optax_optimizer = optax_optimizer
        self.opt_state = None
    
    def step(self,
             state: OptimizerState,
             loss_fn: Callable,
             *args,
             **kwargs) -> Tuple[OptimizerState, float]:
        """执行一步优化"""
        
        x = state.get_optimization_vector()
        
        if self.opt_state is None:
            self.opt_state = self.optax_optimizer.init(x)
        
        # 计算损失和梯度
        loss_value, grad = jax.value_and_grad(
            lambda vec: loss_fn(OptimizerState.from_optimization_vector(vec, state.joint_names))
        )(x)
        
        # Optax 更新
        updates, self.opt_state = self.optax_optimizer.update(grad, self.opt_state, x)
        x_new = optax.apply_updates(x, updates)
        
        new_state = OptimizerState.from_optimization_vector(x_new, state.joint_names)
        
        return new_state, float(loss_value)
    
    def reset(self):
        """重置优化器状态"""
        self.opt_state = None


# ============================================================================
# 预配置的高性能优化器
# ============================================================================

def create_adam(learning_rate: float = 0.01,
                b1: float = 0.9,
                b2: float = 0.999,
                eps: float = 1e-8,
                clip_grad: Optional[float] = 1.0) -> Optimizer:
    """
    创建 Adam 优化器（推荐）
    
    性能：Optax Adam 使用了多项优化技术，性能通常优于 PyTorch
    
    Args:
        learning_rate: 学习率
        b1: 一阶矩估计的指数衰减率
        b2: 二阶矩估计的指数衰减率
        eps: 数值稳定性常数
        clip_grad: 梯度裁剪阈值（None 表示不裁剪）
    """
    if clip_grad is not None:
        opt = optax.chain(
            optax.clip_by_global_norm(clip_grad),
            optax.adam(learning_rate=learning_rate, b1=b1, b2=b2, eps=eps)
        )
    else:
        opt = optax.adam(learning_rate=learning_rate, b1=b1, b2=b2, eps=eps)
    
    return Optimizer(opt)


def create_adamw(learning_rate: float = 0.01,
                 weight_decay: float = 0.0001,
                 clip_grad: Optional[float] = 1.0) -> Optimizer:
    """
    创建 AdamW 优化器（带权重衰减，防止过拟合）
    
    AdamW 是 Adam 的改进版本，权重衰减实现更正确
    """
    if clip_grad is not None:
        opt = optax.chain(
            optax.clip_by_global_norm(clip_grad),
            optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
        )
    else:
        opt = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    
    return Optimizer(opt)


def create_lion(learning_rate: float = 0.001,
                b1: float = 0.9,
                b2: float = 0.99,
                clip_grad: Optional[float] = 1.0) -> Optimizer:
    """
    创建 Lion 优化器（2023年最新，内存效率更高）
    
    Lion 是 Google 开发的新优化器，比 Adam 内存效率高 2 倍
    注意：Lion 通常需要更小的学习率（~10x smaller than Adam）
    """
    if clip_grad is not None:
        opt = optax.chain(
            optax.clip_by_global_norm(clip_grad),
            optax.lion(learning_rate=learning_rate, b1=b1, b2=b2)
        )
    else:
        opt = optax.lion(learning_rate=learning_rate, b1=b1, b2=b2)
    
    return Optimizer(opt)


def create_with_schedule(base_lr: float = 0.01,
                         warmup_steps: int = 100,
                         decay_steps: int = 1000,
                         decay_rate: float = 0.96,
                         optimizer_type: str = "adam",
                         clip_grad: Optional[float] = 1.0) -> Optimizer:
    """
    创建带学习率调度的优化器
    
    使用 warmup + 指数衰减策略
    
    Args:
        base_lr: 基础学习率
        warmup_steps: 预热步数
        decay_steps: 衰减周期
        decay_rate: 衰减率
        optimizer_type: "adam", "adamw", 或 "lion"
        clip_grad: 梯度裁剪
    """
    # 学习率调度
    schedule = optax.warmup_exponential_decay_schedule(
        init_value=0.0,
        peak_value=base_lr,
        warmup_steps=warmup_steps,
        transition_steps=decay_steps,
        decay_rate=decay_rate
    )
    
    # 选择优化器
    if optimizer_type == "adamw":
        base_opt = optax.adamw(learning_rate=schedule)
    elif optimizer_type == "lion":
        base_opt = optax.lion(learning_rate=schedule)
    else:
        base_opt = optax.adam(learning_rate=schedule)
    
    # 添加梯度裁剪
    if clip_grad is not None:
        opt = optax.chain(optax.clip_by_global_norm(clip_grad), base_opt)
    else:
        opt = base_opt
    
    return Optimizer(opt)


# ============================================================================
# 向后兼容别名
# ============================================================================

AdamOptimizer = create_adam
GradientDescentOptimizer = lambda learning_rate=0.01, clip_grad=1.0, **kwargs: Optimizer(
    optax.chain(
        optax.clip_by_global_norm(clip_grad),
        optax.sgd(learning_rate=learning_rate)
    ) if clip_grad else optax.sgd(learning_rate=learning_rate)
)
MomentumOptimizer = lambda learning_rate=0.01, momentum=0.9, clip_grad=1.0, **kwargs: Optimizer(
    optax.chain(
        optax.clip_by_global_norm(clip_grad),
        optax.sgd(learning_rate=learning_rate, momentum=momentum)
    ) if clip_grad else optax.sgd(learning_rate=learning_rate, momentum=momentum)
)
