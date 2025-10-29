# Optimization Module

这个模块提供了一个灵活且可扩展的优化框架，用于机械手抓取位姿优化。

## 架构

```
optimization/
├── __init__.py              # 模块导出
├── optimizer_state.py       # 优化状态容器
├── energy_functions.py      # 能量函数（损失函数）
├── optimizers.py            # 优化算法
└── README.md               # 本文档
```

## 核心概念

### 1. OptimizerState（优化状态）

存储所有需要优化的变量：
- `base_pose_wxyz_xyz`: 机械手基座位姿 (SE3, 7D)
- `joint_values`: 关节角度 (N维数组)

```python
from optimization import OptimizerState

# 创建状态
state = OptimizerState.from_numpy(
    base_pose_4x4=np.eye(4),
    joint_dict={'joint1': 0.0, 'joint2': 0.5},
    joint_names=['joint1', 'joint2']
)

# 获取优化向量
x = state.get_optimization_vector()  # [wxyz(4), xyz(3), joints(N)]
```

### 2. Energy Functions（能量函数）

所有能量函数继承自 `EnergyFunction` 基类，实现 `compute()` 方法。

#### 可用的能量函数：

| 能量函数 | 描述 | 权重建议 |
|---------|------|---------|
| `AnchorPointEnergy` | 锚点匹配能量：最小化手部锚点与物体锚点的距离 | 1.0 |
| `JointLimitEnergy` | 关节限位能量：防止关节超出限位（软约束） | 0.5 |
| `CollisionAvoidanceEnergy` | 碰撞避免能量：惩罚手部与物体的碰撞 | 0.1 |
| `ManipulabilityEnergy` | 可操作性能量：鼓励远离奇异位形 | 0.01 |

#### 使用示例：

```python
from optimization import AnchorPointEnergy, JointLimitEnergy, CompositeEnergy

# 创建单个能量函数
anchor_energy = AnchorPointEnergy(weight=1.0)

# 组合多个能量函数
energy_fn = CompositeEnergy([
    AnchorPointEnergy(weight=1.0),
    JointLimitEnergy(weight=0.5, margin=0.1),
])

# 计算能量
total_energy = energy_fn.compute(state, robot, anchor_pairs=anchors)

# 获取详细能量（用于调试）
detailed = energy_fn.compute_detailed(state, robot, anchor_pairs=anchors)
# {'AnchorPointEnergy': 0.15, 'JointLimitEnergy': 0.03}
```

### 3. Optimizers（优化器）

所有优化器继承自 `Optimizer` 基类，实现 `step()` 方法。

#### 可用的优化器：

| 优化器 | 描述 | 适用场景 |
|-------|------|---------|
| `AdamOptimizer` | **[推荐]** Adam 自适应优化器 | 大多数情况 |
| `GradientDescentOptimizer` | 标准梯度下降 | 简单问题 |
| `MomentumOptimizer` | 带动量的梯度下降 | 需要加速收敛 |
| `LBFGSOptimizer` | 限内存BFGS (二阶优化) | 中等规模、需要快速收敛 |

#### 使用示例：

```python
from optimization import AdamOptimizer

# 创建优化器
optimizer = AdamOptimizer(
    learning_rate=0.01,
    beta1=0.9,
    beta2=0.999,
    clip_grad=1.0
)

# 定义损失函数
def loss_fn(state):
    return energy_fn.compute(state, robot, anchor_pairs=anchors)

# 执行一步优化
new_state, loss_value = optimizer.step(state, loss_fn)

print(f"Loss: {loss_value:.4f}")
```

## 完整使用示例

```python
from optimization import (
    OptimizerState,
    AnchorPointEnergy,
    JointLimitEnergy,
    CompositeEnergy,
    AdamOptimizer
)

# 1. 准备锚点数据
anchors = [
    {
        'hand_point_local': [0.01, 0.0, 0.05],  # 手部局部坐标
        'hand_link_idx': 15,                     # link 索引
        'obj_point': [0.1, 0.2, 0.3]            # 物体世界坐标
    },
    # ... 更多锚点
]

# 2. 创建初始状态
state = OptimizerState.from_numpy(
    base_pose_4x4=np.eye(4),
    joint_dict=initial_joint_dict,
    joint_names=robot.joints.actuated_names
)

# 3. 设置能量函数
energy_fn = CompositeEnergy([
    AnchorPointEnergy(weight=1.0),
    JointLimitEnergy(weight=0.5, margin=0.1),
])

# 4. 创建优化器
optimizer = AdamOptimizer(learning_rate=0.01, clip_grad=1.0)

# 5. 优化循环
for step in range(100):
    # 定义损失函数
    def loss_fn(s):
        return energy_fn.compute(s, robot, anchor_pairs=anchors)
    
    # 执行一步优化
    state, loss = optimizer.step(state, loss_fn)
    
    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss:.4f}")

# 6. 提取结果
final_pose = state.to_base_pose_matrix()  # 4x4 矩阵
final_joints = state.to_joint_dict()       # {joint_name: value}
```

## 添加自定义能量函数

创建新的能量函数非常简单：

```python
from optimization.energy_functions import EnergyFunction
import jax.numpy as jnp

class MyCustomEnergy(EnergyFunction):
    def __init__(self, weight=1.0, my_param=0.1):
        self.weight = weight
        self.my_param = my_param
    
    def compute(self, state, robot, **kwargs):
        # 访问优化变量
        base_pose = state.to_base_pose_se3()
        joints = state.joint_values
        
        # 计算能量（必须可微分！）
        energy = jnp.sum(joints ** 2) * self.my_param
        
        return energy * self.weight
    
    def get_weight(self):
        return self.weight
```

然后在 `energy_functions.py` 中添加并导出即可。

## 优化器参数调优

### Adam 优化器（推荐设置）

```python
AdamOptimizer(
    learning_rate=0.01,    # 较大的学习率适合抓取问题
    beta1=0.9,             # 保持默认
    beta2=0.999,           # 保持默认
    clip_grad=1.0          # 防止梯度爆炸
)
```

### 梯度下降优化器

```python
GradientDescentOptimizer(
    learning_rate=0.005,   # 需要较小的学习率
    clip_grad=1.0
)
```

## 性能优化建议

1. **使用 JAX JIT 编译**（未来改进）：
   ```python
   from jax import jit
   
   @jit
   def loss_fn(state):
       return energy_fn.compute(state, robot, anchor_pairs=anchors)
   ```

2. **批量优化多个锚点**：当前实现已经支持多锚点，使用向量化计算。

3. **调整优化频率**：在 `optimization_thread.py` 中调整 `msleep(16)` 来平衡性能和视觉流畅度。

4. **能量权重平衡**：
   - `AnchorPointEnergy`: 1.0（基准）
   - `JointLimitEnergy`: 0.1-0.5（防止过度约束）
   - `CollisionAvoidanceEnergy`: 0.1-0.2（视情况启用）

## 调试技巧

### 1. 打印详细能量

```python
detailed_energies = energy_fn.compute_detailed(state, robot, anchor_pairs=anchors)
for name, value in detailed_energies.items():
    print(f"{name}: {value:.4f}")
```

### 2. 检查梯度

```python
from jax import grad

def loss_fn(vec):
    s = OptimizerState.from_optimization_vector(vec, joint_names)
    return energy_fn.compute(s, robot, anchor_pairs=anchors)

x = state.get_optimization_vector()
g = grad(loss_fn)(x)
print(f"Gradient norm: {jnp.linalg.norm(g):.4f}")
```

### 3. 可视化优化过程

在 `optimization_thread.py` 中已经实现了每30步打印一次详细能量。

## 常见问题

**Q: 优化不收敛怎么办？**
- 降低学习率
- 检查锚点数据是否正确
- 增加梯度裁剪阈值
- 尝试切换优化器（如 LBFGS）

**Q: 手部位姿抖动？**
- 降低学习率
- 增加 `JointLimitEnergy` 权重
- 使用动量优化器

**Q: 如何加速优化？**
- 使用 Adam 而非梯度下降
- 减少能量函数数量
- 使用 JAX JIT 编译（需要额外实现）

## 扩展阅读

- [JAX Autodiff Documentation](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html)
- [Adam Optimizer Paper](https://arxiv.org/abs/1412.6980)
- [L-BFGS Explanation](https://en.wikipedia.org/wiki/Limited-memory_BFGS)
