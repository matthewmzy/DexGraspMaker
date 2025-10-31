# Shadow手FK可视化脚本

这个Python脚本用于可视化Shadow手的正向运动学(FK)结果，包括mesh、关键点和拟合球体。

## 功能

- ✅ 加载Shadow手URDF
- ✅ 计算正向运动学得到全局姿态
- ✅ 可视化所有link的mesh（使用open3d，半透明浅蓝色）
- ✅ 显示关键点（红色不透明）
- ✅ 显示拟合球体（橙色不透明）
- ✅ 支持关节角度调整

## 依赖

- numpy, trimesh, yourdfpy, jax, jaxlie, pyroki, open3d

## 使用方法

### 基本可视化（默认姿态）
```bash
python visualize_hand_fk.py
```

### 指定文件路径
```bash
python visualize_hand_fk.py \
  --urdf test_assets/shadow/shadow_hand_right.urdf \
  --keypoints test_assets/shadow/shadow_hand_right_keypoints.json \
  --spheres test_assets/shadow/shadow_hand_right_spheres.json
```

### 设置关节角度
```bash
# 设置单个关节
python visualize_hand_fk.py --joint rh_FFJ1 0.5

# 设置多个关节
python visualize_hand_fk.py --joint rh_FFJ1 0.5 --joint rh_FFJ2 0.3 --joint rh_MFJ1 -0.2
```

### 查看帮助
```bash
python visualize_hand_fk.py --help
```

## 可视化说明

- **半透明浅蓝色mesh**: 手的各个link（统一颜色，半透明显示）
- **红色不透明球**: 关键点位置
- **橙色不透明球**: 碰撞检测球体
- **交互式3D视图**: 支持鼠标旋转、缩放、平移

## 示例

```bash
# 可视化握拳姿态
python visualize_hand_fk.py \
  --joint rh_FFJ1 1.0 --joint rh_FFJ2 1.0 --joint rh_FFJ3 1.0 --joint rh_FFJ4 1.0 \
  --joint rh_MFJ1 1.0 --joint rh_MFJ2 1.0 --joint rh_MFJ3 1.0 --joint rh_MFJ4 1.0 \
  --joint rh_RFJ1 1.0 --joint rh_RFJ2 1.0 --joint rh_RFJ3 1.0 --joint rh_RFJ4 1.0 \
  --joint rh_LFJ1 1.0 --joint rh_LFJ2 1.0 --joint rh_LFJ3 1.0 --joint rh_LFJ4 1.0 \
  --joint rh_THJ1 1.0 --joint rh_THJ2 1.0 --joint rh_THJ3 1.0 --joint rh_THJ4 1.0 --joint rh_THJ5 1.0
```

## 注意事项

- 可视化使用open3d库，支持交互式3D操作
- mesh显示为半透明浅蓝色，统一视觉效果
- 关键点和球体为不透明，突出显示
- 如果在无头环境中运行，可能需要设置虚拟显示</content>
<parameter name="filePath">/home/ubuntu/Documents/DexGraspMaker/README_visualize_hand_fk.md