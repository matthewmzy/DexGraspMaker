# optimization_thread.py (migrated)

import time
import numpy as np
from typing import Optional
from PyQt6.QtCore import (
    QThread, QMutex, QWaitCondition, pyqtSignal, pyqtSlot, QMutexLocker, QObject
)
import pyroki as pk
import jaxlie
from jax import numpy as jnp
import os
import yaml

# 导入优化模块（调整为相对导入）
from .optimization import (
    OptimizerState,
    AnchorPointEnergy,
    JointLimitEnergy,
    CollisionAvoidanceEnergy,
    PenetrationAvoidanceEnergy,
    SelfCollisionAvoidanceEnergy,
    CompositeEnergy,
    create_adam,
)
from utils.constants import (
    HAND_DYNAMIC_PREFIX,
    ENERGY_WEIGHTS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_CLIP_GRAD,
    DEFAULT_SCALE_FACTORS,
    OPTIMIZATION_SLEEP_MS,
)
from utils.geometry import euler_rpy_to_matrix
from utils.diagnostics import format_energy_breakdown, pose_delta
from utils.hand_config import load_hand_config, base_pose_from_cfg, init_joints_from_cfg


class OptimizationThread(QThread):
    """
    在一个单独的线程中运行可微分优化，以避免阻塞UI。
    
    信号:
        pose_update_signal(dict): 
            在每一步优化后发射。
            字典格式: {actor_name: 4x4_pose_matrix}
            'actor_name' 必须匹配 VistaWidget 中的名称 (例如 'dyn_hand_link_1')
    """
    
    pose_update_signal = pyqtSignal(dict)
    base_pose_updated_signal = pyqtSignal(list, list)
    joint_values_updated_signal = pyqtSignal(dict)
    joint_info_signal = pyqtSignal(list)
    # 新增：SDF缓存事件提示（命中或生成）
    sdf_cache_message_signal = pyqtSignal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        
        # --- 线程控制 ---
        self.mutex = QMutex()
        self.wait_condition = QWaitCondition()
        self._is_running = True
        self._needs_optimization = False # 优化循环的开关
        self._manual_update = False      # 手动关节更新的开关
        self._is_paused = False          # 优化暂停开关

        # 当加载物体后需要在后台预计算SDF/距离场时使用
        self._pending_precompute_mesh = None  # type: ignore[assignment]

        # --- 状态变量 (由 mutex 保护) ---

        # 1. 存储 JAX-native 机器人模型
        self.robot: pk.Robot | None = None
        
        # 2. 锚点
        self.anchor_pairs: list[dict] = []
        
        # 3. 物体网格（用于穿透避免）
        self.object_mesh = None
        
        # 4. 手关键点（用于穿透避免）
        self.hand_keypoints: dict[str, np.ndarray] = {}
        
        # 5. 手link球体（用于自碰撞避免）
        self.link_spheres: dict[str, list] = {}
        
        # 6. 优化参数 (这是您用 JAX/PyTorch 优化的变量)
        self.current_base_pose: np.ndarray = np.eye(4)
        self.current_joint_values: dict[str, float] = {} # {joint_name: value}
        
        # 4. 手动更新的目标 (来自滑块)
        self.target_joint_values: dict[str, float] = {}
        
        # 5. 名称前缀 (必须与 main_window 中设置的一致)
        self.actor_name_prefix = HAND_DYNAMIC_PREFIX
        
        # 6. 优化器和能量函数
        self.optimizer = None
        self.energy_function = None
        self._setup_optimization()

        # 参数缩放因子（用于平衡旋转/平移/关节的梯度量级）
        self.scale_factors = dict(DEFAULT_SCALE_FACTORS)

        # 仅可视化的 link 名称集合 (用于抑制不存在 actor 的警告)
        self._visual_link_names = set()

        # 待应用的手配置名称（当 robot 尚未就绪时临时保存）
        self._pending_hand_config_name = None  # type: Optional[str]

    def stop(self) -> None:
        """
        请求线程停止。
        """
        with QMutexLocker(self.mutex):
            self._is_running = False
            self.wait_condition.wakeAll() # 唤醒 'run' 循环以便退出
    
    def pause(self) -> None:
        """
        暂停优化。
        """
        with QMutexLocker(self.mutex):
            self._is_paused = True
            self.wait_condition.wakeAll()
    
    def resume(self) -> None:
        """
        恢复优化。
        """
        with QMutexLocker(self.mutex):
            self._is_paused = False
            self.wait_condition.wakeAll()

    def set_scale_factors(self, rotation: float | None = None,
                          translation: float | None = None,
                          joints: float | None = None) -> None:
        """在运行时调整参数缩放因子。传入的值会覆盖当前设置。"""
        with QMutexLocker(self.mutex):
            if rotation is not None:
                self.scale_factors['rotation'] = float(rotation)
            if translation is not None:
                self.scale_factors['translation'] = float(translation)
            if joints is not None:
                self.scale_factors['joints'] = float(joints)
            # 重置优化器以避免缩放变化导致的动量不一致
            if self.optimizer is not None:
                try:
                    self.optimizer.reset()
                except Exception:
                    pass
            # 触发一次手动更新以让可视化立即反映（不必等下一次优化步）
            self._manual_update = True
            self.wait_condition.wakeAll()
    
    def _setup_optimization(self) -> None:
        """
        设置优化器和能量函数
        
        可以通过修改这个方法来切换不同的优化器和能量组合
        """
        # 创建能量函数
        anchor_energy = AnchorPointEnergy(weight=ENERGY_WEIGHTS['anchor'])
        joint_limit_energy = JointLimitEnergy(weight=ENERGY_WEIGHTS['joint_limit'], margin=0.1)
        penetration_energy = PenetrationAvoidanceEnergy(weight=ENERGY_WEIGHTS['penetration'], margin=0.0)
        self_collision_energy = SelfCollisionAvoidanceEnergy(weight=ENERGY_WEIGHTS['self_collision'], margin=0.005)
        # collision_energy = CollisionAvoidanceEnergy(weight=0.1, margin=0.01)
        
        # 组合能量函数
        self.energy_function = CompositeEnergy([
            anchor_energy,
            joint_limit_energy,
            penetration_energy,
            self_collision_energy,
            # collision_energy  # 暂时禁用，需要实现碰撞检测
        ])
        
        # 创建高性能 Optax Adam 优化器
        self.optimizer = create_adam(
            learning_rate=DEFAULT_LEARNING_RATE,
            clip_grad=DEFAULT_CLIP_GRAD
        )
    
    def set_optimizer(self, optimizer_type: str = "adam", **kwargs):
        """
        动态切换优化器
        
        Args:
            optimizer_type: "adam", "adamw", "lion", "sgd"
            **kwargs: 优化器参数
        """
        from .optimization import create_adamw, create_lion, GradientDescentOptimizer
        
        with QMutexLocker(self.mutex):
            if optimizer_type.lower() == "adam":
                self.optimizer = create_adam(**kwargs)
            elif optimizer_type.lower() == "adamw":
                self.optimizer = create_adamw(**kwargs)
            elif optimizer_type.lower() == "lion":
                self.optimizer = create_lion(**kwargs)
            elif optimizer_type.lower() in ["gd", "sgd"]:
                self.optimizer = GradientDescentOptimizer(**kwargs)
            else:
                print(f"OptimizationThread: 未知优化器类型 '{optimizer_type}'")
                return
            
            print(f"OptimizationThread: 优化器已切换为 {optimizer_type}")

    # 新增：应用 hand_config/<hand_name>.yaml 或 default.yaml 进行初始位姿与关节初始化
    @pyqtSlot(str)
    def apply_hand_config(self, hand_name: str) -> None:
        """
        根据 hand_config/<hand_name>.yaml 或 hand_config/default.yaml 初始化基座位姿与关节值。

        YAML 格式：
        base_pose:
          translation_m: [x, y, z]
          rpy_deg: [roll, pitch, yaw]
        joints: "default" 或 {joint_name: value_in_radians}
        """
        # 计算 hand_config 目录
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            config_dir = os.path.join(project_root, 'hand_config')
            specific_cfg = os.path.join(config_dir, f"{hand_name}.yaml")
            default_cfg = os.path.join(config_dir, 'default.yaml')

            cfg_path = specific_cfg if os.path.exists(specific_cfg) else default_cfg
            if not os.path.exists(cfg_path):
                print(f"OptimizationThread: 未找到 hand 配置文件，跳过初始化: {cfg_path}")
                return

            print(f"OptimizationThread: 准备加载 hand_config 文件: {cfg_path} (hand_name请求='{hand_name}') robot_ready={self.robot is not None}")
            cfg = load_hand_config(cfg_path)
            print(f"OptimizationThread: hand_config 内容: {cfg}")

            base_pose_cfg = (cfg.get('base_pose') or {})
            with QMutexLocker(self.mutex):
                self.current_base_pose = base_pose_from_cfg(base_pose_cfg)
                print(f"OptimizationThread: 已设置 current_base_pose 平移={self.current_base_pose[:3,3].tolist()}")

            # 关节初始化（需要 robot 已设置好以获取关节名与上下限）
            with QMutexLocker(self.mutex):
                if self.robot is None:
                    # Robot 尚未就绪，记录待应用配置名称，等 set_pyroki_robot 后再应用
                    self._pending_hand_config_name = hand_name
                    print("OptimizationThread: Robot 尚未设置，记录 _pending_hand_config_name，等待 set_pyroki_robot 后再次 apply。")
                    # 仍然触发一次 FK 以更新基座位姿（若需要）
                    self._manual_update = True
                    self.wait_condition.wakeAll()
                    return

                joints_cfg = cfg.get('joints', 'default')
                print(f"OptimizationThread: 准备初始化关节，joints_cfg 类型={type(joints_cfg)}")
                new_joint_values = init_joints_from_cfg(self.robot, joints_cfg)
                print(f"OptimizationThread: 生成初始关节字典，大小={len(new_joint_values)} 示例={(list(new_joint_values.items())[:3])}")

                # 应用到当前/目标
                self.current_joint_values = new_joint_values.copy()
                self.target_joint_values = new_joint_values.copy()

                # 触发一次手动更新以广播到 UI（关节与基座）
                self._manual_update = True
                self._is_paused = True
                self._needs_optimization = False
                self.wait_condition.wakeAll()

            try:
                bt = self.current_base_pose[:3, 3].tolist()
                print(f"OptimizationThread: hand_config 应用完成: {os.path.basename(cfg_path)} 基座平移={bt} 关节数={len(self.current_joint_values)}")
            except Exception as _e_dbg:
                print(f"OptimizationThread: hand_config 应用完成(获取平移失败): {os.path.basename(cfg_path)} err={_e_dbg}")
        except Exception as e:
            print(f"OptimizationThread: 应用 hand_config 失败: {e}")


    # --- 线程主循环 ---

    def run(self) -> None:
        """
        QThread 的主入口点。
        这个循环会持续运行，等待被唤醒以执行优化或FK。
        """
        print("OptimizationThread: 线程已启动。")
        
        while self._is_running:
            # --- 1. 等待工作 ---
            with QMutexLocker(self.mutex):
                # 如果没有工作 (优化=False, 手动=False) 并且在运行中，则等待
                while (not self._needs_optimization 
                       and not self._manual_update 
                       and self._pending_precompute_mesh is None
                       and self._is_running
                       and not self._is_paused):
                    
                    self.wait_condition.wait(self.mutex) # 自动释放锁并等待
                
                # 被唤醒后，检查是否是
                if not self._is_running:
                    break # 收到停止信号
                
                # --- 2. 准备工作 (快照状态) ---
                local_anchors = list(self.anchor_pairs)
                
                is_optimizing = self._needs_optimization
                is_manual_update = self._manual_update
                local_precompute_mesh = self._pending_precompute_mesh
                # 一次性取走待处理的预计算任务
                self._pending_precompute_mesh = None
                
                if is_manual_update:
                    # 手动更新优先，并停止当前的优化
                    self.current_joint_values = self.target_joint_values.copy()
                    is_optimizing = False
                    self._needs_optimization = False
                
                # 如果没有锚点，自动停止优化
                if not local_anchors:
                    is_optimizing = False
                    self._needs_optimization = False
                    
                # 重置一次性触发器
                self._manual_update = False

            # --- 3. 执行计算 ---
            try:
                # A0. 处理距离场预计算
                if local_precompute_mesh is not None:
                    if hasattr(self.energy_function, 'energy_functions'):
                        for energy in self.energy_function.energy_functions:
                            if isinstance(energy, PenetrationAvoidanceEnergy):
                                print("OptimizationThread: 正在后台预计算物体距离场 (SDF)…")
                                # 提供中断回调用于窗口关闭时尽快结束预计算
                                energy.precompute_distance_field(
                                    local_precompute_mesh,
                                    abort_fn=lambda: (not self._is_running)
                                )
                                # 根据能量对象的 _last_cache_hit 输出 UI 友好提示
                                hit = getattr(energy, '_last_cache_hit', None)
                                if hit is True:
                                    msg = "命中物体SDF缓存，已快速加载距离场。"
                                elif hit is False:
                                    msg = "已生成物体SDF并写入缓存。"
                                else:
                                    msg = "物体距离场处理完成。"
                                print(f"OptimizationThread: {msg}")
                                self.sdf_cache_message_signal.emit(msg)
                                print("OptimizationThread: 物体距离场预计算完成。")
                                break
                
                if is_optimizing:
                    new_pose, new_joints = self._optimization_step(
                        local_anchors, 
                        self.current_base_pose, 
                        self.current_joint_values
                    )
                else:
                    new_pose = self.current_base_pose
                    new_joints = self.current_joint_values
                
                link_poses_dict = self._run_forward_kinematics(
                    new_pose, 
                    new_joints
                )

            except Exception as e:
                print(f"OptimizationThread: 计算错误: {e}")
                import traceback
                traceback.print_exc()
                with QMutexLocker(self.mutex):
                    self._needs_optimization = False
                continue

            # --- 4. 更新状态并发射信号 ---
            with QMutexLocker(self.mutex):
                self.current_base_pose = new_pose
                self.current_joint_values = new_joints
                base_translation = self.current_base_pose[:3, 3].tolist()
                base_rotation = self.current_base_pose[:3, :3].tolist()
                joint_snapshot = dict(self.current_joint_values)
            
            self.pose_update_signal.emit(link_poses_dict)
            self.base_pose_updated_signal.emit(base_translation, base_rotation)
            self.joint_values_updated_signal.emit(joint_snapshot)
            
            self.msleep(OPTIMIZATION_SLEEP_MS)
            
        print("OptimizationThread: 线程已停止。")

    # --- 公共槽 ---

    @pyqtSlot(list)
    def trigger_optimization(self, anchor_pairs: list) -> None:
        with QMutexLocker(self.mutex):
            self.anchor_pairs = anchor_pairs
            if self.anchor_pairs:
                self._needs_optimization = True
                try:
                    if self.optimizer is not None:
                        self.optimizer.reset()
                except Exception:
                    pass
                print(f"OptimizationThread: 已收到 {len(anchor_pairs)} 个锚点，开始优化。")
            else:
                self._needs_optimization = False
                print("OptimizationThread: 锚点列表为空，停止优化。")
            self.wait_condition.wakeAll()

    @pyqtSlot(object)
    def set_object_mesh(self, mesh) -> None:
        with QMutexLocker(self.mutex):
            self.object_mesh = mesh
            self._pending_precompute_mesh = mesh
            self.wait_condition.wakeAll()
            print(f"OptimizationThread: 已设置物体网格（预计算任务已入队）。顶点数: {len(mesh.points) if hasattr(mesh, 'points') else '未知'}")
    
    def set_pyroki_robot(self, robot: pk.Robot) -> None:
        with QMutexLocker(self.mutex):
            if pk is None:
                print("OptimizationThread: 错误: pyroki 未导入，无法设置 robot。")
                return
            self.robot = robot
            if self.robot is None:
                print("OptimizationThread: 警告: 收到了空的 Robot 对象。")
                return
            self.current_joint_values.clear()
            self.target_joint_values.clear()
            try:
                joint_names = self.robot.joints.actuated_names
                lower_limits = self.robot.joints.lower_limits
                upper_limits = self.robot.joints.upper_limits
                average_limits = (lower_limits + upper_limits) / 2.0
                joint_info_list = []
                for i, name in enumerate(joint_names):
                    joint_info = {
                        'name': name,
                        'min': float(lower_limits[i]),
                        'max': float(upper_limits[i]),
                        'default': float(average_limits[i])
                    }
                    joint_info_list.append(joint_info)
                for i, name in enumerate(joint_names):
                    default_val = float(average_limits[i])
                    self.current_joint_values[name] = default_val
                    self.target_joint_values[name] = default_val
                print(f"OptimizationThread: 已成功设置 pyroki.Robot 实例。 关节: {list(self.current_joint_values.keys())}")
                self.joint_info_signal.emit(joint_info_list)
                self._manual_update = True
                self.wait_condition.wakeAll()
            except Exception as e:
                print(f"OptimizationThread: 解析 pyroki.Robot 时出错: {e}")
                self.robot = None
            if self.anchor_pairs:
                self._needs_optimization = True
                self.wait_condition.wakeAll()

        # Robot 就绪后，如果之前有待应用的 hand 配置，则立即应用
        try:
            if self._pending_hand_config_name:
                name = self._pending_hand_config_name
                self._pending_hand_config_name = None
                print(f"OptimizationThread: 检测到待应用 hand 配置 _pending_hand_config_name='{name}'，现在应用。")
                self.apply_hand_config(name)
        except Exception:
            pass

    @pyqtSlot(list)
    def set_visual_link_names(self, link_names: list[str]) -> None:
        """接收仅用于渲染的 link 名称集合，优化线程据此过滤 FK 输出。"""
        with QMutexLocker(self.mutex):
            self._visual_link_names = set(link_names)

    @pyqtSlot(dict)
    def set_hand_keypoints(self, keypoints: dict[str, np.ndarray]) -> None:
        with QMutexLocker(self.mutex):
            self.hand_keypoints = keypoints.copy()
            if hasattr(self.energy_function, 'energy_functions'):
                for energy in self.energy_function.energy_functions:
                    if isinstance(energy, PenetrationAvoidanceEnergy):
                        energy.set_key_points(self.hand_keypoints)
                        print(f"OptimizationThread: 已设置 {sum(len(points) for points in keypoints.values())} 个关键点到穿透避免能量函数")
                        break
            print(f"OptimizationThread: 已接收手关键点，共 {len(keypoints)} 个link，{sum(len(points) for points in keypoints.values())} 个点")

    @pyqtSlot(dict)
    def set_link_spheres(self, spheres: dict[str, list]) -> None:
        with QMutexLocker(self.mutex):
            self.link_spheres = spheres.copy()
            if hasattr(self.energy_function, 'energies'):
                for energy in self.energy_function.energies:
                    if isinstance(energy, SelfCollisionAvoidanceEnergy):
                        energy.set_link_spheres(self.link_spheres)
                        total_spheres = sum(len(sphere_list) for sphere_list in spheres.values())
                        print(f"OptimizationThread: 已设置 {total_spheres} 个球体到自碰撞避免能量函数")
                        break
            total_spheres = sum(len(sphere_list) for sphere_list in spheres.values())
            print(f"OptimizationThread: 已接收link球体，共 {len(spheres)} 个link，{total_spheres} 个球体")

    @pyqtSlot(str, float)
    def set_manual_joint(self, joint_name: str, value: float) -> None:
        with QMutexLocker(self.mutex):
            if joint_name in self.target_joint_values:
                self.target_joint_values[joint_name] = value
                self._manual_update = True
                self._needs_optimization = False
                self._is_paused = True
                self.wait_condition.wakeAll()

    @pyqtSlot(float, float, float)
    def set_base_translation(self, x: float, y: float, z: float) -> None:
        with QMutexLocker(self.mutex):
            self.current_base_pose[:3, 3] = np.array([x, y, z], dtype=float)
            self._manual_update = True
            self._needs_optimization = False
            self._is_paused = True
            self.wait_condition.wakeAll()


    @pyqtSlot(float, float, float)
    def set_base_rotation(self, roll: float, pitch: float, yaw: float) -> None:
        with QMutexLocker(self.mutex):
            rotation_matrix = euler_rpy_to_matrix(roll, pitch, yaw)
            self.current_base_pose[:3, :3] = rotation_matrix
            self._manual_update = True
            self._needs_optimization = False
            self._is_paused = True
            self.wait_condition.wakeAll()

    def _prepare_anchor_data(self, anchors: list[dict]) -> list[dict]:
        if not anchors or self.robot is None:
            return []
        prepared_anchors = []
        link_names = self.robot.links.names
        for anchor in anchors:
            new_anchor = anchor.copy()
            link_name = anchor.get('hand_link_name', '')
            if link_name.startswith('static_hand_'):
                link_name = link_name[len('static_hand_'):]
            try:
                link_idx = link_names.index(link_name)
                new_anchor['hand_link_idx'] = link_idx
                prepared_anchors.append(new_anchor)
            except ValueError:
                print(f"OptimizationThread: 警告: 未找到 link '{link_name}'")
                continue
        return prepared_anchors

    def _optimization_step(self, 
                          anchors: list[dict], 
                          current_pose: np.ndarray, 
                          current_joints: dict[str, float]
                          ) -> tuple[np.ndarray, dict[str, float]]:
        if not anchors or self.robot is None or self.optimizer is None:
            return current_pose.copy(), current_joints.copy()
        prepared_anchors = self._prepare_anchor_data(anchors)
        if not prepared_anchors:
            print("OptimizationThread: 没有有效的锚点，跳过优化")
            return current_pose.copy(), current_joints.copy()
        joint_names = self.robot.joints.actuated_names
        state = OptimizerState.from_numpy(
            current_pose,
            current_joints,
            joint_names,
            self.scale_factors
        )
        def loss_fn(opt_state):
            return self.energy_function.compute(
                opt_state,
                self.robot,
                anchor_pairs=prepared_anchors,
                object_mesh=self.object_mesh
            )
        try:
            new_state, loss_value = self.optimizer.step(state, loss_fn)
            if hasattr(self, '_step_counter'):
                self._step_counter += 1
            else:
                self._step_counter = 0
            if self._step_counter % 30 == 0:
                detailed_energies = self.energy_function.compute_detailed(
                    new_state, self.robot, anchor_pairs=prepared_anchors, object_mesh=self.object_mesh
                )
                energy_str = format_energy_breakdown(detailed_energies)
                print(f"OptimizationThread: Step {self._step_counter}, Total Loss: {loss_value:.4f} ({energy_str})")
            new_pose = np.array(new_state.to_base_pose_matrix())
            new_joints = new_state.to_joint_dict()
            if self._step_counter % 30 == 0:
                try:
                    old_pose = np.array(state.to_base_pose_matrix())
                    dt, drot = pose_delta(old_pose, new_pose)
                    print(f"OptimizationThread: Δtrans={dt:.6f} m, Δrot={drot:.3f} deg (scaled via {self.scale_factors})")
                except Exception:
                    pass
            return new_pose, new_joints
        except Exception as e:
            print(f"OptimizationThread: 优化步骤失败: {e}")
            import traceback
            traceback.print_exc()
            return current_pose.copy(), current_joints.copy()

    def _run_forward_kinematics(self, 
                                base_pose_mat: np.ndarray, 
                                joint_values_dict: dict[str, float]
                                ) -> dict[str, np.ndarray]:
        if self.robot is None or jaxlie is None:
            if self.robot is None: print("FK GZ: Robot not set")
            if jaxlie is None: print("FK GZ: Jaxlie not set")
            return {}
        try:
            ordered_joint_names = self.robot.joints.actuated_names
            if not all(name in joint_values_dict for name in ordered_joint_names):
                return {}
            cfg_array = jnp.array(
                [joint_values_dict[name] for name in ordered_joint_names]
            )
            link_poses_rel_root_wxyz = self.robot.forward_kinematics(cfg_array)
            T_world_base = jaxlie.SE3.from_matrix(base_pose_mat)
            link_poses_dict = {}
            link_names = self.robot.links.names
            for i, link_name in enumerate(link_names):
                # 如果提供了可视化 link 集，且当前 link 不在其中则跳过
                if self._visual_link_names and link_name not in self._visual_link_names:
                    continue
                actor_name = self.actor_name_prefix + link_name
                T_root_link = jaxlie.SE3(link_poses_rel_root_wxyz[i])
                T_world_link = T_world_base @ T_root_link
                link_poses_dict[actor_name] = np.array(T_world_link.as_matrix())
            return link_poses_dict
        except Exception as e:
            print(f"OptimizationThread: FK 计算失败: {e}")
            import traceback
            traceback.print_exc()
            return {}
