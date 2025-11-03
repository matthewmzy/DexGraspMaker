# optimization_thread.py

import time
import numpy as np
from PyQt6.QtCore import (
    QThread, QMutex, QWaitCondition, pyqtSignal, pyqtSlot, QMutexLocker, QObject
)
import pyroki as pk
import jaxlie
from jax import numpy as jnp

# 导入优化模块
from optimization import (
    OptimizerState,
    AnchorPointEnergy,
    JointLimitEnergy,
    CollisionAvoidanceEnergy,
    PenetrationAvoidanceEnergy,
    SelfCollisionAvoidanceEnergy,
    CompositeEnergy,
    create_adam,
)


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

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        
        # --- 线程控制 ---
        self.mutex = QMutex()
        self.wait_condition = QWaitCondition()
        self._is_running = True
        self._needs_optimization = False # 优化循环的开关
        self._manual_update = False      # 手动关节更新的开关
        self._is_paused = False          # 优化暂停开关

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
        self.actor_name_prefix = "dyn_hand_"
        
        # 6. 优化器和能量函数
        self.optimizer = None
        self.energy_function = None
        self._setup_optimization()

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
    
    def _setup_optimization(self) -> None:
        """
        设置优化器和能量函数
        
        可以通过修改这个方法来切换不同的优化器和能量组合
        """
        # 创建能量函数
        anchor_energy = AnchorPointEnergy(weight=1.0)
        joint_limit_energy = JointLimitEnergy(weight=0.5, margin=0.1)
        penetration_energy = PenetrationAvoidanceEnergy(weight=2.0, margin=0.0)  # 直接使用穿透距离
        self_collision_energy = SelfCollisionAvoidanceEnergy(weight=0.3, margin=0.005)
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
            learning_rate=0.01,
            clip_grad=1.0
        )
    
    def set_optimizer(self, optimizer_type: str = "adam", **kwargs):
        """
        动态切换优化器
        
        Args:
            optimizer_type: "adam", "adamw", "lion", "sgd"
            **kwargs: 优化器参数
        """
        from optimization import create_adamw, create_lion, GradientDescentOptimizer
        
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
                       and self._is_running
                       and not self._is_paused):
                    
                    self.wait_condition.wait(self.mutex) # 自动释放锁并等待
                
                # 被唤醒后，检查是否是
                if not self._is_running:
                    break # 收到停止信号
                
                # --- 2. 准备工作 (快照状态) ---
                # 复制状态，以便我们可以在没有锁的情况下执行长时间的计算
                local_anchors = list(self.anchor_pairs)
                
                is_optimizing = self._needs_optimization
                is_manual_update = self._manual_update
                
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
                
                # (锁在 'with' 块结束时自动释放)

            # --- 3. 执行计算 (没有锁！) ---
            # 这是此线程中耗时的部分
            
            try:
                if is_optimizing:
                    # A. 运行一步梯度下降（真实优化）
                    new_pose, new_joints = self._optimization_step(
                        local_anchors, 
                        self.current_base_pose, 
                        self.current_joint_values
                    )
                    
                else:
                    # B. 仅手动更新或空闲
                    # 我们只需要使用当前状态运行FK
                    new_pose = self.current_base_pose
                    new_joints = self.current_joint_values
                
                
                # C. 运行正向运动学 (FK)
                # 无论哪种情况，我们都需要计算FK以进行可视化
                
                link_poses_dict = self._run_forward_kinematics(
                    new_pose, 
                    new_joints
                )

            except Exception as e:
                print(f"OptimizationThread: 计算错误: {e}")
                import traceback
                traceback.print_exc()
                # 发生错误时，停止优化
                with QMutexLocker(self.mutex):
                    self._needs_optimization = False
                continue

            # --- 4. 更新状态并发射信号 ---
            with QMutexLocker(self.mutex):
                # 将计算结果存回状态变量
                self.current_base_pose = new_pose
                self.current_joint_values = new_joints
                base_translation = self.current_base_pose[:3, 3].tolist()
                base_rotation = self.current_base_pose[:3, :3].tolist()
                joint_snapshot = dict(self.current_joint_values)
            
            # 发射信号 (没有锁！)
            # 'vista_widget' 将接收此信号
            self.pose_update_signal.emit(link_poses_dict)
            self.base_pose_updated_signal.emit(base_translation, base_rotation)
            self.joint_values_updated_signal.emit(joint_snapshot)
            
            # 检查是否有待可视化的数据
            self._check_and_perform_visualization()
            
            # 休息 16ms (~60 FPS) 以产生平滑的动画效果
            # 并防止 CPU 100% 占用
            self.msleep(16)
            
        print("OptimizationThread: 线程已停止。")

    # --- 公共槽 (Public Slots) ---
    # 这些槽由 main_thread 调用 (通过信号连接)

    @pyqtSlot(list)
    def trigger_optimization(self, anchor_pairs: list) -> None:
        """
        槽：当锚点列表更新时，由 data_manager 调用。
        """
        with QMutexLocker(self.mutex):
            self.anchor_pairs = anchor_pairs
            if self.anchor_pairs:
                self._needs_optimization = True
                print(f"OptimizationThread: 已收到 {len(anchor_pairs)} 个锚点，开始优化。")
            else:
                self._needs_optimization = False # 没有锚点，停止优化
                print("OptimizationThread: 锚点列表为空，停止优化。")
                
            self.wait_condition.wakeAll() # 唤醒 'run' 循环

    @pyqtSlot(object)
    def set_object_mesh(self, mesh) -> None:
        """
        设置物体网格，用于穿透避免计算
        
        Args:
            mesh: pyvista.PolyData 或 trimesh.Trimesh 对象
        """
        with QMutexLocker(self.mutex):
            self.object_mesh = mesh
            
            # 预计算距离场用于精确的穿透检测
            if hasattr(self.energy_function, 'energy_functions'):
                for energy in self.energy_function.energy_functions:
                    if isinstance(energy, PenetrationAvoidanceEnergy):
                        energy.precompute_distance_field(mesh)
                        break
            
            print(f"OptimizationThread: 设置了物体网格，顶点数: {len(mesh.points) if hasattr(mesh, 'points') else '未知'}")
    
    def set_pyroki_robot(self, robot: pk.Robot) -> None:
        """
        槽：当 data_manager 加载完 URDF 后调用。
        """
        with QMutexLocker(self.mutex):
            if pk is None:
                print("OptimizationThread: 错误: pyroki 未导入，无法设置 robot。")
                return

            self.robot = robot
            if self.robot is None:
                print("OptimizationThread: 警告: 收到了空的 Robot 对象。")
                return

            # 从 pyroki robot 初始化关节状态
            self.current_joint_values.clear()
            self.target_joint_values.clear()
            
            try:
                joint_names = self.robot.joints.actuated_names
                lower_limits = self.robot.joints.lower_limits
                upper_limits = self.robot.joints.upper_limits
                average_limits = (lower_limits + upper_limits) / 2.0
                # 创建关节信息列表，用于发送给 controls_widget
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
                
                # 发射关节信息信号给 controls_widget
                self.joint_info_signal.emit(joint_info_list)
                
                # 加载新模型后，总是触发一次手动更新以显示初始姿势
                self._manual_update = True
                self.wait_condition.wakeAll()
                
            except Exception as e:
                print(f"OptimizationThread: 解析 pyroki.Robot 时出错: {e}")
                self.robot = None # 设置失败

            # 加载新模型后，如果存在锚点，则触发一次优化
            if self.anchor_pairs:
                self._needs_optimization = True
                self.wait_condition.wakeAll()

    @pyqtSlot(dict)
    def set_hand_keypoints(self, keypoints: dict[str, np.ndarray]) -> None:
        """
        槽：当 data_manager 加载完关键点后调用。
        
        Args:
            keypoints: {link_name: points_array} 字典
        """
        with QMutexLocker(self.mutex):
            self.hand_keypoints = keypoints.copy()
            
            # 更新能量函数中的关键点
            if hasattr(self.energy_function, 'energy_functions'):
                for energy in self.energy_function.energy_functions:
                    if isinstance(energy, PenetrationAvoidanceEnergy):
                        energy.set_key_points(self.hand_keypoints)
                        print(f"OptimizationThread: 已设置 {sum(len(points) for points in keypoints.values())} 个关键点到穿透避免能量函数")
                        break
            
            print(f"OptimizationThread: 已接收手关键点，共 {len(keypoints)} 个link，{sum(len(points) for points in keypoints.values())} 个点")

    @pyqtSlot(dict)
    def set_link_spheres(self, spheres: dict[str, list]) -> None:
        """
        槽：当 data_manager 加载完link球体数据后调用。

        Args:
            spheres: {link_name: [(center, radius), ...]} 字典
        """
        with QMutexLocker(self.mutex):
            self.link_spheres = spheres.copy()

            # 更新能量函数中的球体数据
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
        """
        槽：当 controls_widget 中的滑块移动时调用。
        """
        with QMutexLocker(self.mutex):
            if joint_name in self.target_joint_values:
                self.target_joint_values[joint_name] = value
                self._manual_update = True # 标记为手动更新
                self._needs_optimization = False # 手动操作会覆盖优化
                self._is_paused = True # 手动调节时暂停优化

                self.wait_condition.wakeAll() # 唤醒 'run' 循环以应用FK

    @pyqtSlot(float, float, float)
    def set_base_translation(self, x: float, y: float, z: float) -> None:
        """槽：从 UI 调整基座平移"""
        with QMutexLocker(self.mutex):
            self.current_base_pose[:3, 3] = np.array([x, y, z], dtype=float)
            self._manual_update = True
            self._needs_optimization = False
            self._is_paused = True
            self.wait_condition.wakeAll()

    @staticmethod
    def _euler_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
        """将 roll-pitch-yaw (按 ZYX 顺序) 转换为 3x3 旋转矩阵"""
        cx, cy, cz = np.cos(roll), np.cos(pitch), np.cos(yaw)
        sx, sy, sz = np.sin(roll), np.sin(pitch), np.sin(yaw)

        R_x = np.array(
            [[1.0, 0.0, 0.0],
             [0.0, cx, -sx],
             [0.0, sx, cx]]
        )
        R_y = np.array(
            [[cy, 0.0, sy],
             [0.0, 1.0, 0.0],
             [-sy, 0.0, cy]]
        )
        R_z = np.array(
            [[cz, -sz, 0.0],
             [sz, cz, 0.0],
             [0.0, 0.0, 1.0]]
        )

        return R_z @ R_y @ R_x

    @pyqtSlot(float, float, float)
    def set_base_rotation(self, roll: float, pitch: float, yaw: float) -> None:
        """槽：从 UI 调整基座旋转（弧度）"""
        with QMutexLocker(self.mutex):
            rotation_matrix = self._euler_to_matrix(roll, pitch, yaw)
            self.current_base_pose[:3, :3] = rotation_matrix
            self._manual_update = True
            self._needs_optimization = False
            self._is_paused = True
            self.wait_condition.wakeAll()

    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    # ▼▼▼▼▼▼ [真实优化实现] ▼▼▼▼▼▼
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼

    def _prepare_anchor_data(self, anchors: list[dict]) -> list[dict]:
        """
        准备锚点数据，添加 link 索引
        
        Args:
            anchors: 原始锚点列表，包含 hand_link_name
            
        Returns:
            处理后的锚点列表，添加了 hand_link_idx
        """
        if not anchors or self.robot is None:
            return []
        
        prepared_anchors = []
        link_names = self.robot.links.names
        
        for anchor in anchors:
            # 创建副本
            new_anchor = anchor.copy()
            
            # 查找 link 索引
            link_name = anchor.get('hand_link_name', '')
            
            # 移除可能的前缀（如果存在）
            # data_manager 可能使用了 'static_hand_' 前缀
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
        """
        执行一步优化（使用梯度下降和能量函数）
        
        Args:
            anchors: 锚点对列表
            current_pose: 当前基座位姿 (4x4 矩阵)
            current_joints: 当前关节值字典
            
        Returns:
            (new_pose, new_joints): 优化后的位姿和关节值
        """
        if not anchors or self.robot is None or self.optimizer is None:
            return current_pose.copy(), current_joints.copy()
        
        # 1. 准备锚点数据（添加 link 索引）
        prepared_anchors = self._prepare_anchor_data(anchors)
        
        if not prepared_anchors:
            print("OptimizationThread: 没有有效的锚点，跳过优化")
            return current_pose.copy(), current_joints.copy()
        
        # 2. 创建优化状态
        joint_names = self.robot.joints.actuated_names
        state = OptimizerState.from_numpy(
            current_pose,
            current_joints,
            joint_names
        )
        
        # 3. 定义损失函数
        def loss_fn(opt_state):
            return self.energy_function.compute(
                opt_state,
                self.robot,
                anchor_pairs=prepared_anchors,
                object_mesh=self.object_mesh
            )
        
        # 4. 执行一步优化
        try:
            new_state, loss_value = self.optimizer.step(state, loss_fn)
            
            # 可选：打印损失值用于调试
            if hasattr(self, '_step_counter'):
                self._step_counter += 1
            else:
                self._step_counter = 0
            
            if self._step_counter % 30 == 0:  # 每30步打印一次
                # 获取详细能量
                detailed_energies = self.energy_function.compute_detailed(
                    new_state, self.robot, anchor_pairs=prepared_anchors, object_mesh=self.object_mesh
                )
                energy_str = ", ".join([f"{k}: {v:.4f}" for k, v in detailed_energies.items()])
                print(f"OptimizationThread: Step {self._step_counter}, Total Loss: {loss_value:.4f} ({energy_str})")
            
            # 5. 转换回 NumPy 格式
            new_pose = np.array(new_state.to_base_pose_matrix())
            new_joints = new_state.to_joint_dict()
            
            return new_pose, new_joints
            
        except Exception as e:
            print(f"OptimizationThread: 优化步骤失败: {e}")
            import traceback
            traceback.print_exc()
            return current_pose.copy(), current_joints.copy()

    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    # ▲▲▲▲▲▲ [真实优化实现结束] ▲▲▲▲▲▲
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

    
    def _run_forward_kinematics(self, 
                                base_pose_mat: np.ndarray, 
                                joint_values_dict: dict[str, float]
                                ) -> dict[str, np.ndarray]:
        """
        [真实实现] 使用 pyroki 运行正向运动学 (FK)。
        
        这个函数用于*可视化*，所以它在 QThread 中运行，
        但不需要在 JAX @jit 内部。
        """
        if self.robot is None or jaxlie is None:
            if self.robot is None: print("FK GZ: Robot not set")
            if jaxlie is None: print("FK GZ: Jaxlie not set")
            return {}

        try:
            # 1. 将关节字典 {name: val} 转换为有序数组 [val1, val2, ...]
            ordered_joint_names = self.robot.joints.actuated_names
            
            # 确保 joint_values_dict 已经包含了所有需要的关节
            if not all(name in joint_values_dict for name in ordered_joint_names):
                # print("FK 警告: 关节值尚未完全初始化。")
                return {} # 尚未准备好

            cfg_array = jnp.array(
                [joint_values_dict[name] for name in ordered_joint_names]
            )

            # 2. 调用 pyroki FK。返回 (link_count, 7)
            link_poses_rel_root_wxyz = self.robot.forward_kinematics(cfg_array)
            
            # 3. 获取优化的基座位姿 (4x4 NumPy -> jaxlie.SE3)
            T_world_base = jaxlie.SE3.from_matrix(base_pose_mat)
            
            # 4. 转换并应用基座位姿
            link_poses_dict = {}
            link_names = self.robot.links.names # pyroki 存储的 link name 列表
            
            for i, link_name in enumerate(link_names):
                actor_name = self.actor_name_prefix + link_name
                
                # 获取 link i 相对于 root 的位姿
                T_root_link = jaxlie.SE3(link_poses_rel_root_wxyz[i])
                
                # 计算 link 在世界坐标系下的最终位姿
                T_world_link = T_world_base @ T_root_link
                
                # 转换为 4x4 矩阵以进行可视化
                link_poses_dict[actor_name] = np.array(T_world_link.as_matrix())
                
            return link_poses_dict
            
        except Exception as e:
            print(f"OptimizationThread: FK 计算失败: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _check_and_perform_visualization(self):
        """检查是否有待可视化的数据，如果有则执行可视化"""
        if hasattr(self.energy_function, 'energy_functions'):
            for energy in self.energy_function.energy_functions:
                if isinstance(energy, PenetrationAvoidanceEnergy):
                    viz_data = energy.get_pending_visualization_data()
                    if viz_data is not None:
                        # 在后台线程中可视化
                        import threading
                        def visualize_async():
                            try:
                                # 转换为numpy数组
                                points_np = np.array(viz_data['points'])
                                distances_np = np.array(viz_data['distances'])
                                
                                # 保存调试数据
                                with open('/home/ubuntu/Documents/DexGraspMaker/debug_penetration.txt', 'w') as f:
                                    f.write(f"关键点数量: {len(points_np)}\n")
                                    f.write(f"穿透点数量: {np.sum(distances_np > 0)}\n")
                                    f.write(f"最大穿透深度: {distances_np.max():.4f}m\n")
                                    f.write(f"最小距离: {distances_np.min():.4f}m\n")
                                    f.write("前10个关键点和距离:\n")
                                    for i in range(min(10, len(points_np))):
                                        f.write(".4f")
                                
                                # 执行可视化
                                energy._visualize_keypoints_and_mesh(viz_data['object_mesh'], points_np, distances_np)
                            except Exception as e:
                                print(f"可视化失败: {e}")
                        
                        thread = threading.Thread(target=visualize_async, daemon=True)
                        thread.start()
                    break


# --- 用于独立测试 ---
# (独立测试代码保持不变，但会因缺少 set_hand_model 而失败)
# (你需要更新它以使用 set_pyroki_robot)
if __name__ == '__main__':
    from PyQt6.QtWidgets import QApplication, QPushButton, QWidget, QVBoxLayout
    
    app = QApplication([])
    
    print("测试 OptimizationThread...")
    thread = OptimizationThread()
    
    # [修改] 测试代码需要更新
    print("注意: 独立测试代码需要更新以使用 'set_pyroki_robot'")
    print("      你必须提供一个真实的 pk.Robot 对象来进行测试。")
    print("      跳过 set_hand_model 测试。")
    
    # 2. 模拟设置模型 (已过时)
    # test_links = {"base": None, "link1": None, "link2": None}
    # test_joints = [
    #     {'name': 'j1', 'min': -1, 'max': 1, 'default': 0},
    #     {'name': 'j2', 'min': -1, 'max': 1, 'default': 0}
    # ]
    # thread.set_hand_model(test_links, test_joints) 
    
    # 3. 连接信号
    def on_pose_update(poses: dict):
        print(f"--- [UI 线程] 收到位姿更新 (共 {len(poses)} 个 links) ---")
        if poses:
            name, pose = list(poses.items())[0]
            print(f"  (示例) {name}: T=({pose[0:3, 3][0]:.3f}, ...)")
            
    thread.pose_update_signal.connect(on_pose_update)
    thread.start()
    
    # 5. 模拟 UI 交互
    def test_optimization():
        print("\n--- [UI 线程] 触发优化 (将失败，因为 robot=None) ---")
        anchors = [{'hand_point': [0.1, 0.0, 0.0], 'obj_point': [1.0, 0.5, 0.2]}]
        thread.trigger_optimization(anchors)
        
    def test_manual_joint():
        print("\n--- [UI 线程] 触发手动关节 (将失败，因为 robot=None) ---")
        thread.set_manual_joint('j1', 0.5)

    def stop_thread():
        print("\n--- [UI 线程] 停止线程 ---")
        thread.stop()
        thread.wait(1000)
        app.quit()

    window = QWidget()
    layout = QVBoxLayout(window)
    btn_opt = QPushButton("1. 触发优化 (将失败)")
    btn_man = QPushButton("2. 设置关节 'j1' = 0.5 (将失败)")
    btn_stop = QPushButton("3. 停止线程并退出")
    
    btn_opt.clicked.connect(test_optimization)
    btn_man.clicked.connect(test_manual_joint)
    btn_stop.clicked.connect(stop_thread)
    
    layout.addWidget(btn_opt)
    layout.addWidget(btn_man)
    layout.addWidget(btn_stop)
    
    window.show()
    app.exec()