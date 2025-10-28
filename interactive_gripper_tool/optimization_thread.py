# optimization_thread.py

import time
import numpy as np
from PyQt6.QtCore import (
    QThread, QMutex, QWaitCondition, pyqtSignal, pyqtSlot, QMutexLocker
)

# -------------------------------------------------------------------
# [占位符] - 辅助函数 (Numpy-based)
# -------------------------------------------------------------------
# 您将使用您自己的 JAX / PyTorch / ROS 运动学库替换这些。

def _create_translation(x: float, y: float, z: float) -> np.ndarray:
    """创建一个 4x4 平移矩阵。"""
    mat = np.eye(4)
    mat[0:3, 3] = [x, y, z]
    return mat

def _create_z_rotation(angle_rad: float) -> np.ndarray:
    """创建一个 4x4 Z轴 旋转矩阵。"""
    mat = np.eye(4)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    mat[0, 0] = cos_a
    mat[0, 1] = -sin_a
    mat[1, 0] = sin_a
    mat[1, 1] = cos_a
    return mat

def _create_pose(trans: list | np.ndarray, 
                 rot_z_rad: float = 0.0) -> np.ndarray:
    """创建一个简单的位姿。"""
    return _create_translation(*trans) @ _create_z_rotation(rot_z_rad)
    
# -------------------------------------------------------------------

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

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        
        # --- 线程控制 ---
        self.mutex = QMutex()
        self.wait_condition = QWaitCondition()
        self._is_running = True
        self._needs_optimization = False # 优化循环的开关
        self._manual_update = False      # 手动关节更新的开关

        # --- 状态变量 (由 mutex 保护) ---
        
        # 1. 机器人模型
        self.link_names: list[str] = [] # e.g., ['base_link', 'link_1', ...]
        self.joint_info: list[dict] = [] # e.g., [{'name': 'j1', 'min': -1, 'max': 1, 'default': 0}]
        
        # 2. 锚点
        self.anchor_pairs: list[dict] = []
        
        # 3. 优化参数 (这是您用 JAX/PyTorch 优化的变量)
        self.current_base_pose: np.ndarray = np.eye(4)
        self.current_joint_values: dict[str, float] = {} # {joint_name: value}
        
        # 4. 手动更新的目标 (来自滑块)
        self.target_joint_values: dict[str, float] = {}
        
        # 5. 名称前缀 (必须与 main_window 中设置的一致)
        self.actor_name_prefix = "dyn_hand_"

    def stop(self) -> None:
        """
        请求线程停止。
        """
        with QMutexLocker(self.mutex):
            self._is_running = False
            self.wait_condition.wakeAll() # 唤醒 'run' 循环以便退出

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
                       and self._is_running):
                    
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
                    # A. 运行一步梯度下降
                    
                    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ [用户: 在此替换] ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
                    # 这是您调用 JAX/PyTorch model.step() 的地方
                    
                    # `_mock_optimization_step` 只是模拟计算
                    new_pose, new_joints = self._mock_optimization_step(
                        local_anchors, 
                        self.current_base_pose, 
                        self.current_joint_values
                    )
                    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ [用户: 替换结束] ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
                    
                else:
                    # B. 仅手动更新或空闲
                    # 我们只需要使用当前状态运行FK
                    new_pose = self.current_base_pose
                    new_joints = self.current_joint_values
                
                
                # C. 运行正向运动学 (FK)
                # 无论哪种情况，我们都需要计算FK以进行可视化
                
                # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ [用户: 在此替换] ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
                # 这是您调用 JAX/PyTorch FK 的地方
                
                # `_mock_forward_kinematics` 只是模拟FK
                link_poses_dict = self._mock_forward_kinematics(
                    new_pose, 
                    new_joints
                )
                # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ [用户: 替换结束] ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

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
            
            # 发射信号 (没有锁！)
            # 'vista_widget' 将接收此信号
            self.pose_update_signal.emit(link_poses_dict)
            
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

    @pyqtSlot(dict, list)
    def set_hand_model(self, links_mesh_dict: dict, joint_info_list: list) -> None:
        """
        槽：当加载新机械手时，由 data_manager 调用。
        (注意：我们假设 main_window 会连接这个)
        """
        with QMutexLocker(self.mutex):
            self.link_names = list(links_mesh_dict.keys())
            self.joint_info = joint_info_list
            
            # 重置/初始化优化状态
            self.current_base_pose = np.eye(4)
            self.current_joint_values = {j['name']: j['default'] for j in self.joint_info}
            self.target_joint_values = self.current_joint_values.copy()
            
            print(f"OptimizationThread: 已设置机械手模型: {len(self.link_names)} links, {len(self.joint_info)} joints.")
            
            # 加载新模型后，如果存在锚点，则触发一次优化
            if self.anchor_pairs:
                self._needs_optimization = True
                self.wait_condition.wakeAll()

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
                
                self.wait_condition.wakeAll() # 唤醒 'run' 循环以应用FK

    # --- 占位符 (Mock) 方法 ---
    #
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    # ▼▼▼▼▼▼ [用户: 用您自己的 JAX / PyTorch 代码替换以下方法] ▼▼▼▼▼▼
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼

    def _mock_optimization_step(self, 
                                anchors: list[dict], 
                                current_pose: np.ndarray, 
                                current_joints: dict[str, float]
                                ) -> tuple[np.ndarray, dict[str, float]]:
        """
        [占位符] 模拟一步优化。
        
        您的实现应该：
        1. 使用 'current_pose' 和 'current_joints'。
        2. 使用 'anchors' 列表计算损失 (loss)。
        3. 计算梯度 (grads)。
        4. 使用 Adam/Optimizer 更新 'current_pose' 和 'current_joints'。
        5. 返回 (new_pose, new_joints)。
        
        --- 模拟逻辑 ---
        这个模拟只是简单地将 base_pose 缓慢移向第一个锚点的目标位置，
        并轻微摆动第一个关节。
        """
        
        # 复制以避免修改原始值
        new_pose = current_pose.copy()
        new_joints = current_joints.copy()
        
        if not anchors:
            return new_pose, new_joints
            
        # 1. 模拟关节优化
        if new_joints:
            first_joint_name = list(new_joints.keys())[0]
            # 只是一个简单的摆动
            new_joints[first_joint_name] = np.sin(time.time() * 2) * 0.5 

        # 2. 模拟位姿优化
        # 目标：将 'hand_point' (在世界系中) 移动到 'obj_point'
        # (这是一个*错误*的假设，hand_point 应该在 link 坐标系中，
        #  但对于模拟来说这足够了)
        
        # 我们用 FK 计算出手部点在世界坐标系中的当前位置
        # (模拟：假设 'hand_point' 在 'current_pose' 的局部坐标中)
        hand_point_local = np.array(anchors[0]['hand_point'] + [1])
        hand_point_world = (current_pose @ hand_point_local)[:3]
        
        obj_point_world = np.array(anchors[0]['obj_point'])
        
        # 计算误差
        error = obj_point_world - hand_point_world
        
        # 向目标移动 5%
        learning_rate = 0.05
        translation_update = error * learning_rate
        
        # 应用更新
        new_pose[0:3, 3] += translation_update

        return new_pose, new_joints

    def _mock_forward_kinematics(self, 
                                 base_pose: np.ndarray, 
                                 joint_values: dict[str, float]
                                 ) -> dict[str, np.ndarray]:
        """
        [占位符] 模拟正向运动学 (FK)。
        
        您的实现应该：
        1. 接受基座位姿和关节值。
        2. 遍历您的运动学链 (Kinematic Chain)。
        3. 计算*每个* link 在世界坐标系中的 4x4 位姿矩阵。
        4. 返回一个字典，其键必须是 '{prefix}{link_name}'。
        
        --- 模拟逻辑 ---
        这个模拟只是将基座位姿赋予所有 links，
        然后根据关节值给每个 link 一个小的 Z 轴旋转。
        """
        
        link_poses_dict = {}
        
        if not self.link_names:
            return {}
            
        # 1. 将基座位姿应用到所有 links
        for link_name in self.link_names:
            actor_name = self.actor_name_prefix + link_name
            link_poses_dict[actor_name] = base_pose
            
        # 2. 模拟关节运动 (应用到对应的 link)
        current_link_transform = base_pose
        
        # (这是一个非常简化的、不正确的链式FK，仅用于演示)
        for i, (joint_name, joint_val) in enumerate(joint_values.items()):
            
            # 假设 joint 'i' 驱动 link 'i'
            if i < len(self.link_names):
                link_name = self.link_names[i]
                actor_name = self.actor_name_prefix + link_name
                
                # 模拟：每个关节在Z轴上旋转，并在X轴上平移
                joint_rot = _create_z_rotation(joint_val)
                link_offset = _create_translation(0.1, 0, 0) # 假设 link 长度为 0.1
                
                # 链式变换
                current_link_transform = current_link_transform @ link_offset @ joint_rot
                
                # 覆盖基座位姿
                link_poses_dict[actor_name] = current_link_transform

        return link_poses_dict

    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    # ▲▲▲▲▲▲ [用户: 替换以上方法] ▲▲▲▲▲▲
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲


# --- 用于独立测试 ---
if __name__ == '__main__':
    from PyQt6.QtWidgets import QApplication, QPushButton, QWidget, QVBoxLayout
    
    app = QApplication([])
    
    print("测试 OptimizationThread...")
    
    # 1. 创建线程
    thread = OptimizationThread()
    
    # 2. 模拟设置模型
    test_links = {"base": None, "link1": None, "link2": None}
    test_joints = [
        {'name': 'j1', 'min': -1, 'max': 1, 'default': 0},
        {'name': 'j2', 'min': -1, 'max': 1, 'default': 0}
    ]
    thread.set_hand_model(test_links, test_joints)
    
    # 3. 连接信号
    def on_pose_update(poses: dict):
        print(f"--- [UI 线程] 收到位姿更新 ---")
        for name, pose in poses.items():
            print(f"  {name}: T=({pose[0:3, 3][0]:.3f}, {pose[0:3, 3][1]:.3f}, {pose[0:3, 3][2]:.3f})")
            
    thread.pose_update_signal.connect(on_pose_update)
    
    # 4. 启动线程
    thread.start()
    
    # 5. 模拟 UI 交互
    def test_optimization():
        print("\n--- [UI 线程] 触发优化 ---")
        anchors = [
            {'hand_point': [0.1, 0.0, 0.0], 'obj_point': [1.0, 0.5, 0.2]}
        ]
        thread.trigger_optimization(anchors)
        
    def test_manual_joint():
        print("\n--- [UI 线程] 触发手动关节 ---")
        thread.set_manual_joint('j1', 0.5)

    def stop_thread():
        print("\n--- [UI 线程] 停止线程 ---")
        thread.stop()
        thread.wait(1000)
        app.quit()

    # 6. 创建一个简单的
    window = QWidget()
    layout = QVBoxLayout(window)
    btn_opt = QPushButton("1. 触发优化")
    btn_man = QPushButton("2. 设置关节 'j1' = 0.5")
    btn_stop = QPushButton("3. 停止线程并退出")
    
    btn_opt.clicked.connect(test_optimization)
    btn_man.clicked.connect(test_manual_joint)
    btn_stop.clicked.connect(stop_thread)
    
    layout.addWidget(btn_opt)
    layout.addWidget(btn_man)
    layout.addWidget(btn_stop)
    
    window.show()
    
    app.exec()