# main_window.py
import numpy as np
import sys
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QSplitter, QLabel, QFrame, QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread

from vista_widget import VistaWidget
from controls_widget import ControlsWidget  # 使用新版 UI
from data_manager import DataManager
from optimization_thread import OptimizationThread
from keyboard_controller import KeyboardController


class MainWindow(QMainWindow):
    """
    应用程序的主窗口。
    负责初始化所有UI组件和核心逻辑模块，并设置窗口布局。
    """
    
    def __init__(self, parent: QWidget | None = None, load_default: bool = False) -> None:
        super().__init__(parent)
        
        # 1. 初始化核心逻辑组件 (非UI)
        self.init_core_components()
        
        # 2. 初始化UI组件和布局
        self.init_ui()
        
        # 3. 连接所有组件的信号和槽
        self.connect_signals()

        # [新增] 将状态栏消息连接到 QMainWindow 的 statusBar
        self.statusBar().showMessage("欢迎使用可交互式机械手位姿匹配工具")
        self.data_manager.status_message_signal.connect(self.statusBar().showMessage)
        
        # 4. 启动后端线程
        print("启动优化线程...")
        self.optimization_thread.start()
        
        # 5. 如果指定了 load_default，自动加载测试资源
        if load_default:
            self.load_default_assets()

    def init_core_components(self) -> None:
        """
        初始化数据管理器和优化线程。
        """
        print("初始化核心组件...")
        self.data_manager = DataManager(self)
        self.optimization_thread = OptimizationThread(self)
        self.keyboard_controller = KeyboardController(self)

    def init_ui(self) -> None:
        """
        初始化所有UI小部件并设置主窗口布局。
        """
        print("初始化UI布局...")
        
        # 设置一个中心小部件，所有的布局都将建立在它之上
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        
        # 主布局是一个垂直布局，用于容纳顶部的3D视图和底部的工具栏
        main_layout = QVBoxLayout(central_widget)

        # --- 顶部3D视图 (左, 中, 右) ---
        
        # 1. 创建三个3D视窗实例
        self.view_left = VistaWidget(self)
        self.view_left.setObjectName("view_left") # 用于调试和样式表

        self.view_center = VistaWidget(self)
        self.view_center.setObjectName("view_center")

        self.view_right = VistaWidget(self)
        self.view_right.setObjectName("view_right")

        # 2. 创建一个水平分割器来容纳这三个视窗
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.addWidget(self.view_left)
        top_splitter.addWidget(self.view_center)
        top_splitter.addWidget(self.view_right)
        
        # 设置初始大小比例： 左:中:右 = 300:500:300
        top_splitter.setSizes([300, 500, 300])

        # --- 底部工具栏 ---
        self.controls_widget = ControlsWidget(self)
        self.controls_widget.setObjectName("controls_widget")

        # --- 组合布局 ---
        
        # 3. 创建一个垂直分割器，顶部是3D视图，底部是工具栏
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.addWidget(top_splitter)
        main_splitter.addWidget(self.controls_widget)
        
        # 设置初始大小比例：3D视图占 80% 的高度
        main_splitter.setSizes([600, 400])

        # 4. 将主分割器添加到中心小部件的布局中
        main_layout.addWidget(main_splitter)
        
        print("UI布局初始化完成。")

    def connect_signals(self) -> None:
        """
        连接所有组件的信号和槽，定义应用程序的逻辑流程。
        """
        print("连接信号与槽...")

        # --- 流程 1a: 加载文件 ---
        # 控件 (按钮) -> 数据管理器 (处理逻辑)
        self.controls_widget.load_object_signal.connect(self.data_manager.load_object)
        self.controls_widget.load_hand_signal.connect(self.data_manager.load_hand)

        # 数据管理器 (加载成功) -> 3D 视窗 (显示)
        # 我们使用 lambda 函数来为 load_mesh 指定特定的 'name'
        self.data_manager.object_loaded_signal.connect(
            lambda mesh_data: self.view_left.load_mesh(mesh_data, name="object")
        )
        self.data_manager.object_loaded_signal.connect(
            lambda mesh_data: self.view_center.load_mesh(mesh_data, name="object", opacity=0.5) # 物体在中间半透明
        )
        self.data_manager.hand_loaded_signal.connect(
            lambda links_dict: self.view_right.load_hand(links_dict, name_prefix="static_hand_") # 右侧，静态（用于拾取）
        )
        self.data_manager.hand_loaded_signal.connect(
            lambda links_dict: self.view_center.load_hand(links_dict, name_prefix="dyn_hand_") # 中间，动态（优化更新）
        )
        self.data_manager.hand_initial_pose_signal.connect(self.on_hand_initial_pose_received)

        # --- 流程 1b: 将加载的数据连接到 UI 控件 ---
        # DataManager (加载成功) -> ControlsWidget (更新UI)
        self.data_manager.anchor_list_updated_signal.connect(self.controls_widget.update_anchor_list)
        self.data_manager.hand_joint_info_signal.connect(self.controls_widget.create_joint_controls)
        
        # --- 流程 1c: 将 JAX Robot 模型发送到后端 ---
        self.data_manager.pyroki_robot_loaded_signal.connect(self.optimization_thread.set_pyroki_robot)

        # --- 流程 2: 锚点拾取（新版工作流） ---
        # 控件 (添加锚点对按钮) -> 数据管理器 (激活拾取)
        self.controls_widget.add_anchor_pair_signal.connect(
            lambda: self.data_manager.set_picking_mode(True)
        )
        
        # 控件 (确定按钮) -> 数据管理器 (确认添加锚点对)
        self.controls_widget.confirm_anchor_pair_signal.connect(
            self.data_manager.confirm_anchor_pair
        )

        # 数据管理器 (状态改变) -> 3D 视窗 (启用/禁用拾取功能)
        self.data_manager.picking_mode_changed_signal.connect(self.on_picking_mode_changed)
        
        # 数据管理器 (锚点对准备状态) -> 控件 (显示/隐藏确定按钮)
        self.data_manager.anchor_pair_ready_signal.connect(self.controls_widget.show_confirm_button)

        # 3D 视窗 (用户点击) -> 数据管理器 (记录坐标)
        self.view_left.point_picked_signal.connect(self.data_manager.on_object_point_picked)
        self.view_right.point_picked_signal.connect(self.data_manager.on_hand_point_picked)
        
        # 控件 (删除按钮) -> 数据管理器 (删除锚点对)
        self.controls_widget.delete_anchor_signal.connect(self.data_manager.on_delete_anchor)
        
        # 控件 (启用/禁用复选框) -> 数据管理器 (切换锚点状态)
        self.controls_widget.toggle_anchor_signal.connect(self.data_manager.on_toggle_anchor)

        # --- 锚点位置调整 ---
        # 控件 (调整按钮) -> 数据管理器 (开始调整)
        self.controls_widget.adjust_hand_anchor_signal.connect(self.data_manager.on_adjust_hand_anchor)
        self.controls_widget.adjust_object_anchor_signal.connect(self.data_manager.on_adjust_object_anchor)
        
        # 数据管理器 (开始调整) -> 主窗口 (启动键盘控制器)
        self.data_manager.adjust_hand_anchor_signal.connect(self.on_start_adjust_hand_anchor)
        self.data_manager.adjust_object_anchor_signal.connect(self.on_start_adjust_object_anchor)
        
        # 控件 (位置更新) -> 数据管理器 (更新位置)
        self.controls_widget.update_anchor_position_signal.connect(self.data_manager.on_update_anchor_position)
        
        # 键盘控制器 (位置改变) -> 控件 (更新显示)
        self.keyboard_controller.position_changed_signal.connect(self.controls_widget.update_anchor_position)
        
        # 键盘控制器 (控制结束) -> 主窗口 (清理状态)
        self.keyboard_controller.control_ended_signal.connect(self.on_keyboard_control_ended)
        
        # 键盘控制器 (状态变化) -> 控件 (更新按钮状态)
        self.keyboard_controller.control_state_changed_signal.connect(self.controls_widget.update_anchor_adjust_button_state)

        # --- 流程 3: 触发优化 ---
        # 数据管理器 (凑成一对) -> 优化线程 (开始计算)
        self.data_manager.new_anchor_pair_signal.connect(self.optimization_thread.trigger_optimization)

        # --- 流程 4: 实时更新 ---
        # 优化线程 (计算出新位姿) -> 主窗口 (更新渲染和锚点)
        self.optimization_thread.pose_update_signal.connect(self.on_pose_update_with_anchors)

        # --- 流程 5: 可视化设置 ---
        # 控件 (滑块/颜色) -> 中心视窗 (更新渲染)
        self.controls_widget.visualization_settings_changed_signal.connect(self.on_visualization_changed)
        
        # 控件 (锚点大小/显示线) -> 主窗口 (重新渲染锚点)
        self.controls_widget.visualization_settings_changed_signal.connect(
            lambda settings: self.on_anchor_list_updated(self.data_manager.anchor_pairs)
        )

        # --- 流程 6: 手动关节控制 ---
        # 控件 (滑块) -> 优化线程 (设置关节值并计算FK)
        # (优化线程将通过 pose_update_signal 发出新位姿)
        self.controls_widget.manual_joint_changed_signal.connect(self.optimization_thread.set_manual_joint)
        
        # --- 流程 7: 优化控制 ---
        self.controls_widget.optimization_toggle_signal.connect(self.on_optimization_toggle)
        
        # --- 流程 8: 姿态导入/导出 ---
        self.controls_widget.import_pose_signal.connect(self.data_manager.import_hand_pose)
        self.controls_widget.export_pose_signal.connect(self.data_manager.export_hand_pose)
        
        # --- 流程 9: 锚点可视化 ---
        # 数据管理器 (锚点列表更新) -> 主窗口 (更新所有视窗的锚点显示)
        self.data_manager.anchor_list_updated_signal.connect(self.on_anchor_list_updated)
        
        print("信号连接完成。")

    # --- 辅助槽函数 (Helper Slots) ---

    def on_picking_mode_changed(self, is_active: bool) -> None:
        """
        当拾取模式激活或关闭时，更新左右视窗的状态。
        """
        print(f"主窗口：切换拾取模式为 {is_active}")
        self.view_left.enable_picking(is_active)  # 启用物体拾取
        self.view_right.enable_picking(is_active) # 启用机械手拾取
        self.view_center.enable_picking(False)    # 中心视窗始终禁用拾取
        
        if is_active:
            # 设置下一个锚点对的颜色
            next_index = len(self.data_manager.anchor_pairs)
            color = self.controls_widget.get_anchor_color(next_index)
            self.view_left.set_next_anchor_color(color)
            self.view_right.set_next_anchor_color(color)
        else:
            # 清除临时拾取标记
            self.view_left.plotter.remove_actor(f"_pick_marker_{self.view_left.objectName()}")
            self.view_right.plotter.remove_actor(f"_pick_marker_{self.view_right.objectName()}")

    def on_visualization_changed(self, settings: dict) -> None:
        """
        处理来自工具栏的可视化设置更改。
        """
        print(f"主窗口：收到可视化设置: {settings}")
        
        # 示例：更新中心视窗的透明度
        if 'hand_opacity' in settings:
            # 假设 vista_widget.set_actor_properties 支持通配符
            self.view_center.set_actor_properties(
                name_pattern="dyn_hand_*", 
                opacity=settings['hand_opacity']
            )
        
        if 'object_opacity' in settings:
            self.view_center.set_actor_properties(
                name="object", 
                opacity=settings['object_opacity']
            )
        
        # 可以在此处扩展颜色等其他设置...

    def on_hand_initial_pose_received(self, poses_dict: dict[str, np.ndarray]) -> None:
        """
        槽：当 DataManager 加载完机械手并发出其默认姿态时调用。
        """
        if not poses_dict:
            return
            
        print("MainWindow: 收到初始姿态，正在更新视窗...")
        
        # 存储初始姿态（用于静态视图的锚点位置计算）
        self._initial_link_poses = poses_dict.copy()
        
        # 1. 更新右侧 (静态) 视窗 - 添加 'static_hand_' 前缀
        static_poses = {}
        for link_name, pose_matrix in poses_dict.items():
            actor_name = "static_hand_" + link_name
            static_poses[actor_name] = pose_matrix
        self.view_right.update_hand_pose(static_poses)
        
        # 2. 更新中间 (动态) 视窗 - 添加 'dyn_hand_' 前缀
        dyn_poses = {}
        for link_name, pose_matrix in poses_dict.items():
            actor_name = "dyn_hand_" + link_name
            dyn_poses[actor_name] = pose_matrix
            
        self.view_center.update_hand_pose(dyn_poses)
        
        # 3. (重要!) 移动 actor 之后重置相机
        self.view_right.plotter.reset_camera()
        self.view_center.plotter.reset_camera()
        
        print("MainWindow: 视窗姿态已更新。")
    
    def on_anchor_list_updated(self, anchor_pairs: list) -> None:
        """
        当锚点列表更新时，更新所有视窗的锚点显示。
        :param anchor_pairs: 锚点对列表
        """
        if not anchor_pairs:
            # 如果没有锚点，清空显示
            self.view_left.update_anchor_spheres([], lambda p: 'red', 0.005)
            self.view_right.update_anchor_spheres([], lambda p: 'red', 0.005)
            self.view_center.update_anchor_spheres([], lambda p: 'red', 0.005)
            return

        color_func = lambda i: self.get_color_for_pair(i)
        sphere_radius = self.controls_widget.anchor_size_spinbox.value() / 1000.0  # mm转m
        
        self.view_left.update_anchor_spheres(
            anchor_pairs, 
            color_func, 
            sphere_radius
        )
        self.view_right.update_anchor_spheres(
            anchor_pairs, 
            color_func, 
            sphere_radius
        )
        self.view_center.update_anchor_spheres(
            anchor_pairs, 
            color_func, 
            sphere_radius
        )

    def get_color_for_pair(self, index: int) -> str:
        """
        根据锚点对索引返回颜色。
        """
        colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'purple', 'pink', 'brown']
        return colors[index % len(colors)]
    
    def on_pose_update_with_anchors(self, link_poses_dict: dict) -> None:
        """
        当手部位姿更新时，同时更新锚点位置（因为手部锚点需要跟随手移动）
        
        性能优化：手部位姿每帧更新，锚点位置也每帧更新但使用快速方法（仅更新位置）
        
        :param link_poses_dict: {link_name: 4x4_matrix} 字典（来自优化线程，key是纯link名）
        """
        if not link_poses_dict:
            print("MainWindow: on_pose_update_with_anchors 收到空字典！")
            return
        
        # DEBUG: 每100帧打印一次，确认信号被接收
        if not hasattr(self, '_pose_update_counter'):
            self._pose_update_counter = 0
        self._pose_update_counter += 1
        if self._pose_update_counter % 100 == 0:
            print(f"MainWindow: 已接收 {self._pose_update_counter} 次位姿更新，当前有 {len(link_poses_dict)} 个 links")
        
        # 首先更新手部位姿（每帧）
        # 1. 中心视窗（动态手）- link_poses_dict 的 key 已经包含 'dyn_hand_' 前缀
        # 注意：optimization_thread 的 _run_forward_kinematics 已经添加了前缀！
        self.view_center.update_hand_pose(link_poses_dict)
        
        # 2. 更新DataManager中的当前link姿态（用于锚点调整）
        # 移除 'dyn_hand_' 前缀以匹配link名称
        current_link_poses = {}
        for actor_name, pose_matrix in link_poses_dict.items():
            if actor_name.startswith('dyn_hand_'):
                link_name = actor_name[len('dyn_hand_'):]
                current_link_poses[link_name] = pose_matrix
        self.data_manager.update_current_link_poses(current_link_poses)
        
        # 3. 快速更新锚点位置（每帧，但仅更新位置不重建actors）
        anchor_pairs = self.data_manager.anchor_pairs
        if not anchor_pairs:
            return
        
        # 为动态视图（view_center）计算锚点位置 - 使用当前动态姿态
        dynamic_updated_pairs = []
        for pair in anchor_pairs:
            link_name = pair['hand_link_name']
            hand_local = np.array(pair['hand_point_local'])
            
            # 查找对应 link 的当前世界位姿
            actor_name = "dyn_hand_" + link_name
            if actor_name in link_poses_dict:
                T_world_link = link_poses_dict[actor_name]
                
                # 将局部坐标转换为世界坐标
                hand_local_homogeneous = np.append(hand_local, 1.0)
                hand_world = (T_world_link @ hand_local_homogeneous)[:3]
                
                # 创建更新后的锚点对
                updated_pair = pair.copy()
                updated_pair['hand_point'] = hand_world.tolist()
                dynamic_updated_pairs.append(updated_pair)
            else:
                # 如果找不到对应的 link，保持原值
                dynamic_updated_pairs.append(pair)
        
        # 为静态视图（view_left, view_right）计算锚点位置 - 使用初始姿态
        static_updated_pairs = []
        if hasattr(self, '_initial_link_poses') and self._initial_link_poses:
            for pair in anchor_pairs:
                link_name = pair['hand_link_name']
                hand_local = np.array(pair['hand_point_local'])
                
                # 创建更新后的锚点对
                updated_pair = pair.copy()
                
                # 处理手部锚点：使用初始link姿态计算位置
                if link_name in self._initial_link_poses:
                    T_world_link = self._initial_link_poses[link_name]
                    
                    # 将局部坐标转换为世界坐标
                    hand_local_homogeneous = np.append(hand_local, 1.0)
                    hand_world = (T_world_link @ hand_local_homogeneous)[:3]
                    
                    updated_pair['hand_point'] = hand_world.tolist()
                else:
                    # 如果找不到对应的 link，保持原值
                    pass
                
                # 物体锚点直接使用世界坐标（物体在世界坐标系中）
                # obj_point已经是世界坐标，无需转换
                
                static_updated_pairs.append(updated_pair)
        else:
            # 如果没有初始姿态，使用动态姿态作为fallback
            static_updated_pairs = dynamic_updated_pairs.copy()
        
        # 使用快速位置更新（不重建actors，性能更好）
        color_func = lambda i: self.get_color_for_pair(i)
        self.view_left.color_func = color_func
        self.view_right.color_func = color_func
        self.view_center.color_func = color_func
        
        # 静态视图使用初始姿态计算的锚点位置
        self.view_left.update_anchor_positions_fast(static_updated_pairs)
        self.view_right.update_anchor_positions_fast(static_updated_pairs)
        
        # 动态视图使用当前姿态计算的锚点位置
        self.view_center.update_anchor_positions_fast(dynamic_updated_pairs)
        
        # 更新手姿态显示
        self.update_hand_pose_display()

    def load_default_assets(self) -> None:
        """
        自动加载默认测试资源
        """
        import os
        from PyQt6.QtCore import QTimer
        
        # 获取项目根目录（main_window.py 的上级目录）
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        
        object_path = os.path.join(project_root, "test_assets", "objects", "Mug.obj")
        hand_path = os.path.join(project_root, "test_assets", "shadow", "shadow_hand_right.urdf")
        
        print(f"自动加载默认资源...")
        print(f"  物体: {object_path}")
        print(f"  手部: {hand_path}")
        
        # 使用QTimer延迟加载，确保UI已完全初始化
        def delayed_load():
            if os.path.exists(object_path):
                self.data_manager.load_object_from_file(object_path)
                self.statusBar().showMessage(f"✓ 已自动加载物体: Aligned.obj")
            else:
                print(f"警告: 未找到物体文件: {object_path}")
                self.statusBar().showMessage(f"✗ 未找到物体文件: {object_path}")
            
            if os.path.exists(hand_path):
                self.data_manager.load_hand_from_file(hand_path)
                self.statusBar().showMessage(f"✓ 已自动加载手部: shadow_hand_right.urdf")
            else:
                print(f"警告: 未找到手部文件: {hand_path}")
                self.statusBar().showMessage(f"✗ 未找到手部文件: {hand_path}")
        
        # 延迟500ms后加载，确保窗口已显示
        QTimer.singleShot(500, delayed_load)

    def on_start_adjust_hand_anchor(self, anchor_index: int) -> None:
        """
        开始或结束调整手上锚点
        
        :param anchor_index: 锚点对索引
        """
        if 0 <= anchor_index < len(self.data_manager.anchor_pairs):
            anchor = self.data_manager.anchor_pairs[anchor_index]
            # 使用局部坐标进行键盘控制，因为用户想要相对于手移动
            position = np.array(anchor['hand_point_local'])
            
            if self.keyboard_controller.toggle_control(anchor_index, "hand", position):
                self.statusBar().showMessage(f"开始调整锚点对 #{anchor_index+1} 的手上位置 (WASD+空格控制)")
            else:
                self.statusBar().showMessage(f"结束调整锚点对 #{anchor_index+1} 的手上位置")

    def on_start_adjust_object_anchor(self, anchor_index: int) -> None:
        """
        开始或结束调整物体锚点
        
        :param anchor_index: 锚点对索引
        """
        if 0 <= anchor_index < len(self.data_manager.anchor_pairs):
            anchor = self.data_manager.anchor_pairs[anchor_index]
            position = np.array(anchor['obj_point'])
            
            if self.keyboard_controller.toggle_control(anchor_index, "object", position):
                self.statusBar().showMessage(f"开始调整锚点对 #{anchor_index+1} 的物体位置 (WASD+空格控制)")
            else:
                self.statusBar().showMessage(f"结束调整锚点对 #{anchor_index+1} 的物体位置")

    def on_keyboard_control_ended(self) -> None:
        """
        键盘控制结束时的处理
        """
        self.statusBar().showMessage("键盘控制结束")

    def closeEvent(self, event) -> None:
        """
        重写 QMainWindow 的 closeEvent，以确保线程被安全关闭。
        """
        print("关闭应用程序...")
        self.optimization_thread.stop()
        if not self.optimization_thread.wait(1000): # 等待线程1秒钟
            print("警告：优化线程未能正常停止，将强制终止。")
            self.optimization_thread.terminate()
            
        print("再见。")
        event.accept()

    def on_optimization_toggle(self, is_running: bool) -> None:
        """
        优化开启/暂停切换
        
        :param is_running: True=开始优化, False=暂停优化
        """
        if is_running:
            self.optimization_thread.resume()
            print("优化已恢复")
        else:
            self.optimization_thread.pause()
            print("优化已暂停")

    def update_hand_pose_display(self) -> None:
        """
        更新手姿态显示（位置和旋转矩阵）
        """
        translation, rotation_matrix = self.data_manager.get_current_hand_pose()
        self.controls_widget.update_hand_pose_display(translation, rotation_matrix)


# --- 用于独立测试 ---
if __name__ == '__main__':
    # 这允许我们直接运行 main_window.py 来测试布局
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle("MainWindow - 独立测试")
    window.setGeometry(100, 100, 1600, 900)
    window.show()
    sys.exit(app.exec())