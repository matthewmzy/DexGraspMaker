# main_window.py

import sys
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QSplitter, QLabel, QFrame, QApplication
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QThread

# -------------------------------------------------------------------
# [占位符] - 开始
# -------------------------------------------------------------------
# 我们将导入的（尚未实现的）类放在 try/except 块中。
# 这样，即使这些文件还不存在，main_window.py 也能被理解和运行。
# 当我们实现这些文件后，真正的类将被导入。
# -------------------------------------------------------------------

try:
    from vista_widget import VistaWidget
except ImportError:
    print("提示：正在使用 'VistaWidget' 的占位符。")
    class VistaWidget(QFrame):
        """VistaWidget 的占位符"""
        # 定义一个空的信号，以便 connect_signals 不会出错
        point_picked_signal = pyqtSignal(dict)
        
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setLayout(QVBoxLayout())
            self.label = QLabel(f"占位符：\n{self.__class__.__name__}\n(3D 视窗)")
            self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
            self.layout().addWidget(self.label)
        
        # 定义空的槽函数，以便 connect_signals 不会出错
        def load_mesh(self, *args, **kwargs):
            self.label.setText(f"{self.label.text()}\n- 已加载物体")
        
        def load_hand(self, *args, **kwargs):
            self.label.setText(f"{self.label.text()}\n- 已加载机械手")
            
        def update_hand_pose(self, pose_dict: dict):
            # 这个会被高频调用，所以不打印
            pass 
        
        def set_actor_properties(self, *args, **kwargs): pass
        def enable_picking(self, enable: bool):
             self.label.setText(f"{self.label.text()}\n- 拾取模式: {enable}")

try:
    from controls_widget import ControlsWidget
except ImportError:
    print("提示：正在使用 'ControlsWidget' 的占位符。")
    class ControlsWidget(QFrame):
        """ControlsWidget 的占位符"""
        load_object_signal = pyqtSignal()
        load_hand_signal = pyqtSignal()
        start_picking_signal = pyqtSignal(bool)
        visualization_settings_changed_signal = pyqtSignal(dict)
        manual_joint_changed_signal = pyqtSignal(str, float)
        
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setLayout(QHBoxLayout())
            label = QLabel("占位符：ControlsWidget (工具栏)")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Sunken)
            self.setMinimumHeight(100)
            self.layout().addWidget(label)

try:
    from data_manager import DataManager
except ImportError:
    print("提示：正在使用 'DataManager' 的占位符。")
    class DataManager(QObject):
        """DataManager 的占位符"""
        object_loaded_signal = pyqtSignal(object)
        hand_loaded_signal = pyqtSignal(dict)
        new_anchor_pair_signal = pyqtSignal(list)
        # 我们需要一个信号来告诉主窗口何时切换拾取模式
        picking_mode_changed_signal = pyqtSignal(bool)

        def __init__(self, parent=None):
            super().__init__(parent)
            self._picking_mode = False
            
        def load_object(self): print("DataManager: (占位符) 触发加载物体")
        def load_hand(self): print("DataManager: (占位符) 触发加载机械手")
        def set_picking_mode(self, is_active: bool):
            print(f"DataManager: (占位符) 设置拾取模式为 {is_active}")
            self._picking_mode = is_active
            self.picking_mode_changed_signal.emit(is_active)
        def on_object_point_picked(self, pick_data: dict): pass
        def on_hand_point_picked(self, pick_data: dict): pass
        def on_manual_joint_change(self, joint_name: str, value: float): pass

try:
    from optimization_thread import OptimizationThread
except ImportError:
    print("提示：正在使用 'OptimizationThread' 的占位符。")
    class OptimizationThread(QThread):
        """OptimizationThread 的占位符"""
        pose_update_signal = pyqtSignal(dict)
        
        def __init__(self, parent=None):
            super().__init__(parent)
            self._is_running = True
            
        def run(self):
            print("OptimizationThread: (占位符) 优化线程已启动。")
            while self._is_running:
                self.msleep(500) # 保持线程活动
        
        def stop(self):
            print("OptimizationThread: (占位符) 停止线程。")
            self._is_running = False
            self.quit()
            
        def trigger_optimization(self, anchor_pairs: list):
            print(f"OptimizationThread: (占位符) 收到 {len(anchor_pairs)} 个锚点，触发优化。")
        
        def set_manual_joint(self, joint_name: str, value: float):
            pass

# -------------------------------------------------------------------
# [占位符] - 结束
# -------------------------------------------------------------------


class MainWindow(QMainWindow):
    """
    应用程序的主窗口。
    负责初始化所有UI组件和核心逻辑模块，并设置窗口布局。
    """
    
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        
        # 1. 初始化核心逻辑组件 (非UI)
        self.init_core_components()
        
        # 2. 初始化UI组件和布局
        self.init_ui()
        
        # 3. 连接所有组件的信号和槽
        self.connect_signals()
        
        # 4. 启动后端线程
        print("启动优化线程...")
        self.optimization_thread.start()

    def init_core_components(self) -> None:
        """
        初始化数据管理器和优化线程。
        """
        print("初始化核心组件...")
        self.data_manager = DataManager(self)
        self.optimization_thread = OptimizationThread(self)

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
        
        # 设置初始大小比例：中间的(核心)视窗是左右两边的2倍宽
        top_splitter.setSizes([200, 400, 200])

        # --- 底部工具栏 ---
        self.controls_widget = ControlsWidget(self)
        self.controls_widget.setObjectName("controls_widget")

        # --- 组合布局 ---
        
        # 3. 创建一个垂直分割器，顶部是3D视图，底部是工具栏
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.addWidget(top_splitter)
        main_splitter.addWidget(self.controls_widget)
        
        # 设置初始大小比例：3D视图占 80% 的高度
        main_splitter.setSizes([800, 200])

        # 4. 将主分割器添加到中心小部件的布局中
        main_layout.addWidget(main_splitter)
        
        print("UI布局初始化完成。")

    def connect_signals(self) -> None:
        """
        连接所有组件的信号和槽，定义应用程序的逻辑流程。
        """
        print("连接信号与槽...")

        # --- 流程 1: 加载文件 ---
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
            lambda links_dict: self.view_right.load_hand(links_dict) # 右侧，静态
        )
        self.data_manager.hand_loaded_signal.connect(
            lambda links_dict: self.view_center.load_hand(links_dict, name_prefix="dyn_hand_") # 中间，动态
        )

        # --- 流程 2: 锚点拾取 ---
        # 控件 (按钮) -> 数据管理器 (切换状态)
        self.controls_widget.start_picking_signal.connect(self.data_manager.set_picking_mode)

        # 数据管理器 (状态改变) -> 3D 视窗 (启用/禁用拾取功能)
        # (注意：我们使用了占位符中添加的 'picking_mode_changed_signal')
        self.data_manager.picking_mode_changed_signal.connect(self.on_picking_mode_changed)

        # 3D 视窗 (用户点击) -> 数据管理器 (记录坐标)
        self.view_left.point_picked_signal.connect(self.data_manager.on_object_point_picked)
        self.view_right.point_picked_signal.connect(self.data_manager.on_hand_point_picked)

        # --- 流程 3: 触发优化 ---
        # 数据管理器 (凑成一对) -> 优化线程 (开始计算)
        self.data_manager.new_anchor_pair_signal.connect(self.optimization_thread.trigger_optimization)

        # --- 流程 4: 实时更新 ---
        # 优化线程 (计算出新位姿) -> 中心视窗 (更新渲染)
        self.optimization_thread.pose_update_signal.connect(self.view_center.update_hand_pose)

        # --- 流程 5: 可视化设置 ---
        # 控件 (滑块/颜色) -> 中心视窗 (更新渲染)
        self.controls_widget.visualization_settings_changed_signal.connect(self.on_visualization_changed)

        # --- 流程 6: 手动关节控制 ---
        # 控件 (滑块) -> 优化线程 (设置关节值并计算FK)
        # (优化线程将通过 pose_update_signal 发出新位姿)
        self.controls_widget.manual_joint_changed_signal.connect(self.optimization_thread.set_manual_joint)
        
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


# --- 用于独立测试 ---
if __name__ == '__main__':
    # 这允许我们直接运行 main_window.py 来测试布局
    app = QApplication(sys.argv)
    window = MainWindow()
    window.setWindowTitle("MainWindow - 独立测试")
    window.setGeometry(100, 100, 1600, 900)
    window.show()
    sys.exit(app.exec())