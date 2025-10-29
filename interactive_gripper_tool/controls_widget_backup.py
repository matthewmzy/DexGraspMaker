# controls_widget.py

import sys
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QTabWidget, QPushButton, QSlider, QListWidget, 
    QVBoxLayout, QHBoxLayout, QGroupBox, QDoubleSpinBox, QLabel,
    QGridLayout, QScrollArea, QFrame, QApplication
)
from PyQt6.QtCore import pyqtSignal, Qt, pyqtSlot

class ControlsWidget(QWidget):
    """
    封装所有 2D 控制按钮、滑块和列表的工具栏面板。
    """
    
    # --- 信号定义 ---
    
    # 流程 1: 加载
    load_object_signal = pyqtSignal()
    load_hand_signal = pyqtSignal()
    
    # 流程 2: 拾取
    start_picking_signal = pyqtSignal(bool) # 发送 (is_checked)
    
    # 流程 3: 锚点列表
    delete_anchor_signal = pyqtSignal(int) # 发送 (row_index)
    anchor_settings_changed_signal = pyqtSignal(dict) # 发送 (settings_dict)
    
    # 流程 5: 可视化
    visualization_settings_changed_signal = pyqtSignal(dict) # 发送 (settings_dict)
    
    # 流程 6: 关节调试
    manual_joint_changed_signal = pyqtSignal(str, float) # 发送 (joint_name, value)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        
        # 存储动态创建的关节控件
        self.joint_controls = {} # {joint_name: {'slider': QSlider, 'spinbox': QDoubleSpinBox}}
        
        # 主布局
        main_layout = QVBoxLayout(self)
        
        # 创建选项卡界面
        self.tab_widget = QTabWidget(self)
        main_layout.addWidget(self.tab_widget)
        
        # 1. 创建 "文件 & 控制" 选项卡
        self.tab_widget.addTab(self._create_control_tab(), "文件 & 控制")
        
        # 2. 创建 "锚点" 选项卡
        self.tab_widget.addTab(self._create_anchors_tab(), "锚点设置")
        
        # 3. 创建 "可视化" 选项卡
        self.tab_widget.addTab(self._create_viz_tab(), "可视化")
        
        # 4. 创建 "关节调试" 选项卡
        self.tab_widget.addTab(self._create_joints_tab(), "关节调试")
        
        # 设置最小高度，使其不会被过度压缩
        self.setMinimumHeight(200)

    # --- 选项卡创建 (私有方法) ---

    def _create_control_tab(self) -> QWidget:
        """创建 "文件 & 控制" 选项卡的内容"""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # 文件加载组
        load_group = QGroupBox("加载模型")
        load_layout = QVBoxLayout(load_group)
        
        self.load_obj_button = QPushButton("1. 加载物体 (Mesh)")
        self.load_hand_button = QPushButton("2. 加载机械手 (URDF)")
        
        load_layout.addWidget(self.load_obj_button)
        load_layout.addWidget(self.load_hand_button)
        
        # 拾取控制组
        picking_group = QGroupBox("锚点设置")
        picking_layout = QVBoxLayout(picking_group)
        
        self.start_picking_button = QPushButton("3. 开始设置锚点")
        self.start_picking_button.setCheckable(True) # 设置为可切换的按钮
        
        picking_layout.addWidget(self.start_picking_button)
        
        layout.addWidget(load_group)
        layout.addWidget(picking_group)
        
        # --- 连接信号 ---
        self.load_obj_button.clicked.connect(self.load_object_signal)
        self.load_hand_button.clicked.connect(self.load_hand_signal)
        self.start_picking_button.toggled.connect(self.start_picking_signal)
        
        return tab_widget

    def _create_anchors_tab(self) -> QWidget:
        """创建 "锚点设置" 选项卡的内容"""
        tab_widget = QWidget()
        layout = QHBoxLayout(tab_widget)
        
        # 左侧：列表
        list_group = QGroupBox("锚点对列表")
        list_layout = QVBoxLayout(list_group)
        
        self.anchor_list_widget = QListWidget()
        self.delete_anchor_button = QPushButton("删除选中锚点")
        
        list_layout.addWidget(self.anchor_list_widget)
        list_layout.addWidget(self.delete_anchor_button)
        
        # 右侧：设置
        settings_group = QGroupBox("可视化设置")
        settings_layout = QGridLayout(settings_group)
        
        self.anchor_size_label = QLabel("锚点球大小 (m):")
        self.anchor_size_spinbox = QDoubleSpinBox()
        self.anchor_size_spinbox.setRange(0.001, 0.1)
        self.anchor_size_spinbox.setValue(0.005)
        self.anchor_size_spinbox.setSingleStep(0.001)
        
        settings_layout.addWidget(self.anchor_size_label, 0, 0)
        settings_layout.addWidget(self.anchor_size_spinbox, 0, 1)
        # (未来可以添加颜色按钮等)
        settings_layout.setRowStretch(2, 1) # 占位
        
        layout.addWidget(list_group, stretch=2)
        layout.addWidget(settings_group, stretch=1)
        
        # --- 连接信号 ---
        self.delete_anchor_button.clicked.connect(self._on_delete_anchor)
        self.anchor_size_spinbox.valueChanged.connect(self._on_anchor_settings_changed)
        
        return tab_widget

    def _create_viz_tab(self) -> QWidget:
        """创建 "可视化" 选项卡的内容"""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # 辅助函数创建滑块
        def create_opacity_slider() -> QSlider:
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100) # 0% to 100%
            slider.setValue(100)
            return slider

        # 中间视图 - 机械手
        hand_group = QGroupBox("中间视图: 机械手 (动态)")
        hand_layout = QGridLayout(hand_group)
        self.hand_opacity_slider = create_opacity_slider()
        self.hand_opacity_slider.setValue(70) # 默认半透明
        hand_layout.addWidget(QLabel("透明度:"), 0, 0)
        hand_layout.addWidget(self.hand_opacity_slider, 0, 1)
        
        # 中间视图 - 物体
        obj_group = QGroupBox("中间视图: 物体")
        obj_layout = QGridLayout(obj_group)
        self.object_opacity_slider = create_opacity_slider()
        self.object_opacity_slider.setValue(100) # 默认不透明
        obj_layout.addWidget(QLabel("透明度:"), 0, 0)
        obj_layout.addWidget(self.object_opacity_slider, 0, 1)

        layout.addWidget(hand_group)
        layout.addWidget(obj_group)
        
        # --- 连接信号 ---
        self.hand_opacity_slider.valueChanged.connect(self._on_viz_settings_changed)
        self.object_opacity_slider.valueChanged.connect(self._on_viz_settings_changed)
        
        return tab_widget

    def _create_joints_tab(self) -> QWidget:
        """创建 "关节调试" 选项卡的内容"""
        # 这个选项卡是动态填充的
        tab_widget = QWidget()
        main_layout = QVBoxLayout(tab_widget)
        
        # 占位符
        self.joints_placeholder_label = QLabel(
            "加载机械手 (URDF) 后，将在此处显示所有可动关节的控制器。"
        )
        self.joints_placeholder_label.setWordWrap(True)
        self.joints_placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 滚动区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        # 滚动区域的内容
        self.scroll_content_widget = QWidget()
        self.joints_layout = QVBoxLayout(self.scroll_content_widget) # 关键布局
        self.joints_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.joints_layout.addWidget(self.joints_placeholder_label)
        
        scroll_area.setWidget(self.scroll_content_widget)
        main_layout.addWidget(scroll_area)
        
        return tab_widget

    # --- 公共槽 (Public Slots) ---
    # 这些方法将由 main_window 或 data_manager 调用

    @pyqtSlot(list)
    def update_anchor_list(self, anchor_pairs: list[dict]) -> None:
        """
        清空并重新填充锚点列表。
        :param anchor_pairs: 锚点对的完整列表。
        """
        self.anchor_list_widget.clear()
        for i, pair in enumerate(anchor_pairs):
            # 格式化坐标
            h_pt = np.array(pair['hand_point']).round(3)
            o_pt = np.array(pair['obj_point']).round(3)
            # 在 'hand_point' 中找到 'link_name'
            link_name = pair.get('hand_link_name', 'unknown_link') # 优雅地处理
            
            display_text = f"{i+1}: [{link_name}] @ {h_pt}  ->  [Object] @ {o_pt}"
            self.anchor_list_widget.addItem(display_text)

    @pyqtSlot(list)
    def create_joint_controls(self, joint_info_list: list[dict]) -> None:
        """
        动态创建所有关节的滑块和输入框。
        :param joint_info_list: [{'name': str, 'min': float, 'max': float, 'default': float}]
        """
        # 1. 清除旧控件
        for joint_name, controls in self.joint_controls.items():
            controls['slider'].deleteLater()
            controls['spinbox'].deleteLater()
            controls['label'].deleteLater()
            controls['layout_widget'].deleteLater()
        self.joint_controls.clear()
        
        if not joint_info_list:
            self.joints_placeholder_label.show()
            return
            
        self.joints_placeholder_label.hide()
        
        # 2. 创建新控件
        for joint_info in joint_info_list:
            name = joint_info['name']
            min_val = joint_info['min']
            max_val = joint_info['max']
            default_val = joint_info['default']
            
            # 使用一个 widget 来容纳布局，方便删除
            layout_widget = QWidget()
            control_layout = QHBoxLayout(layout_widget)
            
            label = QLabel(f"{name}:")
            label.setMinimumWidth(100)
            
            # 滑块使用整数，精度为 0.01
            slider = QSlider(Qt.Orientation.Horizontal)
            slider_min = int(min_val * 100)
            slider_max = int(max_val * 100)
            slider.setRange(slider_min, slider_max)
            
            spinbox = QDoubleSpinBox()
            spinbox.setRange(min_val, max_val)
            spinbox.setSingleStep(0.01)
            spinbox.setDecimals(3)
            
            # 设置默认值
            spinbox.setValue(default_val) # 这会触发 _on_spinbox_moved
            slider.setValue(int(default_val * 100))
            
            # 添加到布局
            control_layout.addWidget(label, stretch=1)
            control_layout.addWidget(spinbox, stretch=1)
            control_layout.addWidget(slider, stretch=3)
            
            self.joints_layout.addWidget(layout_widget)
            
            # 存储控件
            self.joint_controls[name] = {
                'label': label,
                'slider': slider,
                'spinbox': spinbox,
                'layout_widget': layout_widget
            }
            
            # --- 连接信号 ---
            # 关键：连接滑块和输入框以实现同步
            slider.valueChanged.connect(self._on_slider_moved)
            spinbox.valueChanged.connect(self._on_spinbox_moved)

    # --- 内部槽 (Private Slots) ---
    # 这些方法用于响应 UI 交互并发射信号

    @pyqtSlot(int)
    def _on_slider_moved(self, value: int) -> None:
        """当滑块移动时，更新对应的 spinbox 并发射信号。"""
        float_value = value / 100.0
        
        # 找到是哪个滑块
        sender_slider = self.sender()
        for joint_name, controls in self.joint_controls.items():
            if controls['slider'] == sender_slider:
                # 1. 更新 SpinBox (并阻止其发信号，防止循环)
                controls['spinbox'].blockSignals(True)
                controls['spinbox'].setValue(float_value)
                controls['spinbox'].blockSignals(False)
                
                # 2. 发射主信号
                self.manual_joint_changed_signal.emit(joint_name, float_value)
                return

    @pyqtSlot(float)
    def _on_spinbox_moved(self, value: float) -> None:
        """当 spinbox 值改变时，更新对应的滑块并发射信号。"""
        int_value = int(value * 100)
        
        # 找到是哪个 spinbox
        sender_spinbox = self.sender()
        for joint_name, controls in self.joint_controls.items():
            if controls['spinbox'] == sender_spinbox:
                # 1. 更新 Slider (并阻止其发信号)
                controls['slider'].blockSignals(True)
                controls['slider'].setValue(int_value)
                controls['slider'].blockSignals(False)
                
                # 2. 发射主信号 (SpinBox 是信号的"真实来源")
                self.manual_joint_changed_signal.emit(joint_name, value)
                return

    @pyqtSlot()
    def _on_viz_settings_changed(self) -> None:
        """收集所有可视化设置并发射信号。"""
        settings = {
            'hand_opacity': self.hand_opacity_slider.value() / 100.0,
            'object_opacity': self.object_opacity_slider.value() / 100.0
        }
        self.visualization_settings_changed_signal.emit(settings)

    @pyqtSlot()
    def _on_delete_anchor(self) -> None:
        """当"删除"按钮被点击时，获取所选行并发射信号。"""
        current_row = self.anchor_list_widget.currentRow()
        if current_row >= 0: # -1 表示没有选中
            self.delete_anchor_signal.emit(current_row)
            
    @pyqtSlot(float)
    def _on_anchor_settings_changed(self, value: float) -> None:
        """当锚点可视化设置改变时，发射信号。"""
        settings = {
            'size': self.anchor_size_spinbox.value()
            # 可以在此添加 'color' 等
        }
        self.anchor_settings_changed_signal.emit(settings)


# --- 用于独立测试 ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    window = QMainWindow()
    controls = ControlsWidget()
    
    # --- 测试动态关节创建 ---
    test_joints = [
        {'name': 'joint_1', 'min': -3.14, 'max': 3.14, 'default': 0.0},
        {'name': 'joint_2', 'min': -1.57, 'max': 1.57, 'default': 0.5},
        {'name': 'long_joint_name_6', 'min': 0.0, 'max': 1.0, 'default': 0.2},
    ]
    controls.create_joint_controls(test_joints)
    
    # --- 测试信号连接 ---
    def on_joint_change(name, val):
        print(f"[信号] 关节: {name} | 值: {val:.3f}")
        
    def on_viz_change(settings):
        print(f"[信号] 可视化: {settings}")
        
    def on_pick_toggle(is_active):
        print(f"[信号] 拾取模式: {is_active}")

    controls.manual_joint_changed_signal.connect(on_joint_change)
    controls.visualization_settings_changed_signal.connect(on_viz_change)
    controls.start_picking_signal.connect(on_pick_toggle)
    
    # --- 测试列表更新 ---
    test_anchors = [
        {'hand_point': [0.1, 0.2, 0.3], 'obj_point': [1.1, 1.2, 1.3], 'hand_link_name': 'link_1'},
        {'hand_point': [0.4, 0.5, 0.6], 'obj_point': [1.4, 1.5, 1.6], 'hand_link_name': 'link_2'},
    ]
    controls.update_anchor_list(test_anchors)

    window.setCentralWidget(controls)
    window.setWindowTitle("ControlsWidget - 独立测试")
    window.setGeometry(300, 300, 500, 400)
    window.show()
    sys.exit(app.exec())