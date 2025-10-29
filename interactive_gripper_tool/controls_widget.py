# controls_widget_v2.py
# 重新设计的控件面板，支持更直观的多锚点工作流

import sys
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QTabWidget, QPushButton, QSlider, QListWidget, QListWidgetItem,
    QVBoxLayout, QHBoxLayout, QGroupBox, QDoubleSpinBox, QLabel,
    QGridLayout, QScrollArea, QFrame, QApplication, QColorDialog, QCheckBox
)
from PyQt6.QtCore import pyqtSignal, Qt, pyqtSlot
from PyQt6.QtGui import QColor, QBrush

class ControlsWidget(QWidget):
    """
    改进的控制面板 - 支持多锚点对的直观管理
    
    主要改进：
    1. 简化的锚点添加流程 - 一键添加新锚点对
    2. 每个锚点对有独立的颜色
    3. 实时优化 - 任何修改都立即触发优化
    4. 更好的视觉反馈
    """
    
    # --- 信号定义 ---
    
    # 流程 1: 加载
    load_object_signal = pyqtSignal()
    load_hand_signal = pyqtSignal()
    
    # 流程 2: 锚点管理（新设计）
    add_anchor_pair_signal = pyqtSignal()  # 开始添加新锚点对
    confirm_anchor_pair_signal = pyqtSignal()  # 确认添加锚点对
    delete_anchor_signal = pyqtSignal(int)  # 删除指定索引的锚点对
    toggle_anchor_signal = pyqtSignal(int, bool)  # 启用/禁用锚点对
    
    # 流程 3: 可视化
    visualization_settings_changed_signal = pyqtSignal(dict)
    
    # 流程 4: 关节调试
    manual_joint_changed_signal = pyqtSignal(str, float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        
        # 预定义的颜色方案（用于不同的锚点对）
        self.anchor_colors = [
            QColor(255, 0, 0),      # 红色
            QColor(0, 128, 255),    # 蓝色
            QColor(0, 255, 0),      # 绿色
            QColor(255, 165, 0),    # 橙色
            QColor(255, 0, 255),    # 品红
            QColor(0, 255, 255),    # 青色
            QColor(255, 255, 0),    # 黄色
            QColor(128, 0, 128),    # 紫色
            QColor(255, 192, 203),  # 粉色
            QColor(128, 128, 0),    # 橄榄绿
        ]
        
        # 存储动态创建的关节控件
        self.joint_controls = {}
        
        # 主布局
        main_layout = QVBoxLayout(self)
        
        # 创建选项卡界面
        self.tab_widget = QTabWidget(self)
        main_layout.addWidget(self.tab_widget)
        
        # 1. 创建 "文件 & 控制" 选项卡
        self.tab_widget.addTab(self._create_control_tab(), "文件 & 控制")
        
        # 2. 创建 "锚点" 选项卡（重新设计）
        self.tab_widget.addTab(self._create_anchors_tab(), "锚点配对")
        
        # 3. 创建 "可视化" 选项卡
        self.tab_widget.addTab(self._create_viz_tab(), "可视化")
        
        # 4. 创建 "关节调试" 选项卡
        self.tab_widget.addTab(self._create_joints_tab(), "关节调试")
        
        self.setMinimumHeight(200)

    # --- 选项卡创建 ---

    def _create_control_tab(self) -> QWidget:
        """创建 "文件 & 控制" 选项卡"""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # 文件加载组
        load_group = QGroupBox("1. 加载模型")
        load_layout = QVBoxLayout(load_group)
        
        self.load_obj_button = QPushButton("📦 加载物体 (OBJ/STL)")
        self.load_hand_button = QPushButton("🤖 加载机械手 (URDF)")
        
        self.load_obj_button.setMinimumHeight(40)
        self.load_hand_button.setMinimumHeight(40)
        
        load_layout.addWidget(self.load_obj_button)
        load_layout.addWidget(self.load_hand_button)
        
        # 说明文本
        info_label = QLabel(
            "💡 提示：\n"
            "1. 先加载物体和机械手\n"
            "2. 切换到「锚点配对」标签\n"
            "3. 点击「添加锚点对」开始配对\n"
            "4. 点击手部和物体上的对应点\n"
            "5. 优化会实时更新"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("QLabel { background-color: #f0f8ff; padding: 10px; border-radius: 5px; }")
        
        layout.addWidget(load_group)
        layout.addWidget(info_label)
        layout.addStretch()
        
        # 连接信号
        self.load_obj_button.clicked.connect(self.load_object_signal.emit)
        self.load_hand_button.clicked.connect(self.load_hand_signal.emit)
        
        return tab_widget

    def _create_anchors_tab(self) -> QWidget:
        """创建改进的 "锚点配对" 选项卡"""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        
        # 顶部：添加锚点对按钮（醒目）
        add_button_layout = QHBoxLayout()
        self.add_anchor_button = QPushButton("➕ 添加新锚点对")
        self.add_anchor_button.setMinimumHeight(50)
        self.add_anchor_button.setStyleSheet(
            "QPushButton { "
            "background-color: #4CAF50; "
            "color: white; "
            "font-size: 16px; "
            "font-weight: bold; "
            "border-radius: 8px; "
            "} "
            "QPushButton:hover { background-color: #45a049; } "
            "QPushButton:pressed { background-color: #3d8b40; }"
        )
        add_button_layout.addWidget(self.add_anchor_button)
        
        # 确定按钮（初始隐藏）
        self.confirm_anchor_button = QPushButton("✅ 确定添加锚点对")
        self.confirm_anchor_button.setMinimumHeight(50)
        self.confirm_anchor_button.setStyleSheet(
            "QPushButton { "
            "background-color: #2196F3; "
            "color: white; "
            "font-size: 16px; "
            "font-weight: bold; "
            "border-radius: 8px; "
            "} "
            "QPushButton:hover { background-color: #1976D2; } "
            "QPushButton:pressed { background-color: #1565C0; }"
        )
        self.confirm_anchor_button.setVisible(False)
        add_button_layout.addWidget(self.confirm_anchor_button)
        
        layout.addLayout(add_button_layout)
        
        # 说明
        instruction_label = QLabel(
            "🎯 点击「添加新锚点对」，然后：\n"
            "   1️⃣ 在右侧视窗点击手部位置\n"
            "   2️⃣ 在左侧视窗点击物体对应位置\n"
            "   3️⃣ 点击「确定」按钮完成添加\n"
            "   ✨ 然后可以继续添加更多锚点对！"
        )
        instruction_label.setStyleSheet("QLabel { background-color: #fffacd; padding: 8px; border-radius: 5px; }")
        instruction_label.setWordWrap(True)
        layout.addWidget(instruction_label)
        
        # 中间：锚点对列表
        list_group = QGroupBox("锚点对列表")
        list_layout = QVBoxLayout(list_group)
        
        self.anchor_list_widget = QListWidget()
        self.anchor_list_widget.setMinimumHeight(200)
        list_layout.addWidget(self.anchor_list_widget)
        
        # 列表操作按钮
        button_layout = QHBoxLayout()
        self.delete_anchor_button = QPushButton("🗑️ 删除")
        self.clear_all_button = QPushButton("🧹 清空所有")
        
        button_layout.addWidget(self.delete_anchor_button)
        button_layout.addWidget(self.clear_all_button)
        list_layout.addLayout(button_layout)
        
        layout.addWidget(list_group)
        
        # 连接信号
        self.add_anchor_button.clicked.connect(self.add_anchor_pair_signal.emit)
        self.confirm_anchor_button.clicked.connect(self.confirm_anchor_pair_signal.emit)
        self.delete_anchor_button.clicked.connect(self._on_delete_anchor)
        self.clear_all_button.clicked.connect(self._on_clear_all_anchors)
        
        return tab_widget

    def _create_viz_tab(self) -> QWidget:
        """创建 "可视化" 选项卡"""
        tab_widget = QWidget()
        layout = QVBoxLayout(tab_widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # 辅助函数
        def create_opacity_slider() -> QSlider:
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 100)
            slider.setValue(100)
            return slider

        # 中间视图 - 机械手
        hand_group = QGroupBox("中间视图: 机械手 (动态)")
        hand_layout = QGridLayout(hand_group)
        self.hand_opacity_slider = create_opacity_slider()
        self.hand_opacity_slider.setValue(70)
        hand_layout.addWidget(QLabel("透明度:"), 0, 0)
        hand_layout.addWidget(self.hand_opacity_slider, 0, 1)
        
        # 中间视图 - 物体
        obj_group = QGroupBox("中间视图: 物体")
        obj_layout = QGridLayout(obj_group)
        self.object_opacity_slider = create_opacity_slider()
        self.object_opacity_slider.setValue(100)
        obj_layout.addWidget(QLabel("透明度:"), 0, 0)
        obj_layout.addWidget(self.object_opacity_slider, 0, 1)
        
        # 锚点可视化
        anchor_group = QGroupBox("锚点可视化")
        anchor_layout = QGridLayout(anchor_group)
        
        self.anchor_size_label = QLabel("球体大小 (mm):")
        self.anchor_size_spinbox = QDoubleSpinBox()
        self.anchor_size_spinbox.setRange(1.0, 20.0)
        self.anchor_size_spinbox.setValue(8.0)
        self.anchor_size_spinbox.setSingleStep(0.5)
        self.anchor_size_spinbox.setSuffix(" mm")
        
        self.show_lines_checkbox = QCheckBox("显示连接线")
        self.show_lines_checkbox.setChecked(True)
        
        anchor_layout.addWidget(self.anchor_size_label, 0, 0)
        anchor_layout.addWidget(self.anchor_size_spinbox, 0, 1)
        anchor_layout.addWidget(self.show_lines_checkbox, 1, 0, 1, 2)

        layout.addWidget(hand_group)
        layout.addWidget(obj_group)
        layout.addWidget(anchor_group)
        layout.addStretch()
        
        # 连接信号
        self.hand_opacity_slider.valueChanged.connect(self._on_viz_settings_changed)
        self.object_opacity_slider.valueChanged.connect(self._on_viz_settings_changed)
        self.anchor_size_spinbox.valueChanged.connect(self._on_viz_settings_changed)
        self.show_lines_checkbox.stateChanged.connect(self._on_viz_settings_changed)
        
        return tab_widget

    def _create_joints_tab(self) -> QWidget:
        """创建 "关节调试" 选项卡"""
        tab_widget = QWidget()
        main_layout = QVBoxLayout(tab_widget)
        
        self.joints_placeholder_label = QLabel(
            "加载机械手后，这里会显示所有关节的控制器"
        )
        self.joints_placeholder_label.setWordWrap(True)
        self.joints_placeholder_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        
        self.scroll_content_widget = QWidget()
        self.joints_layout = QVBoxLayout(self.scroll_content_widget)
        self.joints_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.joints_layout.addWidget(self.joints_placeholder_label)
        
        scroll_area.setWidget(self.scroll_content_widget)
        main_layout.addWidget(scroll_area)
        
        return tab_widget

    # --- 槽函数 ---

    def _on_delete_anchor(self) -> None:
        """删除选中的锚点对"""
        current_row = self.anchor_list_widget.currentRow()
        if current_row >= 0:
            self.delete_anchor_signal.emit(current_row)

    def _on_clear_all_anchors(self) -> None:
        """清空所有锚点对"""
        count = self.anchor_list_widget.count()
        # 从后向前删除
        for i in range(count - 1, -1, -1):
            self.delete_anchor_signal.emit(i)

    def _on_viz_settings_changed(self) -> None:
        """可视化设置改变"""
        settings = {
            'hand_opacity': self.hand_opacity_slider.value() / 100.0,
            'object_opacity': self.object_opacity_slider.value() / 100.0,
            'anchor_size': self.anchor_size_spinbox.value() / 1000.0,  # mm to m
            'show_lines': self.show_lines_checkbox.isChecked()
        }
        self.visualization_settings_changed_signal.emit(settings)

    # --- 公共方法 (由 main_window/data_manager 调用) ---

    @pyqtSlot(list)
    def update_anchor_list(self, anchor_pairs: list) -> None:
        """
        更新锚点对列表的显示
        
        :param anchor_pairs: 锚点对列表
        """
        self.anchor_list_widget.clear()
        
        for i, pair in enumerate(anchor_pairs):
            # 获取颜色
            color = self.anchor_colors[i % len(self.anchor_colors)]
            
            # 创建列表项
            link_name = pair.get('hand_link_name', '未知')
            enabled = pair.get('enabled', True)
            
            item_text = f"锚点对 {i+1}: {link_name}"
            if not enabled:
                item_text += " (已禁用)"
            
            item = QListWidgetItem(item_text)
            
            # 设置颜色
            item.setBackground(QBrush(color.lighter(180)))
            item.setForeground(QBrush(QColor(0, 0, 0)))
            
            # 如果禁用，设置灰色
            if not enabled:
                item.setForeground(QBrush(QColor(128, 128, 128)))
            
            self.anchor_list_widget.addItem(item)

    @pyqtSlot(list)
    def populate_joint_controls(self, joint_info_list: list) -> None:
        """
        根据机械手的关节信息，动态生成关节控制滑块
        
        :param joint_info_list: 列表，每个元素是一个字典:
                                {'name': str, 'min': float, 'max': float, 'default': float}
        """
        # 清除占位符
        self.joints_placeholder_label.setParent(None)
        
        # 清除旧控件
        for joint_name in list(self.joint_controls.keys()):
            controls = self.joint_controls.pop(joint_name)
            controls['frame'].setParent(None)
        
        # 为每个关节创建控件
        for joint_info in joint_info_list:
            name = joint_info['name']
            min_val = joint_info['min']
            max_val = joint_info['max']
            default_val = joint_info.get('default', 0.0)
            
            # 创建框架
            frame = QFrame()
            frame.setFrameShape(QFrame.Shape.StyledPanel)
            frame_layout = QGridLayout(frame)
            
            # 标签
            label = QLabel(f"{name}:")
            label.setMinimumWidth(100)
            
            # 滑块
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(int(min_val * 1000), int(max_val * 1000))
            slider.setValue(int(default_val * 1000))
            
            # 数值框
            spinbox = QDoubleSpinBox()
            spinbox.setRange(min_val, max_val)
            spinbox.setValue(default_val)
            spinbox.setSingleStep(0.01)
            spinbox.setDecimals(3)
            
            # 布局
            frame_layout.addWidget(label, 0, 0)
            frame_layout.addWidget(slider, 0, 1)
            frame_layout.addWidget(spinbox, 0, 2)
            
            # 连接信号
            slider.valueChanged.connect(
                lambda val, sb=spinbox: sb.setValue(val / 1000.0)
            )
            spinbox.valueChanged.connect(
                lambda val, sl=slider: sl.setValue(int(val * 1000))
            )
            spinbox.valueChanged.connect(
                lambda val, jname=name: self.manual_joint_changed_signal.emit(jname, val)
            )
            
            # 添加到布局
            self.joints_layout.addWidget(frame)
            
            # 存储
            self.joint_controls[name] = {
                'frame': frame,
                'slider': slider,
                'spinbox': spinbox
            }

    def get_anchor_color(self, index: int) -> tuple:
        """
        获取指定索引的锚点对颜色 (RGB, 0-1范围)
        
        :param index: 锚点对索引
        :return: (r, g, b) 元组
        """
        color = self.anchor_colors[index % len(self.anchor_colors)]
        return (color.redF(), color.greenF(), color.blueF())
    
    def show_confirm_button(self, show: bool = True) -> None:
        """
        显示或隐藏确定按钮
        
        :param show: True 显示，False 隐藏
        """
        self.confirm_anchor_button.setVisible(show)
    
    # 向后兼容：create_joint_controls 是 populate_joint_controls 的别名
    create_joint_controls = populate_joint_controls


# --- 测试代码 ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    widget = ControlsWidget()
    widget.setWindowTitle("改进的控制面板")
    widget.resize(400, 600)
    widget.show()
    
    # 模拟添加锚点
    def test_add_anchor():
        test_pairs = [
            {'hand_link_name': 'forearm', 'enabled': True},
            {'hand_link_name': 'palm', 'enabled': True},
            {'hand_link_name': 'finger1', 'enabled': False},
        ]
        widget.update_anchor_list(test_pairs)
    
    widget.add_anchor_button.clicked.connect(test_add_anchor)
    
    sys.exit(app.exec())
