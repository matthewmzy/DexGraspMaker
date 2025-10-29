# controls_widget_v2.py
# 重新设计的控件面板，支持更直观的多锚点工作流

import sys
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QTabWidget, QPushButton, QSlider, QListWidget, QListWidgetItem,
    QVBoxLayout, QHBoxLayout, QGroupBox, QDoubleSpinBox, QLabel,
    QGridLayout, QScrollArea, QFrame, QApplication, QColorDialog, QCheckBox,
    QStyledItemDelegate, QHBoxLayout
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
    
    # 锚点位置调整
    adjust_hand_anchor_signal = pyqtSignal(int)  # 开始调整手上锚点 (anchor_index)
    adjust_object_anchor_signal = pyqtSignal(int)  # 开始调整物体锚点 (anchor_index)
    update_anchor_position_signal = pyqtSignal(int, str, list)  # 更新锚点位置 (anchor_index, point_type, position)
    
    # 键盘控制状态
    keyboard_control_state_changed_signal = pyqtSignal(int, str, bool)  # anchor_index, point_type, is_active
    
    # 流程 3: 可视化
    visualization_settings_changed_signal = pyqtSignal(dict)
    
    # 流程 4: 关节调试
    manual_joint_changed_signal = pyqtSignal(str, float)
    
    # 优化控制
    optimization_toggle_signal = pyqtSignal(bool)  # True=开始, False=暂停
    
    # 姿态导入/导出
    import_pose_signal = pyqtSignal()
    export_pose_signal = pyqtSignal()

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
        
        # 存储锚点调整按钮的引用，用于状态更新
        self.anchor_adjust_buttons = {}  # {(anchor_index, point_type): button}
        
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
        main_layout = QHBoxLayout(tab_widget)  # 改为水平布局
        
        # 左侧：操作区域
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
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
        
        left_layout.addLayout(add_button_layout)
        
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
        left_layout.addWidget(instruction_label)
        
        # 添加伸缩空间
        left_layout.addStretch()
        
        main_layout.addWidget(left_widget)
        
        # 右侧：锚点对列表
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        list_group = QGroupBox("锚点对列表")
        list_layout = QVBoxLayout(list_group)
        
        self.anchor_list_widget = QListWidget()
        self.anchor_list_widget.setMinimumHeight(300)
        list_layout.addWidget(self.anchor_list_widget)
        
        # 列表操作按钮
        button_layout = QHBoxLayout()
        self.delete_anchor_button = QPushButton("🗑️ 删除")
        self.clear_all_button = QPushButton("🧹 清空所有")
        
        button_layout.addWidget(self.delete_anchor_button)
        button_layout.addWidget(self.clear_all_button)
        list_layout.addLayout(button_layout)
        
        right_layout.addWidget(list_group)
        
        main_layout.addWidget(right_widget)
        
        # 设置左右两部分的宽度比例为 3:2
        main_layout.setStretchFactor(left_widget, 3)
        main_layout.setStretchFactor(right_widget, 2)
        
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
        main_layout = QHBoxLayout(tab_widget)  # 改为水平布局
        
        # 左侧控制面板 (40%)
        left_panel = self._create_joints_left_panel()
        left_panel.setFixedWidth(300)  # 固定宽度约40%
        
        # 右侧关节滑块面板 (60%)
        right_panel = self._create_joints_right_panel()
        
        main_layout.addWidget(left_panel, 2)  # 比例 2
        main_layout.addWidget(right_panel, 3)  # 比例 3
        
        return tab_widget

    def _create_joints_left_panel(self) -> QWidget:
        """创建关节调试左侧控制面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 优化控制组
        opt_group = QGroupBox("优化控制")
        opt_layout = QVBoxLayout(opt_group)
        
        # 优化开启/暂停按钮
        self.optimization_toggle_button = QPushButton("▶️ 开始优化")
        self.optimization_toggle_button.setCheckable(True)
        self.optimization_toggle_button.setChecked(True)  # 默认开启
        self.optimization_toggle_button.clicked.connect(self._on_optimization_toggle)
        opt_layout.addWidget(self.optimization_toggle_button)
        
        layout.addWidget(opt_group)
        
        # 手姿态显示组
        pose_group = QGroupBox("手姿态")
        pose_layout = QVBoxLayout(pose_group)
        
        # 平移矩阵
        translation_label = QLabel("平移 (X, Y, Z):")
        pose_layout.addWidget(translation_label)
        
        self.translation_display = QLabel("0.000, 0.000, 0.000")
        self.translation_display.setStyleSheet("font-family: monospace; background-color: #f0f0f0; padding: 5px;")
        pose_layout.addWidget(self.translation_display)
        
        # 旋转矩阵
        rotation_label = QLabel("旋转矩阵:")
        pose_layout.addWidget(rotation_label)
        
        self.rotation_display = QLabel(
            "1.000  0.000  0.000\n"
            "0.000  1.000  0.000\n"
            "0.000  0.000  1.000"
        )
        self.rotation_display.setStyleSheet("font-family: monospace; background-color: #f0f0f0; padding: 5px;")
        pose_layout.addWidget(self.rotation_display)
        
        layout.addWidget(pose_group)
        
        # 姿态导入/导出组
        io_group = QGroupBox("姿态管理")
        io_layout = QVBoxLayout(io_group)
        
        self.import_pose_button = QPushButton("📥 导入姿态")
        self.import_pose_button.clicked.connect(self._on_import_pose)
        io_layout.addWidget(self.import_pose_button)
        
        self.export_pose_button = QPushButton("📤 导出姿态")
        self.export_pose_button.clicked.connect(self._on_export_pose)
        io_layout.addWidget(self.export_pose_button)
        
        layout.addWidget(io_group)
        
        # 添加伸缩空间
        layout.addStretch()
        
        return panel

    def _create_joints_right_panel(self) -> QWidget:
        """创建关节调试右侧滑块面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
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
        layout.addWidget(scroll_area)
        
        return panel

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
            
            # 创建列表项容器
            item_widget = QWidget()
            item_layout = QHBoxLayout(item_widget)
            item_layout.setContentsMargins(5, 5, 5, 5)
            
            # 左侧：文本信息
            link_name = pair.get('hand_link_name', '未知')
            enabled = pair.get('enabled', True)
            
            item_text = f"锚点对 {i+1}: {link_name}"
            if not enabled:
                item_text += " (已禁用)"
            
            text_label = QLabel(item_text)
            text_label.setStyleSheet(f"font-weight: bold; color: {color.name()};")
            if not enabled:
                text_label.setStyleSheet("color: #808080; font-style: italic;")
            
            item_layout.addWidget(text_label)
            
            # 添加伸缩空间
            item_layout.addStretch()
            
            # 右侧：调整按钮
            adjust_hand_btn = QPushButton("🤖 调手上")
            adjust_hand_btn.setFixedSize(80, 30)
            adjust_hand_btn.setStyleSheet(
                "QPushButton { background-color: #2196F3; color: white; border-radius: 3px; }"
                "QPushButton:hover { background-color: #1976D2; }"
                "QPushButton:pressed { background-color: #1565C0; }"
            )
            adjust_hand_btn.clicked.connect(lambda checked, idx=i: self.adjust_hand_anchor_signal.emit(idx))
            
            adjust_obj_btn = QPushButton("📦 调物体")
            adjust_obj_btn.setFixedSize(80, 30)
            adjust_obj_btn.setStyleSheet(
                "QPushButton { background-color: #4CAF50; color: white; border-radius: 3px; }"
                "QPushButton:hover { background-color: #45a049; }"
                "QPushButton:pressed { background-color: #3d8b40; }"
            )
            adjust_obj_btn.clicked.connect(lambda checked, idx=i: self.adjust_object_anchor_signal.emit(idx))
            
            # 保存按钮引用
            self.anchor_adjust_buttons[(i, "hand")] = adjust_hand_btn
            self.anchor_adjust_buttons[(i, "object")] = adjust_obj_btn
            
            item_layout.addWidget(adjust_hand_btn)
            item_layout.addWidget(adjust_obj_btn)
            
            # 创建列表项
            item = QListWidgetItem()
            item.setSizeHint(item_widget.sizeHint())
            
            # 设置背景色
            item.setBackground(QBrush(color.lighter(190)))
            
            self.anchor_list_widget.addItem(item)
            self.anchor_list_widget.setItemWidget(item, item_widget)

    def update_anchor_adjust_button_state(self, anchor_index: int, point_type: str, is_active: bool) -> None:
        """
        更新锚点调整按钮的状态
        
        :param anchor_index: 锚点对索引
        :param point_type: 点类型 ("hand" 或 "object")
        :param is_active: 是否正在调整
        """
        button_key = (anchor_index, point_type)
        if button_key in self.anchor_adjust_buttons:
            button = self.anchor_adjust_buttons[button_key]
            
            if is_active:
                # 激活状态：改变颜色和文本
                if point_type == "hand":
                    button.setText("🔵 调整中")
                    button.setStyleSheet(
                        "QPushButton { background-color: #FF9800; color: white; border-radius: 3px; font-weight: bold; }"
                        "QPushButton:hover { background-color: #F57C00; }"
                        "QPushButton:pressed { background-color: #EF6C00; }"
                    )
                else:  # object
                    button.setText("🔵 调整中")
                    button.setStyleSheet(
                        "QPushButton { background-color: #FF9800; color: white; border-radius: 3px; font-weight: bold; }"
                        "QPushButton:hover { background-color: #F57C00; }"
                        "QPushButton:pressed { background-color: #EF6C00; }"
                    )
            else:
                # 非激活状态：恢复原始样式
                if point_type == "hand":
                    button.setText("🤖 调手上")
                    button.setStyleSheet(
                        "QPushButton { background-color: #2196F3; color: white; border-radius: 3px; }"
                        "QPushButton:hover { background-color: #1976D2; }"
                        "QPushButton:pressed { background-color: #1565C0; }"
                    )
                else:  # object
                    button.setText("📦 调物体")
                    button.setStyleSheet(
                        "QPushButton { background-color: #4CAF50; color: white; border-radius: 3px; }"
                        "QPushButton:hover { background-color: #45a049; }"
                        "QPushButton:pressed { background-color: #3d8b40; }"
                    )

    def update_anchor_position(self, anchor_index: int, point_type: str, position: list) -> None:
        """
        更新锚点位置（由键盘控制器调用）
        
        :param anchor_index: 锚点对索引
        :param point_type: 点类型 ("hand" 或 "object")
        :param position: 新位置 [x, y, z]
        """
        self.update_anchor_position_signal.emit(anchor_index, point_type, position)

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

    # --- 关节调试相关的槽函数 ---

    def _on_optimization_toggle(self) -> None:
        """优化开启/暂停按钮点击"""
        is_checked = self.optimization_toggle_button.isChecked()
        if is_checked:
            self.optimization_toggle_button.setText("⏸️ 暂停优化")
        else:
            self.optimization_toggle_button.setText("▶️ 开始优化")
        self.optimization_toggle_signal.emit(is_checked)

    def _on_import_pose(self) -> None:
        """导入姿态按钮点击"""
        self.import_pose_signal.emit()

    def _on_export_pose(self) -> None:
        """导出姿态按钮点击"""
        self.export_pose_signal.emit()

    # --- 公共方法 ---

    def update_hand_pose_display(self, translation: list, rotation_matrix: list) -> None:
        """
        更新手姿态显示
        
        :param translation: 平移向量 [x, y, z]
        :param rotation_matrix: 3x3旋转矩阵
        """
        # 更新平移显示
        self.translation_display.setText(".3f")
        
        # 更新旋转矩阵显示
        rot_text = ".3f"
        self.rotation_display.setText(rot_text)

    def set_optimization_state(self, is_running: bool) -> None:
        """
        设置优化状态显示
        
        :param is_running: 优化是否正在运行
        """
        self.optimization_toggle_button.setChecked(is_running)
        if is_running:
            self.optimization_toggle_button.setText("⏸️ 暂停优化")
        else:
            self.optimization_toggle_button.setText("▶️ 开始优化")


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
