# keyboard_controller.py
# 键盘控制模块，用于调整锚点位置

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, Qt
from PyQt6.QtWidgets import QApplication


class KeyboardController(QObject):
    """
    键盘控制器，用于调整锚点位置

    控制键：
    - W/S: X轴方向移动 (±1mm)
    - A/D: Y轴方向移动 (±1mm)
    - 空格/Shift: Z轴方向移动 (±1mm)
    """

    # 信号定义
    position_changed_signal = pyqtSignal(int, str, np.ndarray)  # anchor_index, point_type, new_position
    control_ended_signal = pyqtSignal()  # 控制结束
    control_state_changed_signal = pyqtSignal(int, str, bool)  # anchor_index, point_type, is_active

    def __init__(self, parent=None):
        super().__init__(parent)

        self.is_active = False
        self.current_anchor_index = -1
        self.current_point_type = ""  # "hand" 或 "object"
        self.original_position = None

        # 移动步长：1mm = 0.001m
        self.step_size = 0.001

        # 安装事件过滤器
        QApplication.instance().installEventFilter(self)

    def start_control(self, anchor_index: int, point_type: str, current_position: np.ndarray):
        """
        开始键盘控制

        :param anchor_index: 锚点对索引
        :param point_type: 点类型 ("hand" 或 "object")
        :param current_position: 当前位置 (numpy array) - 对于hand是局部坐标，对于object是世界坐标
        """
        self.is_active = True
        self.current_anchor_index = anchor_index
        self.current_point_type = point_type
        self.original_position = current_position.copy()

        print(f"开始键盘控制 - 锚点对 {anchor_index + 1}, {point_type} 点")
        print("控制键: W/S=X轴, A/D=Y轴, 空格/Shift=Z轴, 再次点击按钮结束")
        
        # 发射状态变化信号
        self.control_state_changed_signal.emit(anchor_index, point_type, True)

    def is_controlling(self, anchor_index: int, point_type: str) -> bool:
        """
        检查是否正在控制指定的锚点
        
        :param anchor_index: 锚点对索引
        :param point_type: 点类型 ("hand" 或 "object")
        :return: True 如果正在控制该锚点
        """
        return (self.is_active and 
                self.current_anchor_index == anchor_index and 
                self.current_point_type == point_type)

    def toggle_control(self, anchor_index: int, point_type: str, current_position: np.ndarray) -> bool:
        """
        切换控制状态：如果正在控制指定的锚点则结束，否则先停止当前所有控制然后开始新控制
        
        :param anchor_index: 锚点对索引
        :param point_type: 点类型 ("hand" 或 "object")
        :param current_position: 当前位置 (numpy array)
        :return: True 如果开始控制，False 如果结束控制
        """
        if self.is_controlling(anchor_index, point_type):
            # 如果正在控制指定的锚点，结束控制
            self.end_control()
            return False  # 结束控制
        else:
            # 如果没有控制指定的锚点，先停止任何当前的控制，然后开始新控制
            if self.is_active:
                self.end_control()
            self.start_control(anchor_index, point_type, current_position)
            return True   # 开始控制

    def end_control(self):
        """结束键盘控制"""
        if self.is_active:
            # 保存当前状态用于发射信号
            anchor_index = self.current_anchor_index
            point_type = self.current_point_type
            
            self.is_active = False
            self.current_anchor_index = -1
            self.current_point_type = ""
            self.original_position = None
            self.control_ended_signal.emit()
            print("键盘控制结束")
            
            # 发射状态变化信号
            self.control_state_changed_signal.emit(anchor_index, point_type, False)

    def eventFilter(self, obj, event):
        """事件过滤器，处理键盘事件"""
        if not self.is_active:
            return False

        if event.type() == event.Type.KeyPress:
            key = event.key()
            delta = np.zeros(3)

            # X轴控制 (W/S)
            if key == Qt.Key.Key_W:
                delta[0] = self.step_size
            elif key == Qt.Key.Key_S:
                delta[0] = -self.step_size

            # Y轴控制 (A/D)
            elif key == Qt.Key.Key_A:
                delta[1] = self.step_size
            elif key == Qt.Key.Key_D:
                delta[1] = -self.step_size

            # Z轴控制 (空格/Shift)
            elif key == Qt.Key.Key_Space:
                delta[2] = self.step_size
            elif key == Qt.Key.Key_Shift:
                delta[2] = -self.step_size

            # 如果有移动，更新位置
            if np.any(delta != 0):
                new_position = self.original_position + delta
                self.position_changed_signal.emit(
                    self.current_anchor_index,
                    self.current_point_type,
                    new_position
                )
                self.original_position = new_position
                return True  # 事件已处理

        return False

    def update_current_position(self, new_position: np.ndarray):
        """
        更新当前控制的位置（当外部修改时调用）

        :param new_position: 新的位置
        """
        if self.is_active:
            self.original_position = new_position.copy()