# main_window.py (migrated)
import numpy as np
import sys
from PyQt6.QtWidgets import (
	QMainWindow, QWidget, QApplication, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt

from .vista_widget import VistaWidget
from .controls_widget import ControlsWidget  # 使用新版 UI
from .data_manager import DataManager
from .optimization_thread import OptimizationThread
from utils.constants import ANCHOR_COLOR_CYCLE
from utils.hand_config import ensure_hand_config  # new utility (to be added)
from .ui.anchor_view import AnchorViewCoordinator
from .ui.wiring import connect_all
from .keyboard_controller import KeyboardController


class MainWindow(QMainWindow):
	"""
	应用程序的主窗口。
	负责初始化所有UI组件和核心逻辑模块，并设置窗口布局。 
	"""
    
	def __init__(self, parent: QWidget | None = None, load_default: bool = False, selected_hand: str = "shadow") -> None:
		super().__init__(parent)
		self.selected_hand = selected_hand
        
		# 1. 初始化核心逻辑组件 (非UI)
		self.init_core_components() 
        
		# 2. 初始化UI组件和布局
		self.init_ui()
        
		# 3. 连接所有组件的信号和槽（拆分至 wiring 模块）
		self.anchors = AnchorViewCoordinator(self)
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
		# 将选择的手配置名通知 DataManager（用于 hand_config 名称）
		if hasattr(self, 'selected_hand') and self.selected_hand:
			self.data_manager.set_selected_hand_name(self.selected_hand)

	def init_ui(self) -> None:
		"""初始化所有UI小部件并设置主窗口布局 (委托给 ui_builder)."""
		print("初始化UI布局...")
		from .ui.ui_builder import build_ui
		build_ui(self)
		print("UI布局初始化完成。")

	def connect_signals(self) -> None:
		"""连接所有组件的信号和槽，定义应用程序的逻辑流程。"""
		print("连接信号与槽...")
		connect_all(self, self.anchors)
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
			# 重置按钮状态
			self.controls_widget.reset_anchor_button_state()

	def _clear_temp_anchors(self) -> None:
		"""
		清空所有视窗的临时锚点球体
		"""
		self.view_left.clear_temp_anchors()
		self.view_right.clear_temp_anchors()
		self.view_center.clear_temp_anchors()
    
	def _show_temp_anchor(self, anchor_data: dict) -> None:
		"""
		显示临时锚点球体
		"""
		# 获取下一个锚点对的颜色
		next_index = len(self.data_manager.anchor_pairs)
		color_rgb = self.controls_widget.get_anchor_color(next_index)
        
		# 先清除所有临时锚点
		self._clear_temp_anchors()
        
		# 显示所有已选择的临时锚点
		if self.data_manager._temp_hand_anchor:
			point = self.data_manager._temp_hand_anchor['world_coord']
			self.view_left.add_temp_anchor(point, color_rgb, True)
			self.view_right.add_temp_anchor(point, color_rgb, True)
			self.view_center.add_temp_anchor(point, color_rgb, True)
        
		if self.data_manager._temp_object_anchor:
			point = self.data_manager._temp_object_anchor['world_coord']
			self.view_left.add_temp_anchor(point, color_rgb, False)
			self.view_right.add_temp_anchor(point, color_rgb, False)
			self.view_center.add_temp_anchor(point, color_rgb, False)

	def on_visualization_changed(self, settings: dict) -> None:
		"""
		处理来自工具栏的可视化设置更改。
		"""
		print(f"主窗口：收到可视化设置: {settings}")
        
		# 示例：更新中心视窗的透明度
		if 'hand_opacity' in settings:
			# 假设 vista_widget.set_actor_properties 支持通配符
			self.view_center.set_actor_properties(
				name_pattern="dyn_hand_", 
				opacity=settings['hand_opacity']
			)
        
		if 'object_opacity' in settings:
			self.view_center.set_actor_properties(
				name="object", 
				opacity=settings['object_opacity']
			)
        
		# 可以在此处扩展颜色等其他设置...

	def on_new_anchor_pair_auto_start(self, anchor_pairs: list) -> None:
		"""
		当确认添加锚点对后，自动开始优化并同步按钮状态。
		"""
		if not anchor_pairs:
			return
		# 同步UI：切换到“运行中”
		self.controls_widget.set_optimization_state(True)
		# 恢复线程并触发优化（使用当前锚点）
		self.optimization_thread.resume()
		self.optimization_thread.trigger_optimization(anchor_pairs)

	# on_hand_initial_pose_received moved to AnchorViewCoordinator
    
	# on_anchor_list_updated moved to AnchorViewCoordinator

	def get_color_for_pair(self, index: int) -> str:
		"""
		根据锚点对索引返回颜色。
		"""
		return ANCHOR_COLOR_CYCLE[index % len(ANCHOR_COLOR_CYCLE)]
    
	# on_pose_update_with_anchors moved to AnchorViewCoordinator

	def on_base_pose_updated(self, translation: list, rotation_matrix: list) -> None:
		"""优化线程同步的基座姿态更新"""
		self.data_manager.update_base_pose(translation, rotation_matrix)
		self.controls_widget.update_hand_pose_display(translation, rotation_matrix)

	def load_default_assets(self) -> None:
		"""
		自动加载默认测试资源
		"""
		import os, yaml
		from PyQt6.QtCore import QTimer
        
		# 获取项目根目录（main_window.py 的上级目录）
		current_dir = os.path.dirname(os.path.abspath(__file__))
		project_root = os.path.dirname(current_dir)
        
		object_path = os.path.join(project_root, "test_assets", "objects", "Mug.obj")
		# 从 hand_config/<selected_hand>.yaml 读取 URDF 路径（如不存在则回退到 shadow）
		cfg_dir = os.path.join(project_root, 'hand_config')
		selected = getattr(self, 'selected_hand', 'shadow') or 'shadow'
		cfg_path = ensure_hand_config(selected, cfg_dir, project_root, parent=self)
		if not cfg_path:
			# Fallback to default.yaml if user skipped creation
			cfg_path = os.path.join(cfg_dir, 'default.yaml')
		cfg = {}
		if os.path.exists(cfg_path):
			try:
				with open(cfg_path, 'r') as f:
					cfg = yaml.safe_load(f) or {}
			except Exception as e:
				print(f"加载 hand 配置失败: {e}")
		urdf_rel = cfg.get('urdf_path') or os.path.join('test_assets', 'shadow', 'shadow_hand_right.urdf')
		hand_path = urdf_rel if os.path.isabs(urdf_rel) else os.path.join(project_root, urdf_rel)
        
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
				self.statusBar().showMessage(f"✓ 已自动加载手部: {os.path.basename(hand_path)}")
			else:
				print(f"警告: 未找到手部文件: {hand_path}")
				self.statusBar().showMessage(f"✗ 未找到手部文件: {hand_path}")
        
		# 延迟500ms后加载，确保窗口已显示
		QTimer.singleShot(500, delayed_load)

	def _ensure_hand_config(self, hand_name: str, cfg_dir: str, project_root: str) -> str | None:
		"""确保 hand_config/<hand_name>.yaml 存在；如果不存在则询问是否创建并选择URDF路径创建。

		返回配置文件的绝对路径；若用户取消则返回 None。
		"""
		import os, yaml
		os.makedirs(cfg_dir, exist_ok=True)
		cfg_path = os.path.join(cfg_dir, f"{hand_name}.yaml")
		if os.path.exists(cfg_path):
			return cfg_path

		# 询问是否创建
		reply = QMessageBox.question(
			self,
			"创建手配置",
			f"未找到手配置文件: {cfg_path}\n是否现在创建并选择 URDF 路径?",
			QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
			QMessageBox.StandardButton.Yes,
		)
		if reply != QMessageBox.StandardButton.Yes:
			return None

		# 选择 URDF 文件
		urdf_path, _ = QFileDialog.getOpenFileName(
			self,
			"选择手 URDF 文件",
			project_root,
			"URDF 文件 (*.urdf)"
		)
		if not urdf_path:
			return None

		# 写入 YAML（尽量使用相对路径）
		try:
			rel_path = os.path.relpath(urdf_path, project_root)
		except Exception:
			rel_path = urdf_path

		cfg = {
			'urdf_path': rel_path,
			'base_pose': {
				'translation_m': [0.0, 0.0, 0.0],
				'rpy_deg': [0.0, 0.0, 0.0],
			},
			'joints': 'default'
		}

		try:
			with open(cfg_path, 'w') as f:
				yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
			self.statusBar().showMessage(f"✓ 已创建手配置: {cfg_path}")
			return cfg_path
		except Exception as e:
			QMessageBox.warning(self, "写入失败", f"无法写入配置文件:\n{e}")
			return None

	def on_hand_identity_loaded(self, hand_name: str) -> None:
		"""当 DataManager 发出手身份后，确保配置存在，再通知优化线程应用该配置。"""
		import os
		current_dir = os.path.dirname(os.path.abspath(__file__))
		project_root = os.path.dirname(current_dir)
		cfg_dir = os.path.join(project_root, 'hand_config')
		ensure_hand_config(hand_name, cfg_dir, project_root, parent=self)
		# 继续让优化线程应用该 hand 的配置
		self.optimization_thread.apply_hand_config(hand_name)

	def on_start_adjust_hand_anchor(self, anchor_index: int) -> None:
		"""
		开始或结束调整手上锚点
        
		:param anchor_index: 锚点对索引
		"""
		print(f"DEBUG: on_start_adjust_hand_anchor called for anchor {anchor_index}")
        
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
			# 如果有锚点，开始优化
			if self.data_manager.anchor_pairs:
				self.optimization_thread.trigger_optimization(self.data_manager.anchor_pairs)
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
