# data_manager.py

import sys
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QUrl
from PyQt6.QtWidgets import QFileDialog

# -------------------------------------------------------------------
# [依赖项导入]
# -------------------------------------------------------------------

try:
    import pyvista
except ImportError:
    print("错误：'pyvista' 库未找到。请运行: pip install pyvista")
    # 模拟一个 PolyData 类，以便代码在语法上有效
    class PolyData: pass
    pyvista = type('PyVistaMock', (object,), {'PolyData': PolyData, 'read': lambda: None, 'wrap': lambda: None})()

try:
    import trimesh
except ImportError:
    print("错误：'trimesh' 库未找到。请运行: pip install trimesh")
    # 模拟 trimesh
    trimesh = type('TrimeshMock', (object,), {'load': lambda: None, 'Scene': type('Scene', (object,), {})})()

# -------------------------------------------------------------------

class DataManager(QObject):
    """
    管理应用程序的状态、数据加载和业务逻辑。
    这是 UI 和后端优化线程之间的主要协调者。
    """
    
    # --- 信号定义 ---
    
    # 信号 1: 加载成功后，将 mesh 数据发射给 main_window
    object_loaded_signal = pyqtSignal(pyvista.PolyData)
    # {link_name: pyvista.PolyData}
    hand_loaded_signal = pyqtSignal(dict) 
    
    # 信号 2: 加载 URDF 成功后，将关节信息发射给 controls_widget
    # [{'name': str, 'min': float, 'max': float, 'default': float}]
    hand_joint_info_signal = pyqtSignal(list)
    
    # 信号 3: 锚点列表发生变化时
    # 信号发给 optimization_thread
    new_anchor_pair_signal = pyqtSignal(list)
    # 信号发给 controls_widget 的 QListWidget
    anchor_list_updated_signal = pyqtSignal(list)
    
    # 信号 4: 拾取模式状态改变
    picking_mode_changed_signal = pyqtSignal(bool)
    
    # 信号 5: （可选）状态栏消息
    status_message_signal = pyqtSignal(str) # (用于提示用户)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        
        # --- 内部状态 ---
        
        # 1. 加载的数据
        self.object_mesh: pyvista.PolyData | None = None
        self.hand_links_mesh_dict: dict[str, pyvista.PolyData] = {}
        self.joint_info: list[dict] = []
        
        # 2. 锚点状态
        self.anchor_pairs: list[dict] = []
        
        # 3. 拾取状态机
        self.is_picking_mode: bool = False
        self._current_pick_stage: str = 'hand' # 'hand' 或 'object'
        self._temp_hand_anchor: dict | None = None # 存储临时的手部点信息

    # --- 公共槽 (Public Slots) ---
    # 这些槽由 main_window 连接到 controls_widget 的信号

    @pyqtSlot()
    def load_object(self) -> None:
        """
        打开文件对话框以加载物体 mesh。
        """
        file_path, _ = QFileDialog.getOpenFileName(
            None, # parent
            "加载物体 Mesh", 
            "", # directory
            "Mesh 文件 (*.stl *.obj *.ply *.vtk *.vtp)"
        )
        
        if not file_path:
            return # 用户取消
            
        try:
            self.object_mesh = pyvista.read(file_path)
            print(f"DataManager: 已加载物体: {file_path}")
            # 发射信号，将 mesh 数据传递给视窗
            self.object_loaded_signal.emit(self.object_mesh)
            self.status_message_signal.emit(f"已加载物体: {file_path}")
        except Exception as e:
            print(f"DataManager: 加载物体失败: {e}")
            self.status_message_signal.emit(f"加载物体失败: {e}")

    @pyqtSlot()
    def load_hand(self) -> None:
        """
        打开文件对话框以加载机械手 URDF。
        这将解析 URDF，提取 link 的 meshes 和 joint 的信息。
        """
        file_path, _ = QFileDialog.getOpenFileName(
            None, 
            "加载机械手 URDF", 
            "", 
            "URDF 文件 (*.urdf)"
        )
        
        if not file_path:
            return # 用户取消
            
        try:
            # 1. 使用 trimesh 加载 URDF
            # trimesh 会自动加载所有关联的 mesh (stl, dae, obj...)
            # 它返回一个 trimesh.Scene 对象
            
            # URDFs 经常使用 'package://' 路径，我们需要提供一个解析器
            # 我们假设 mesh 文件位于 URDF 文件的相对目录中
            urdf_url = QUrl.fromLocalFile(file_path)
            
            def resolve_package_path(path):
                # 简单的解析器：假设 'package://' 指向 URDF 文件的父目录
                if path.startswith('package://'):
                    # TBD: 这部分可能需要根据实际的包结构进行调整
                    # 一个更健壮的实现会搜索 ROS_PACKAGE_PATH
                    base_dir = QUrl(urdf_url).adjusted(QUrl.UrlFormattingOption.RemoveFilename)
                    # 移除 'package://' 和包名 (假设包名是第一个路径组件)
                    relative_path = "/".join(path.split('/')[2:]) 
                    resolved_url = base_dir.resolved(QUrl(relative_path))
                    return resolved_url.toLocalFile()
                return path

            # trimesh.load 返回一个 Scene 对象
            scene = trimesh.load(file_path, file_resolver=resolve_package_path)

            # 2. 提取 Link Meshes (几何)
            self.hand_links_mesh_dict.clear()
            
            # 遍历场景图，找到哪个节点(link)对应哪个几何体(mesh)
            for link_name, geom_key in scene.graph.nodes_geometry.items():
                trimesh_mesh = scene.geometry[geom_key]
                
                # 将 trimesh.Trimesh 转换为 pyvista.PolyData
                # 我们需要确保 trimesh_mesh 是 Trimesh 对象，而不是 Path3D 等
                if isinstance(trimesh_mesh, trimesh.Trimesh):
                    pv_mesh = pyvista.wrap(trimesh_mesh)
                    self.hand_links_mesh_dict[link_name] = pv_mesh
                else:
                    print(f"DataManager: 跳过非 Mesh 的几何体 '{link_name}' (类型: {type(trimesh_mesh)})")

            if not self.hand_links_mesh_dict:
                raise ValueError("未能在 URDF 中找到任何可加载的 link meshes。")

            print(f"DataManager: 已加载 {len(self.hand_links_mesh_dict)} 个 link meshes。")
            self.hand_loaded_signal.emit(self.hand_links_mesh_dict)

            # 3. 提取 Joint 信息
            self.joint_info.clear()
            
            # trimesh (如果安装了 urdfpy) 会将 urdf 对象存储在元数据中
            if '_loaded_urdf' in scene.metadata:
                urdf_robot = scene.metadata['_loaded_urdf']
                for joint in urdf_robot.joints:
                    if joint.joint_type in ['revolute', 'prismatic']:
                        info = {
                            'name': joint.name,
                            'min': -np.pi * 2, # 默认值
                            'max': np.pi * 2,  # 默认值
                            'default': 0.0
                        }
                        if joint.limit:
                            info['min'] = joint.limit.lower if joint.limit.lower is not None else -np.pi * 2
                            info['max'] = joint.limit.upper if joint.limit.upper is not None else np.pi * 2
                        
                        # 确保 min < max
                        if info['min'] >= info['max']:
                            info['min'] = -np.pi * 2
                            info['max'] = np.pi * 2

                        self.joint_info.append(info)
            else:
                print("DataManager: 警告: 未安装 'urdfpy'。无法加载关节限制信息。")
                print("DataManager: 请运行: pip install urdfpy")

            print(f"DataManager: 已找到 {len(self.joint_info)} 个可动关节。")
            self.hand_joint_info_signal.emit(self.joint_info)
            
            self.status_message_signal.emit(f"已加载机械手: {file_path}")

        except Exception as e:
            print(f"DataManager: 加载机械手失败: {e}")
            import traceback
            traceback.print_exc()
            self.status_message_signal.emit(f"加载机械手失败: {e}")

    @pyqtSlot(bool)
    def set_picking_mode(self, is_active: bool) -> None:
        """
        切换锚点拾取模式。
        """
        self.is_picking_mode = is_active
        self._current_pick_stage = 'hand' # 每次都重置为先选 'hand'
        self._temp_hand_anchor = None
        
        self.picking_mode_changed_signal.emit(is_active)
        
        if is_active:
            self.status_message_signal.emit("拾取模式已激活：请在 [右侧] 视窗点击机械手上的一个点。")
        else:
            self.status_message_signal.emit("拾取模式已关闭。")

    @pyqtSlot(dict)
    def on_hand_point_picked(self, pick_data: dict) -> None:
        """
        槽：当 [右侧] 静态手视窗被点击时调用。
        :param pick_data: {'actor_name': str, 'world_coord': [x, y, z]}
        """
        if not (self.is_picking_mode and self._current_pick_stage == 'hand'):
            return # 状态不正确，忽略
            
        # 1. 存储临时锚点信息
        self._temp_hand_anchor = pick_data
        
        # 2. 转换状态
        self._current_pick_stage = 'object'
        
        print(f"DataManager: 已拾取手部点: {pick_data}")
        self.status_message_signal.emit(f"已选定手部点 ({pick_data['actor_name']})。请在 [左侧] 视窗点击物体上的对应点。")

    @pyqtSlot(dict)
    def on_object_point_picked(self, pick_data: dict) -> None:
        """
        槽：当 [左侧] 物体视窗被点击时调用。
        :param pick_data: {'actor_name': str, 'world_coord': [x, y, z]}
        """
        if not (self.is_picking_mode and self._current_pick_stage == 'object'):
            return # 状态不正确，忽略
            
        # 1. 创建锚点对
        new_pair = {
            'hand_point': self._temp_hand_anchor['world_coord'],
            'hand_link_name': self._temp_hand_anchor['actor_name'], # 这是 'static_hand_link_x'
            'obj_point': pick_data['world_coord']
        }
        
        # 2. 添加到列表
        self.anchor_pairs.append(new_pair)
        print(f"DataManager: 已创建新锚点对: {new_pair}")
        
        # 3. 重置状态机，准备下一次拾取
        self._current_pick_stage = 'hand'
        self._temp_hand_anchor = None
        
        # 4. 发射信号
        self.new_anchor_pair_signal.emit(self.anchor_pairs) # 发给优化器
        self.anchor_list_updated_signal.emit(self.anchor_pairs) # 发给 UI 列表
        
        self.status_message_signal.emit("锚点对已添加。请在 [右侧] 视窗点击下一个手部点。")

    @pyqtSlot(int)
    def on_delete_anchor(self, row_index: int) -> None:
        """
        槽：当 controls_widget 请求删除一个锚点时调用。
        :param row_index: 要删除的行号。
        """
        if 0 <= row_index < len(self.anchor_pairs):
            removed = self.anchor_pairs.pop(row_index)
            print(f"DataManager: 已删除锚点 {row_index}: {removed}")
            
            # 重新发射更新后的列表
            self.new_anchor_pair_signal.emit(self.anchor_pairs)
            self.anchor_list_updated_signal.emit(self.anchor_pairs)
            self.status_message_signal.emit(f"已删除锚点 {row_index}")
        else:
            print(f"DataManager: 警告: 尝试删除无效的锚点索引 {row_index}")

# --- 用于独立测试 ---
if __name__ == '__main__':
    # DataManager 依赖 QApplication 来运行 QFileDialog
    # 我们可以进行有限的单元测试
    
    app = QApplication(sys.argv)
    
    dm = DataManager()
    
    print("测试 DataManager...")
    
    # 1. 测试拾取逻辑
    dm.set_picking_mode(True)
    
    # 模拟手部拾取
    dm.on_hand_point_picked({'actor_name': 'hand_link_1', 'world_coord': [0.1, 0.2, 0.3]})
    
    # 模拟物体拾取
    dm.on_object_point_picked({'actor_name': 'object', 'world_coord': [1.0, 1.1, 1.2]})
    
    assert len(dm.anchor_pairs) == 1
    assert dm.anchor_pairs[0]['hand_link_name'] == 'hand_link_1'
    assert dm.anchor_pairs[0]['obj_point'] == [1.0, 1.1, 1.2]
    assert dm._current_pick_stage == 'hand' # 检查是否重置
    
    print("锚点添加测试通过。")
    
    # 2. 测试删除逻辑
    dm.on_delete_anchor(0)
    assert len(dm.anchor_pairs) == 0
    print("锚点删除测试通过。")
    
    # 3. 测试文件加载 (手动)
    # print("测试加载物体 (将打开对话框)...")
    # dm.load_object()
    # assert dm.object_mesh is not None
    # print("物体加载测试通过。")
    
    # print("测试加载机械手 (将打开对话框)...")
    # dm.load_hand()
    # assert len(dm.hand_links_mesh_dict) > 0
    # print("机械手加载测试通过。")
    
    print("DataManager 单元测试完成 (文件加载需手动验证)。")