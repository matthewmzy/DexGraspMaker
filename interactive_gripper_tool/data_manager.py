# data_manager.py

import sys
import os
import json
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QUrl
from PyQt6.QtWidgets import QFileDialog

# -------------------------------------------------------------------
# [依赖项导入]
# -------------------------------------------------------------------

import pyvista
import trimesh
import yourdfpy
import pyroki as pk

# 导入球体拟合模块
from sphere_fitting import generate_link_spheres, save_link_spheres, load_link_spheres

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
    
    # 信号 4.5: 锚点对准备状态改变（控制确定按钮显示）
    anchor_pair_ready_signal = pyqtSignal(bool)
    
    # 信号 4.6: 清空临时锚点球体
    clear_temp_anchors_signal = pyqtSignal()
    
    # 信号 4.7: 显示临时锚点
    show_temp_anchor_signal = pyqtSignal(dict)  # {'type': 'hand'/'object', 'point': [x,y,z], 'link_name': str}
    
    # 信号 5: （可选）状态栏消息
    status_message_signal = pyqtSignal(str) # (用于提示用户)

    # 信号 6: 将 JAX-native 的 pyroki robot 对象发送给优化线程
    pyroki_robot_loaded_signal = pyqtSignal(object) # 发射 pk.Robot 实例

    # 信号 7: 关键点加载成功
    hand_keypoints_loaded_signal = pyqtSignal(dict) # 发射 {link_name: points_array} 字典

    # 信号 8: link球体加载成功
    hand_link_spheres_loaded_signal = pyqtSignal(dict) # 发射 {link_name: [(center, radius), ...]} 字典

    # 信号 9: 加载 URDF 成功后，将 link 的初始姿态字典发射出去
    # {link_name: np.ndarray(4,4)}
    hand_initial_pose_signal = pyqtSignal(dict)

    # 信号 8: 锚点调整相关
    adjust_hand_anchor_signal = pyqtSignal(int)  # 开始调整手上锚点 (anchor_index)
    adjust_object_anchor_signal = pyqtSignal(int)  # 开始调整物体锚点 (anchor_index)
    
    # 信号 9: 手姿态导入/导出
    import_hand_pose_signal = pyqtSignal()
    export_hand_pose_signal = pyqtSignal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        
        # --- 内部状态 ---
        
        # 1. 加载的数据
        self.object_mesh: pyvista.PolyData | None = None
        self.hand_links_mesh_dict: dict[str, pyvista.PolyData] = {}
        self.joint_info: list[dict] = []

        # 2. 存储 pyroki robot 实例
        self.pyroki_robot: pk.Robot | None = None
        
        # 3. 锚点状态
        self.anchor_pairs: list[dict] = []

        # 4. 拾取状态机
        self.is_picking_mode: bool = False
        self._current_pick_stage: str = 'hand' # 'hand' 或 'object'
        self._temp_hand_anchor: dict | None = None # 存储临时的手部点信息
        self._temp_object_anchor: dict | None = None # 存储临时的物体点信息

        # 5. 当前link姿态 (用于锚点调整)
        self.current_link_poses: dict[str, np.ndarray] = {}  # {link_name: 4x4 transform matrix}
        self.current_base_translation: list[float] | None = None
        self.current_base_rotation: list[list[float]] | None = None

    # --- 公共槽 (Public Slots) ---
    # 这些槽由 main_window 连接到 controls_widget 的信号

    def update_current_link_poses(self, link_poses: dict[str, np.ndarray]) -> None:
        """
        更新当前link姿态（由main_window调用）
        
        :param link_poses: {link_name: 4x4 transform matrix}
        """
        self.current_link_poses = link_poses

    def update_base_pose(self, translation: list[float], rotation_matrix: list[list[float]]) -> None:
        """记录最新的基座位姿"""
        self.current_base_translation = translation
        self.current_base_rotation = rotation_matrix

    def load_object_from_file(self, file_path: str) -> bool:
        """
        从指定路径加载物体mesh（无对话框）
        
        :param file_path: 物体文件路径
        :return: 加载成功返回True
        """
        try:
            self.object_mesh = pyvista.read(file_path)
            print(f"DataManager: 已加载物体: {file_path}")
            # 发射信号，将 mesh 数据传递给视窗
            if max(self.object_mesh.bounds_size) > 10: # 简单检查单位
                print("DataManager: 物体尺寸较大，正在缩放至米级单位...")
                self.object_mesh.scale(0.001, inplace=True)
            self.object_loaded_signal.emit(self.object_mesh)
            self.status_message_signal.emit(f"已加载物体: {file_path}")
            return True
        except Exception as e:
            print(f"DataManager: 加载物体失败: {e}")
            self.status_message_signal.emit(f"加载物体失败: {e}")
            return False

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
            if max(self.object_mesh.bounds_size) > 10: # 简单检查单位
                print("DataManager: 物体尺寸较大，正在缩放至米级单位...")
                self.object_mesh.scale(0.001, inplace=True)
            self.object_loaded_signal.emit(self.object_mesh)
            self.status_message_signal.emit(f"已加载物体: {file_path}")
        except Exception as e:
            print(f"DataManager: 加载物体失败: {e}")
            self.status_message_signal.emit(f"加载物体失败: {e}")

    @pyqtSlot()
    def load_hand(self) -> None:
        """
        打开文件对话框以加载机械手 URDF。
        """
        file_path, _ = QFileDialog.getOpenFileName(
            None, "加载机械手 URDF", "", "URDF 文件 (*.urdf)"
        )
        
        if not file_path:
            return # 用户取消
        
        self.load_hand_from_file(file_path)
    
    def load_hand_from_file(self, file_path: str) -> bool:
        """
        从指定路径加载机械手URDF（无对话框）
        
        1. 使用 yourdfpy 加载 URDF (它会处理 'package://' 并加载 meshes)。
        2. 使用 pyroki 创建 JAX-native 的 Robot 对象。
        3. 提取可视化 meshes 发送给 PyVista。
        4. 提取关节信息发送给 ControlsWidget。
        5. 发送 pyroki.Robot 对象给 OptimizationThread。
        
        :param file_path: URDF文件路径
        :return: 加载成功返回True
        """
        
        if not file_path:
            return False
            
        try:
            # 1. 使用 yourdfpy 加载 URDF
            # yourdfpy 会自动处理 package:// 路径并加载 trimesh 场景
            print(f"DataManager: 正在使用 yourdfpy 加载: {file_path}")
            self.urdf_obj = yourdfpy.URDF.load(file_path)
            
            # 2. 使用 pyroki 创建 JAX-native Robot
            print("DataManager: 正在创建 pyroki.Robot 实例...")
            self.pyroki_robot = pk.Robot.from_urdf(self.urdf_obj)
            print("DataManager: pyroki.Robot 创建成功。")

            # 3. 提取 Link Meshes (用于 PyVista 可视化)
            trimesh_scene = self.urdf_obj.scene
            if trimesh_scene is None:
                raise ValueError("yourdfpy 未能加载场景 (urdf_obj.scene 为空)。")

            self.hand_links_mesh_dict.clear()
            
            scene_graph = trimesh_scene.graph
            transform_graph = scene_graph.transforms
            all_node_data = scene_graph.transforms.node_data

            geometry_dict = trimesh_scene.geometry

            # debug_tm_scene = trimesh.Scene()

            for link_name in self.pyroki_robot.links.names:
                if link_name not in transform_graph.nodes:
                    print(f"DataManager: 警告: Pyroki link '{link_name}' 不在 trimesh 场景图中。")
                    continue

                link_meshes = []
                # 查找作为此 link_name 子节点的所有 "visual" 节点
                # (link_name) -> (visual_name)
                child_nodes = [to_node for from_node, to_node in transform_graph.edge_data if from_node == link_name]
                for child_node_name in child_nodes:
                    # 从 all_node_data (而不是 graph) 访问节点数据
                    if child_node_name in all_node_data and "geometry" in all_node_data[child_node_name]:
                        
                        geom_key = all_node_data[child_node_name]["geometry"]
                        if geom_key not in geometry_dict:
                            print(f"DataManager: 警告: 找不到 '{geom_key}' (来自 {child_node_name}) 的几何体。")
                            continue
                            
                        trimesh_geom = geometry_dict[geom_key]
                        
                        transform_matrix = scene_graph.get(child_node_name, link_name)[0]

                        if hasattr(trimesh_geom, 'to_mesh'):
                            # 如果是 Box, Sphere, Cylinder，调用 .to_mesh()
                            trimesh_mesh = trimesh_geom.to_mesh()
                        else:
                            trimesh_mesh = trimesh_geom.copy()

                        trimesh_mesh.apply_transform(transform_matrix)
                        link_meshes.append(trimesh_mesh)

                if not link_meshes:
                    continue # 此 link 没有可视化 meshes
                
                # 将此 link 的所有 visual meshes 合并为一个
                if len(link_meshes) > 1:
                    combined_mesh = trimesh.util.concatenate(link_meshes)
                else:
                    combined_mesh = link_meshes[0]

                # debug_tm_scene.add_geometry(combined_mesh)

                # 转换为 PyVista 并存储
                pv_mesh = pyvista.wrap(combined_mesh)
                self.hand_links_mesh_dict[link_name] = pv_mesh

            # debug_tm_scene.show()

            if not self.hand_links_mesh_dict:
                raise ValueError("未能在 URDF 场景中提取任何 link meshes。")

            print(f"DataManager: 已提取 {len(self.hand_links_mesh_dict)} 个 link meshes。")
            self.hand_loaded_signal.emit(self.hand_links_mesh_dict)

            # 4. 提取 Joint 信息 (用于 ControlsWidget 滑块)
            self.joint_info.clear()
            pyroki_joints = self.pyroki_robot.joints
            
            # pyroki.joints.lower_limits/upper_limits 已经是被驱动关节的列表
            # 我们需要从 yourdfpy 中获取它们的名字以保持一致
            actuated_names = self.urdf_obj.actuated_joint_names

            if len(actuated_names) != pyroki_joints.num_actuated_joints:
                print(f"DataManager: 警告: 'yourdfpy' ({len(actuated_names)} joints) 和 'pyroki' ({pyroki_joints.num_actuated_joints} joints) 的驱动关节数量不匹配。")
                # 这种情况不应该发生，但作为后备
                if pyroki_joints.num_actuated_joints > 0:
                     actuated_names = [f"joint_{i}" for i in range(pyroki_joints.num_actuated_joints)]
                else:
                    raise ValueError("未找到驱动关节。")

            for i, joint_name in enumerate(actuated_names):
                lower = float(pyroki_joints.lower_limits[i])
                upper = float(pyroki_joints.upper_limits[i])
                info = {
                    'name': joint_name,
                    'min': lower,
                    'max': upper,
                    'default': (lower + upper) / 2.0
                }
                self.joint_info.append(info)

            print(f"DataManager: 已提取 {len(self.joint_info)} 个可动关节信息。")
            self.hand_joint_info_signal.emit(self.joint_info)
            
            # 5. 发射 JAX-native robot
            self.pyroki_robot_loaded_signal.emit(self.pyroki_robot)

            # 6. 计算并
            print("DataManager: 正在计算初始姿态 (Default FK)...")
            initial_poses = {}
            base_link_name = self.urdf_obj.base_link
            scene_graph = self.urdf_obj.scene.graph

            # 我们需要为 hand_links_mesh_dict 中的每个 link 计算其全局姿态
            # 注意：urdf_obj.scene.graph 已经包含了默认姿态的变换
            for link_name in self.hand_links_mesh_dict.keys():
                try:
                    # 获取 T_world_link (其中 'world' 是 base_link)
                    transform_matrix = scene_graph.get(link_name, base_link_name)[0]
                    initial_poses[link_name] = transform_matrix
                except:
                    # 这可能发生在 link_name == base_link_name 时
                    if link_name == base_link_name:
                        initial_poses[link_name] = np.eye(4)
                    else:
                        print(f"DataManager: 警告: 无法获取 '{link_name}' 相对于 '{base_link_name}' 的变换。")
            
            print(f"DataManager: 已计算 {len(initial_poses)} 个 links 的初始姿态。")
            # 发射初始姿态
            self.hand_initial_pose_signal.emit(initial_poses)

            # 7. 生成或加载关键点
            self._generate_or_load_keypoints(file_path)

            # 8. 生成或加载link球体
            self._generate_or_load_link_spheres(file_path)
            
            self.status_message_signal.emit(f"已加载机械手: {file_path}")
            return True

        except Exception as e:
            print(f"DataManager: 加载机械手失败: {e}")
            import traceback
            traceback.print_exc()
            self.status_message_signal.emit(f"加载机械手失败: {e}")
            return False

    @pyqtSlot(bool)
    def set_picking_mode(self, is_active: bool) -> None:
        """
        切换锚点拾取模式
        
        注意：新设计中，每次添加锚点对后不会自动关闭拾取模式，
        用户可以连续添加多个锚点对
        """
        self.is_picking_mode = is_active
        self._temp_hand_anchor = None
        self._temp_object_anchor = None
        
        self.picking_mode_changed_signal.emit(is_active)
        self.anchor_pair_ready_signal.emit(False)  # 隐藏确定按钮
        
        if is_active:
            self.status_message_signal.emit("拾取模式已激活：请在任意视窗点击选择点（手或物体均可）。")
        else:
            self.status_message_signal.emit("拾取模式已关闭。")

    @pyqtSlot(dict)
    def on_hand_point_picked(self, pick_data: dict) -> None:
        """
        当在右侧视窗（手部）点击时调用
        """
        if not self.is_picking_mode:
            return
        
        actor_name = pick_data['actor_name']
        
        # main_window 中设置了前缀 "static_hand_"
        prefix = "static_hand_" 
        if actor_name.startswith(prefix):
             link_name = actor_name[len(prefix):]
        else:
             print(f"DataManager: 警告: 拾取到的 actor '{actor_name}' 没有预期的 '{prefix}' 前缀。")
             link_name = actor_name

        # 检查此 link_name 是否在我们的 mesh 字典中
        if link_name not in self.hand_links_mesh_dict:
             print(f"DataManager: 错误: 拾取到的 link '{link_name}' 不在已加载的 meshes 列表中。")
             self.status_message_signal.emit(f"错误: 拾取到未知 link '{link_name}'。")
             return

        self._temp_hand_anchor = {
            'world_coord': pick_data['world_coord'],
            'local_coord': pick_data.get('relative_coord', pick_data['world_coord']),
            'link_name': link_name
        }
        print(f"DataManager: 已拾取手部点: link={link_name}, 局部坐标={self._temp_hand_anchor['local_coord']}")
        
        # 显示临时锚点
        self.show_temp_anchor_signal.emit({
            'type': 'hand',
            'point': self._temp_hand_anchor['world_coord'],
            'link_name': link_name
        })
        
        # 检查是否可以显示确定按钮
        self._check_anchor_pair_ready()

    @pyqtSlot(dict)
    def on_object_point_picked(self, pick_data: dict) -> None:
        """
        当在左侧视窗（物体）点击时调用
        
        改进：只存储物体点，等待用户确认后再创建锚点对
        """
        if not self.is_picking_mode:
            return
            
        # 1. 存储物体点
        self._temp_object_anchor = {
            'world_coord': pick_data['world_coord'],
            'local_coord': pick_data.get('relative_coord', pick_data['world_coord'])
        }
        
        print(f"DataManager: 已拾取物体点: 世界坐标={self._temp_object_anchor['world_coord']}")
        
        # 显示临时锚点
        self.show_temp_anchor_signal.emit({
            'type': 'object',
            'point': self._temp_object_anchor['world_coord']
        })
        
        # 2. 检查是否可以显示确定按钮
        self._check_anchor_pair_ready()
    
    def _check_anchor_pair_ready(self) -> None:
        """
        检查锚点对是否准备好，更新UI状态
        """
        if self._temp_hand_anchor is not None and self._temp_object_anchor is not None:
            self.anchor_pair_ready_signal.emit(True)
            self.status_message_signal.emit("✓ 两个点都已选定！点击「确定」按钮添加锚点对，或继续选择其他点覆盖当前选择。")
        elif self._temp_hand_anchor is not None:
            self.status_message_signal.emit(f"✓ 手部点已选定 ({self._temp_hand_anchor['link_name']})。请在任意视窗选择物体对应点。")
        elif self._temp_object_anchor is not None:
            self.status_message_signal.emit("✓ 物体点已选定。请在任意视窗选择手部对应点。")
        else:
            self.status_message_signal.emit("请在任意视窗点击选择第一个点（手或物体均可）。")

    @pyqtSlot()
    def confirm_anchor_pair(self) -> None:
        """
        确认添加锚点对
        """
        if not self.is_picking_mode or self._temp_hand_anchor is None or self._temp_object_anchor is None:
            return
            
        # 1. 创建锚点对
        new_pair = {
            'hand_point': self._temp_hand_anchor['world_coord'],
            'hand_point_local': self._temp_hand_anchor['local_coord'],
            'hand_link_name': self._temp_hand_anchor['link_name'],
            'obj_point': self._temp_object_anchor['world_coord'],
            'obj_point_local': self._temp_object_anchor['local_coord'],
            'enabled': True
        }
        
        # 2. 添加到列表
        self.anchor_pairs.append(new_pair)
        print(f"DataManager: 已创建新锚点对 #{len(self.anchor_pairs)}: {new_pair}")
        
        # 3. 重置状态机并退出拾取模式
        self._temp_hand_anchor = None
        self._temp_object_anchor = None
        self.is_picking_mode = False
        
        # 4. 隐藏确定按钮
        self.anchor_pair_ready_signal.emit(False)
        
        # 5. 清除临时锚点显示
        self.clear_temp_anchors_signal.emit()
        
        # 6. 发射信号触发优化
        self.new_anchor_pair_signal.emit(self.anchor_pairs)
        self.anchor_list_updated_signal.emit(self.anchor_pairs)
        self.picking_mode_changed_signal.emit(False)
        
        # 6. 提示用户
        self.status_message_signal.emit(f"✓ 锚点对 #{len(self.anchor_pairs)} 已添加！优化已开始。")

    @pyqtSlot()
    def cancel_anchor_adding(self) -> None:
        """
        取消添加锚点对
        """
        if not self.is_picking_mode:
            return
            
        # 1. 清空临时状态
        self._temp_hand_anchor = None
        self._temp_object_anchor = None
        self.is_picking_mode = False
        
        # 2. 隐藏确定按钮
        self.anchor_pair_ready_signal.emit(False)
        
        # 3. 发射信号关闭拾取模式
        self.picking_mode_changed_signal.emit(False)
        
        # 4. 清空所有视窗的锚点球体
        self.clear_temp_anchors_signal.emit()
        
        self.status_message_signal.emit("已取消添加锚点对。")

    @pyqtSlot(int)
    def on_delete_anchor(self, row_index: int) -> None:
        """
        删除指定的锚点对，并立即触发优化更新
        """
        if 0 <= row_index < len(self.anchor_pairs):
            removed = self.anchor_pairs.pop(row_index)
            print(f"DataManager: 已删除锚点 #{row_index+1}: {removed}")
            
            # 立即触发优化更新
            self.new_anchor_pair_signal.emit(self.anchor_pairs)
            self.anchor_list_updated_signal.emit(self.anchor_pairs)
            
            self.status_message_signal.emit(f"✓ 锚点对 #{row_index+1} 已删除，优化已更新。")
        else:
            print(f"DataManager: 删除失败，索引 {row_index} 超出范围")
    
    @pyqtSlot(int, bool)
    def on_toggle_anchor(self, row_index: int, enabled: bool) -> None:
        """
        启用/禁用指定的锚点对，并立即触发优化更新
        
        :param row_index: 锚点对索引
        :param enabled: True=启用, False=禁用
        """
        if 0 <= row_index < len(self.anchor_pairs):
            self.anchor_pairs[row_index]['enabled'] = enabled
            status = "启用" if enabled else "禁用"
            print(f"DataManager: 已{status}锚点对 #{row_index+1}")
            
            # 立即触发优化更新
            self.new_anchor_pair_signal.emit(self.anchor_pairs)
            self.anchor_list_updated_signal.emit(self.anchor_pairs)
            
            self.status_message_signal.emit(f"✓ 锚点对 #{row_index+1} 已{status}，优化已更新。")

    @pyqtSlot(int)
    def on_adjust_hand_anchor(self, anchor_index: int) -> None:
        """
        开始调整手上锚点位置
        
        :param anchor_index: 锚点对索引
        """
        if 0 <= anchor_index < len(self.anchor_pairs):
            # 发射信号给main_window，让其启动键盘控制器
            self.adjust_hand_anchor_signal.emit(anchor_index)

    @pyqtSlot(int)
    def on_adjust_object_anchor(self, anchor_index: int) -> None:
        """
        开始调整物体锚点位置
        
        :param anchor_index: 锚点对索引
        """
        if 0 <= anchor_index < len(self.anchor_pairs):
            # 发射信号给main_window，让其启动键盘控制器
            self.adjust_object_anchor_signal.emit(anchor_index)

    @pyqtSlot(int, str, list)
    def on_update_anchor_position(self, anchor_index: int, point_type: str, position: list) -> None:
        """
        更新锚点位置
        
        :param anchor_index: 锚点对索引
        :param point_type: 点类型 ("hand" 或 "object")
        :param position: 新位置 [x, y, z] (对于hand是局部坐标，对于object是世界坐标)
        """
        if 0 <= anchor_index < len(self.anchor_pairs):
            if point_type == "hand":
                # 手部锚点：直接更新局部坐标，世界坐标会在下一帧pose更新时自动计算
                self.anchor_pairs[anchor_index]['hand_point_local'] = position
                
                # 立即计算并更新世界坐标（用于一致性）
                link_name = self.anchor_pairs[anchor_index]['hand_link_name']
                if link_name in self.current_link_poses:
                    T_world_link = self.current_link_poses[link_name]
                    local_pos_homogeneous = np.append(position, 1.0)
                    world_pos = (T_world_link @ local_pos_homogeneous)[:3]
                    self.anchor_pairs[anchor_index]['hand_point'] = world_pos.tolist()
                    
            elif point_type == "object":
                # 物体锚点：直接使用世界坐标
                self.anchor_pairs[anchor_index]['obj_point'] = position
                self.anchor_pairs[anchor_index]['obj_point_local'] = position
            
            print(f"DataManager: 已更新锚点对 #{anchor_index+1} 的 {point_type} 点位置: {position}")
            
            # 立即触发优化更新（只发给优化线程，不重建UI列表）
            self.new_anchor_pair_signal.emit(self.anchor_pairs)
            # 注意：不发射 anchor_list_updated_signal 以避免重建UI列表和重置按钮状态
            
            self.status_message_signal.emit(f"✓ 锚点对 #{anchor_index+1} 的 {point_type} 点已更新，优化已重新开始。")

    def import_hand_pose(self) -> None:
        """
        导入手姿态（关节角度）
        """
        from PyQt6.QtWidgets import QFileDialog
        file_path, _ = QFileDialog.getOpenFileName(
            None, "导入手姿态", "", "JSON files (*.json);;All files (*)"
        )
        
        if file_path:
            try:
                import json
                with open(file_path, 'r') as f:
                    pose_data = json.load(f)
                
                # 验证数据格式
                if not isinstance(pose_data, dict) or 'joint_values' not in pose_data:
                    self.status_message_signal.emit("❌ 姿态文件格式错误：缺少 'joint_values' 字段")
                    return
                
                joint_values = pose_data['joint_values']
                if not isinstance(joint_values, dict):
                    self.status_message_signal.emit("❌ 姿态文件格式错误：'joint_values' 应为字典")
                    return
                
                # 更新关节值
                for joint_name, value in joint_values.items():
                    if joint_name in self.optimization_thread.target_joint_values:
                        self.optimization_thread.set_manual_joint(joint_name, float(value))
                
                self.status_message_signal.emit(f"✓ 已导入手姿态：{len(joint_values)} 个关节")
                
            except Exception as e:
                self.status_message_signal.emit(f"❌ 导入姿态失败：{str(e)}")

    def export_hand_pose(self) -> None:
        """
        导出当前手姿态（关节角度）
        """
        from PyQt6.QtWidgets import QFileDialog
        
        # 获取当前关节值
        from datetime import datetime
        current_pose = {
            'joint_values': self.optimization_thread.current_joint_values.copy(),
            'timestamp': datetime.now().isoformat()
        }
        
        file_path, _ = QFileDialog.getSaveFileName(
            None, "导出手姿态", "hand_pose.json", "JSON files (*.json);;All files (*)"
        )
        
        if file_path:
            try:
                import json
                with open(file_path, 'w') as f:
                    json.dump(current_pose, f, indent=2)
                
                self.status_message_signal.emit(f"✓ 已导出手姿态到：{file_path}")
                
            except Exception as e:
                self.status_message_signal.emit(f"❌ 导出姿态失败：{str(e)}")

    def get_current_hand_pose(self) -> tuple[list, list]:
        """
        获取当前手的位置和旋转矩阵
        
        :return: (translation, rotation_matrix) 元组
        """
        if self.current_base_translation is not None and self.current_base_rotation is not None:
            return (
                list(self.current_base_translation),
                [row[:] for row in self.current_base_rotation]
            )

        # 获取手掌的位姿（假设有一个名为 'palm' 或类似的基础link）
        palm_link_name = None
        for link_name in self.current_link_poses.keys():
            if 'palm' in link_name.lower() or 'wrist' in link_name.lower():
                palm_link_name = link_name
                break
        
        if palm_link_name and palm_link_name in self.current_link_poses:
            pose_matrix = self.current_link_poses[palm_link_name]
            translation = pose_matrix[:3, 3].tolist()
            rotation_matrix = pose_matrix[:3, :3].tolist()
            return translation, rotation_matrix
        
        # 如果找不到手掌link，返回默认值
        return [0.0, 0.0, 0.0], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    def _generate_or_load_keypoints(self, urdf_path: str) -> None:
        """
        生成或加载手模型的关键点文件
        
        基于体积的层级采样策略：
        - 体积 < 10 cm³: 6 个点
        - 10-100 cm³: 8 个点  
        - 100-1000 cm³: 16 个点
        - ≥1000 cm³: 32 个点
        
        :param urdf_path: URDF文件路径，用于生成关键点文件名
        """
        try:
            # 生成关键点文件名 (与URDF同名，但扩展名为.json)
            base_name = os.path.splitext(os.path.basename(urdf_path))[0]
            keypoints_file = os.path.join(os.path.dirname(urdf_path), f"{base_name}_keypoints.json")
            
            # 检查关键点文件是否已存在
            if os.path.exists(keypoints_file):
                print(f"DataManager: 找到现有关键点文件: {keypoints_file}")
                with open(keypoints_file, 'r') as f:
                    keypoints_data = json.load(f)
                
                # 转换为 numpy 数组格式
                hand_keypoints = {}
                for link_name, points_list in keypoints_data.items():
                    hand_keypoints[link_name] = np.array(points_list)
                
                print(f"DataManager: 已加载 {sum(len(points) for points in hand_keypoints.values())} 个关键点")
            else:
                print(f"DataManager: 未找到关键点文件，正在生成: {keypoints_file}")
                hand_keypoints = self._generate_keypoints()
                
                # 保存到文件
                keypoints_data = {link_name: points.tolist() for link_name, points in hand_keypoints.items()}
                with open(keypoints_file, 'w') as f:
                    json.dump(keypoints_data, f, indent=2)
                
                print(f"DataManager: 已生成并保存 {sum(len(points) for points in hand_keypoints.values())} 个关键点到 {keypoints_file}")
            
            # 发射关键点加载成功信号
            self.hand_keypoints_loaded_signal.emit(hand_keypoints)
            
        except Exception as e:
            print(f"DataManager: 关键点生成/加载失败: {e}")
            import traceback
            traceback.print_exc()
            self.status_message_signal.emit(f"关键点生成失败: {e}")

    def _generate_keypoints(self) -> dict[str, np.ndarray]:
        """
        使用FPS算法基于体积生成关键点
        
        :return: {link_name: points_array} 字典
        """
        hand_keypoints = {}
        
        for link_name, pv_mesh in self.hand_links_mesh_dict.items():
            # 计算网格体积 (cm³)
            try:
                # PyVista mesh 体积计算
                volume = pv_mesh.volume * 1e6  # 转换为 cm³ (PyVista使用米)
                if volume <= 0:
                    # 如果体积计算失败，使用边界框体积作为近似
                    bounds = pv_mesh.bounds
                    volume = (bounds[1] - bounds[0]) * (bounds[3] - bounds[2]) * (bounds[5] - bounds[4]) * 1e6
                
                # 基于体积确定采样点数
                if volume < 10:
                    num_points = 6
                elif volume < 100:
                    num_points = 8
                elif volume < 1000:
                    num_points = 16
                else:
                    num_points = 32
                
                # 使用FPS算法采样关键点
                points = self._farthest_point_sampling(pv_mesh, num_points)
                hand_keypoints[link_name] = points
                
                print(f"DataManager: {link_name} (体积: {volume:.1f} cm³) -> {num_points} 个关键点")
                
            except Exception as e:
                print(f"DataManager: 为 {link_name} 生成关键点失败: {e}")
                # 使用边界框中心作为后备
                bounds = pv_mesh.bounds
                center = np.array([
                    (bounds[0] + bounds[1]) / 2,
                    (bounds[2] + bounds[3]) / 2,
                    (bounds[4] + bounds[5]) / 2
                ])
                hand_keypoints[link_name] = center.reshape(1, -1)
                print(f"DataManager: 为 {link_name} 使用边界框中心作为关键点")
        
        return hand_keypoints

    def _farthest_point_sampling(self, pv_mesh: pyvista.PolyData, num_points: int) -> np.ndarray:
        """
        使用最远点采样(FPS)算法从网格采样关键点
        
        :param pv_mesh: PyVista网格
        :param num_points: 采样点数
        :return: 采样点数组 (N, 3)
        """
        # 获取网格顶点
        vertices = pv_mesh.points
        
        if len(vertices) == 0:
            raise ValueError("网格没有顶点")
        
        if len(vertices) <= num_points:
            return vertices
        
        # 初始化：选择第一个点（随机或几何中心）
        selected_indices = [0]  # 从第一个顶点开始
        min_distances = np.full(len(vertices), np.inf)
        
        for _ in range(1, num_points):
            # 计算所有点到已选点的最近距离
            for i, selected_idx in enumerate(selected_indices):
                distances = np.linalg.norm(vertices - vertices[selected_idx], axis=1)
                min_distances = np.minimum(min_distances, distances)
            
            # 选择距离最远的点
            farthest_idx = np.argmax(min_distances)
            selected_indices.append(farthest_idx)
            
            # 重置该点的距离为0
            min_distances[farthest_idx] = 0
        
        return vertices[selected_indices]

    def _generate_or_load_link_spheres(self, urdf_path: str) -> None:
        """
        生成或加载手模型的link球体数据

        基于体积的层级策略：
        - 体积 < 10 cm³: 2 个球体
        - 10-100 cm³: 3 个球体
        - 100-1000 cm³: 4 个球体
        - ≥1000 cm³: 6 个球体

        :param urdf_path: URDF文件路径，用于生成球体文件名
        """
        try:
            # 生成球体文件名 (与URDF同名，但扩展名为_spheres.json)
            base_name = os.path.splitext(os.path.basename(urdf_path))[0]
            spheres_file = os.path.join(os.path.dirname(urdf_path), f"{base_name}_spheres.json")

            # 检查球体文件是否已存在
            if os.path.exists(spheres_file):
                print(f"DataManager: 找到现有球体文件: {spheres_file}")
                link_spheres = load_link_spheres(spheres_file)
            else:
                print(f"DataManager: 未找到球体文件，正在生成: {spheres_file}")
                link_spheres = generate_link_spheres(self.hand_links_mesh_dict, method='bounding_box_packing', urdf_obj=self.urdf_obj)

                # 保存到文件
                save_link_spheres(link_spheres, spheres_file)
                print(f"DataManager: 已生成并保存球体数据到 {spheres_file}")

            total_spheres = sum(len(spheres) for spheres in link_spheres.values())
            print(f"DataManager: 已加载 {total_spheres} 个球体用于 {len(link_spheres)} 个links")

            # 发射球体加载成功信号
            self.hand_link_spheres_loaded_signal.emit(link_spheres)

        except Exception as e:
            print(f"DataManager: 球体生成/加载失败: {e}")
            import traceback
            traceback.print_exc()
            self.status_message_signal.emit(f"球体生成失败: {e}")

# --- 用于独立测试 ---
if __name__ == '__main__':
    # DataManager 依赖 QApplication 来运行 QFileDialog
    # 我们可以进行有限的单元测试
    
    from PyQt6.QtWidgets import QApplication
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