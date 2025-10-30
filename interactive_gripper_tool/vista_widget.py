# vista_widget.py

import sys
import numpy as np
import pyvista
from pyvistaqt import QtInteractor
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QApplication, QMainWindow, QLabel
from PyQt6.QtCore import pyqtSignal, Qt

# 确保 PyVista 使用 Qt 后端
pyvista.set_plot_theme("document")  # 使用一个干净的主题
pyvista.global_theme.interactive = True

class VistaWidget(QWidget):
    """
    一个封装了 PyVista QtInteractor 的 QWidget。
    
    这个小部件提供了加载/更新 3D MESH 和处理点拾取的功能。
    
    信号:
        point_picked_signal(dict): 当一个点被拾取时发射。
                                     发出的字典格式为:
                                     {'actor_name': str, 'world_coord': [x, y, z], 'relative_coord': [x, y, z]}
    """
    
    # 定义信号
    point_picked_signal = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        
        # 1. 设置布局
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # 2. 创建 PyVista Plotter (使用 QtInteractor)
        # QtInteractor 本身就是一个 QWidget
        self.plotter = QtInteractor(self)
        self.layout.addWidget(self.plotter.interactor)
        
        # 3. 存储 actors 的字典
        #    键是我们在 'main_window.py' 中设置的 'name' (例如 'object', 'dyn_hand_link_1')
        #    值是 PyVista 的 Actor 对象
        self.actors = {}
        
        # 4. 锚点相关
        self.anchor_actors = []
        self.color_func = None
        self.sphere_radius = 0.005
        
        # 5. 拾取相关
        self._picking_enabled = False
        self._next_anchor_color = (1.0, 0.0, 0.0)  # 默认红色
        
        # 5. 配置渲染器
        self.plotter.set_background("gray", top="white") # 渐变背景
        self.plotter.add_axes() # 添加世界坐标系
        self.plotter.enable_shadows() # 启用阴影
        self.plotter.camera.azimuth = 30
        self.plotter.camera.elevation = 30
        
        # 6. 启用 mesh 拾取
        self.plotter.enable_mesh_picking(
            callback=self._on_click_callback, 
            use_actor=True, 
            left_clicking=True, 
            show=False, 
            show_message=False
        )

    # --- 公共 API (槽 / Public Methods) ---

    def load_mesh(self, mesh_data: pyvista.PolyData, name: str, 
                  color: str = 'white', opacity: float = 1.0, 
                  camera_reset: bool = True, **kwargs) -> None:
        """
        加载一个 PyVista Mesh (PolyData) 到场景中。
        
        :param mesh_data: 要加载的 PyVista PolyData 对象。
        :param name: 赋予该 mesh 的唯一名称 (用于更新和拾取)。
        :param color: MESH 的颜色。
        :param opacity: MESH 的透明度 (1.0 = 不透明)。
        :param camera_reset: 是否在加载后重置相机视角。
        """
        # 如果已存在同名 actor，先移除
        if name in self.actors:
            self.plotter.remove_actor(self.actors[name])
            
        # 添加 mesh 并获取返回的 actor
        actor = self.plotter.add_mesh(
            mesh_data,
            name=name,
            color=color,
            opacity=opacity,
            **kwargs
        )
        
        # 存储 actor
        self.actors[name] = actor
        
        if camera_reset:
            self.plotter.reset_camera()
        print(f"VistaWidget: 已加载 actor '{name}'。")

    def load_hand(self, links_dict_data: dict[str, pyvista.PolyData], 
                  name_prefix: str = "", camera_reset: bool = True, **kwargs) -> None: # [修改] 1. 添加 camera_reset 参数
        """
        加载一个由多个 link 组成的机械手。
        
        :param links_dict_data: 字典 {link_name: mesh_data}
        :param name_prefix: 添加到 actor 名称的前缀 (例如 'dyn_hand_')
        :param camera_reset: [新增] 是否在加载最后一个 link 后重置相机
        :param kwargs: 传递给 load_mesh 的其他参数 (例如 color, opacity)
        """
        print(f"VistaWidget: 正在加载机械手，前缀: '{name_prefix}'...")
        if not links_dict_data:
            print("VistaWidget: 警告: 传入了空的 links_dict_data。")
            return

        # 批量加载，只在最后重置一次相机
        for i, (link_name, mesh_data) in enumerate(links_dict_data.items()):
            actor_name = name_prefix + link_name
            
            # 仅在最后一个link加载时重置相机
            is_last = (i == len(links_dict_data) - 1)
            
            self.load_mesh(
                mesh_data, 
                name=actor_name, 
                camera_reset=(is_last and camera_reset),
                **kwargs
            )
        
        print(f"VistaWidget: 机械手加载完成。")

    def update_actor_pose(self, name: str, pose_matrix: np.ndarray) -> None:
        """
        更新单个 actor (mesh) 的位姿。
        
        :param name: 要更新的 actor 的名称。
        :param pose_matrix: 4x4 的 NumPy 变换矩阵。
        """
        if name not in self.actors:
            # 这是一个高频函数，所以我们只在第一次遇到时打印警告
            if not hasattr(self, '_warned_actors'): self._warned_actors = set()
            if name not in self._warned_actors:
                print(f"VistaWidget: 警告: 尝试更新不存在的 actor '{name}' 的位姿。")
                self._warned_actors.add(name)
            return
            
        actor = self.actors[name]
        
        # PyVista actor 可以直接接受一个 4x4 NumPy 数组作为其 'user_transform'
        actor.user_matrix = pose_matrix

    def update_hand_pose(self, link_poses_dict: dict[str, np.ndarray]) -> None:
        """
        高频更新：更新机械手所有 links 的位姿。
        
        :param link_poses_dict: 字典 {actor_name: pose_matrix}
                               (注意: key 必须是带前缀的完整 actor 名称)
        """
        if not link_poses_dict:
            return
            
        # 批量更新所有 actor 的位姿
        # PyVista/VTK 足够智能，只会在下一个渲染帧中统一重绘
        for actor_name, pose_matrix in link_poses_dict.items():
            self.update_actor_pose(actor_name, pose_matrix)
    
    def enable_picking(self, enable: bool = True) -> None:
        """
        启用或禁用此视窗的点拾取功能。
        """
        print(f"VistaWidget ({self.objectName()}): 拾取模式设置为 {enable}")
        self._picking_enabled = enable
        
        # 可选：添加视觉提示
        if enable:
            self.plotter.add_text("拾取模式已激活", name="_pick_text", position='upper_left')
        else:
            self.plotter.remove_actor("_pick_text")

    def set_next_anchor_color(self, color: tuple) -> None:
        """
        设置下一个锚点对的颜色 (RGB, 0-1范围)
        
        :param color: (r, g, b) 元组
        """
        self._next_anchor_color = color

    def set_actor_properties(self, name: str | None = None, 
                           name_pattern: str | None = None, **kwargs) -> None:
        """
        更改一个或多个 actors 的可视化属性。
        
        :param name: 要修改的 actor 的确切名称。
        :param name_pattern: 要修改的 actors 的名称前缀 (例如 'dyn_hand_')。
        :param kwargs: 要设置的属性 (例如 opacity=0.5, color='red')。
        """
        actors_to_modify = []
        
        if name and name in self.actors:
            actors_to_modify.append(self.actors[name])
        elif name_pattern:
            for actor_name, actor in self.actors.items():
                if actor_name.startswith(name_pattern):
                    actors_to_modify.append(actor)
        else:
            print(f"VistaWidget: 警告: set_actor_properties 未指定 name 或 name_pattern。")
            return

        if not actors_to_modify:
            print(f"VistaWidget: 警告: 未找到匹配 '{name or name_pattern}' 的 actor。")
            return
            
        # 暂时抑制渲染以提高性能
        original_suppress = self.plotter.suppress_rendering
        self.plotter.suppress_rendering = True
        
        # 应用属性
        for actor in actors_to_modify:
            if 'opacity' in kwargs:
                actor.prop.opacity = kwargs['opacity']
            if 'color' in kwargs:
                actor.prop.color = kwargs['color']
            if 'visibility' in kwargs:
                actor.prop.visibility = kwargs['visibility']
        
        # 恢复渲染状态并手动渲染一次
        self.plotter.suppress_rendering = original_suppress
        if not original_suppress:
            self.plotter.render()
                
        print(f"VistaWidget: 已更新 {len(actors_to_modify)} 个 actors 的属性。")

    def update_anchor_spheres(self, anchor_pairs: list, color_func, sphere_radius: float) -> None:
        """
        更新锚点球体显示。
        
        :param anchor_pairs: 锚点对列表，每个包含 'obj_point', 'hand_point' 等
        :param color_func: 函数，输入 pair，返回颜色
        :param sphere_radius: 球体半径
        """
        # 移除旧的锚点球体
        for actors in self.anchor_actors:
            if 'obj_actor' in actors:
                self.plotter.remove_actor(actors['obj_actor'])
            if 'hand_actor' in actors:
                self.plotter.remove_actor(actors['hand_actor'])
        
        self.anchor_actors = []
        self.color_func = color_func
        self.sphere_radius = sphere_radius
        
        for i, pair in enumerate(anchor_pairs):
            obj_point = pair['obj_point']
            hand_point = pair.get('hand_point', pair.get('obj_point', [0, 0, 0]))
            color = color_func(i)
            
            # 创建物体球体（如果位置不是原点）
            obj_actor = None
            if not np.allclose(obj_point, [0, 0, 0], atol=1e-6):
                obj_actor = self.plotter.add_mesh(
                    pyvista.Sphere(radius=sphere_radius, center=(0, 0, 0)),
                    color=color,
                    name=f"anchor_obj_{i}"
                )
                obj_translation = np.eye(4)
                obj_translation[:3, 3] = obj_point
                obj_actor.user_matrix = obj_translation
            
            # 创建手部球体（如果位置不是原点）
            hand_actor = None
            if not np.allclose(hand_point, [0, 0, 0], atol=1e-6):
                hand_actor = self.plotter.add_mesh(
                    pyvista.Sphere(radius=sphere_radius, center=(0, 0, 0)),
                    color=color,
                    name=f"anchor_hand_{i}"
                )
                hand_translation = np.eye(4)
                hand_translation[:3, 3] = hand_point
                hand_actor.user_matrix = hand_translation
            
            self.anchor_actors.append({
                'obj_actor': obj_actor,
                'hand_actor': hand_actor,
                'color': color,
                'pair': pair
            })
        
        print(f"VistaWidget: 已更新 {len(anchor_pairs)} 个锚点对的球体显示。")

    def add_temp_anchor(self, point: list, color_rgb: tuple, is_hand: bool = True) -> None:
        """
        添加单个临时锚点球体，不清除现有的锚点
        
        :param point: 锚点位置 [x, y, z]
        :param color_rgb: 颜色 (r, g, b) 0-1范围
        :param is_hand: True为手部锚点，False为物体锚点
        """
        name_suffix = "hand" if is_hand else "obj"
        actor_name = f"temp_anchor_{name_suffix}"
        
        # 移除现有的临时锚点（如果有）
        if actor_name in [actor.GetObjectName() for actor in self.plotter.actors.values()]:
            self.plotter.remove_actor(actor_name)
        
        # 创建新的临时锚点
        actor = self.plotter.add_mesh(
            pyvista.Sphere(radius=0.008, center=(0, 0, 0)),
            color=color_rgb,
            name=actor_name
        )
        translation = np.eye(4)
        translation[:3, 3] = point
        actor.user_matrix = translation

    def clear_temp_anchors(self) -> None:
        """
        清除所有临时锚点球体
        """
        actors_to_remove = []
        for name, actor in self.plotter.actors.items():
            if name.startswith("temp_anchor_"):
                actors_to_remove.append(name)
        
        for name in actors_to_remove:
            self.plotter.remove_actor(name)

    def update_anchor_positions_fast(self, updated_pairs: list) -> None:
        """
        快速更新锚点位置（同时更新手部点和物体点位置，不重建actors）。
        
        :param updated_pairs: 更新后的锚点对列表
        """
        if not self.color_func or not self.anchor_actors:
            return
        
        for i, pair in enumerate(updated_pairs):
            if i >= len(self.anchor_actors):
                continue
            
            actors = self.anchor_actors[i]
            hand_point = pair['hand_point']
            obj_point = pair['obj_point']
            color = self.color_func(i)
            
            # 更新手部锚点位置
            if actors['hand_actor'] is not None:
                hand_translation = np.eye(4)
                hand_translation[:3, 3] = hand_point
                actors['hand_actor'].user_matrix = hand_translation
            
            # 更新物体锚点位置
            if actors['obj_actor'] is not None:
                obj_translation = np.eye(4)
                obj_translation[:3, 3] = obj_point
                actors['obj_actor'].user_matrix = obj_translation
            
            # 更新颜色如果改变
            if actors['color'] != color:
                if actors['hand_actor'] is not None:
                    actors['hand_actor'].prop.color = color
                if actors['obj_actor'] is not None:
                    actors['obj_actor'].prop.color = color
                actors['color'] = color
            
            actors['pair'] = pair
        

    # --- 内部回调 (Internal Callbacks) ---

    def _on_click_callback(self, picked_actor) -> None:
        """
        当 PyVista 视窗被点击时由 enable_mesh_picking 调用的内部回调。
        """
        # 1. 检查拾取模式是否激活
        if not self._picking_enabled:
            return

        # 2. 获取拾取到的点和 mesh
        picked_point = self.plotter.picked_point
        if picked_point is None:
            print("VistaWidget: 未拾取到点。")
            return
        picked_point = np.array(picked_point)
        picked_mesh = picked_actor.mapper.dataset

        # 3. 反向查找 actor 的名称 (虽然 picked_actor 已知，但为了保持一致)
        actor_name = None
        for name, actor in self.actors.items():
            if actor is picked_actor:
                actor_name = name
                break
                
        if actor_name is None:
            print(f"VistaWidget: 拾取到一个未被管理的 actor。")
            return

        # 4. 计算相对位置 (相对于 actor 的本地坐标系)
        pose_matrix = picked_actor.user_matrix
        if pose_matrix is None:
            pose_matrix = np.eye(4)
        pose_inv = np.linalg.inv(pose_matrix)
        relative_point = pose_inv @ np.append(picked_point, 1.0)
        relative_point = relative_point[:3]

        # 5. 添加一个视觉标记 (使用下一个锚点对的颜色)
        #    使用一个固定的名字，这样下次点击时会自动替换
        self.plotter.add_mesh(
            pyvista.Sphere(radius=0.005, center=picked_point),
            color=self._next_anchor_color,
            name=f"_pick_marker_{self.objectName()}"
        )

        # 6. 准备要发射的数据
        pick_data = {
            'actor_name': actor_name,
            'world_coord': list(picked_point),  # 世界坐标
            'relative_coord': list(relative_point)  # 相对于 link 的本地坐标
        }
        
        print(f"VistaWidget ({self.objectName()}): 成功拾取! 发射信号: {pick_data}")
        
        # 7. 发射信号
        self.point_picked_signal.emit(pick_data)


# --- 用于独立测试 ---
if __name__ == '__main__':
    # 这允许我们直接运行 vista_widget.py 来测试其功能
    
    app = QApplication(sys.argv)
    
    # 创建一个主窗口来容纳我们的 widget
    test_window = QMainWindow()
    test_window.setWindowTitle("VistaWidget - 独立测试")
    test_window.setGeometry(100, 100, 800, 600)
    
    main_widget = QWidget()
    main_layout = QVBoxLayout(main_widget)
    test_window.setCentralWidget(main_widget)
    
    # 创建一个标签和 VistaWidget 实例
    label = QLabel("这是一个 VistaWidget 的独立测试。点击立方体。")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    
    vista_widget_instance = VistaWidget()
    # 在 main_window 中我们会设置 objectName
    vista_widget_instance.setObjectName("TestWidget") 
    
    main_layout.addWidget(label)
    main_layout.addWidget(vista_widget_instance)
    
    # 1. 加载一个测试用的立方体
    cube = pyvista.Cube()
    vista_widget_instance.load_mesh(cube, name="test_cube", color="cyan")
    
    # 2. 启用拾取
    vista_widget_instance.enable_picking(True)
    
    # 3. 连接信号到一个简单的槽函数
    def on_pick(data):
        print(f"--- [测试] 信号已接收 ---")
        print(f"Actor: {data['actor_name']}")
        print(f"世界坐标: {data['world_coord']}")
        print(f"相对坐标: {data['relative_coord']}")
        label.setText(f"拾取到: {data['actor_name']} @ 世界: {np.array(data['world_coord']).round(2)}, 相对: {np.array(data['relative_coord']).round(2)}")
        
    vista_widget_instance.point_picked_signal.connect(on_pick)
    
    # 4. 测试 'set_actor_properties'
    # vista_widget_instance.set_actor_properties(name="test_cube", opacity=0.5)
    
    # 5. 测试 'load_hand' (模拟)
    # hand_data = {
    #     "link1": pyvista.Sphere(center=(0, 0, 0.5)),
    #     "link2": pyvista.Sphere(center=(0, 0, 1.0))
    # }
    # vista_widget_instance.load_hand(hand_data, name_prefix="my_hand_")
    
    test_window.show()
    sys.exit(app.exec())