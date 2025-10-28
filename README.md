# DexGraspMaker

项目文件结构（前端部分）

```text
interactive_gripper_tool/
├── main.py                 # 应用程序入口
├── main_window.py          # 主窗口（负责整体布局）
├── vista_widget.py         # 可重用的 PyVista 3D 视窗组件
├── controls_widget.py      # 底部/侧边的工具栏和设置面板
├── data_manager.py         # 核心：状态管理和信号中枢
├── optimization_thread.py  # 核心：在独立线程运行优化（JAX/PyTorch）
├── icons/                  # 存放按钮图标（例如：load.png）
└── resources/              # 存放默认的 mesh / urdf（可选）
```

下面是各个文件/模块的说明（用于前端/UI 结构）：

## 1. `main.py`（应用程序入口）

- 目的：启动整个 PyQt 应用程序。

- 功能：
    - 导入 `QApplication` 和 `MainWindow`。
    - 创建 `QApplication` 实例。
    - 创建 `MainWindow` 实例并显示（`window.show()`）。
    - 启动应用循环（`app.exec()`）。

## 2. `main_window.py`（主窗口）

- 目的：搭建“左-中-右 + 工具栏”的整体布局。

- 类：`MainWindow(QMainWindow)`。

- 功能：
    - 布局：使用 `QSplitter` 实现可拖拽调整大小的子窗口：
        - 一个水平 `QSplitter` 容纳 [Left | Center | Right] 视图。
        - 一个垂直 `QSplitter` 将上述水平分割器放在上面，将 `ControlsWidget` 放在下面。
    - 初始化组件：创建三个 `VistaWidget`（左/中/右）、`ControlsWidget`、`DataManager` 和 `OptimizationThread`。
    - 信号与槽（Signal & Slot）连接示例：
        - `controls.load_object_button.clicked.connect(data_manager.load_object)`
        - `controls.load_hand_button.clicked.connect(data_manager.load_hand)`
        - `controls.start_picking_button.toggled.connect(data_manager.set_picking_mode)`
        - `data_manager.object_loaded_signal.connect(left_view.load_mesh)`
        - `data_manager.object_loaded_signal.connect(center_view.load_object)`
        - `data_manager.hand_loaded_signal.connect(right_view.load_hand)`
        - `data_manager.hand_loaded_signal.connect(center_view.load_hand)`
        - `left_view.point_picked_signal.connect(data_manager.on_object_point_picked)`
        - `right_view.point_picked_signal.connect(data_manager.on_hand_point_picked)`
        - `data_manager.new_anchor_pair_signal.connect(optimization_thread.trigger_optimization)`
        - `optimization_thread.pose_update_signal.connect(center_view.update_hand_pose)`
        - `controls.visualization_settings_changed.connect(...)`（用于更新透明度等）
        - `controls.manual_joint_slider_moved.connect(...)`（用于手动控制中间视图的关节）

## 3. `vista_widget.py`（3D 视窗组件）

- 目的：封装 PyVista 渲染器的可重用 `QWidget` 组件。

- 类：`VistaWidget(QWidget)`。

- 功能：
    - `__init__()`：初始化 `pyvistaqt.QtInteractor`（PyVista 的 Qt 组件）。
    - `self.actors`：使用字典存储场景中的 mesh actor，便于按名字索引和更新。
    - 公共接口（槽 / methods）：
        - `load_mesh(mesh_data, name, color='white', opacity=1.0)`：加载 mesh 并添加到场景，存到 `self.actors[name]`。
        - `load_hand(links_dict_data)`：加载 URDF 解析出的多个 link，每个 link 为单独的 actor（例如 `self.actors['hand_link_1']`）。
        - `update_actor_pose(name, pose_matrix)`：使用 `actor.SetUserTransform(matrix)` 更新 actor 位姿。
        - `update_hand_pose(link_poses_dict)`：循环调用 `update_actor_pose` 更新手上所有 link 的位姿（实现 FK 可视化）。
        - `set_actor_properties(name, color=None, opacity=None)`：改变 actor 的可视属性。
        - `enable_picking(enable=True)`：启用点拾取（`plotter.enable_point_picking()`）并连接内部回调。
    - 内部回调：`_on_point_picked(point_coord, **kwargs)`：当 PyVista 拾取到点时被调用并处理。
    - 信号：`point_picked_signal = pyqtSignal(dict)`；拾取到点时发射，例如：
        ```py
        {'world_coord': [x, y, z], 'mesh_name': 'hand_base_link'}
        ```

## 4. `controls_widget.py`（工具栏 / 控制面板）

- 目的：封装所有 2D 控件（按钮、滑块、列表等）。

- 类：`ControlsWidget(QWidget)`。

- 功能与布局建议：
    - 使用 `QTabWidget` 或 `QGroupBox` 组织 UI：
        - Tab 1（文件 & 控制）：`QPushButton('加载物体')`、`QPushButton('加载机械手')`、`QPushButton('开始设置锚点', checkable=True)`。
        - Tab 2（锚点列表）：`QListWidget`（显示 `Hand_Point -> Obj_Point`）、`QPushButton('删除选中锚点')`、用于调整锚点球大小/颜色的 `QSlider` / `QColorDialog` 按钮。
        - Tab 3（可视化）：中间视图手/物体的透明度滑块和颜色按钮。
        - Tab 4（关节调试）：动态生成 N 个 `QSlider` 或 `QDoubleSpinBox`，用于手动设置和覆盖中间视图手的关节角度。
    - 信号（此 Widget 只负责发射信号，不处理业务逻辑）：
        - `load_object_signal = pyqtSignal()`
        - `load_hand_signal = pyqtSignal()`
        - `start_picking_signal = pyqtSignal(bool)`
        - `anchor_settings_changed_signal = pyqtSignal(dict)`
        - `visualization_settings_changed_signal = pyqtSignal(dict)`
        - `manual_joint_changed_signal = pyqtSignal(str, float)`：（`joint_name, value`）
        - `delete_anchor_signal = pyqtSignal(int)`：（列表索引）

## 5. `data_manager.py`（状态管理器）

- 目的：作为应用的“数据大脑”和逻辑中枢，持有状态并处理 UI 事件逻辑，避免窗口直接互相通信。

- 类：`DataManager(QObject)`（继承 `QObject` 以便发射信号）。

- 主要属性：
    - `self.object_mesh = None`
    - `self.hand_links_mesh_dict = None`（存储手部 URDF 解析出的原始 mesh）
    - `self.anchor_pairs = []`（存储锚点对，例如 `[{ 'hand_point': P1, 'obj_point': P2 }, ...]`）
    - `self.is_picking_mode = False`
    - 内部状态：`self._current_pick_stage = 'hand'`（先选手，再选物体）
    - `self._temp_hand_anchor = None`

- 主要方法与槽（slots）：
    - `load_object()`：弹出 `QFileDialog`，使用 `pyvista.read()` 或 `trimesh.load()` 加载文件，加载成功后发射 `object_loaded_signal`。
    - `load_hand()`：弹出 `QFileDialog`，解析 URDF（例如用 trimesh），提取 link mesh 并发射 `hand_loaded_signal`。
    - `set_picking_mode(is_active: bool)`：设置 `self.is_picking_mode` 并把 `_current_pick_stage` 设为 `'hand'`。
    - `on_hand_point_picked(pick_data: dict)`：
        - 在拾取模式且当前阶段为 `'hand'` 时，将 `pick_data['world_coord']` 保存到 `_temp_hand_anchor`，并将阶段设为 `'object'`（提示用户在物体上选择对应点）。
    - `on_object_point_picked(pick_data: dict)`：
        - 在拾取模式且当前阶段为 `'object'` 时，创建新锚点对并加入 `self.anchor_pairs`，将阶段重置为 `'hand'`，并发射 `new_anchor_pair_signal`（将完整列表发送给优化线程与 UI 列表）。

- 信号：
    - `object_loaded_signal = pyqtSignal(object)`（例如 `pyvista.PolyData`）
    - `hand_loaded_signal = pyqtSignal(dict)`（例如 `{ link_name: mesh_data }`）
    - `new_anchor_pair_signal = pyqtSignal(list)`

## 6. `optimization_thread.py`（后端优化线程）

- 目的：在独立线程中运行 JAX / PyTorch 的优化，避免阻塞 UI。

- 类：`OptimizationThread(QThread)`。

- 功能要点：
    - 初始化：准备 JAX/PyTorch 模型、FK 函数、Adam 优化器等。
    - 属性示例：`self.current_anchor_pairs = []`，`self.lock = QMutex()`（线程安全）。
    - `run()`：线程主循环，等待 `trigger_optimization` 信号；触发后执行若干步梯度下降：
        - 每步计算 `loss, grads = your_jax_model(self.current_anchor_pairs, current_pose_params)`，用优化器更新参数。
        - 运行 FK，得到 `link_poses_dict`，并通过 `self.pose_update_signal.emit(link_poses_dict)` 将位姿发送回主线程以更新中间视图。
        - 每步可调用 `self.msleep(16)`（约 60 FPS）以获得流畅动画。
    - 槽：
        - `trigger_optimization(anchor_pairs: list)`：在锁内更新 `self.current_anchor_pairs` 并唤醒 `run()` 循环开始优化。

- 信号：
    - `pose_update_signal = pyqtSignal(dict)`：在每一步优化中发射 `{ link_name: pose_matrix }`，由 `main_window` 的中间视图接收并用于更新可视化。

---

以上内容为前端（UI）部分的说明与模块接口建议，便于实现信号连接与职责划分。后端具体实现（例如 JAX/PyTorch 的模型、URDF 解析细节）可按项目需要补充。