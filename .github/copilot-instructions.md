## DexGraspMaker — AI coding instructions (concise)

Interactive PyQt6 app for robotic grasp pose optimization. Three PyVista views (left/object, center/live hand, right/static hand). Data flows via Qt signals; optimization runs in a background thread with JAX/pyroki.

Architecture (key files)
- `scripts/main.py`: entrypoint; parses `--load-default` and creates `MainWindow`.
- `scripts/main_window.py`: thin orchestrator; delegates UI build to `scripts/ui/ui_builder.py`, wiring to `scripts/ui/wiring.py`, pose/anchor updates to `scripts/ui/anchor_view.py`.
- `scripts/vista_widget.py`: 3D view. Stores actors in a dict; update via `actor.user_matrix`. Emits `point_picked_signal({'actor_name', 'world_coord', 'relative_coord'})`.
- `scripts/data_manager.py`: state hub. Loads meshes/URDF, manages anchor signals; anchor CRUD/picking delegated to `scripts/anchor_manager.py`.
- `scripts/optimization_thread.py`: QThread doing optimization + FK; emits link poses and updated base/joints at ~60 FPS.
- `scripts/optimization/*`: optimization primitives; see `scripts/optimization/README.md` for energy/optimizer APIs.

Communication & naming contracts
- Only use signals/slots across components (no direct method calls).
- Actor names are stable keys in `VistaWidget.actors`.
  - Dynamic hand pose updates use prefix `dyn_hand_` (see `OptimizationThread.actor_name_prefix`).
  - Static right-hand picking uses prefix `static_hand_` (parsed in `DataManager.on_hand_point_picked`). Keep these prefixes consistent across views.
- Picking signal payload example: `{'actor_name': 'static_hand_<link>', 'world_coord': [x,y,z], 'relative_coord': [x,y,z]}`.

Loading pipelines (project-specific behavior)
- Object mesh: `DataManager.load_object*()` reads with PyVista and auto scales to meters if `max(bounds_size) > 10` (assumes mm→m).
- URDF hand: `yourdfpy.URDF.load(...)` → build `pk.Robot` (JAX-native) → extract per-link meshes from trimesh scene → emit:
  - `hand_loaded_signal({link_name: pyvista.PolyData})`
  - `hand_joint_info_signal([{'name','min','max','default'}])`
  - `pyroki_robot_loaded_signal(pk.Robot)` and initial link poses via `hand_initial_pose_signal`.
- Keypoints/spheres: generated or loaded per link (`hand_keypoints_loaded_signal`, `hand_link_spheres_loaded_signal`) for penetration/self-collision energies.

Optimization loop (defaults and where to change)
- Energies combined via `CompositeEnergy`: `AnchorPointEnergy(1.0) + JointLimitEnergy(0.5, margin=0.1) + PenetrationAvoidanceEnergy(2.0) + SelfCollisionAvoidanceEnergy(0.3, margin=0.005)` (set in `_setup_optimization`).
- Optimizer: Optax-based Adam via `create_adam(learning_rate=0.01, clip_grad=1.0)`. Switch via `OptimizationThread.set_optimizer('adam'|'adamw'|'lion'|'sgd', **kwargs)`.
- Variables managed with `OptimizerState` (SE(3) + joints). Scale factors used to balance gradients: translation scaled by 0.1.
- FK via `pyroki.Robot.forward_kinematics`; poses emitted as `{ 'dyn_hand_<link>': 4x4 }`. Rate throttled with `msleep(16)`.

3D rendering patterns
- Use `VistaWidget.load_hand(links_dict, name_prefix='dyn_hand_')` for dynamic view and `name_prefix='static_hand_'` for static/picking view.
- Update all link poses in bulk with `VistaWidget.update_hand_pose(link_poses_dict)`; individual with `update_actor_pose(name, pose)`.
- Picking is enabled per view; ensure `_picking_enabled` is true before expecting `point_picked_signal`.

Developer workflows
- Launch (recommended): `./run.sh` (sets OpenGL env, activates conda `dgm`, forwards args). Alt: `conda activate dgm && python scripts/main.py`.
- Quick validation: run with `--load-default` to auto-load test assets from `test_assets/`.
- Tuning: edit energies/weights in `OptimizationThread._setup_optimization()`; switch optimizer via `set_optimizer(...)`.
- Debugging: watch console prints and `status_message_signal` updates; detailed energies print every 30 steps.

Gotchas
- Keep actor prefixes consistent across producer/consumer; mismatches break pose updates and picking mapping.
- Always communicate via signals to avoid cross-thread/UI blocking.
- Respect mesh units (mm vs m) and avoid re-scaling already scaled data.
- Ensure joint dicts include all `robot.joints.actuated_names` before FK; otherwise FK returns empty.