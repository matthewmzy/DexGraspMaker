# DexGraspMaker AI Coding Instructions

## Architecture Overview

DexGraspMaker is a PyQt6-based interactive tool for robotic grasping pose optimization. The app features three synchronized 3D views (left/center/right) using PyVista, with a DataManager coordinating state between UI components and a separate OptimizationThread running JAX-based differentiable optimization.

**Core Components:**
- `DataManager`: Central state hub using Qt signals/slots for inter-component communication
- `OptimizationThread`: Runs JAX optimization in background thread to avoid UI blocking
- `VistaWidget`: PyVista-based 3D rendering widgets with actor-based scene management
- `ControlsWidget`: Qt controls organized in tabs (files/controls, anchors, visualization, joint debugging)

## Key Patterns & Conventions

### Signal-Slot Communication
Use Qt signals/slots for all inter-component communication. Never have components directly call methods on each other.

```python
# DataManager emits signals, UI components connect slots
self.object_loaded_signal = pyqtSignal(pyvista.PolyData)
data_manager.object_loaded_signal.connect(left_view.load_mesh)
```

### Actor-Based 3D Scene Management
3D scenes use named actors stored in dictionaries. Always use consistent naming prefixes:

```python
# In VistaWidget
self.actors = {}  # dict[str, pyvista.Actor]
self.actors[f"{self.actor_name_prefix}link_{i}"] = actor

# In OptimizationThread
self.actor_name_prefix = "dyn_hand_"  # Must match VistaWidget
```

### Two-Stage Anchor Point Picking
Anchor points are created in two stages: pick hand point first, then corresponding object point.

```python
# DataManager manages picking state
self._current_pick_stage = 'hand'  # or 'object'
self._temp_hand_anchor = None
```

### Composite Energy Functions
Optimization uses weighted combinations of energy functions. Always use `CompositeEnergy` for multiple objectives:

```python
from optimization import CompositeEnergy, AnchorPointEnergy, JointLimitEnergy

energy_fn = CompositeEnergy([
    AnchorPointEnergy(weight=1.0),      # Primary: anchor matching
    JointLimitEnergy(weight=0.5),       # Secondary: joint limits
    CollisionAvoidanceEnergy(weight=0.1) # Tertiary: collision avoidance
])
```

### SE(3) Pose Optimization
Use `OptimizerState` for managing optimization variables with proper SE(3) representations:

```python
from optimization import OptimizerState

# Create state from 4x4 matrix and joint values
state = OptimizerState.from_numpy(
    base_pose_4x4=np.eye(4),
    joint_dict={'joint1': 0.0, 'joint2': 0.5},
    joint_names=['joint1', 'joint2']
)

# Get optimization vector [wxyz(4), xyz(3), joints(N)]
x = state.get_optimization_vector()
```

## Critical Developer Workflows

### Launching the Application
Always use `./run.sh` which sets required OpenGL environment variables:

```bash
# Standard launch
./run.sh

# Load default test assets automatically
./run.sh --load-default
```

The script activates conda environment `dgm` and handles OpenGL driver paths.

### Testing Changes
Run with `--load-default` flag to automatically load test hand/object models for validation:

```bash
./run.sh --load-default
```

### Optimization Debugging
Monitor optimization progress through status bar messages and pose update signals. The optimization thread runs at ~60 FPS with `msleep(16)`.

## Dependencies & Integration Points

### Robotics Stack
- **pyroki**: Custom differentiable kinematics library (forward kinematics, collision detection)
- **JAX**: Automatic differentiation for optimization
- **jaxlie**: Lie group operations for SE(3) poses

### 3D Rendering & Meshes
- **PyVista**: 3D rendering in Qt widgets
- **trimesh**: Mesh loading and processing
- **yourdfpy**: URDF parsing for robot models

### UI Framework
- **PyQt6**: Qt6 Python bindings for desktop UI
- **pyvistaqt**: PyVista-Qt integration

## Common Implementation Patterns

### Thread-Safe State Updates
Use `QMutexLocker` for thread-safe access to shared state:

```python
def update_anchor_pairs(self, anchors: list):
    with QMutexLocker(self.mutex):
        self.anchor_pairs = anchors.copy()
        self._needs_optimization = True
        self.wait_condition.wakeAll()
```

### Error Handling
Use try/catch in main components with status bar feedback:

```python
try:
    # risky operation
    self.status_message_signal.emit("Operation completed")
except Exception as e:
    self.status_message_signal.emit(f"Error: {str(e)}")
```

### File Loading with Unit Scaling
Always use `QFileDialog` for user file selection, emit signals on success, and handle unit scaling:

```python
def load_object(self):
    file_path, _ = QFileDialog.getOpenFileName(
        self.parent(), "Select Object Mesh",
        "", "Mesh files (*.stl *.ply *.obj)"
    )
    if file_path:
        mesh = pyvista.read(file_path)
        # Auto-scale large meshes (assuming mm to m conversion)
        if max(mesh.bounds_size) > 10:
            mesh.scale(0.001, inplace=True)
        self.object_loaded_signal.emit(mesh)
```

### URDF Loading Pipeline
Follow the complete URDF loading sequence:

```python
def load_hand_from_file(self, file_path: str):
    # 1. Parse URDF with yourdfpy (handles package:// paths)
    urdf = yourdfpy.URDF.load(file_path)
    
    # 2. Create pyroki Robot for optimization
    robot = pk.Robot.from_urdf(file_path)
    
    # 3. Extract meshes for visualization
    link_meshes = {}
    for link in urdf.links:
        if link.visuals:
            link_meshes[link.name] = link.visuals[0].geometry.mesh
    
    # 4. Extract joint info for UI controls
    joint_info = [{'name': j.name, 'min': j.limit.lower, 'max': j.limit.upper} 
                  for j in urdf.joints if j.limit]
    
    # Emit all signals
    self.hand_loaded_signal.emit(link_meshes)
    self.hand_joint_info_signal.emit(joint_info)
    self.pyroki_robot_loaded_signal.emit(robot)
```

## Testing & Validation

Focus validation on:
1. Signal connections between components
2. Actor naming consistency across views (critical for pose updates)
3. Thread synchronization in optimization updates
4. Energy function weight tuning for convergence
5. Unit scaling in mesh loading (mm vs meters)

Use `--load-default` flag for quick iteration testing with known-good assets from `test_assets/`.