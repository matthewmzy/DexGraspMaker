"""Global constants (macro-style) centralizing magic numbers and strings.
Modify here to tune behavior; avoid scattering literals across codebase.
"""

# Actor name prefixes
HAND_DYNAMIC_PREFIX = "dyn_hand_"
HAND_STATIC_PREFIX = "static_hand_"

# Visualization defaults
DEFAULT_HAND_OPACITY = 1.0
DEFAULT_OBJECT_OPACITY = 1.0
DEFAULT_ANCHOR_SPHERE_RADIUS_M = 0.005  # 5mm
ANCHOR_COLOR_CYCLE = [
    'red','blue','green','yellow','cyan','magenta','orange','purple','pink','brown'
]

# UI slider ranges
TRANSLATION_RANGE_MM = (-200, 200)
ROTATION_RANGE_DEG = (-180, 180)
ANCHOR_SIZE_MM_DEFAULT = 8.0

# Optimization timing
OPTIMIZATION_SLEEP_MS = 16  # ~60 FPS

# Energy weights
ENERGY_WEIGHTS = {
    'anchor': 1.0,
    'joint_limit': 0.5,
    'penetration': 2.0,
    'self_collision': 0.3,
}

# Optimizer defaults
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_CLIP_GRAD = 1.0

# Scale factors (used by OptimizationThread.set_scale_factors)
DEFAULT_SCALE_FACTORS = {
    'rotation': 0.5,    # make rotation relatively easier than translation if translation=1.0
    'translation': 1.0,
    'joints': 1.0,
}

