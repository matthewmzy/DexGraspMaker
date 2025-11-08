"""Hand configuration helpers.

Provides ensure_hand_config used by MainWindow to guarantee a YAML config exists.
"""
from __future__ import annotations
import os
import yaml
from typing import Optional, Dict
import numpy as np
from PyQt6.QtWidgets import QMessageBox, QFileDialog, QWidget

DEFAULT_HAND_CONFIG = {
    'urdf_path': 'test_assets/shadow/shadow_hand_right.urdf',
    'base_pose': {
        'translation_m': [0.0, 0.0, 0.0],
        'rpy_deg': [0.0, 0.0, 0.0],
    },
    'joints': 'default'
}


def ensure_hand_config(hand_name: str, cfg_dir: str, project_root: str, parent: Optional[QWidget] = None) -> Optional[str]:
    """Ensure hand_config/<hand_name>.yaml exists; optionally prompt to create.

    Returns absolute path or None if user aborted.
    """
    os.makedirs(cfg_dir, exist_ok=True)
    cfg_path = os.path.join(cfg_dir, f"{hand_name}.yaml")
    if os.path.exists(cfg_path):
        return cfg_path

    # Ask user to create
    reply = QMessageBox.question(
        parent,
        "创建手配置",
        f"未找到手配置文件: {cfg_path}\n是否现在创建并选择 URDF 路径?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.Yes,
    )
    if reply != QMessageBox.StandardButton.Yes:
        return None

    # 选择 URDF 文件
    urdf_path, _ = QFileDialog.getOpenFileName(
        parent,
        "选择手 URDF 文件",
        project_root,
        "URDF 文件 (*.urdf)"
    )
    if not urdf_path:
        return None

    # 生成配置并写入
    cfg_path = os.path.join(cfg_dir, f"{hand_name}.yaml")
    try:
        rel_path = os.path.relpath(urdf_path, project_root)
    except Exception:
        rel_path = urdf_path

    cfg = dict(DEFAULT_HAND_CONFIG)
    cfg['urdf_path'] = rel_path

    try:
        with open(cfg_path, 'w') as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        # 友好提示
        try:
            if parent is not None and hasattr(parent, 'statusBar'):
                parent.statusBar().showMessage(f"✓ 已创建手配置: {cfg_path}")
        except Exception:
            pass
        return cfg_path
    except Exception as e:
        QMessageBox.warning(parent, "写入失败", f"无法写入配置文件:\n{e}")
        return None


def load_hand_config(cfg_path: str) -> Dict:
    """Load YAML config file into a dict (empty dict on failure)."""
    try:
        with open(cfg_path, 'r') as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        print(f"hand_config: failed to load {cfg_path}: {e}")
        return {}


def base_pose_from_cfg(base_pose_cfg: Dict) -> np.ndarray:
    """Build 4x4 base pose matrix from cfg {translation_m, rpy_deg}."""
    from .geometry import euler_rpy_to_matrix
    T = np.eye(4, dtype=float)
    t = base_pose_cfg.get('translation_m', [0.0, 0.0, 0.0])
    rpy_deg = base_pose_cfg.get('rpy_deg', [0.0, 0.0, 0.0])
    T[:3, 3] = np.array(t, dtype=float)
    roll, pitch, yaw = [np.deg2rad(v) for v in rpy_deg]
    T[:3, :3] = euler_rpy_to_matrix(roll, pitch, yaw)
    return T


def init_joints_from_cfg(robot, joints_cfg) -> Dict[str, float]:
    """Return joint dict initialized from cfg ('default' or explicit dict)."""
    names = list(robot.joints.actuated_names)
    lower = robot.joints.lower_limits
    upper = robot.joints.upper_limits
    mid = (lower + upper) / 2.0
    result: Dict[str, float] = {}
    if isinstance(joints_cfg, dict):
        for i, n in enumerate(names):
            try:
                result[n] = float(joints_cfg.get(n, mid[i]))
            except Exception:
                result[n] = float(mid[i])
    else:
        for i, n in enumerate(names):
            result[n] = float(mid[i])
    return result
