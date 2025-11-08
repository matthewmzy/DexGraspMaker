"""Geometry helpers.

Small collection of math utilities used across the app.
"""
from __future__ import annotations
import numpy as np


def euler_rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return rotation matrix from RPY (XYZ intrinsic) Euler angles.

    Args:
        roll: rotation around X (rad)
        pitch: rotation around Y (rad)
        yaw: rotation around Z (rad)
    Returns:
        3x3 numpy array
    """
    cx, cy, cz = np.cos(roll), np.cos(pitch), np.cos(yaw)
    sx, sy, sz = np.sin(roll), np.sin(pitch), np.sin(yaw)
    R_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    R_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    R_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return R_z @ R_y @ R_x


def matrix_to_euler_rpy(R: np.ndarray) -> tuple[float, float, float]:
    """Convert rotation matrix to RPY (XYZ intrinsic) Euler angles.

    Returns (roll, pitch, yaw) in radians.
    """
    # Guard numerical issues
    R = np.asarray(R, dtype=float)
    sy = -R[2, 0]
    sy_clamped = np.clip(sy, -1.0, 1.0)
    pitch = np.arcsin(sy_clamped)

    if np.abs(np.cos(pitch)) < 1e-8:  # Gimbal lock
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    return float(roll), float(pitch), float(yaw)
