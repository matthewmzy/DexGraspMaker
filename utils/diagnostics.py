"""Diagnostics helpers for logging optimization details.
"""
from __future__ import annotations
import numpy as np
from typing import Dict

__all__ = [
    'format_energy_breakdown',
    'pose_delta'
]

def format_energy_breakdown(energies: Dict[str, float]) -> str:
    """Return a compact key: value string with 4-decimal formatting."""
    parts = [f"{k}: {v:.4f}" for k, v in energies.items()]
    return ", ".join(parts)


def pose_delta(old_T: np.ndarray, new_T: np.ndarray) -> tuple[float, float]:
    """Compute (translation_norm_m, rotation_deg) between two 4x4 poses.

    Rotation computed from trace formula with clamped cosine.
    """
    dt = float(np.linalg.norm(new_T[:3, 3] - old_T[:3, 3]))
    R_old = old_T[:3, :3]
    R_new = new_T[:3, :3]
    cos_theta = (np.trace(R_old.T @ R_new) - 1.0) * 0.5
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    dtheta = float(np.degrees(np.arccos(cos_theta)))
    return dt, dtheta
