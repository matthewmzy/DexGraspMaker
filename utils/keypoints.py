"""Keypoint generation utilities for hand links.

Provides volume-based point count policy and FPS-based keypoint generation per-link.
"""
from __future__ import annotations
from typing import Dict
import numpy as np
import pyvista
from .sampling import farthest_point_sampling

__all__ = [
    'points_count_for_volume',
    'generate_keypoints'
]

def _mesh_volume_cm3(pv_mesh: pyvista.PolyData) -> float:
    """Compute approximate mesh volume in cm^3; fallback to bbox volume if needed."""
    try:
        vol_m3 = float(pv_mesh.volume)
        vol_cm3 = vol_m3 * 1e6
        if vol_cm3 > 0:
            return vol_cm3
    except Exception:
        pass
    bounds = pv_mesh.bounds
    vol_m3 = (bounds[1]-bounds[0]) * (bounds[3]-bounds[2]) * (bounds[5]-bounds[4])
    return float(vol_m3 * 1e6)


def points_count_for_volume(volume_cm3: float) -> int:
    """Policy mapping from link volume to number of keypoints."""
    if volume_cm3 < 10:
        return 16
    if volume_cm3 < 100:
        return 32
    if volume_cm3 < 1000:
        return 100
    return 200


def generate_keypoints(hand_links_mesh_dict: Dict[str, pyvista.PolyData]) -> Dict[str, np.ndarray]:
    """Generate keypoints for each link using FPS with a volume-based count policy.

    Returns {link_name: (N,3) ndarray}; on error for a link, returns bbox center.
    """
    hand_keypoints: Dict[str, np.ndarray] = {}
    for link_name, pv_mesh in hand_links_mesh_dict.items():
        try:
            vol_cm3 = _mesh_volume_cm3(pv_mesh)
            n = points_count_for_volume(vol_cm3)
            pts = farthest_point_sampling(pv_mesh, n)
            hand_keypoints[link_name] = pts
            print(f"keypoints: {link_name} (体积: {vol_cm3:.1f} cm³) -> {n} 个关键点")
        except Exception as e:
            print(f"keypoints: warn generating for {link_name} failed: {e}")
            bounds = pv_mesh.bounds
            center = np.array([
                (bounds[0] + bounds[1]) / 2,
                (bounds[2] + bounds[3]) / 2,
                (bounds[4] + bounds[5]) / 2,
            ], dtype=float)
            hand_keypoints[link_name] = center.reshape(1, -1)
    return hand_keypoints
