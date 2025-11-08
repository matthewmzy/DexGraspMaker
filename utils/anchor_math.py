"""Anchor pose update math utilities.

Separates computation of updated anchor positions for different views.
"""
from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple

__all__ = ["compute_view_anchor_sets"]

def _transform_local_point(T_world_link: np.ndarray, local_point: np.ndarray) -> np.ndarray:
    local_h = np.append(local_point, 1.0)
    return (T_world_link @ local_h)[:3]


def compute_view_anchor_sets(anchor_pairs: List[dict], link_poses_dict: Dict[str, np.ndarray], initial_link_poses: Dict[str, np.ndarray]) -> Tuple[List[dict], List[dict], List[dict]]:
    """Compute per-view anchor pair lists with updated hand anchor world positions.

    Returns (left_pairs, right_pairs, center_pairs).
    right_pairs have obj_point set to origin to hide object anchor.
    center_pairs use dynamic hand poses; left_pairs include both anchors.
    """
    if not anchor_pairs:
        return [], [], []

    # Map dyn link actor names back to link_name
    dynamic_link_pose = {}
    for actor_name, pose in link_poses_dict.items():
        if actor_name.startswith("dyn_hand_"):
            link_name = actor_name[len("dyn_hand_"):]
            dynamic_link_pose[link_name] = pose

    left_pairs = []
    right_pairs = []
    center_pairs = []

    for pair in anchor_pairs:
        link_name = pair['hand_link_name']
        local = np.array(pair['hand_point_local'])
        # Dynamic world position if present
        if link_name in dynamic_link_pose:
            dyn_world = _transform_local_point(dynamic_link_pose[link_name], local)
        else:
            dyn_world = np.array(pair['hand_point'])
        # Static world position via initial pose
        if link_name in initial_link_poses:
            static_world = _transform_local_point(initial_link_poses[link_name], local)
        else:
            static_world = np.array(pair['hand_point'])
        # Build updated copies
        left_updated = pair.copy()
        left_updated['hand_point'] = dyn_world.tolist()
        center_updated = left_updated.copy()
        right_updated = pair.copy()
        right_updated['hand_point'] = static_world.tolist()
        right_updated['obj_point'] = [0,0,0]
        left_pairs.append(left_updated)
        center_pairs.append(center_updated)
        right_pairs.append(right_updated)
    return left_pairs, right_pairs, center_pairs
