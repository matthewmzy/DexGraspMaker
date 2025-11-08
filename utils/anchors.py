"""Anchor utilities: creation, deletion, toggling, and position updates.

These helpers are pure transformations over anchor lists and points, to keep
DataManager slimmer and more testable.
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np

__all__ = [
    'create_anchor_pair',
    'delete_anchor_inplace',
    'toggle_anchor_inplace',
    'update_anchor_position_inplace'
]


def create_anchor_pair(temp_hand_anchor: dict, temp_object_anchor: dict) -> dict:
    """Build a canonical anchor pair dict from temporary picks.

    temp_hand_anchor: {'world_coord','local_coord','link_name'}
    temp_object_anchor: {'world_coord','local_coord'}
    """
    return {
        'hand_point': temp_hand_anchor['world_coord'],
        'hand_point_local': temp_hand_anchor['local_coord'],
        'hand_link_name': temp_hand_anchor['link_name'],
        'obj_point': temp_object_anchor['world_coord'],
        'obj_point_local': temp_object_anchor['local_coord'],
        'enabled': True,
    }


def delete_anchor_inplace(anchor_pairs: List[dict], index: int):
    """Delete anchor at index; return removed or None if out of range."""
    if 0 <= index < len(anchor_pairs):
        return anchor_pairs.pop(index)
    return None


def toggle_anchor_inplace(anchor_pairs: List[dict], index: int, enabled: bool) -> bool:
    """Set enabled flag for an anchor; return success flag."""
    if 0 <= index < len(anchor_pairs):
        anchor_pairs[index]['enabled'] = enabled
        return True
    return False


def update_anchor_position_inplace(anchor_pairs: List[dict], index: int, point_type: str, position: list, current_link_poses: dict) -> bool:
    """Update anchor position in-place; for hand point also recompute world position if FK available.

    point_type: 'hand' updates local coord; 'object' updates world (and local) coord.
    Returns success flag.
    """
    if not (0 <= index < len(anchor_pairs)):
        return False
    pair = anchor_pairs[index]
    if point_type == 'hand':
        pair['hand_point_local'] = position
        link = pair['hand_link_name']
        if link in current_link_poses:
            T = current_link_poses[link]
            local_h = np.append(np.array(position), 1.0)
            world = (T @ local_h)[:3]
            pair['hand_point'] = world.tolist()
        return True
    elif point_type == 'object':
        pair['obj_point'] = position
        pair['obj_point_local'] = position
        return True
    return False
