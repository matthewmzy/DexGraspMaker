"""AnchorManager: encapsulates anchor picking state and CRUD operations.

MainWindow/DataManager delegate anchor-related mutations here to reduce
DataManager size and improve testability.
"""
from __future__ import annotations
from typing import List, Dict, Optional, Tuple
import numpy as np
from utils.anchors import (
    create_anchor_pair,
    delete_anchor_inplace,
    toggle_anchor_inplace,
    update_anchor_position_inplace,
)


class AnchorManager:
    def __init__(self) -> None:
        self.anchor_pairs: List[Dict] = []
        self.is_picking_mode: bool = False
        self._temp_hand_anchor: Optional[Dict] = None
        self._temp_object_anchor: Optional[Dict] = None

    # --- Picking control ---
    def start_picking(self) -> None:
        self.is_picking_mode = True
        self._temp_hand_anchor = None
        self._temp_object_anchor = None

    def stop_picking(self) -> None:
        self.is_picking_mode = False
        self._temp_hand_anchor = None
        self._temp_object_anchor = None

    # --- Temp selections ---
    def set_hand_temp(self, world_coord, local_coord, link_name: str) -> None:
        if not self.is_picking_mode:
            return
        self._temp_hand_anchor = {
            'world_coord': world_coord,
            'local_coord': local_coord,
            'link_name': link_name,
        }

    def set_object_temp(self, world_coord, local_coord) -> None:
        if not self.is_picking_mode:
            return
        self._temp_object_anchor = {
            'world_coord': world_coord,
            'local_coord': local_coord,
        }

    def pair_ready(self) -> bool:
        return self._temp_hand_anchor is not None and self._temp_object_anchor is not None

    # --- Confirm / cancel ---
    def confirm_pair(self) -> Tuple[bool, Optional[Dict]]:
        if not self.is_picking_mode or not self.pair_ready():
            return False, None
        new_pair = create_anchor_pair(self._temp_hand_anchor, self._temp_object_anchor)
        self.anchor_pairs.append(new_pair)
        self.stop_picking()
        return True, new_pair

    def cancel(self) -> bool:
        if not self.is_picking_mode:
            return False
        self.stop_picking()
        return True

    # --- CRUD ---
    def delete(self, index: int) -> Optional[Dict]:
        return delete_anchor_inplace(self.anchor_pairs, index)

    def toggle(self, index: int, enabled: bool) -> bool:
        return toggle_anchor_inplace(self.anchor_pairs, index, enabled)

    def update_position(self, index: int, point_type: str, position: List[float], current_link_poses: Dict[str, np.ndarray]) -> bool:
        return update_anchor_position_inplace(self.anchor_pairs, index, point_type, position, current_link_poses)
