"""Anchor and pose coordination helpers for MainWindow.

Encapsulates anchor visualization updates and per-frame hand pose updates
to keep MainWindow lean.
"""
from __future__ import annotations
from typing import Dict
import numpy as np

from utils.anchor_math import compute_view_anchor_sets


class AnchorViewCoordinator:
    def __init__(self, mw) -> None:
        self.mw = mw

    # Slots moved from MainWindow
    def on_hand_initial_pose_received(self, poses_dict: Dict[str, np.ndarray]) -> None:
        if not poses_dict:
            return
        print("MainWindow: 收到初始姿态，正在更新视窗...")

        # Store initial poses on MainWindow for static view math
        self.mw._initial_link_poses = poses_dict.copy()

        # Right (static) view
        static_poses = {f"static_hand_{ln}": T for ln, T in poses_dict.items()}
        self.mw.view_right.update_hand_pose(static_poses)

        # Center (dynamic) view
        dyn_poses = {f"dyn_hand_{ln}": T for ln, T in poses_dict.items()}
        self.mw.view_center.update_hand_pose(dyn_poses)

        # Limit FK outputs to visual links to avoid warnings
        try:
            self.mw.optimization_thread.set_visual_link_names(list(self.mw.data_manager.hand_links_mesh_dict.keys()))
        except Exception:
            pass

        # Reset cameras after move
        self.mw.view_right.plotter.reset_camera()
        self.mw.view_center.plotter.reset_camera()
        print("MainWindow: 视窗姿态已更新。")

    def on_anchor_list_updated(self, anchor_pairs: list) -> None:
        print(f"DEBUG: on_anchor_list_updated called with {len(anchor_pairs)} pairs")
        if not anchor_pairs:
            self.mw.view_left.update_anchor_spheres([], lambda p: 'red', 0.005)
            self.mw.view_right.update_anchor_spheres([], lambda p: 'red', 0.005)
            self.mw.view_center.update_anchor_spheres([], lambda p: 'red', 0.005)
            return

        color_func = lambda i: self.mw.get_color_for_pair(i)
        sphere_radius = self.mw.controls_widget.anchor_size_spinbox.value() / 1000.0

        # Left: both anchors; Center: both anchors; Right: hand-only (hide object)
        left_pairs = anchor_pairs.copy()
        right_pairs = []
        for pair in anchor_pairs:
            right_pairs.append({
                'hand_point': pair['hand_point'],
                'hand_point_local': pair['hand_point_local'],
                'hand_link_name': pair['hand_link_name'],
                'obj_point': [0, 0, 0],
                'obj_point_local': pair.get('obj_point_local'),
                'enabled': pair['enabled']
            })

        center_pairs = anchor_pairs.copy()

        self.mw.view_left.update_anchor_spheres(left_pairs, color_func, sphere_radius)
        self.mw.view_right.update_anchor_spheres(right_pairs, color_func, sphere_radius)
        self.mw.view_center.update_anchor_spheres(center_pairs, color_func, sphere_radius)

    def on_pose_update_with_anchors(self, link_poses_dict: dict) -> None:
        if not link_poses_dict:
            print("MainWindow: on_pose_update_with_anchors 收到空字典！")
            return

        # Debug throttling
        if not hasattr(self.mw, '_pose_update_counter'):
            self.mw._pose_update_counter = 0
        self.mw._pose_update_counter += 1
        if self.mw._pose_update_counter % 100 == 0:
            print(f"MainWindow: 已接收 {self.mw._pose_update_counter} 次位姿更新，当前有 {len(link_poses_dict)} 个 links")

        # Update dynamic hand poses (keys already prefixed dyn_hand_)
        self.mw.view_center.update_hand_pose(link_poses_dict)

        # Update DataManager with current link poses for anchor adjustments
        current_link_poses = {
            actor_name[len('dyn_hand_'):]: pose
            for actor_name, pose in link_poses_dict.items() if actor_name.startswith('dyn_hand_')
        }
        self.mw.data_manager.update_current_link_poses(current_link_poses)

        anchor_pairs = self.mw.data_manager.anchor_pairs
        if not anchor_pairs:
            return

        left_pairs, right_pairs, center_pairs = compute_view_anchor_sets(
            anchor_pairs, link_poses_dict, getattr(self.mw, "_initial_link_poses", {})
        )

        color_func = lambda i: self.mw.get_color_for_pair(i)
        self.mw.view_left.color_func = color_func
        self.mw.view_right.color_func = color_func
        self.mw.view_center.color_func = color_func

        self.mw.view_left.update_anchor_positions_fast(left_pairs)
        self.mw.view_right.update_anchor_positions_fast(right_pairs)
        self.mw.view_center.update_anchor_positions_fast(center_pairs)

        # Update pose readout panel
        self.mw.update_hand_pose_display()
