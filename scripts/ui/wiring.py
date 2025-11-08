"""Signal wiring for MainWindow.

Connects signals/slots across DataManager, OptimizationThread, ControlsWidget,
and views, delegating anchor-related slots to AnchorViewCoordinator.
"""
from __future__ import annotations


def connect_all(mw, anchors):
    # 1. Load object/hand
    mw.controls_widget.load_object_signal.connect(mw.data_manager.load_object)
    mw.controls_widget.load_hand_signal.connect(mw.data_manager.load_hand)

    # 2. Object loaded -> render and optimizer
    mw.data_manager.object_loaded_signal.connect(
        lambda mesh_data: mw.view_left.load_mesh(mesh_data, name="object")
    )
    mw.data_manager.object_loaded_signal.connect(
        lambda mesh_data: mw.view_center.load_mesh(mesh_data, name="object", opacity=1.0)
    )
    mw.data_manager.object_loaded_signal.connect(mw.optimization_thread.set_object_mesh)

    # 3. Hand loaded -> render; initial poses; joints; pyroki robot; identity & collision helpers
    mw.data_manager.hand_loaded_signal.connect(
        lambda links_dict: mw.view_right.load_hand(links_dict, name_prefix="static_hand_")
    )
    mw.data_manager.hand_loaded_signal.connect(
        lambda links_dict: mw.view_center.load_hand(links_dict, name_prefix="dyn_hand_", opacity=1.0)
    )
    mw.data_manager.hand_initial_pose_signal.connect(anchors.on_hand_initial_pose_received)
    mw.optimization_thread.joint_info_signal.connect(mw.controls_widget.create_joint_controls)
    mw.data_manager.pyroki_robot_loaded_signal.connect(mw.optimization_thread.set_pyroki_robot)
    mw.data_manager.hand_identity_loaded_signal.connect(mw.on_hand_identity_loaded)
    mw.data_manager.hand_keypoints_loaded_signal.connect(mw.optimization_thread.set_hand_keypoints)
    mw.data_manager.hand_link_spheres_loaded_signal.connect(mw.optimization_thread.set_link_spheres)

    # 4. Anchors picking
    mw.controls_widget.add_anchor_pair_signal.connect(lambda: mw.data_manager.set_picking_mode(True))
    mw.controls_widget.confirm_anchor_pair_signal.connect(mw.data_manager.confirm_anchor_pair)
    mw.controls_widget.cancel_anchor_adding_signal.connect(mw.data_manager.cancel_anchor_adding)
    mw.data_manager.picking_mode_changed_signal.connect(mw.on_picking_mode_changed)
    mw.data_manager.anchor_pair_ready_signal.connect(mw.controls_widget.show_confirm_button)
    mw.data_manager.anchor_list_updated_signal.connect(mw.controls_widget.update_anchor_list)
    mw.data_manager.anchor_list_updated_signal.connect(anchors.on_anchor_list_updated)
    mw.data_manager.clear_temp_anchors_signal.connect(mw._clear_temp_anchors)
    mw.data_manager.show_temp_anchor_signal.connect(mw._show_temp_anchor)
    mw.view_left.point_picked_signal.connect(mw.data_manager.on_object_point_picked)
    mw.view_right.point_picked_signal.connect(mw.data_manager.on_hand_point_picked)
    mw.controls_widget.delete_anchor_signal.connect(mw.data_manager.on_delete_anchor)
    mw.controls_widget.toggle_anchor_signal.connect(mw.data_manager.on_toggle_anchor)

    # 5. Adjust anchors
    mw.controls_widget.adjust_hand_anchor_signal.connect(mw.data_manager.on_adjust_hand_anchor)
    mw.controls_widget.adjust_object_anchor_signal.connect(mw.data_manager.on_adjust_object_anchor)
    mw.data_manager.adjust_hand_anchor_signal.connect(mw.on_start_adjust_hand_anchor)
    mw.data_manager.adjust_object_anchor_signal.connect(mw.on_start_adjust_object_anchor)
    mw.controls_widget.update_anchor_position_signal.connect(mw.data_manager.on_update_anchor_position)
    mw.keyboard_controller.position_changed_signal.connect(mw.controls_widget.update_anchor_position)
    mw.keyboard_controller.control_ended_signal.connect(mw.on_keyboard_control_ended)
    mw.keyboard_controller.control_state_changed_signal.connect(mw.controls_widget.update_anchor_adjust_button_state)

    # 6. Optimization
    mw.data_manager.new_anchor_pair_signal.connect(mw.optimization_thread.trigger_optimization)
    mw.data_manager.new_anchor_pair_signal.connect(mw.on_new_anchor_pair_auto_start)
    mw.optimization_thread.pose_update_signal.connect(anchors.on_pose_update_with_anchors)

    # 7. Visualization settings
    mw.controls_widget.visualization_settings_changed_signal.connect(mw.on_visualization_changed)
    mw.controls_widget.visualization_settings_changed_signal.connect(
        lambda settings: anchors.on_anchor_list_updated(mw.data_manager.anchor_pairs)
    )

    # 8. Manual joint/base control
    mw.controls_widget.manual_joint_changed_signal.connect(mw.optimization_thread.set_manual_joint)
    mw.controls_widget.base_translation_changed_signal.connect(mw.optimization_thread.set_base_translation)
    mw.controls_widget.base_rotation_changed_signal.connect(mw.optimization_thread.set_base_rotation)
    mw.controls_widget.manual_joint_changed_signal.connect(lambda *_: mw.controls_widget.set_optimization_state(False))
    mw.controls_widget.base_translation_changed_signal.connect(lambda *_: mw.controls_widget.set_optimization_state(False))
    mw.controls_widget.base_rotation_changed_signal.connect(lambda *_: mw.controls_widget.set_optimization_state(False))

    # 9. State sync
    mw.optimization_thread.base_pose_updated_signal.connect(mw.on_base_pose_updated)
    mw.optimization_thread.joint_values_updated_signal.connect(mw.controls_widget.update_joint_controls)

    # 10. Optimization toggle
    mw.controls_widget.optimization_toggle_signal.connect(mw.on_optimization_toggle)

    # 11. Pose import/export
    mw.controls_widget.import_pose_signal.connect(mw.data_manager.import_hand_pose)
    mw.controls_widget.export_pose_signal.connect(mw.data_manager.export_hand_pose)
