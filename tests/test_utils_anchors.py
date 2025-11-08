import numpy as np
from utils.anchors import (
    create_anchor_pair,
    delete_anchor_inplace,
    toggle_anchor_inplace,
    update_anchor_position_inplace,
)


def test_create_anchor_pair():
    hand_tmp = {'world_coord': [1,2,3], 'local_coord': [0.1,0.2,0.3], 'link_name': 'finger_link'}
    obj_tmp = {'world_coord': [4,5,6], 'local_coord': [4,5,6]}
    anchor = create_anchor_pair(hand_tmp, obj_tmp)
    assert anchor['hand_point'] == [1,2,3]
    assert anchor['obj_point'] == [4,5,6]
    assert anchor['enabled'] is True
    assert anchor['hand_link_name'] == 'finger_link'


def test_delete_anchor_inplace():
    anchors = [ {'hand_point': [0,0,0]} for _ in range(3) ]
    removed = delete_anchor_inplace(anchors, 1)
    assert removed is not None
    assert len(anchors) == 2
    assert delete_anchor_inplace(anchors, 10) is None


def test_toggle_anchor_inplace():
    anchors = [ {'hand_point': [0,0,0], 'enabled': True} ]
    ok = toggle_anchor_inplace(anchors, 0, False)
    assert ok and anchors[0]['enabled'] is False
    assert toggle_anchor_inplace(anchors, 5, True) is False


def test_update_anchor_position_inplace_hand():
    anchors = [ {
        'hand_point': [0,0,0],
        'hand_point_local': [0,0,0],
        'hand_link_name': 'finger',
        'obj_point': [1,1,1],
        'obj_point_local': [1,1,1]
    }]
    # FK pose: translate by (1,2,3)
    T = np.eye(4)
    T[:3,3] = [1,2,3]
    ok = update_anchor_position_inplace(anchors, 0, 'hand', [0.5,0.0,0.0], {'finger': T})
    assert ok
    # world = translation + local
    assert np.allclose(anchors[0]['hand_point'], [1.5,2.0,3.0])
    assert anchors[0]['hand_point_local'] == [0.5,0.0,0.0]


def test_update_anchor_position_inplace_object():
    anchors = [ {
        'hand_point': [0,0,0],
        'hand_point_local': [0,0,0],
        'hand_link_name': 'finger',
        'obj_point': [1,1,1],
        'obj_point_local': [1,1,1]
    }]
    ok = update_anchor_position_inplace(anchors, 0, 'object', [9,9,9], {})
    assert ok
    assert anchors[0]['obj_point'] == [9,9,9]
    assert anchors[0]['obj_point_local'] == [9,9,9]


def test_update_anchor_position_inplace_out_of_range():
    anchors = []
    ok = update_anchor_position_inplace(anchors, 0, 'hand', [0,0,0], {})
    assert ok is False
