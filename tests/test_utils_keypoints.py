import numpy as np
import pyvista as pv
from utils.keypoints import points_count_for_volume, generate_keypoints


def test_points_count_for_volume_policy():
    assert points_count_for_volume(0.0) == 16
    assert points_count_for_volume(9.9) == 16
    assert points_count_for_volume(10.0) == 32
    assert points_count_for_volume(99.9) == 32
    assert points_count_for_volume(100.0) == 100
    assert points_count_for_volume(999.9) == 100
    assert points_count_for_volume(1000.0) == 200


essential_tol = 1e-9


def test_generate_keypoints_on_sphere_small_cm3():
    # radius=1cm -> volume ~ 4.19 cm^3 -> expect 16 points by policy
    sphere = pv.Sphere(radius=0.01)  # meters
    kps = generate_keypoints({'link': sphere})['link']
    assert isinstance(kps, np.ndarray) and kps.ndim == 2 and kps.shape[1] == 3
    assert kps.shape[0] == 16
    # Ensure points are from mesh vertices
    verts = sphere.points
    # For each keypoint, find a matching vertex
    for p in kps:
        diffs = np.linalg.norm(verts - p[None, :], axis=1)
        assert diffs.min() <= essential_tol
