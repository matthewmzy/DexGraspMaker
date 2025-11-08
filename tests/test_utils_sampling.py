import numpy as np
import pyvista as pv
from utils.sampling import farthest_point_sampling


def test_fps_returns_all_when_few_vertices():
    # Create a polydata with 5 points, no faces
    pts = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ], dtype=float)
    mesh = pv.PolyData(pts)
    out = farthest_point_sampling(mesh, 10)
    assert out.shape == (5, 3)
    assert np.allclose(np.sort(out, axis=0), np.sort(pts, axis=0))


def test_fps_max_count_respected():
    # Dense grid
    xs, ys, zs = np.meshgrid(np.linspace(0,1,6), np.linspace(0,1,6), np.linspace(0,1,6))
    pts = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])
    mesh = pv.PolyData(pts)
    out = farthest_point_sampling(mesh, 20)
    assert out.shape == (20, 3)
    # Ensure pairwise distances are not zero (distinct points)
    d = np.linalg.norm(out[:, None, :] - out[None, :, :], axis=2)
    assert np.all(d + np.eye(20) * 1e9 > 0)
