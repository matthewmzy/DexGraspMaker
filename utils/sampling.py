"""Sampling utilities (e.g., farthest point sampling)."""
from __future__ import annotations
import numpy as np
import pyvista


def farthest_point_sampling(pv_mesh: pyvista.PolyData, num_points: int) -> np.ndarray:
    """Perform farthest point sampling (FPS) on a mesh's vertices.

    Returns at most num_points points (N,3). If mesh has fewer points, returns all.
    """
    vertices = pv_mesh.points
    if len(vertices) == 0:
        raise ValueError("网格没有顶点")
    if len(vertices) <= num_points:
        return vertices

    selected_indices = [0]
    min_distances = np.full(len(vertices), np.inf)
    for _ in range(1, num_points):
        # Update min distance to selected set
        for selected_idx in selected_indices:
            distances = np.linalg.norm(vertices - vertices[selected_idx], axis=1)
            min_distances = np.minimum(min_distances, distances)
        farthest_idx = int(np.argmax(min_distances))
        selected_indices.append(farthest_idx)
        min_distances[farthest_idx] = 0.0
    return vertices[selected_indices]
