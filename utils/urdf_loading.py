"""URDF and robot loading helpers.

Encapsulates logic for reading a URDF via yourdfpy, constructing a pyroki.Robot,
extracting per-link visual meshes (merged), and computing initial link poses.
"""
from __future__ import annotations
import os
import numpy as np
import pyvista
import trimesh
import yourdfpy
import pyroki as pk
from typing import Dict, Tuple

__all__ = [
    'load_urdf_and_robot',
    'extract_link_meshes',
    'compute_initial_link_poses'
]

def load_urdf_and_robot(file_path: str) -> Tuple[yourdfpy.URDF, pk.Robot]:
    """Load URDF and build pyroki.Robot.

    Raises Exception on failure.
    """
    urdf_obj = yourdfpy.URDF.load(file_path)
    robot = pk.Robot.from_urdf(urdf_obj)
    return urdf_obj, robot


def extract_link_meshes(urdf_obj: yourdfpy.URDF, robot: pk.Robot) -> Dict[str, pyvista.PolyData]:
    """Extract and merge visual meshes for each link in robot.

    Returns {link_name: pyvista.PolyData}. Links without visual geometry are skipped.
    """
    scene = urdf_obj.scene
    if scene is None:
        raise ValueError("URDF scene is None; cannot extract meshes.")
    graph = scene.graph
    transform_graph = graph.transforms
    node_data = transform_graph.node_data
    geometry_dict = scene.geometry

    link_meshes: Dict[str, pyvista.PolyData] = {}
    for link_name in robot.links.names:
        if link_name not in transform_graph.nodes:
            # Skip absent
            continue
        child_nodes = [to_node for from_node, to_node in transform_graph.edge_data if from_node == link_name]
        collected = []
        for child in child_nodes:
            if child in node_data and 'geometry' in node_data[child]:
                geom_key = node_data[child]['geometry']
                if geom_key not in geometry_dict:
                    continue
                geom = geometry_dict[geom_key]
                T = graph.get(child, link_name)[0]
                if hasattr(geom, 'to_mesh'):
                    tm = geom.to_mesh()
                else:
                    tm = geom.copy()
                tm.apply_transform(T)
                collected.append(tm)
        if not collected:
            continue
        if len(collected) > 1:
            merged = trimesh.util.concatenate(collected)
        else:
            merged = collected[0]
        link_meshes[link_name] = pyvista.wrap(merged)
    if not link_meshes:
        raise ValueError("No visual meshes extracted from URDF.")
    return link_meshes


def compute_initial_link_poses(urdf_obj: yourdfpy.URDF, link_meshes: Dict[str, pyvista.PolyData]) -> Dict[str, np.ndarray]:
    """Compute default FK/world poses for each link using the URDF scene graph transforms.
    Uses base_link as reference; if transform missing, identity for base link, otherwise skip with warning.
    """
    initial: Dict[str, np.ndarray] = {}
    base_link = urdf_obj.base_link
    graph = urdf_obj.scene.graph
    for link_name in link_meshes.keys():
        try:
            T = graph.get(link_name, base_link)[0]
            initial[link_name] = T
        except Exception:
            if link_name == base_link:
                initial[link_name] = np.eye(4)
            else:
                print(f"urdf_loading: warn cannot get transform {link_name}->{base_link}")
    return initial
