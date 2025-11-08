#!/usr/bin/env python3
"""
Fast SDF demo: visualize and compare the new 'fast' SDF method.

- Loads test_assets/objects/Mug.obj (auto scale mm->m)
- Builds a fast SDF at a coarse resolution (default 5mm)
- Shows 2D slices similar to tests/test_sdf_visualization.py
- Prints timing and distance range

Run:
  python scripts/fast_sdf_demo.py --resolution 0.005 --method fast

Optional env:
  DGM_FAST_SDF_MAX_SLICES=8
"""
from __future__ import annotations
import os
import time
import argparse
import numpy as np
import trimesh
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.optimization.energy_functions import PenetrationAvoidanceEnergy


def visualize_sdf_slices(field: np.ndarray, bounds: np.ndarray, num_slices: int = 5, title: str = "Fast SDF Slices"):
    fig, axes = plt.subplots(2, num_slices, figsize=(15, 6))
    fig.suptitle(title, fontsize=16)
    # X slices
    for i in range(num_slices):
        idx = i * (field.shape[0] - 1) // max(1, (num_slices - 1))
        slice_data = field[idx, :, :]
        y_coords = np.linspace(bounds[0, 1], bounds[1, 1], field.shape[1])
        z_coords = np.linspace(bounds[0, 2], bounds[1, 2], field.shape[2])
        Y, Z = np.meshgrid(y_coords, z_coords)
        im1 = axes[0, i].contourf(Y, Z, slice_data.T, levels=20, cmap='RdYlBu_r')
        axes[0, i].set_title(f'X-slice {idx}/{field.shape[0]-1}')
        axes[0, i].set_xlabel('Y')
        axes[0, i].set_ylabel('Z')
        plt.colorbar(im1, ax=axes[0, i])
        binary_slice = (slice_data > 0).astype(float)
        im2 = axes[1, i].contourf(Y, Z, binary_slice.T, levels=[-0.5, 0.5, 1.5], colors=['blue', 'red'])
        axes[1, i].set_title(f'Binary (X-slice {idx})')
        axes[1, i].set_xlabel('Y')
        axes[1, i].set_ylabel('Z')
    plt.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--object', default='test_assets/objects/Mug.obj')
    ap.add_argument('--resolution', type=float, default=0.005)
    ap.add_argument('--method', type=str, default='fast', choices=['fast', 'signed'])
    args = ap.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    obj_path = args.object if os.path.isabs(args.object) else os.path.join(project_root, args.object)

    print(f"加载网格: {obj_path}")
    mesh = trimesh.load(obj_path)
    # auto mm->m
    if mesh.bounds.ptp(axis=0).max() > 10:  # heuristic
        mesh.apply_scale(0.001)
        print("已自动从mm缩放到m")
    print(f"网格: {len(mesh.vertices)} 顶点, {len(mesh.faces)} 面; bounds={mesh.bounds}")

    energy = PenetrationAvoidanceEnergy(
        weight=1.0,
        margin=0.002,
        resolution=args.resolution,
        cache_enabled=False,
        sdf_method=args.method,
    )

    print(f"\n构建SDF method={args.method} resolution={args.resolution}...")
    t0 = time.time()
    energy.precompute_distance_field(mesh, mesh_source=obj_path)
    dt = time.time() - t0

    if energy.distance_field is None:
        print("SDF 构建失败")
        return

    print(f"SDF shape={energy.field_shape}, range=[{energy.distance_field.min():.4f}, {energy.distance_field.max():.4f}], time={dt:.2f}s")

    visualize_sdf_slices(energy.distance_field, energy.field_bounds, num_slices=5,
                         title=f"SDF Slices ({args.method}, res={args.resolution}m)")


if __name__ == '__main__':
    main()
