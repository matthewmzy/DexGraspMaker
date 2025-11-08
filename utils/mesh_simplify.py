"""Mesh simplify & cache utilities.

- Uses PyVista (VTK) decimation for robust, dependency-free simplification.
- Caches simplified meshes on disk keyed by geometry hash + target cap.
- Provides helpers for PyVista PolyData inputs and returns PolyData outputs.

Cache layout (default): .cache/mesh/
  mesh_<hash>_cap<faces>.vtp  (VTK PolyData XML)

Env overrides:
  DGM_MESH_CACHE             - cache root directory
  DGM_MAX_OBJECT_FACES       - object face cap (int)
  DGM_MAX_HAND_LINK_FACES    - hand-link face cap (int)

"""
from __future__ import annotations
import os
import hashlib
import inspect
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pyvista as pv

from .constants import MESH_CACHE_DIR


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _polydata_hash(mesh: pv.PolyData) -> str:
    """Hash vertices+faces (connectivity) to identify geometry.
    Use float32/int32 to stabilize.
    """
    v = np.asarray(mesh.points, dtype=np.float32)
    # VTK faces: [n, i0, i1, i2, n, j0, ...]; convert to triangle index array
    faces = mesh.faces.reshape(-1, 4)[:, 1:].astype(np.int32, copy=False)
    h = hashlib.sha256()
    h.update(v.tobytes())
    h.update(faces.tobytes())
    return h.hexdigest()


def _cache_dir() -> Path:
    root = os.environ.get("DGM_MESH_CACHE", MESH_CACHE_DIR)
    p = Path(root)
    _ensure_dir(p)
    return p


def _cache_path(mesh_hash: str, face_cap: int) -> Path:
    return _cache_dir() / f"mesh_{mesh_hash}_cap{face_cap}.vtp"


def _face_count(mesh: pv.PolyData) -> int:
    # faces stored as n+indices, triangles expected: 4 stride but could be mixed
    # robustly count using number of cells
    return int(mesh.n_cells)


def _mesh_length(mesh: pv.PolyData) -> float:
    try:
        return float(mesh.length)
    except Exception:
        # bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
        b = mesh.bounds
        return float(((b[1]-b[0])**2 + (b[3]-b[2])**2 + (b[5]-b[4])**2) ** 0.5)


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    v = v.strip().lower()
    return v in ("1", "true", "yes", "y", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def _clean_with_tol(mesh: pv.PolyData) -> pv.PolyData:
    # Merge near-duplicate points to avoid tiny seams from OBJ per-face attributes.
    rel = _env_float('DGM_MESH_CLEAN_TOL_REL', 1e-5)
    absolute = _env_bool('DGM_MESH_CLEAN_TOL_ABS', True)
    tol = rel if absolute else rel * max(1e-12, _mesh_length(mesh))
    try:
        return mesh.clean(point_merging=True, tolerance=tol, absolute=absolute, inplace=False)
    except Exception:
        # Fallback to default clean
        return mesh.clean(inplace=False)


def _maybe_keep_largest(mesh: pv.PolyData) -> pv.PolyData:
    keep = os.environ.get('DGM_MESH_CONNECTIVITY_KEEP', 'none').lower()
    if keep not in ('largest', 'none'):
        keep = 'none'
    if keep == 'largest':
        try:
            return mesh.connectivity(largest=True)
        except Exception:
            return mesh
    return mesh


def simplify_polydata(mesh: pv.PolyData, face_cap: int) -> Tuple[pv.PolyData, bool]:
    """Simplify a PolyData to be at or below face_cap.

    Strategy:
      1. Try decimate_pro with version-agnostic parameter name detection.
      2. If decimate_pro unavailable or fails, try basic decimate (coarser).
      3. If still above cap, perform an additional pass (pro or basic) with updated ratio.
      4. Always clean() before returning.

    Returns (result_mesh, did_simplify)
    """
    # Preprocess: ensure triangles and weld nearly coincident vertices
    try:
        work = mesh.triangulate()
    except Exception:
        work = mesh
    work = _clean_with_tol(work)

    original_faces = _face_count(work)
    if original_faces <= face_cap or original_faces == 0:
        return work, False

    # fraction to remove (0..0.99). Example: cap 10k from 50k => remove 80% => ratio=0.8
    def _compute_ratio(cur_faces: int) -> float:
        return max(0.0, min(0.99, 1.0 - float(face_cap) / float(cur_faces)))

    ratio = _compute_ratio(original_faces)
    simplified = mesh
    used_method = None

    def _try_decimate_pro(m: pv.PolyData, r: float) -> Optional[pv.PolyData]:
        if not hasattr(m, 'decimate_pro'):
            return None
        fn = m.decimate_pro
        try:
            sig = inspect.signature(fn)
            kwargs = {}
            # Common options to reduce fragmentation across sharp edges
            if 'preserve_topology' in sig.parameters:
                kwargs['preserve_topology'] = _env_bool('DGM_MESH_PRESERVE_TOPOLOGY', False)
            if 'splitting' in sig.parameters:
                kwargs['splitting'] = _env_bool('DGM_MESH_SPLITTING', False)
            if 'feature_angle' in sig.parameters:
                kwargs['feature_angle'] = _env_float('DGM_MESH_FEATURE_ANGLE_DEG', 60.0)
            if 'boundary_deletion' in sig.parameters:
                kwargs['boundary_deletion'] = _env_bool('DGM_MESH_BOUNDARY_DELETION', True)

            if 'reduction' in sig.parameters:
                return fn(r, **kwargs)
            elif 'target_reduction' in sig.parameters:
                return fn(target_reduction=r, **kwargs)
            else:
                # Fallback: attempt positional
                return fn(r)
        except Exception:
            return None

    def _try_decimate_basic(m: pv.PolyData, r: float) -> Optional[pv.PolyData]:
        if not hasattr(m, 'decimate'):
            return None
        try:
            # decimate wants target_reduction
            return m.decimate(target_reduction=r)
        except Exception:
            return None

    # First attempt: decimate_pro
    # Backend selection (optional Open3D)
    backend = os.environ.get('DGM_MESH_SIMPLIFY_METHOD', 'auto').lower()

    out = None
    used_method = None
    def _use_vtk(m: pv.PolyData, r: float) -> Optional[pv.PolyData]:
        o = _try_decimate_pro(m, r)
        if o is not None:
            return o
        return _try_decimate_basic(m, r)

    # Try Open3D first if requested/available
    tried_open3d = False
    if backend in ('auto', 'open3d'):
        try:
            import open3d as o3d  # type: ignore
            tried_open3d = True
            # Build Open3D mesh
            pts = np.asarray(work.points)
            faces_arr = work.faces.reshape(-1, 4)[:, 1:4]
            o3 = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(pts),
                o3d.utility.Vector3iVector(faces_arr.astype(np.int32, copy=False)),
            )
            target_tris = max(4, int(face_cap))
            o3 = o3.simplify_quadric_decimation(target_tris)
            # Optional smoothing
            smooth_iters = int(_env_float('DGM_MESH_SMOOTH_ITERS', 0))
            if smooth_iters > 0:
                try:
                    o3 = o3.filter_smooth_taubin(number_of_iterations=smooth_iters)
                except Exception:
                    pass
            # Convert back to PyVista
            out_pts = np.asarray(o3.vertices)
            out_tris = np.asarray(o3.triangles, dtype=np.int32)
            if len(out_tris) == 0:
                out = None
            else:
                faces = np.hstack([np.full((out_tris.shape[0], 1), 3, dtype=np.int32), out_tris]).ravel()
                out = pv.PolyData(out_pts, faces)
                used_method = 'open3d'
        except Exception:
            out = None
    if out is None:
        out = _use_vtk(work, ratio)
        if out is not None and used_method is None:
            used_method = 'decimate_pro' if hasattr(work, 'decimate_pro') else 'decimate'
    if out is not None:
        simplified = out
    else:
        print(f"mesh_simplify: no working simplification backend; returning original. faces={original_faces}")
        return work, False

    # If still above cap, second pass with updated ratio
    passes = 1
    while _face_count(simplified) > face_cap and passes < 3:
        passes += 1
        cur_faces = _face_count(simplified)
        ratio2 = _compute_ratio(cur_faces)
        out2 = None
        if used_method == 'decimate_pro':
            out2 = _try_decimate_pro(simplified, ratio2)
        if out2 is None:
            out2 = _try_decimate_basic(simplified, ratio2)
            if out2 is not None and used_method != 'decimate_pro':
                used_method = 'decimate'
        if out2 is None:
            break
        simplified = out2

    final_faces = _face_count(simplified)
    if final_faces >= original_faces:
        # Did not reduce effectively
        print(f"mesh_simplify: reduction ineffective (method={used_method}); original={original_faces} final={final_faces}")
        return work, False

    # Post-process: weld, optionally keep largest component
    simplified = _clean_with_tol(simplified)
    simplified = _maybe_keep_largest(simplified)

    # Optional smoothing (VTK)
    smooth_iters = int(_env_float('DGM_MESH_SMOOTH_ITERS', 0))
    if smooth_iters > 0 and hasattr(simplified, 'smooth_taubin'):
        try:
            simplified = simplified.smooth_taubin(n_iter=smooth_iters)
        except Exception:
            pass
    pct = 100.0 * (original_faces - final_faces) / float(original_faces)
    print(f"mesh_simplify: simplified via {used_method} passes={passes} faces {original_faces}->{final_faces} (-{pct:.1f}%) cap={face_cap}")
    return simplified, True


def simplify_with_cache(mesh: pv.PolyData, face_cap: int) -> Tuple[pv.PolyData, bool, str]:
    """Simplify mesh with on-disk cache.

    Returns (mesh_out, cache_hit, cache_path_str)
    """
    try:
        mesh_hash = _polydata_hash(mesh)
        cap = max(1, int(face_cap))
        path = _cache_path(mesh_hash, cap)
        if path.exists():
            try:
                out = pv.read(str(path))
                return out, True, str(path)
            except Exception:
                pass
        out, did = simplify_polydata(mesh, cap)
        try:
            out.save(str(path))
        except Exception as e:
            print(f"mesh_simplify: save cache failed: {e}")
        return out, False, str(path)
    except Exception as e:
        print(f"mesh_simplify: unexpected error {e}; returning original.")
        return mesh, False, ""


def env_face_caps(default_object: int, default_link: int) -> Tuple[int, int]:
    def _get(name: str, default: int) -> int:
        try:
            v = int(os.environ.get(name, str(default)))
            return max(1, v)
        except Exception:
            return default
    return _get('DGM_MAX_OBJECT_FACES', default_object), _get('DGM_MAX_HAND_LINK_FACES', default_link)
