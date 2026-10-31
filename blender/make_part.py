"""Stage 1 (part variant): put the 48 defects on a real part and export USD + defects.json + part.json.

  blender -b -P blender/make_part.py -- --config config/experiment.yaml --out assets/crankcase/

Steps: import the source mesh (glTF/STL/OBJ), join and stitch it, find the largest planar region
that can face the camera and rotate the part so that region is horizontal at z = 0 (the machined
zone, optionally flattened to the fitted plane because a 1 mm scan cannot represent a machined
finish), split machined / cast faces into two objects, choose defect sites by ray casting (flat,
camera-facing, clear of edges and holes), cut pores and scratches with exact booleans (material
transfer, then separate by material so every cavity is its own prim), add burrs as ridges, bind
every defect to its zone material, and export in metres.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from ads.config import load_config  # noqa: E402
from ads.layout import ROW_ORDER, size_list, defect_name, Defect, is_round  # noqa: E402
from blender.make_coupon import (get_material, new_object, select_only, prism_along_x, finalize_mesh,  # noqa: E402
                                 apply_boolean, separate_by_material, CUTTER_ABOVE_MM)

ZONE_MAT = {"cast": "M_cast", "machined": "M_machined"}
PART_NAME = {"cast": "Part_Cast", "machined": "Part_Machined"}


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "config", "experiment.yaml"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "assets", "crankcase"))
    ap.add_argument("--condition", default=None)
    return ap.parse_args(argv)


def log(*a):
    print("[make_part]", *a, flush=True)


# ----------------------------------------------------------------------------- import + stitch
def import_source(path: str, units_per_mm: float, merge_mm: float):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".stl":
        bpy.ops.wm.stl_import(filepath=path)
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=path)
    else:
        raise RuntimeError(f"unsupported source format {ext}")
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    select_only(meshes)
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.join()
    part = bpy.context.view_layer.objects.active
    part.name = "Part"
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    for o in [o for o in bpy.data.objects if o is not part]:
        bpy.data.objects.remove(o, do_unlink=True)
    # drop the scan's imported per-vertex normals: after flattening they would shade a planar face as
    # bumpy; shading is decided per zone below (flat on machined, smooth on cast)
    try:
        part.data.normals_split_custom_set([(0.0, 0.0, 0.0)] * len(part.data.loops))
    except Exception:  # noqa: BLE001
        pass
    bm = bmesh.new()
    bm.from_mesh(part.data)
    if abs(units_per_mm - 1.0) > 1e-9:
        bmesh.ops.scale(bm, vec=(1 / units_per_mm,) * 3, verts=bm.verts)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge_mm)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    boundary = sum(1 for e in bm.edges if e.is_boundary)
    bm.to_mesh(part.data)
    bm.free()
    part.data.update()
    part.data.materials.clear()
    return part, boundary


# ----------------------------------------------------------------------------- machined plane
def face_arrays(me):
    me.calc_loop_triangles()
    N = np.array([p.normal[:] for p in me.polygons], dtype=float)
    A = np.array([p.area for p in me.polygons], dtype=float)
    C = np.array([p.center[:] for p in me.polygons], dtype=float)
    return N, A, C


AXES = {"+z": (0, 0, 1), "-z": (0, 0, -1), "+y": (0, 1, 0), "-y": (0, -1, 0), "+x": (1, 0, 0), "-x": (-1, 0, 0)}


def find_machined_plane(me, normal_tol_deg: float, band_mm: float, prefer_axis: str | None = None):
    """Largest planar region among the six axis directions (or the one named by `prefer_axis`, in the
    source mesh frame): returns (axis unit vector, offset d, face mask, area)."""
    N, A, C = face_arrays(me)
    best = None
    cos_tol = math.cos(math.radians(normal_tol_deg))
    axes = [AXES[prefer_axis]] if prefer_axis else list(AXES.values())
    for axis in axes:
        ax = np.array(axis, float)
        sel = (N @ ax) > cos_tol
        if sel.sum() < 10:
            continue
        d = C[sel] @ ax
        bins = np.arange(d.min(), d.max() + band_mm, band_mm)
        h, edges = np.histogram(d, bins=bins, weights=A[sel])
        i = int(np.argmax(h))
        d0 = 0.5 * (edges[i] + edges[i + 1])
        # refine: least-squares plane through the faces of that slab, two iterations
        n = ax.copy()
        for _ in range(2):
            m = sel & (np.abs(C @ n - d0) < band_mm)
            if m.sum() < 10:
                break
            pts = C[m]
            w = A[m]
            cen = (pts * w[:, None]).sum(0) / w.sum()
            cov = ((pts - cen) * w[:, None]).T @ (pts - cen)
            evals, evecs = np.linalg.eigh(cov)
            n_new = evecs[:, 0]
            if n_new @ ax < 0:
                n_new = -n_new
            n = n_new / np.linalg.norm(n_new)
            d0 = float(cen @ n)
        m = ((N @ n) > cos_tol) & (np.abs(C @ n - d0) < band_mm)
        area = float(A[m].sum())
        if best is None or area > best[3]:
            best = (n, d0, m, area)
    if best is None:
        raise RuntimeError("no planar region found")
    return best


def rotation_to_z(n: np.ndarray) -> Matrix:
    """Rotation matrix taking unit vector n to +z."""
    v = Vector(n.tolist())
    return v.rotation_difference(Vector((0, 0, 1))).to_matrix().to_4x4()


# ----------------------------------------------------------------------------- site selection
class Sampler:
    """Ray caster that looks from the camera position (perspective), so a site is only valid when the
    camera can actually see it; a vertical ray would accept the floor of a deep bay that the camera's
    slanted line of sight cannot reach."""

    def __init__(self, dg, cam_pos):
        self.dg = dg
        self.cam = Vector(cam_pos)

    def hit(self, x, y, z=0.0):
        target = Vector((x, y, z))
        direction = (target - self.cam).normalized()
        ok, loc, nrm, idx, obj, _ = bpy.context.scene.ray_cast(self.dg, self.cam, direction)
        return (loc, nrm, idx, obj) if ok else None


def farthest_point_pick(cands: np.ndarray, k: int, min_spacing: float) -> list:
    """Greedy farthest-point selection of k indices from candidate xy points."""
    if len(cands) == 0:
        return []
    chosen = [int(np.argmin(np.hypot(cands[:, 0] - cands[:, 0].mean(), cands[:, 1] - cands[:, 1].mean())))]
    dmin = np.hypot(*(cands - cands[chosen[0]]).T)
    while len(chosen) < k:
        i = int(np.argmax(dmin))
        if dmin[i] < min_spacing:
            break
        chosen.append(i)
        dmin = np.minimum(dmin, np.hypot(*(cands - cands[i]).T))
    return chosen


def choose_sites(part_objs: dict, cfg_sites: dict, n_per_zone: int, bbox, cam_pos):
    """Grid-sample the part as seen from the camera; a site is valid when its clearance ring lands on
    the same zone, is locally flat (height tolerance) and faces the camera."""
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    smp = Sampler(dg, cam_pos)
    step = float(cfg_sites["grid_mm"])
    clear = float(cfg_sites["clearance_mm"])
    tilt_cos = math.cos(math.radians(float(cfg_sites["cast_max_tilt_deg"])))
    htol = float(cfg_sites["cast_height_tol_mm"])
    xs = np.arange(bbox[0][0] + clear, bbox[1][0] - clear, step)
    ys = np.arange(bbox[0][1] + clear, bbox[1][1] - clear, step)
    ring = [(clear * math.cos(t), clear * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 12, endpoint=False)]
    ring += [(clear * 0.5 * math.cos(t), clear * 0.5 * math.sin(t)) for t in np.linspace(0, 2 * math.pi, 6, endpoint=False)]
    cands = {"machined": [], "cast": []}
    for x in xs:
        for y in ys:
            h = smp.hit(x, y)
            if h is None:
                continue
            loc, nrm, idx, obj = h
            zone = "machined" if obj.name == PART_NAME["machined"] else "cast"
            if zone == "cast" and nrm.z < tilt_cos:
                continue
            ok = True
            normals = [np.array(nrm[:])]
            for dx, dy in ring:
                hr = smp.hit(loc.x + dx, loc.y + dy, loc.z)
                if hr is None or hr[3] is not obj:
                    ok = False
                    break
                if zone == "machined":
                    if abs(hr[0].z - loc.z) > 0.2:
                        ok = False
                        break
                else:
                    # local plane: height along the site normal must stay within tolerance
                    rel = np.array((hr[0] - loc)[:])
                    if abs(rel @ np.array(nrm[:])) > htol or hr[1].z < tilt_cos:
                        ok = False
                        break
                    normals.append(np.array(hr[1][:]))
            if not ok:
                continue
            n = np.mean(normals, axis=0)
            n /= np.linalg.norm(n)
            cands[zone].append((float(loc.x), float(loc.y), float(loc.z), n))
    sites = {}
    for zone in ("machined", "cast"):
        arr = np.array([(c[0], c[1]) for c in cands[zone]]) if cands[zone] else np.zeros((0, 2))
        idx = farthest_point_pick(arr, n_per_zone, float(cfg_sites["min_spacing_mm"]))
        log(f"zone {zone}: {len(cands[zone])} candidate sites, picked {len(idx)}")
        if len(idx) < n_per_zone:
            raise RuntimeError(f"only {len(idx)} valid sites for zone {zone} (need {n_per_zone}); relax part.sites")
        sites[zone] = [cands[zone][i] for i in idx]
    return sites


# ----------------------------------------------------------------------------- defects
def local_frame(n: np.ndarray) -> Matrix:
    """Rotation whose z axis is n and whose x axis is world x projected on the tangent plane."""
    z = Vector(n.tolist()).normalized()
    x = Vector((1, 0, 0))
    x = (x - z * x.dot(z)).normalized()
    y = z.cross(x).normalized()
    m = Matrix.Identity(4)
    m.col[0][:3] = x
    m.col[1][:3] = y
    m.col[2][:3] = z
    return m


def place(ob, loc: Vector, n: np.ndarray):
    ob.matrix_world = Matrix.Translation(loc) @ local_frame(n)
    select_only([ob])
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def pore_cutter_local(d: Defect, mat):
    r = d.size_mm / 2.0
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=64, v_segments=32, radius=1.0)
    zs = d.depth_mm / r
    for v in bm.verts:
        v.co = Vector((v.co.x * r, v.co.y * r, v.co.z * r * zs))
    return new_object("CUT_" + d.prim, bm, mat)


def scratch_cutter_local(d: Defect, mat):
    w, depth, L = d.width_mm, d.depth_mm, d.length_mm
    e = CUTTER_ABOVE_MM
    profile = [(-w / 2, e), (w / 2, e), (w / 2, 0.0), (0.0, -depth), (-w / 2, 0.0)]
    return new_object("CUT_" + d.prim, prism_along_x(profile, L, 0.0, 0.0), mat)


def burr_local(d: Defect, mat, root_mm: float):
    h, w, L = d.size_mm, d.width_mm, d.length_mm
    profile = [(-w / 2, -root_mm), (w / 2, -root_mm), (w / 2, 0.0), (0.0, h), (-w / 2, 0.0)]
    return new_object(d.prim, prism_along_x(profile, L, 0.0, 0.0), mat)


def build_defect_list(cfg, sites) -> list:
    dcfg = cfg["defects"]
    rows = [r for r in ROW_ORDER if r in dcfg]
    out = []
    for zone in cfg["coupon"]["zones"]:
        k = 0
        for r, dtype in enumerate(rows):
            for c, s in enumerate(size_list(dcfg, dtype)):
                x, y, z, n = sites[zone][k]
                k += 1
                spec = dcfg[dtype]
                if is_round(dtype):
                    depth, length, width = float(spec["depth_ratio"]) * s, None, None
                elif dtype.startswith("scratch"):
                    depth, length, width = float(spec["depth_ratio"]) * s, float(spec["length_mm"]), s
                else:
                    depth, length, width = s, float(spec["length_mm"]), float(spec["width_ratio"]) * s
                out.append((Defect(prim=defect_name(zone, dtype, s), zone=zone, type=dtype, size_mm=float(s), depth_mm=float(depth),
                                   length_mm=length, width_mm=width, center_mm=[float(x), float(y), float(z)], row=r, col=c), n))
    return out


def _box_max(m: np.ndarray, r: int) -> np.ndarray:
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            out |= np.roll(np.roll(m, dy, axis=0), dx, axis=1)
    return out


def _box_min(m: np.ndarray, r: int) -> np.ndarray:
    return ~_box_max(~m, r)


def replanarize(ob, cell_mm: float = 0.5, close_mm: float = 1.5, halo_cells: int = 1, lift_mm: float = 0.02):
    """Replace the flattened (self-overlapping) machined region by a clean planar quad mesh built from
    a raster of its footprint: rasterise the faces at `cell_mm`, close gaps smaller than `close_mm`
    (scan notches), grow by `halo_cells` so the plane overlaps the cast boundary instead of leaving
    gaps, and lift it by `lift_mm` so it wins over coplanar slivers. Returns (n_cells, n_holes_est)."""
    me = ob.data
    co = np.array([v.co[:] for v in me.vertices])
    tris = np.array([p.vertices[:] for p in me.polygons if len(p.vertices) == 3])
    area_before = sum(p.area for p in me.polygons)
    xmin, ymin = co[:, 0].min() - 2 * cell_mm, co[:, 1].min() - 2 * cell_mm
    xmax, ymax = co[:, 0].max() + 2 * cell_mm, co[:, 1].max() + 2 * cell_mm
    nx, ny = int(np.ceil((xmax - xmin) / cell_mm)), int(np.ceil((ymax - ymin) / cell_mm))
    mask = np.zeros((ny, nx), bool)
    cx = xmin + (np.arange(nx) + 0.5) * cell_mm
    cy = ymin + (np.arange(ny) + 0.5) * cell_mm
    for t in tris:
        P = co[t][:, :2]
        i0, i1 = int((P[:, 0].min() - xmin) / cell_mm), int((P[:, 0].max() - xmin) / cell_mm) + 1
        j0, j1 = int((P[:, 1].min() - ymin) / cell_mm), int((P[:, 1].max() - ymin) / cell_mm) + 1
        if i1 <= i0 or j1 <= j0:
            continue
        X, Y = np.meshgrid(cx[i0:i1], cy[j0:j1])
        (x1, y1), (x2, y2), (x3, y3) = P
        det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if abs(det) < 1e-12:
            continue
        l1 = ((y2 - y3) * (X - x3) + (x3 - x2) * (Y - y3)) / det
        l2 = ((y3 - y1) * (X - x3) + (x1 - x3) * (Y - y3)) / det
        l3 = 1 - l1 - l2
        inside = (l1 >= -1e-6) & (l2 >= -1e-6) & (l3 >= -1e-6)
        mask[j0:j1, i0:i1] |= inside
    r = max(int(round(close_mm / cell_mm)), 1)
    closed = _box_min(_box_max(mask, r), r)
    filled = _box_max(closed, halo_cells) if halo_cells > 0 else closed
    holes_est = int((closed & ~mask).sum())
    nb = bmesh.new()
    vcache = {}

    def vert(i, j):
        k = (i, j)
        if k not in vcache:
            vcache[k] = nb.verts.new((xmin + i * cell_mm, ymin + j * cell_mm, lift_mm))
        return vcache[k]

    js, is_ = np.where(filled)
    for j, i in zip(js.tolist(), is_.tolist()):
        nb.faces.new((vert(i, j), vert(i + 1, j), vert(i + 1, j + 1), vert(i, j + 1)))
    bmesh.ops.triangulate(nb, faces=nb.faces)
    nb.normal_update()
    bmesh.ops.reverse_faces(nb, faces=[f for f in nb.faces if f.normal.z < 0])
    nb.to_mesh(me)
    nb.free()
    me.update()
    for p in me.polygons:
        p.use_smooth = False
    log(f"replanarize: footprint {area_before:.0f} mm2 -> raster {int(mask.sum()) * cell_mm**2:.0f} mm2, after closing {int(closed.sum()) * cell_mm**2:.0f} mm2, plane {int(filled.sum()) * cell_mm**2:.0f} mm2 at {cell_mm} mm cells")
    return int(filled.sum()), holes_est


# ----------------------------------------------------------------------------- carving (no booleans)
def cup_mesh(d: Defect, segments: int = 64, rings: int = 16):
    """Open lower half-ellipsoid in the local frame (rim circle at z = 0). Returns (verts, faces, rim_idx)."""
    r, depth = d.size_mm / 2.0, d.depth_mm
    verts, faces = [], []
    for k in range(rings):  # k = 0 is the rim
        phi = 0.5 * math.pi * k / rings
        rho, z = r * math.cos(phi), -depth * math.sin(phi)
        for i in range(segments):
            t = 2 * math.pi * i / segments
            verts.append((rho * math.cos(t), rho * math.sin(t), z))
    apex = len(verts)
    verts.append((0.0, 0.0, -depth))
    for k in range(rings - 1):
        for i in range(segments):
            a, b = k * segments + i, k * segments + (i + 1) % segments
            c, dd = (k + 1) * segments + (i + 1) % segments, (k + 1) * segments + i
            faces.append((a, b, c, dd))
    k = rings - 1
    for i in range(segments):
        faces.append((k * segments + i, k * segments + (i + 1) % segments, apex))
    return verts, faces, list(range(segments))


def groove_mesh(d: Defect, stations: int = 10):
    """Open V-groove along x in the local frame: two walls + end caps. Rim = outline at z = 0."""
    w, depth, L = d.width_mm, d.depth_mm, d.length_mm
    verts, faces = [], []
    for s_ in range(stations + 1):
        x = -L / 2 + L * s_ / stations
        verts += [(x, -w / 2, 0.0), (x, 0.0, -depth), (x, w / 2, 0.0)]  # TL, B, TR
    TL = lambda i: 3 * i  # noqa: E731
    B = lambda i: 3 * i + 1  # noqa: E731
    TR = lambda i: 3 * i + 2  # noqa: E731
    for i in range(stations):
        faces.append((TL(i), TL(i + 1), B(i + 1), B(i)))
        faces.append((B(i), B(i + 1), TR(i + 1), TR(i)))
    faces.append((TL(0), B(0), TR(0)))
    faces.append((TR(stations), B(stations), TL(stations)))
    rim = [TL(i) for i in range(stations + 1)] + [TR(i) for i in range(stations, -1, -1)]
    return verts, faces, rim


def _single_loop(edges) -> bool:
    """True when the edges form exactly one closed loop (every vertex of degree 2, all connected)."""
    if len(edges) < 3:
        return False
    deg = {}
    for e in edges:
        for v in e.verts:
            deg[v] = deg.get(v, 0) + 1
    if any(d != 2 for d in deg.values()):
        return False
    adj = {}
    for e in edges:
        a, b = e.verts
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    start = edges[0].verts[0]
    seen, stack = {start}, [start]
    while stack:
        v = stack.pop()
        for w in adj[v]:
            if w not in seen:
                seen.add(w)
                stack.append(w)
    return len(seen) == len(deg)


def carve_zone(ob, items, zone_mat, margin_mm: float) -> dict:
    """Cut every pore/scratch of `items` into the zone mesh without booleans: delete the scan faces under
    the footprint, build the exact cavity surface as its own object, and bridge its rim to the hole with
    a collar that stays part of the zone mesh (shared rim coordinates, no cracks). The footprint margin is
    widened until the hole outline is a single clean loop."""
    me = ob.data
    N, A, C = face_arrays(me)
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    faces0 = bm.faces[:]
    cups = {}
    for d, nrm in items:
        M = Matrix.Translation(Vector(d.center_mm)) @ local_frame(nrm)
        R = np.array(M.to_3x3())
        local = (C - np.array(d.center_mm)) @ R
        if is_round(d.type):
            base_hx = base_hy = d.size_mm / 2.0
            verts, faces, rim = cup_mesh(d)
        else:
            base_hx, base_hy = d.length_mm / 2.0, d.width_mm / 2.0
            verts, faces, rim = groove_mesh(d)
        chosen = None
        for factor in (1.0, 1.5, 2.2, 3.0):
            m = margin_mm * factor
            sel = np.where((np.abs(local[:, 0]) < base_hx + m) & (np.abs(local[:, 1]) < base_hy + m) & (np.abs(local[:, 2]) < 3.0))[0]
            fsel = [faces0[int(i)] for i in sel if faces0[int(i)].is_valid]
            fset = set(fsel)
            hole_edges, open_edges = {}, 0
            for f in fsel:
                for e in f.edges:
                    lf = e.link_faces
                    if len(lf) == 1:
                        open_edges += 1
                    elif sum(1 for ff in lf if ff in fset) == 1:
                        hole_edges[e.index] = e
            hole_edges = list(hole_edges.values())
            if fsel and not open_edges and _single_loop(hole_edges):
                chosen = (fsel, hole_edges, m)
                break
        if chosen is None:
            raise RuntimeError(f"{d.prim}: no clean single-loop footprint found around {np.round(d.center_mm, 1)} (tried margins up to {margin_mm*3:g} mm)")
        fsel, hole_edges, m = chosen
        # ordered hole outline (every vertex has degree 2: walk it)
        nbr = {}
        for e in hole_edges:
            a, b = e.verts
            nbr.setdefault(a, []).append(b)
            nbr.setdefault(b, []).append(a)
        start_v = hole_edges[0].verts[0]
        Hs, prev, cur = [start_v], None, start_v
        while True:
            nxt = [w for w in nbr[cur] if w is not prev]
            if not nxt or nxt[0] is start_v:
                break
            prev, cur = cur, nxt[0]
            Hs.append(cur)
        bmesh.ops.delete(bm, geom=fsel, context="FACES")
        Minv = M.inverted()
        ring = [bm.verts.new(M @ Vector(verts[i])) for i in rim]  # cyclic order along the rim outline

        def local_xy(v):
            lc = Minv @ v.co
            return lc.x, lc.y

        def signed_area(vs):
            pts = [local_xy(v) for v in vs]
            return 0.5 * sum(pts[k][0] * pts[(k + 1) % len(pts)][1] - pts[(k + 1) % len(pts)][0] * pts[k][1] for k in range(len(pts)))

        if signed_area(Hs) * signed_area(ring) < 0:
            Hs.reverse()
        # align starts: the hole vertex closest in direction to the rim's first vertex
        rx, ry = local_xy(ring[0])
        a0 = math.atan2(ry, rx)
        k0 = min(range(len(Hs)), key=lambda k: abs((math.atan2(*reversed(local_xy(Hs[k]))) - a0 + math.pi) % (2 * math.pi) - math.pi))
        Hs = Hs[k0:] + Hs[:k0]

        def params(vs):
            pts = [Vector(local_xy(v)) for v in vs]
            seg = [(pts[(k + 1) % len(pts)] - pts[k]).length for k in range(len(pts))]
            tot = sum(seg) or 1.0
            t, acc = [], 0.0
            for sl in seg:
                t.append(acc / tot)
                acc += sl
            return t + [1.0]

        th, tr = params(Hs), params(ring)
        up = M.to_3x3() @ Vector((0, 0, 1))
        new_faces = []
        i = j = 0
        nh, nr = len(Hs), len(ring)
        while i < nh or j < nr:
            take_hole = j >= nr or (i < nh and th[i + 1] <= tr[j + 1])
            tri = (Hs[i % nh], Hs[(i + 1) % nh], ring[j % nr]) if take_hole else (Hs[i % nh], ring[(j + 1) % nr], ring[j % nr])
            if take_hole:
                i += 1
            else:
                j += 1
            if len({tri[0], tri[1], tri[2]}) < 3:
                continue
            try:
                f = bm.faces.new(tri)
            except ValueError:
                continue
            f.normal_update()
            if f.normal.dot(up) < 0:
                bmesh.ops.reverse_faces(bm, faces=[f])
            new_faces.append(f)
        if len(new_faces) < nr:
            raise RuntimeError(f"{d.prim}: collar has only {len(new_faces)} faces for a {nr}-vertex rim (margin {m:g} mm)")
        smooth = bool(me.polygons[0].use_smooth) if len(me.polygons) else False
        for f in new_faces:
            f.smooth = smooth
        cb = bmesh.new()
        vs = [cb.verts.new(M @ Vector(v)) for v in verts]
        for f in faces:
            try:
                cb.faces.new([vs[i] for i in f])
            except ValueError:
                pass
        cb.normal_update()
        up = M.to_3x3() @ Vector((0, 0, 1))
        bmesh.ops.reverse_faces(cb, faces=[f for f in cb.faces if up.dot(f.normal) < 0])
        cups[d.prim] = new_object(d.prim, cb, zone_mat)
    bm.to_mesh(me)
    bm.free()
    me.update()
    # verify: no zone face centre may remain inside a cavity footprint (the cavity would be capped)
    N2, A2, C2 = face_arrays(me)
    for d, nrm in items:
        M = Matrix.Translation(Vector(d.center_mm)) @ local_frame(nrm)
        local = (C2 - np.array(d.center_mm)) @ np.array(M.to_3x3())
        if is_round(d.type):
            inside = (np.hypot(local[:, 0], local[:, 1]) < 0.8 * d.size_mm / 2.0) & (np.abs(local[:, 2]) < 1.0)
        else:
            inside = (np.abs(local[:, 0]) < 0.8 * d.length_mm / 2.0) & (np.abs(local[:, 1]) < 0.8 * d.width_mm / 2.0) & (np.abs(local[:, 2]) < 1.0)
        if inside.any():
            raise RuntimeError(f"{d.prim}: {int(inside.sum())} zone faces still cover the cavity footprint")
    return cups


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    cfg = load_config(args.config, args.condition)
    pc = cfg["part"]
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # import with default (metre) scene units: the glTF importer rescales geometry into the scene's
    # unit system, so millimetre display units must be set only after the import
    src = os.path.join(REPO_ROOT, pc["source"])
    part, boundary = import_source(src, float(pc.get("source_units_per_mm", 1.0)), float(pc.get("merge_mm", 0.02)))
    sc = bpy.context.scene
    sc.unit_settings.system = "METRIC"
    sc.unit_settings.scale_length = 0.001
    sc.unit_settings.length_unit = "MILLIMETERS"
    co0 = np.array([v.co[:] for v in part.data.vertices])
    ext0 = co0.max(0) - co0.min(0)
    lo, hi = (float(v) for v in pc.get("expected_extent_mm", [100, 2000]))
    if not (lo <= ext0.max() <= hi):
        raise RuntimeError(f"imported part extent {np.round(ext0, 1)} mm is outside the expected {lo}-{hi} mm; check part.source_units_per_mm")
    log(f"imported {src}: {len(part.data.polygons)} tris, {boundary} boundary edges after stitching, extent {np.round(ext0, 1)} mm ({time.time()-t0:.0f}s)")

    # ---- machined plane -> rotate so it is horizontal at z = 0, long axis along x
    mach = pc["machined"]
    n, d0, mask, area = find_machined_plane(part.data, float(mach["normal_tol_deg"]), float(mach["plane_band_mm"]), mach.get("prefer_axis"))
    log(f"machined plane normal {np.round(n, 3)} offset {d0:.2f} mm, area {area:.0f} mm2, {int(mask.sum())} faces")
    R = rotation_to_z(n)
    part.data.transform(R)
    part.data.update()
    co = np.array([v.co[:] for v in part.data.vertices])
    # plane is now z = d0; shift so it is z = 0
    co[:, 2] -= d0
    # long axis along x
    ext = co.max(0) - co.min(0)
    if pc.get("long_axis", "x") == "x" and ext[1] > ext[0]:
        co[:, [0, 1]] = np.c_[co[:, 1], -co[:, 0]]
        log("rotated 90 deg about z so the long axis runs along x")
    co[:, 0] -= 0.5 * (co[:, 0].max() + co[:, 0].min())
    co[:, 1] -= 0.5 * (co[:, 1].max() + co[:, 1].min())
    part.data.vertices.foreach_set("co", co.ravel())
    part.data.update()
    # recompute the machined mask in the new frame: every face whose centre lies within the band of
    # the plane and that is not a wall (normal within `fill_tol_deg` of +z). The wider normal tolerance
    # than the one used to *find* the plane fills the notches a noisy scan leaves inside the flange,
    # so the flattened zone is a clean plane instead of a ragged one.
    N, A, C = face_arrays(part.data)
    cos_fill = math.cos(math.radians(float(mach.get("fill_tol_deg", 45))))
    mask = (N[:, 2] > cos_fill) & (np.abs(C[:, 2]) < float(mach["plane_band_mm"]))
    if mach.get("flatten", True):
        vidx = np.unique(np.array([i for p in part.data.polygons if mask[p.index] for i in p.vertices]))
        co[vidx, 2] = 0.0
        part.data.vertices.foreach_set("co", co.ravel())
        part.data.update()
        log(f"flattened {len(vidx)} vertices of the machined region onto z = 0")
    bbox = (co.min(0).tolist(), co.max(0).tolist())
    log(f"part bbox mm: {np.round(bbox[0], 1)} .. {np.round(bbox[1], 1)}")

    # ---- split zones into two objects
    zone_mats = {z: get_material(ZONE_MAT[z]) for z in cfg["coupon"]["zones"]}
    part.data.materials.append(zone_mats["cast"])
    part.data.materials.append(zone_mats["machined"])
    mi = np.where(mask, 1, 0).astype(np.int32)
    part.data.polygons.foreach_set("material_index", mi)
    part.data.update()
    select_only([part])
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="MATERIAL")
    bpy.ops.object.mode_set(mode="OBJECT")
    zone_obj = {}
    for ob in [o for o in bpy.data.objects if o.type == "MESH"]:
        used = {ob.data.materials[p.material_index].name for p in ob.data.polygons}
        z = "machined" if used == {ZONE_MAT["machined"]} else "cast"
        ob.name = PART_NAME[z]
        ob.data.name = PART_NAME[z]
        ob.data.materials.clear()
        ob.data.materials.append(zone_mats[z])
        zone_obj[z] = ob
    # shading: the flattened machined zone is a plane (flat shading, exact normals); the cast zone is a
    # scanned surface (smooth shading, angle-limited so real edges stay sharp)
    zone_obj["machined"].data.polygons.foreach_set("use_smooth", [False] * len(zone_obj["machined"].data.polygons))
    zone_obj["cast"].data.polygons.foreach_set("use_smooth", [True] * len(zone_obj["cast"].data.polygons))
    for ob in (zone_obj["machined"], zone_obj["cast"]):
        ob.data.update()
    select_only([zone_obj["cast"]])
    try:
        bpy.ops.object.shade_smooth_by_angle(angle=math.radians(30))
    except Exception:  # noqa: BLE001
        bpy.ops.object.shade_smooth()
    log(f"zones: machined {len(zone_obj['machined'].data.polygons)} tris, cast {len(zone_obj['cast'].data.polygons)} tris")
    # A scan is a surface shell with no thickness, and an exact boolean needs a volume to decide which
    # cutter faces lie "inside". Give each zone an inward skin (Solidify, rim closed) so cavities keep
    # their walls everywhere. Normals must point outward first.
    def faces_camera_side_fraction(ob) -> float:
        """Cast rays down from above a grid of points: the fraction of hit faces whose normal points
        up (towards the camera). The visible skin must have outward normals, so this should be ~1."""
        bpy.context.view_layer.update()
        dg = bpy.context.evaluated_depsgraph_get()
        zmax = bbox[1][2] + 50.0
        up, n_hit = 0, 0
        for x in np.linspace(bbox[0][0] + 5, bbox[1][0] - 5, 40):
            for y in np.linspace(bbox[0][1] + 5, bbox[1][1] - 5, 28):
                ok, loc, nrm, idx, obj, _ = bpy.context.scene.ray_cast(dg, Vector((x, y, zmax)), Vector((0, 0, -1)))
                if ok and obj is ob:
                    n_hit += 1
                    up += 1 if nrm.z > 0 else 0
        return up / max(n_hit, 1)

    # the machined zone is rebuilt as a clean planar mesh from its outline: the flattened scan
    # triangles overlap where notch walls were folded into the plane
    rp = pc.get("replanarize", {})
    replanarize(zone_obj["machined"], cell_mm=float(rp.get("cell_mm", 0.5)), close_mm=float(rp.get("close_mm", 1.5)),
                halo_cells=int(rp.get("halo_cells", 1)), lift_mm=float(rp.get("lift_mm", 0.02)))
    log(f"machined zone re-meshed as a plane: {len(zone_obj['machined'].data.polygons)} tris")
    for z, ob in zone_obj.items():
        select_only([ob])
        if z != "machined":  # the re-meshed plane already has all normals +z; per-island consistency would undo that
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.normals_make_consistent(inside=False)
            bpy.ops.object.mode_set(mode="OBJECT")
        frac_up = faces_camera_side_fraction(ob)
        if frac_up < 0.5:
            bpy.ops.object.mode_set(mode="EDIT")
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.mesh.flip_normals()
            bpy.ops.object.mode_set(mode="OBJECT")
            frac_up2 = faces_camera_side_fraction(ob)
            log(f"{z}: flipped normals (camera-side fraction {frac_up:.2f} -> {frac_up2:.2f})")
            frac_up = frac_up2
        if frac_up < 0.8:
            raise RuntimeError(f"{z}: only {frac_up:.2f} of the camera-facing skin has outward normals")
        log(f"{z}: camera-side normals {frac_up:.2f}, {len(ob.data.polygons)} tris")
    # machined extent (for the scale check)
    mco = np.array([v.co[:] for v in zone_obj["machined"].data.vertices])
    machined_extent = (mco.max(0) - mco.min(0)).tolist()

    # ---- sites and defects
    n_per_zone = sum(len(size_list(cfg["defects"], t)) for t in ROW_ORDER if t in cfg["defects"])
    cam_pos = (0.0, 0.0, float(cfg.get("framing", {}).get("working_distance_mm", 800.0)))
    sites = choose_sites(zone_obj, pc["sites"], n_per_zone, bbox, cam_pos)
    defects = build_defect_list(cfg, sites)
    root_mm = float(cfg["coupon"].get("burr_root_mm", 0.01))
    defect_objects = {}
    margin = float(pc.get("carve_margin_mm", 1.2))
    for zone in cfg["coupon"]["zones"]:
        items = [x for x in defects if x[0].zone == zone]
        for d, nrm in items:
            if d.type == "burr":
                ob = burr_local(d, zone_mats[zone], root_mm)
                place(ob, Vector(d.center_mm), nrm)
                defect_objects[d.prim] = ob
        t1 = time.time()
        cups = carve_zone(zone_obj[zone], [x for x in items if x[0].type != "burr"], zone_mats[zone], margin)
        defect_objects.update(cups)
        log(f"carved {len(cups)} cavities into {zone} in {time.time()-t1:.0f}s")

    uv_tile = float(cfg["coupon"].get("uv_tile_mm", 20))
    all_obs = [zone_obj["cast"], zone_obj["machined"]] + [defect_objects[d.prim] for d, _ in defects]
    for ob in all_obs:
        finalize_mesh(ob, uv_tile)

    records = []
    for d, nrm in defects:
        rec = d.to_record()
        ob = defect_objects[d.prim]
        vs = np.array([v.co[:] for v in ob.data.vertices])
        rec["bbox_center_mm"] = [round(float(v), 4) for v in 0.5 * (vs.max(0) + vs.min(0))]
        rec["normal"] = [round(float(v), 4) for v in nrm]
        rec["n_faces"] = len(ob.data.polygons)
        records.append(rec)
    with open(os.path.join(out_dir, "defects.json"), "w") as fh:
        json.dump(records, fh, indent=1)
    part_info = {
        "kind": "part", "extent_mm": (np.array(bbox[1]) - np.array(bbox[0])).tolist(), "bbox_min_mm": bbox[0], "bbox_max_mm": bbox[1],
        "machined_extent_mm": machined_extent, "machined_area_mm2": float(area), "zone_prims": dict(PART_NAME), "reference_plane_z_mm": 0.0,
        "source": {"file": pc["source"], "name": pc.get("source_name"), "author": pc.get("source_author"), "url": pc.get("source_url"),
                   "license": pc.get("source_license")},
        "stitch_boundary_edges": int(boundary), "tris": {z: len(zone_obj[z].data.polygons) for z in zone_obj},
        "flattened_machined": bool(mach.get("flatten", True)),
    }
    with open(os.path.join(out_dir, "part.json"), "w") as fh:
        json.dump(part_info, fh, indent=1)

    for ob in all_obs:
        ob.data.transform(Matrix.Diagonal((0.001, 0.001, 0.001, 1.0)))
        ob.data.update()
    usd_path = os.path.join(out_dir, cfg["asset"].get("usd", "crankcase.usd") if cfg["asset"].get("kind") == "part" else "crankcase.usd")
    bpy.ops.wm.usd_export(filepath=usd_path, selected_objects_only=False, export_animation=False, export_uvmaps=True, rename_uvmaps=True,
                          export_normals=True, export_materials=True, generate_preview_surface=True, use_instancing=False,
                          convert_scene_units="METERS", root_prim_path="/Coupon", merge_parent_xform=True, triangulate_meshes=False,
                          export_custom_properties=False, relative_paths=True, export_lights=False, export_cameras=False)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, "crankcase.blend"))
    log(f"wrote {usd_path}: {len(defect_objects)} defects, part extent {np.round(part_info['extent_mm'], 1)} mm, in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
