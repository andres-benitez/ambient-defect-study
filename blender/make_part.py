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
from ads.layout import ROW_ORDER, size_list, defect_name, Defect  # noqa: E402
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


def find_machined_plane(me, normal_tol_deg: float, band_mm: float):
    """Largest planar region among the six axis directions: returns (axis unit vector, offset d, face mask, area)."""
    N, A, C = face_arrays(me)
    best = None
    cos_tol = math.cos(math.radians(normal_tol_deg))
    for axis in [(0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0)]:
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
    def __init__(self, dg, zmax):
        self.dg = dg
        self.zmax = zmax

    def hit(self, x, y):
        ok, loc, nrm, idx, obj, _ = bpy.context.scene.ray_cast(self.dg, Vector((x, y, self.zmax)), Vector((0, 0, -1)))
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


def choose_sites(part_objs: dict, cfg_sites: dict, n_per_zone: int, bbox):
    """Grid-sample the part from above; a site is valid when its clearance ring lands on the same
    zone, is locally flat (height tolerance) and faces the camera."""
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    zmax = bbox[1][2] + 50.0
    smp = Sampler(dg, zmax)
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
                hr = smp.hit(x + dx, y + dy)
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
            cands[zone].append((x, y, float(loc.z), n))
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
                if dtype.startswith("pore"):
                    depth, length, width = float(spec["depth_ratio"]) * s, None, None
                elif dtype == "scratch":
                    depth, length, width = float(spec["depth_ratio"]) * s, float(spec["length_mm"]), s
                else:
                    depth, length, width = s, float(spec["length_mm"]), float(spec["width_ratio"]) * s
                out.append((Defect(prim=defect_name(zone, dtype, s), zone=zone, type=dtype, size_mm=float(s), depth_mm=float(depth),
                                   length_mm=length, width_mm=width, center_mm=[float(x), float(y), float(z)], row=r, col=c), n))
    return out


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
    n, d0, mask, area = find_machined_plane(part.data, float(mach["normal_tol_deg"]), float(mach["plane_band_mm"]))
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
    # recompute the machined mask in the new frame (normals within tol of +z, |z| within band)
    N, A, C = face_arrays(part.data)
    cos_tol = math.cos(math.radians(float(mach["normal_tol_deg"])))
    mask = (N[:, 2] > cos_tol) & (np.abs(C[:, 2]) < float(mach["plane_band_mm"]))
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
    log(f"zones: machined {len(zone_obj['machined'].data.polygons)} tris, cast {len(zone_obj['cast'].data.polygons)} tris")
    # machined extent (for the scale check)
    mco = np.array([v.co[:] for v in zone_obj["machined"].data.vertices])
    machined_extent = (mco.max(0) - mco.min(0)).tolist()

    # ---- sites and defects
    n_per_zone = sum(len(size_list(cfg["defects"], t)) for t in ROW_ORDER if t in cfg["defects"])
    sites = choose_sites(zone_obj, pc["sites"], n_per_zone, bbox)
    defects = build_defect_list(cfg, sites)
    root_mm = float(cfg["coupon"].get("burr_root_mm", 0.01))
    defect_objects = {}
    for zone in cfg["coupon"]["zones"]:
        cutters, cmap = [], {}
        for d, nrm in [x for x in defects if x[0].zone == zone]:
            loc = Vector(d.center_mm)
            if d.type == "burr":
                ob = burr_local(d, zone_mats[zone], root_mm)
                place(ob, loc, nrm)
                defect_objects[d.prim] = ob
                continue
            cmat = get_material("C_" + d.prim)
            cmap[cmat.name] = d.prim
            ob = pore_cutter_local(d, cmat) if d.type.startswith("pore") else scratch_cutter_local(d, cmat)
            place(ob, loc, nrm)
            cutters.append(ob)
        select_only(cutters)
        bpy.ops.object.join()
        cutter = bpy.context.view_layer.objects.active
        t1 = time.time()
        target = zone_obj[zone]
        mod = target.modifiers.new("cut", "BOOLEAN")
        mod.operation, mod.solver, mod.material_mode, mod.use_self, mod.use_hole_tolerant, mod.object = "DIFFERENCE", "EXACT", "TRANSFER", False, True, cutter
        select_only([target])
        bpy.ops.object.modifier_apply(modifier=mod.name)
        bpy.data.objects.remove(cutter, do_unlink=True)
        log(f"boolean on {zone}: {time.time()-t1:.0f}s")
        plate, pieces = separate_by_material(target, zone_mats[zone], cmap)
        missing = set(cmap.values()) - set(pieces)
        if missing:
            raise RuntimeError(f"boolean produced no cavity for: {sorted(missing)}")
        defect_objects.update(pieces)
        zone_obj[zone] = plate
    for m in [m for m in bpy.data.materials if m.name.startswith("C_D_")]:
        bpy.data.materials.remove(m)

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
