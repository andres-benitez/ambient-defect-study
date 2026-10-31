"""Stage 1: build the test coupon with real defect geometry and export USD + defects.json.

Run headless:
  blender -b -P blender/make_coupon.py -- --config config/experiment.yaml --out assets/coupon/

Geometry rules (see README):
  * Modelled at 1 Blender unit = 1 mm, scaled to metres right before export (Blender's USD
    exporter ignores the scene unit scale, so the scale is applied to the geometry itself).
  * Two plates, Coupon_Cast and Coupon_Machined, top face at z = 0.
  * Pores and scratches are booleaned out (Exact solver, material transfer), then separated by
    material so every cavity becomes its own mesh object named D_<zone>_<type>_<size_um>.
  * Burrs are raised ridges kept as their own objects (rooted slightly into the plate).
  * Every defect object is re-bound to its zone material: contrast must come from geometry only.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import bmesh
import bpy
from mathutils import Matrix, Vector

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ads.config import load_config  # noqa: E402
from ads.layout import build_layout, zone_extent_x, is_round  # noqa: E402

ZONE_MAT = {"cast": "M_cast", "machined": "M_machined"}
PLATE_NAME = {"cast": "Coupon_Cast", "machined": "Coupon_Machined"}
CUTTER_ABOVE_MM = 0.05  # how far cutters poke above the surface so no cutter face is coplanar with the top


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "config", "experiment.yaml"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "assets", "coupon"))
    ap.add_argument("--condition", default=None)
    return ap.parse_args(argv)


# ----------------------------------------------------------------------------- helpers
def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene
    sc.unit_settings.system = "METRIC"
    sc.unit_settings.scale_length = 0.001
    sc.unit_settings.length_unit = "MILLIMETERS"


def get_material(name: str):
    m = bpy.data.materials.get(name)
    if m is None:
        m = bpy.data.materials.new(name)
    return m


def new_object(name: str, bm: bmesh.types.BMesh, material):
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    me.materials.append(material)
    return ob


def select_only(obs):
    bpy.ops.object.select_all(action="DESELECT")
    for o in obs:
        o.select_set(True)
    bpy.context.view_layer.objects.active = obs[0]


def make_plate(zone: str, cfg: dict):
    plate_w, plate_h, plate_t = (float(v) for v in cfg["coupon"]["plate_mm"])
    xmin, xmax = zone_extent_x(cfg, zone)
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    for v in bm.verts:
        v.co = Vector((v.co.x * (xmax - xmin) + 0.5 * (xmin + xmax), v.co.y * plate_h, v.co.z * plate_t - plate_t / 2))
    ob = new_object(PLATE_NAME[zone], bm, get_material(ZONE_MAT[zone]))
    return ob


def make_pore_cutter(d, mat):
    """Half-ellipsoid cavity: opening diameter = size, depth = depth_ratio * size.
    Built as a full ellipsoid centred on the surface; the upper half is outside the plate and
    disappears in the boolean."""
    r = d.size_mm / 2.0
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=64, v_segments=32, radius=1.0)
    zscale = d.depth_mm / r if r > 0 else 1.0
    for v in bm.verts:
        v.co = Vector((v.co.x * r + d.center_mm[0], v.co.y * r + d.center_mm[1], v.co.z * r * zscale))
    return new_object("CUT_" + d.prim, bm, mat)


def prism_along_x(profile_yz, length, cx, cy):
    """Closed prism from a convex/simple polygon profile in the (y, z) plane, extruded along x."""
    bm = bmesh.new()
    x0, x1 = cx - length / 2.0, cx + length / 2.0
    front = [bm.verts.new((x0, cy + y, z)) for (y, z) in profile_yz]
    back = [bm.verts.new((x1, cy + y, z)) for (y, z) in profile_yz]
    bm.faces.new(front[::-1])
    bm.faces.new(back)
    n = len(profile_yz)
    for i in range(n):
        j = (i + 1) % n
        bm.faces.new((front[i], front[j], back[j], back[i]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    return bm


def make_scratch_cutter(d, mat):
    w, depth, L = d.width_mm, d.depth_mm, d.length_mm
    e = CUTTER_ABOVE_MM
    # pentagon: vertical walls above the surface, V-groove below it (width w at z = 0)
    profile = [(-w / 2, e), (w / 2, e), (w / 2, 0.0), (0.0, -depth), (-w / 2, 0.0)]
    bm = prism_along_x(profile, L, d.center_mm[0], d.center_mm[1])
    return new_object("CUT_" + d.prim, bm, mat)


def make_burr(d, mat, root_mm: float):
    h, w, L = d.size_mm, d.width_mm, d.length_mm
    # raised triangular ridge (apex at z = h, base width w at z = 0) rooted `root_mm` into the plate
    profile = [(-w / 2, -root_mm), (w / 2, -root_mm), (w / 2, 0.0), (0.0, h), (-w / 2, 0.0)]
    bm = prism_along_x(profile, L, d.center_mm[0], d.center_mm[1])
    return new_object(d.prim, bm, mat)


def apply_boolean(plate, cutter):
    mod = plate.modifiers.new("cut", "BOOLEAN")
    mod.operation = "DIFFERENCE"
    mod.solver = "EXACT"
    mod.material_mode = "TRANSFER"
    mod.use_self = False
    mod.use_hole_tolerant = False
    mod.object = cutter
    select_only([plate])
    bpy.ops.object.modifier_apply(modifier=mod.name)
    bpy.data.objects.remove(cutter, do_unlink=True)


def separate_by_material(plate, zone_mat, cutter_mat_to_prim: dict):
    """Split cavity faces (carrying cutter materials) into their own objects named after the defect."""
    before = set(bpy.data.objects)
    plate_name = plate.name
    select_only([plate])
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="MATERIAL")
    bpy.ops.object.mode_set(mode="OBJECT")
    pieces = [o for o in bpy.data.objects if o not in before] + [plate]
    found = {}
    plate_piece = None
    for ob in pieces:
        used = {ob.data.materials[p.material_index].name for p in ob.data.polygons if ob.data.materials}
        if len(used) != 1:
            raise RuntimeError(f"{ob.name}: expected one material after separation, got {used}")
        mname = used.pop()
        if mname == zone_mat.name:
            plate_piece = ob
            continue
        prim = cutter_mat_to_prim[mname]
        ob.name = prim
        ob.data.name = prim
        found[prim] = ob
    if plate_piece is None:
        raise RuntimeError("plate piece lost during separation")
    # rename plate piece back, re-bind every piece to the zone material only
    plate_piece.name = plate_name
    for ob in [plate_piece] + list(found.values()):
        ob.data.materials.clear()
        ob.data.materials.append(zone_mat)
        if ob is plate_piece:
            ob.data.name = ob.name
    return plate_piece, found


def finalize_mesh(ob, uv_tile_mm: float):
    select_only([ob])
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.cube_project(cube_size=uv_tile_mm, correct_aspect=False, scale_to_bounds=False)
    bpy.ops.mesh.quads_convert_to_tris(quad_method="BEAUTY", ngon_method="BEAUTY")
    bpy.ops.object.mode_set(mode="OBJECT")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def bbox_center_mm(ob):
    xs = [v.co.x for v in ob.data.vertices]
    ys = [v.co.y for v in ob.data.vertices]
    return [(min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0]


# ----------------------------------------------------------------------------- main
def main():
    args = parse_args()
    cfg = load_config(args.config, args.condition)
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    layout = build_layout(cfg)
    reset_scene()

    zone_mats = {z: get_material(ZONE_MAT[z]) for z in cfg["coupon"]["zones"]}
    defect_objects = {}
    plates = {}
    for zone in cfg["coupon"]["zones"]:
        plate = make_plate(zone, cfg)
        cutters, cutter_mat_to_prim = [], {}
        for d in [d for d in layout if d.zone == zone]:
            if d.type == "burr":
                defect_objects[d.prim] = make_burr(d, zone_mats[zone], float(cfg["coupon"].get("burr_root_mm", 0.01)))
                continue
            cmat = get_material("C_" + d.prim)
            cutter_mat_to_prim[cmat.name] = d.prim
            cutters.append(make_pore_cutter(d, cmat) if is_round(d.type) else make_scratch_cutter(d, cmat))
        # one boolean per zone: join all cutters first (they never overlap)
        select_only(cutters)
        bpy.ops.object.join()
        cutter = bpy.context.view_layer.objects.active
        apply_boolean(plate, cutter)
        plate, pieces = separate_by_material(plate, zone_mats[zone], cutter_mat_to_prim)
        missing = set(cutter_mat_to_prim.values()) - set(pieces)
        if missing:
            raise RuntimeError(f"boolean produced no cavity for: {sorted(missing)}")
        defect_objects.update(pieces)
        plates[zone] = plate
    # drop placeholder cutter materials
    for m in [m for m in bpy.data.materials if m.name.startswith("C_D_")]:
        bpy.data.materials.remove(m)

    uv_tile = float(cfg["coupon"].get("uv_tile_mm", 20))
    all_obs = list(plates.values()) + [defect_objects[d.prim] for d in layout]
    for ob in all_obs:
        finalize_mesh(ob, uv_tile)

    # defects.json (centres measured from the actual geometry, in mm, before scaling)
    records = []
    for d in layout:
        rec = d.to_record()
        rec["bbox_center_mm"] = [round(v, 6) for v in bbox_center_mm(defect_objects[d.prim])]
        rec["n_faces"] = len(defect_objects[d.prim].data.polygons)
        records.append(rec)
    with open(os.path.join(out_dir, "defects.json"), "w") as fh:
        json.dump(records, fh, indent=1)
    pw, ph, pt = (float(v) for v in cfg["coupon"]["plate_mm"])
    with open(os.path.join(out_dir, "part.json"), "w") as fh:
        json.dump({"kind": "coupon", "extent_mm": [pw, ph, pt], "bbox_min_mm": [-pw / 2, -ph / 2, -pt], "bbox_max_mm": [pw / 2, ph / 2, 0.0],
                   "zone_prims": {"cast": PLATE_NAME["cast"], "machined": PLATE_NAME["machined"]}, "reference_plane_z_mm": 0.0,
                   "source": "generated by blender/make_coupon.py"}, fh, indent=1)

    # scale mm -> m and apply, then export USD in metres
    for ob in all_obs:  # exact double-precision scale on the vertex data, not the transform operator
        ob.data.transform(Matrix.Diagonal((0.001, 0.001, 0.001, 1.0)))
        ob.data.update()
    usd_path = os.path.join(out_dir, "coupon.usd")
    bpy.ops.wm.usd_export(
        filepath=usd_path,
        selected_objects_only=False,
        export_animation=False,
        export_uvmaps=True,
        rename_uvmaps=True,
        export_normals=True,
        export_materials=True,
        generate_preview_surface=True,
        use_instancing=False,
        convert_scene_units="METERS",
        root_prim_path="/Coupon",
        merge_parent_xform=True,
        triangulate_meshes=False,
        export_custom_properties=False,
        relative_paths=True,
        export_lights=False,
        export_cameras=False,
    )
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, "coupon.blend"))
    n_def = sum(1 for o in bpy.data.objects if o.name.startswith("D_"))
    n_plate = sum(1 for o in bpy.data.objects if o.name.startswith("Coupon_"))
    print(f"[make_coupon] wrote {usd_path}: {n_def} defect objects, {n_plate} plates, {len(records)} records")


if __name__ == "__main__":
    main()
