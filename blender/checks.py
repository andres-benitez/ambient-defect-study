"""Geometry checks for Stage 1. Fails loudly (non-zero exit) on the first violated rule.

  blender -b -P blender/checks.py -- --config config/experiment.yaml --assets assets/coupon/

Checks:
  * exactly N defect objects (from the layout) + 2 plates in coupon.blend
  * every zone assembly (plate + its cavities, vertices merged) is watertight; every burr is watertight
  * plate bounding box = plate_mm +- plate_tol_mm
  * every defect's bbox centre matches its center_mm within center_tol_mm
  * coupon.usd re-imports with the same mesh object count, in metres
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import bmesh
import bpy

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from ads.config import load_config  # noqa: E402
from ads.layout import build_layout  # noqa: E402


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "config", "experiment.yaml"))
    ap.add_argument("--assets", default=os.path.join(REPO_ROOT, "assets", "coupon"))
    ap.add_argument("--condition", default=None)
    return ap.parse_args(argv)


def fail(msg):
    print("[checks] FAIL:", msg)
    sys.exit(1)


def bm_from_objects(obs, merge_dist=1e-5):
    bm = bmesh.new()
    for ob in obs:
        me = ob.data.copy()
        me.transform(ob.matrix_world)
        bm.from_mesh(me)
        bpy.data.meshes.remove(me)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=merge_dist)
    return bm


def is_watertight(bm):
    bad = [e for e in bm.edges if not e.is_manifold]
    nonc = [e for e in bm.edges if e.is_manifold and not e.is_contiguous]
    return len(bad) == 0 and len(nonc) == 0, len(bad), len(nonc)


def main():
    args = parse_args()
    cfg = load_config(args.config, args.condition)
    layout = build_layout(cfg)
    chk = cfg["checks"]
    blend = os.path.join(args.assets, "coupon.blend")
    usd = os.path.join(args.assets, "coupon.usd")
    with open(os.path.join(args.assets, "defects.json")) as fh:
        records = json.load(fh)

    bpy.ops.wm.open_mainfile(filepath=blend)
    objs = {o.name: o for o in bpy.data.objects if o.type == "MESH"}
    defects = [n for n in objs if n.startswith("D_")]
    plates = [n for n in objs if n.startswith("Coupon_")]
    if len(defects) != len(layout) or len(plates) != len(cfg["coupon"]["zones"]):
        fail(f"object count: {len(defects)} defects (want {len(layout)}), {len(plates)} plates (want {len(cfg['coupon']['zones'])})")
    if {d.prim for d in layout} != set(defects):
        fail(f"defect names differ from layout: {sorted(set(defects) ^ {d.prim for d in layout})}")
    if len(records) != len(layout):
        fail(f"defects.json has {len(records)} records, want {len(layout)}")

    # scale: the .blend is stored in metres (post-export state)
    SCALE = 1000.0  # m -> mm
    plate_mm = [float(v) for v in cfg["coupon"]["plate_mm"]]
    allv = [objs[p].matrix_world @ v.co for p in plates for v in objs[p].data.vertices]
    dims = [(max(v[i] for v in allv) - min(v[i] for v in allv)) * SCALE for i in range(3)]
    for got, want in zip(dims, plate_mm):
        if abs(got - want) > float(chk["plate_tol_mm"]):
            fail(f"plate bbox {dims} mm, want {plate_mm} +- {chk['plate_tol_mm']}")
    top_z = max(v[2] for v in allv) * SCALE
    if abs(top_z) > float(chk["plate_tol_mm"]):
        fail(f"plate top face at z={top_z} mm, want 0")

    # watertightness per zone assembly and per burr
    zone_of = {d.prim: d.zone for d in layout}
    plate_of_zone = {"cast": "Coupon_Cast", "machined": "Coupon_Machined"}
    for zone, pname in plate_of_zone.items():
        cav = [objs[n] for n in defects if zone_of[n] == zone and not n.split("_")[2] == "burr"]
        bm = bm_from_objects([objs[pname]] + cav, merge_dist=1e-7)
        ok, nbad, nnc = is_watertight(bm)
        nfaces = len(bm.faces)
        bm.free()
        if not ok:
            fail(f"zone {zone} assembly not watertight: {nbad} non-manifold edges, {nnc} non-contiguous")
        print(f"[checks] zone {zone}: plate+{len(cav)} cavities watertight ({nfaces} tris)")
    for n in defects:
        if "_burr_" in n:
            bm = bm_from_objects([objs[n]], merge_dist=1e-9)
            ok, nbad, nnc = is_watertight(bm)
            bm.free()
            if not ok:
                fail(f"burr {n} not watertight ({nbad} non-manifold edges)")

    # centres
    tol = float(chk["center_tol_mm"])
    for rec in records:
        ob = objs[rec["prim"]]
        vs = [ob.matrix_world @ v.co for v in ob.data.vertices]
        cx = (max(v[0] for v in vs) + min(v[0] for v in vs)) / 2 * SCALE
        cy = (max(v[1] for v in vs) + min(v[1] for v in vs)) / 2 * SCALE
        dx, dy = cx - rec["center_mm"][0], cy - rec["center_mm"][1]
        if abs(dx) > tol or abs(dy) > tol:
            fail(f"{rec['prim']}: bbox centre ({cx:.4f},{cy:.4f}) vs center_mm {rec['center_mm']} (tol {tol})")
        # material rule: exactly one slot, the zone material
        mats = [m.name for m in ob.data.materials]
        want = "M_cast" if rec["zone"] == "cast" else "M_machined"
        if mats != [want]:
            fail(f"{rec['prim']}: materials {mats}, want [{want}]")

    # USD round trip
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.usd_import(filepath=usd, import_materials=False, import_lights=False, import_cameras=False)
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if len(meshes) != len(layout) + len(plates):
        fail(f"USD re-import has {len(meshes)} meshes, want {len(layout) + len(plates)}")
    names = {o.name for o in meshes}
    if not {d.prim for d in layout} <= names:
        fail(f"USD re-import is missing defect prims: {sorted({d.prim for d in layout} - names)[:5]} ...")
    allv = [o.matrix_world @ v.co for o in meshes if o.name.startswith("Coupon_") for v in o.data.vertices]
    w = (max(v[0] for v in allv) - min(v[0] for v in allv))
    if abs(w - plate_mm[0] / 1000.0) > float(chk["plate_tol_mm"]) / 1000.0:
        fail(f"USD plate width {w} m, want {plate_mm[0]/1000} m")
    print(f"[checks] PASS: {len(defects)} defects + {len(plates)} plates, plate {dims[0]:.4f} x {dims[1]:.4f} x {dims[2]:.4f} mm, USD width {w:.6f} m")


if __name__ == "__main__":
    main()
