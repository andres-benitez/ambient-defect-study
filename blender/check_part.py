"""Geometry checks for a real-part asset built by blender/make_part.py. Fails loudly.

  blender -b -P blender/check_part.py -- --config config/experiment.yaml --assets assets/crankcase/

Checks: 48 defect objects + 2 zone objects in the .blend; every defect has faces; every defect's
bounding-box centre matches its recorded centre within `checks.part_center_tol_mm` (defects on
tilted cast surfaces are allowed the extra offset of their tilt); every defect carries exactly its
zone material; the USD re-imports with the same mesh count and the recorded extent.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import bpy
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from ads.config import load_config  # noqa: E402


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "config", "experiment.yaml"))
    ap.add_argument("--assets", default=os.path.join(REPO_ROOT, "assets", "crankcase"))
    ap.add_argument("--condition", default="part_5mp")
    return ap.parse_args(argv)


def fail(msg):
    print("[check_part] FAIL:", msg)
    sys.exit(1)


def main():
    a = parse_args()
    cfg = load_config(a.config, a.condition)
    tol = float(cfg["checks"].get("part_center_tol_mm", 0.5))
    with open(os.path.join(a.assets, "defects.json")) as fh:
        records = json.load(fh)
    with open(os.path.join(a.assets, "part.json")) as fh:
        part = json.load(fh)
    zone_prims = part["zone_prims"]
    blend = os.path.join(a.assets, "crankcase.blend")
    bpy.ops.wm.open_mainfile(filepath=blend)
    objs = {o.name: o for o in bpy.data.objects if o.type == "MESH"}
    defects = [n for n in objs if n.startswith("D_")]
    zones = [n for n in objs if n in zone_prims.values()]
    if len(defects) != len(records) or len(zones) != 2:
        fail(f"{len(defects)} defect objects (want {len(records)}), {len(zones)} zone objects (want 2)")
    SCALE = 1000.0
    for rec in records:
        ob = objs.get(rec["prim"])
        if ob is None:
            fail(f"missing object {rec['prim']}")
        if len(ob.data.polygons) == 0:
            fail(f"{rec['prim']} has no faces")
        vs = np.array([(ob.matrix_world @ v.co)[:] for v in ob.data.vertices]) * SCALE
        c = 0.5 * (vs.max(0) + vs.min(0))
        want = np.array(rec["center_mm"], float)
        n = np.array(rec.get("normal", [0, 0, 1]), float)
        # along the surface normal the cavity centre sits below the surface (pores) or above (burrs)
        off = c - want
        lateral = np.linalg.norm(off - n * (off @ n))
        if lateral > tol:
            fail(f"{rec['prim']}: lateral centre offset {lateral:.3f} mm > {tol} (bbox centre {np.round(c, 3)}, recorded {np.round(want, 3)})")
        mats = [m.name for m in ob.data.materials]
        want_m = "M_cast" if rec["zone"] == "cast" else "M_machined"
        if mats != [want_m]:
            fail(f"{rec['prim']}: materials {mats}, want [{want_m}]")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    usd = os.path.join(a.assets, cfg["asset"].get("usd", "crankcase.usd"))
    bpy.ops.wm.usd_import(filepath=usd, import_materials=False, import_lights=False, import_cameras=False)
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    if len(meshes) != len(records) + 2:
        fail(f"USD re-import has {len(meshes)} meshes, want {len(records) + 2}")
    allv = np.array([(o.matrix_world @ v.co)[:] for o in meshes if o.name in zone_prims.values() for v in o.data.vertices])
    ext = (allv.max(0) - allv.min(0)) * SCALE
    want_ext = np.array(part["extent_mm"])
    if np.any(np.abs(ext - want_ext) > 0.5):
        fail(f"USD extent {np.round(ext, 2)} mm, part.json says {np.round(want_ext, 2)}")
    print(f"[check_part] PASS: {len(defects)} defects + 2 zones, extent {np.round(ext, 1)} mm, "
          f"source '{part['source']['name']}' by {part['source']['author']} ({part['source']['license']})")


if __name__ == "__main__":
    main()
