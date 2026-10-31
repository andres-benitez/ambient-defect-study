"""Stage 4a: simulated camera image + per-defect metrics -> out/metrics.csv (appended per condition)

  python analysis/compute_cnr.py --config config/experiment.yaml --condition ambient_nadir_012

For every frame: luminance -> gain (solved once from frame 0 so the machined background sits at
sensor.target_bg_level) -> shot/read noise -> N-bit quantisation -> sim_<f>.png; then every defect
is measured against its own local background. Also runs the exposure and mask checks and writes
mask_overlay.png and preview_<f>.png.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import ndimage

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from ads.config import load_config  # noqa: E402
from ads.layout import build_layout  # noqa: E402
from ads.camera import build_camera, project_mm  # noqa: E402
from analysis.sensor_model import luminance, solve_gain, simulate, to_uint8, expected_noise_std  # noqa: E402
from analysis.metrics import measure_defect, dilate  # noqa: E402
from analysis.io_utils import load_hdr, tonemap_preview, write_png, convert_npy_renders_to_exr  # noqa: E402
from analysis.render_noise import flat_patch_std  # noqa: E402

ZONE_CLASS = {"cast": "zone_cast", "machined": "zone_machined"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "config", "experiment.yaml"))
    ap.add_argument("--condition", default=None)
    ap.add_argument("--out", default=None, help="condition dir (default out/<condition>)")
    ap.add_argument("--metrics", default=os.path.join(REPO_ROOT, "out", "metrics.csv"))
    ap.add_argument("--no-checks", action="store_true", help="report instead of fail on exposure/mask checks")
    return ap.parse_args()


def ids_for(id_map: dict, kind: str, key: str) -> list:
    table = id_map["prim_to_ids"] if kind == "prim" else id_map["class_to_ids"]
    return [int(i) for i in table.get(key, [])]


def mask_of(arr: np.ndarray, ids) -> np.ndarray:
    m = np.zeros(arr.shape, bool)
    for i in ids:
        m |= arr == i
    return m


def main():
    args = parse_args()
    cfg = load_config(args.config, args.condition)
    cond = cfg["condition"]["name"]
    is_control = bool(cfg["condition"]["control"])
    out_dir = os.path.abspath(args.out or os.path.join(REPO_ROOT, "out", cond))
    renders = os.path.join(out_dir, "renders")
    with open(os.path.join(out_dir, "frame_manifest.json")) as fh:
        manifest = json.load(fh)
    with open(os.path.join(renders, "id_map.json")) as fh:
        id_map = json.load(fh)
    with open(os.path.join(REPO_ROOT, "assets", "coupon", "defects.json")) as fh:
        records = json.load(fh)
    layout = {d.prim: d for d in build_layout(cfg)}
    cam = build_camera(cfg["camera"])
    convert_npy_renders_to_exr(renders)
    n_frames = int(manifest["frames"])
    sensor, thr, an, chk = cfg["sensor"], cfg["threshold"], cfg["analysis"], cfg["checks"]
    checks = {}

    def fail(name, msg):
        checks.setdefault(name, {})
        checks[name].update({"pass": False, "detail": msg})
        if args.no_checks or (is_control and name == "exposure"):
            print(f"[compute_cnr] WARN {name} ({'expected for a control condition' if is_control else 'ignored'}): {msg}")
        else:
            raise SystemExit(f"[compute_cnr] FAIL {name}: {msg}")

    # ---- masks (frame 0; the scene is static so masks are identical across frames)
    inst0 = np.load(os.path.join(renders, "mask_0.npy"))
    sem0 = np.load(os.path.join(renders, "zone_0.npy"))
    zone_masks = {z: mask_of(sem0, ids_for(id_map, "class", ZONE_CLASS[z])) for z in cfg["coupon"]["zones"]}
    defect_class = mask_of(sem0, ids_for(id_map, "class", "defect"))
    defect_masks = {prim: mask_of(inst0, ids_for(id_map, "prim", prim)) for prim in layout}
    # a defect's zone mask must include the defect pixels themselves (labelled "defect" semantically)
    for z in zone_masks:
        zone_masks[z] = zone_masks[z] | (defect_class & mask_of(inst0, sum((ids_for(id_map, "prim", p) for p, d in layout.items() if d.zone == z), [])))
    dil = int(an.get("mask_dilate_px", 1))
    dilated = {prim: dilate(m, dil + 1) for prim, m in defect_masks.items()}
    all_defects_dil = np.zeros(inst0.shape, bool)
    for m in dilated.values():
        all_defects_dil |= m

    # mask check: every defect that should cover >= 1 px appears in the id map
    centers = {prim: project_mm(cam, np.array([[d.center_mm[0], d.center_mm[1], 0.0]]))[0] for prim, d in layout.items()}
    missing_big = []
    for prim, d in layout.items():
        footprint_px = (d.size_mm if d.type.startswith("pore") else max(d.width_mm, 0) ) / cam.mm_per_px
        if prim not in id_map["prim_to_ids"] and footprint_px >= 2.0:
            missing_big.append(prim)
    if missing_big:
        fail("masks", f"defects with footprint >= 2 px missing from id_map.json: {missing_big}")
    # mask alignment: centroid vs projected centre for defects with area >= 9 px. Raised burrs sit
    # above z = 0, so their apex parallaxes outward by height x off-axis distance / working distance;
    # that amount is allowed on top of the 2 px tolerance.
    # Only defects at least 2 px wide are checked: a sub-pixel-wide line rasterises to a sparse
    # set of pixels whose centroid wanders along its length.
    offsets, excess = [], []
    for prim, m in defect_masks.items():
        d0 = layout[prim]
        width_px = (d0.size_mm if d0.type.startswith("pore") else d0.width_mm) / cam.mm_per_px
        if m.sum() >= 9 and width_px >= 2.0:
            cy, cx = ndimage.center_of_mass(m)
            u, v = centers[prim]
            off = float(np.hypot(cx + 0.5 - u, cy + 0.5 - v))
            d = layout[prim]
            height_mm = d.size_mm if d.type == "burr" else 0.0
            parallax_px = height_mm * float(np.hypot(*d.center_mm)) / cam.distance_mm / cam.mm_per_px
            offsets.append(off)
            excess.append(off - parallax_px)
    max_off = max(offsets) if offsets else float("nan")
    max_excess = max(excess) if excess else float("nan")
    if excess and max_excess > 2.0:
        fail("mask_alignment", f"mask centroid vs projected centre differs by up to {max_excess:.2f} px beyond the expected burr parallax")
    checks["masks"] = {"pass": not missing_big, "n_prims_in_id_map": sum(p in id_map["prim_to_ids"] for p in layout),
                       "n_defects": len(layout), "missing": [p for p in layout if p not in id_map["prim_to_ids"]],
                       "max_centroid_offset_px": max_off, "max_offset_beyond_parallax_px": max_excess, "n_aligned_checked": len(offsets)}

    # ---- gain from frame 0
    Y0 = luminance(load_hdr(renders, 0))
    bg_mach = zone_masks["machined"] & ~all_defects_dil & ~ndimage.binary_dilation(defect_class, iterations=4)
    gain = solve_gain(Y0, bg_mach, float(sensor["target_bg_level"]))
    noise = flat_patch_std(Y0 * gain, bg_mach, seed=int(cfg["seed"]))
    noise["sensor_std"] = expected_noise_std(noise["level"], sensor)
    noise["ratio"] = noise["std"] / noise["sensor_std"]
    checks["render_noise"] = {"pass": noise["ratio"] <= float(chk["render_noise_frac"]), **noise}
    if noise["ratio"] > float(chk["render_noise_frac"]):
        fail("render_noise", f"render noise {noise['std']:.5f} is {noise['ratio']:.2f}x the sensor noise (limit {chk['render_noise_frac']})")

    rows = []
    rng_base = int(cfg["seed"])
    plate_mask = zone_masks["cast"] | zone_masks["machined"] | defect_class
    for f in range(n_frames):
        Y = luminance(load_hdr(renders, f)) if f else Y0
        rng = np.random.default_rng(rng_base * 1000 + f)
        sim = simulate(Y, gain, sensor, rng)
        write_png(os.path.join(out_dir, f"sim_{f}.png"), to_uint8(sim))
        write_png(os.path.join(out_dir, f"preview_{f}.png"), tonemap_preview(load_hdr(renders, f), float(sensor["target_bg_level"]), bg_mach))
        # exposure check (frame 0)
        if f == 0:
            bg_mean = float(sim[bg_mach].mean())
            sat = float((sim[plate_mask] >= 1.0).mean())
            lo, hi = (float(v) for v in chk["exposure_range"])
            ok = lo <= bg_mean <= hi and sat < float(chk["max_saturated_frac"])
            checks["exposure"] = {"pass": ok, "machined_bg_mean": bg_mean, "saturated_frac": sat, "gain": gain}
            if not ok:
                fail("exposure", f"machined bg mean {bg_mean:.3f} (want {lo}-{hi}), saturated fraction {sat:.5f}")
        for prim, d in layout.items():
            others = all_defects_dil & ~dilated[prim]
            try:
                r = measure_defect(sim, defect_masks[prim], zone_masks[d.zone], others, an, thr, fallback_center_uv=centers[prim])
            except ValueError as exc:
                r = {"area_px": int(defect_masks[prim].sum()), "n_bg_px": 0, "mean_in": np.nan, "mean_bg": np.nan, "std_bg": np.nan,
                     "cnr": np.nan, "cnr_peak": np.nan, "weber": np.nan, "detectable": False, "marginal": False, "mask_empty": True, "error": str(exc)}
            r.update({
                "condition": cond, "control": is_control, "frame": f, "prim": prim, "zone": d.zone, "type": d.type,
                "size_mm": d.size_mm, "depth_mm": d.depth_mm, "length_mm": d.length_mm, "width_mm": d.width_mm,
                "row": d.row, "col": d.col, "center_u_px": float(centers[prim][0]), "center_v_px": float(centers[prim][1]),
                "mm_per_px": cam.mm_per_px, "view": cam.view, "working_distance_mm": cam.distance_mm,
                "lighting_mode": manifest["lighting_mode"], "panel_intensity": manifest["solved_intensity"].get("panel"),
                "dome_intensity": manifest["solved_intensity"].get("dome"), "directional_intensity": manifest["solved_intensity"].get("directional"),
                "spp": manifest["spp"], "gain": gain, "seed": cfg["seed"],
            })
            rows.append(r)
    df = pd.DataFrame(rows)

    # positive-control check: under the raking light every machined-zone scratch and burr must
    # cross the threshold (mean CNR when it has a mask, peak CNR when it is sub-pixel). If not, the
    # geometry, masks or metric are wrong and no ambient result can be trusted.
    if is_control:
        f0 = df[(df.frame == 0) & (df.zone == "machined") & (df.type.isin(["scratch", "burr"]))]
        thr_cnr = float(thr["cnr"])
        rows_pc = []
        for r in f0.itertuples():
            ok = (r.cnr >= thr_cnr) if r.area_px > 0 else (max(r.cnr, r.cnr_peak) >= thr_cnr)
            rows_pc.append({"prim": r.prim, "area_px": int(r.area_px), "cnr": float(r.cnr), "cnr_peak": float(r.cnr_peak), "pass": bool(ok)})
        pc_pass = all(x["pass"] for x in rows_pc)
        checks["positive_control"] = {"pass": pc_pass, "rule": "machined scratches/burrs: cnr >= thr if masked, else max(cnr, cnr_peak) >= thr", "rows": rows_pc}
        if not pc_pass:
            fail("positive_control", "defects failing: " + ", ".join(x["prim"] for x in rows_pc if not x["pass"]))
        else:
            print(f"[compute_cnr] positive control passed: {len(rows_pc)} machined scratches/burrs cross CNR {thr_cnr:g} under the raking light")

    cols = ["condition", "frame", "prim", "zone", "type", "size_mm", "depth_mm", "length_mm", "width_mm", "row", "col",
            "area_px", "mean_in", "mean_bg", "std_bg", "cnr", "cnr_peak", "weber", "detectable", "marginal", "mask_empty", "saturated_bg",
            "n_bg_px", "center_u_px", "center_v_px", "mm_per_px", "view", "working_distance_mm", "lighting_mode",
            "panel_intensity", "dome_intensity", "directional_intensity", "spp", "gain", "seed", "control"]
    df = df[[c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]]

    # ---- mask overlay on the preview
    prev = tonemap_preview(load_hdr(renders, 0), float(sensor["target_bg_level"]), bg_mach)
    overlay = prev.copy()
    edges = np.zeros(inst0.shape, bool)
    for prim, m in defect_masks.items():
        if m.any():
            edges |= dilate(m, 1) & ~m
    overlay[edges] = [255, 40, 40]
    def hollow_square(img, cu, cv, r, color):
        h, w = img.shape[:2]
        u0, u1, v0, v1 = max(cu - r, 0), min(cu + r, w - 1), max(cv - r, 0), min(cv + r, h - 1)
        img[v0, u0:u1 + 1] = color
        img[v1, u0:u1 + 1] = color
        img[v0:v1 + 1, u0] = color
        img[v0:v1 + 1, u1] = color

    for prim in layout:
        if not defect_masks[prim].any():  # sub-pixel defect: mark the fallback window in blue
            u, v = centers[prim]
            hollow_square(overlay, int(np.floor(u)), int(np.floor(v)), 4, [40, 120, 255])
    ys, xs = np.where(plate_mask)
    pad = 40
    crop = overlay[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]
    write_png(os.path.join(out_dir, "mask_overlay.png"), crop)
    checks["masks"]["overlay"] = os.path.relpath(os.path.join(out_dir, "mask_overlay.png"), REPO_ROOT)

    # ---- append to metrics.csv (replace rows of this condition)
    os.makedirs(os.path.dirname(args.metrics), exist_ok=True)
    if os.path.exists(args.metrics):
        old = pd.read_csv(args.metrics)
        old = old[old["condition"] != cond]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(args.metrics, index=False)
    with open(os.path.join(out_dir, "analysis_checks.json"), "w") as fh:
        json.dump(checks, fh, indent=1, default=float)
    this = df[df["condition"] == cond]
    print(f"[compute_cnr] {cond}: {len(this)} rows ({n_frames} frames x {len(layout)} defects), gain {gain:.4f}, "
          f"bg {checks['exposure']['machined_bg_mean']:.3f}, render-noise ratio {noise['ratio']:.2f}, "
          f"detectable (frame 0): {int(this[this.frame == 0].detectable.sum())}/{len(layout)}")


if __name__ == "__main__":
    main()
