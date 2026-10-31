"""Experiment 7: false alarms. Apply the same detection rule used for defects (local contrast >= 3
sigma over >= 3 connected pixels) to every non-defect pixel of a simulated image and count what it
flags, per zone and per 100 cm2 of surface.

  python analysis/false_alarms.py --config config/experiment.yaml [--conditions ...]

Local background = median filter (21 px); local noise = 1.4826 x MAD of the residual within the
zone; candidates = connected components of |residual| >= 3 sigma with >= 3 px, after removing the
dilated true-defect masks and a 6 px band along zone edges and part silhouette. Writes
out/false_alarms.csv and out/<condition>/false_alarms.png (overlay).
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
from ads.config import load_config, resolve_condition  # noqa: E402
from ads.layout import load_defects  # noqa: E402
from ads.camera import build_camera  # noqa: E402
from analysis.io_utils import write_png  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "config", "experiment.yaml"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "out"))
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--lens", default="none")
    a = ap.parse_args()
    cfg = load_config(a.config)
    thr = float(cfg["threshold"]["cnr"])
    min_area = int(cfg["threshold"]["min_area_px"])
    conds = a.conditions or [c for c, s in cfg["conditions"].items() if os.path.exists(os.path.join(a.out, c, "sim_0.png"))]
    import imageio.v3 as iio

    rows = []
    for cond in conds:
        cc = resolve_condition(cfg, cond)
        cam = build_camera(cc["camera"])
        cdir = os.path.join(a.out, cond)
        path = os.path.join(cdir, "sim_0.png" if a.lens == cfg["lens"].get("default_scenario", "none") else f"sim_0_{a.lens}.png")
        if not os.path.exists(path):
            continue
        img = iio.imread(path)
        if img.ndim == 3:
            img = img[..., 0]
        img = img.astype(float) / 255.0
        inst = np.load(os.path.join(cdir, "renders", "mask_0.npy"))
        sem = np.load(os.path.join(cdir, "renders", "zone_0.npy"))
        with open(os.path.join(cdir, "renders", "id_map.json")) as fh:
            idm = json.load(fh)
        defs = load_defects(cc)
        defect_ids = sum((idm["prim_to_ids"].get(d.prim, []) for d in defs), [])
        defect_mask = np.isin(inst, defect_ids)
        # defects too small to have a mask are still real: exclude a 7 x 7 window at each projected centre
        from ads.camera import project_mm
        for d in defs:
            if not idm["prim_to_ids"].get(d.prim):
                u, v = project_mm(cam, np.array([d.center3_mm]))[0]
                cu, cv = int(u), int(v)
                defect_mask[max(cv - 3, 0):cv + 4, max(cu - 3, 0):cu + 4] = True
        # the per-defect rule, applied densely: core = 3 x 3 mean, background = mean of an annulus
        # 3..10 px away, noise = that annulus' standard deviation; z = (core - bg) / std. Alarms are
        # connected components with |z| >= k. Using the identical statistic keeps the comparison with
        # the defects' CNR honest, including the effect of 8-bit quantisation on dark zones.
        from scipy.signal import fftconvolve
        r_in, r_out = (int(v) for v in cfg["analysis"]["bg_annulus_px"])
        yy, xx = np.mgrid[-r_out:r_out + 1, -r_out:r_out + 1]
        ring = ((np.maximum(np.abs(yy), np.abs(xx)) > r_in) & (np.maximum(np.abs(yy), np.abs(xx)) <= r_out)).astype(float)
        n_ring = ring.sum()
        bg_mean = fftconvolve(img, ring, mode="same") / n_ring
        bg_sq = fftconvolve(img * img, ring, mode="same") / n_ring
        bg_std = np.sqrt(np.maximum(bg_sq - bg_mean ** 2, 0) * n_ring / (n_ring - 1))
        core = ndimage.uniform_filter(img, size=3)
        zmap = (core - bg_mean) / np.maximum(bg_std, 1e-6)
        overlay = (np.stack([img] * 3, axis=-1) * 255).astype(np.uint8)
        ks = (3.0, 4.0, 5.0, 6.0, 8.0)
        for z in ("cast", "machined"):
            zone = np.isin(sem, idm["class_to_ids"].get(f"zone_{z}", []))
            valid = ndimage.binary_erosion(zone, iterations=r_out + 2) & ~ndimage.binary_dilation(defect_mask, iterations=r_out + 2)
            if valid.sum() < 1000:
                continue
            area_cm2 = float(valid.sum()) * cam.mm_per_px ** 2 / 100.0
            rec = {"condition": cond, "lens": a.lens, "zone": z, "mm_per_px": cam.mm_per_px, "area_cm2": area_cm2,
                   "median_bg_std": float(np.median(bg_std[valid])), "z_p999": float(np.percentile(np.abs(zmap[valid]), 99.9))}
            quiet_k = None
            for k in ks:
                cand = (np.abs(zmap) >= k) & valid
                n_fa = int(ndimage.label(cand, structure=np.ones((3, 3)))[1])
                rec[f"fa_per_100cm2_k{k:g}"] = 100.0 * n_fa / area_cm2 if area_cm2 else np.nan
                if quiet_k is None and rec[f"fa_per_100cm2_k{k:g}"] <= 1.0:
                    quiet_k = k
                if k == 4.0:
                    edge = ndimage.binary_dilation(cand, iterations=2) & ~ndimage.binary_dilation(cand, iterations=1)
                    overlay[edge] = [255, 60, 60]
            rec["quiet_threshold_k"] = quiet_k if quiet_k is not None else float("nan")
            rows.append(rec)
        ys, xs = np.where(np.isin(sem, idm["class_to_ids"].get("zone_cast", []) + idm["class_to_ids"].get("zone_machined", [])))
        if len(ys):
            pad = 20
            crop = overlay[max(ys.min() - pad, 0):ys.max() + pad, max(xs.min() - pad, 0):xs.max() + pad]
            write_png(os.path.join(cdir, "false_alarms.png"), crop)
    df = pd.DataFrame(rows)
    out_csv = os.path.join(a.out, "false_alarms.csv")
    if os.path.exists(out_csv) and a.conditions:
        old = pd.read_csv(out_csv)
        old = old[~old.condition.isin(conds)]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(out_csv, index=False)
    pd.set_option("display.width", 200)
    cols = ["condition", "zone", "mm_per_px", "area_cm2", "z_p999"] + [c for c in df.columns if c.startswith("fa_per_100cm2_k")] + ["quiet_threshold_k"]
    print(df[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
