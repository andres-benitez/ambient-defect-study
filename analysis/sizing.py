"""Experiment 4: dimensional quantification. Measure each defect's size from the simulated camera
image and compare with the true geometry.

  python analysis/sizing.py --config config/experiment.yaml [--conditions a b ...]

Method per defect: in a window around the projected centre, estimate the local background (median
of the annulus used for CNR), compute the contrast image |img - bg|, threshold it at half the peak
contrast inside the window (a full-width-at-half-maximum boundary, the usual way a vision system
delineates a blob), keep the connected component at the centre, and report its equivalent diameter
(sqrt(4 A / pi)) and its axis-aligned bounding box in millimetres. Truth: the opening diameter for
round defects, length x width for scratches and burrs. Written to out/sizing.csv.
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
from ads.layout import load_defects, is_round  # noqa: E402
from ads.camera import build_camera, project_mm  # noqa: E402


def measure(img: np.ndarray, cu: float, cv: float, half: int, bg_ring=(3, 10)) -> dict | None:
    h, w = img.shape
    u0, u1 = max(int(cu) - half, 0), min(int(cu) + half + 1, w)
    v0, v1 = max(int(cv) - half, 0), min(int(cv) + half + 1, h)
    win = img[v0:v1, u0:u1].astype(float)
    if win.size < 25:
        return None
    # background: the window's outer ring
    ring = np.ones_like(win, bool)
    ring[bg_ring[0]:-bg_ring[0], bg_ring[0]:-bg_ring[0]] = False
    bg = float(np.median(win[ring]))
    sigma = float(1.4826 * np.median(np.abs(win[ring] - bg))) or 1e-6
    contrast = np.abs(win - bg)
    cy, cx = int(cv) - v0, int(cu) - u0
    core = contrast[max(cy - 2, 0):cy + 3, max(cx - 2, 0):cx + 3]
    peak = float(core.max())
    if peak < 3 * sigma:
        return {"found": False, "peak_cnr": peak / sigma}
    mask = contrast >= 0.5 * peak
    lab, n = ndimage.label(mask)
    if n == 0:
        return {"found": False, "peak_cnr": peak / sigma}
    # component containing or nearest to the centre
    lc = lab[cy, cx]
    if lc == 0:
        ys, xs = np.where(lab > 0)
        k = np.argmin((ys - cy) ** 2 + (xs - cx) ** 2)
        lc = lab[ys[k], xs[k]]
    comp = lab == lc
    ys, xs = np.where(comp)
    area = int(comp.sum())
    return {"found": True, "peak_cnr": peak / sigma, "area_px": area, "eq_diam_px": float(np.sqrt(4 * area / np.pi)),
            "bbox_w_px": float(xs.max() - xs.min() + 1), "bbox_h_px": float(ys.max() - ys.min() + 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "config", "experiment.yaml"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "out"))
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--lenses", nargs="*", default=["none", "good"])
    a = ap.parse_args()
    cfg = load_config(a.config)
    conds = a.conditions or [c for c, s in cfg["conditions"].items() if not (s or {}).get("control") and os.path.exists(os.path.join(a.out, c, "sim_0.png"))]
    import imageio.v3 as iio

    rows = []
    for cond in conds:
        cc = resolve_condition(cfg, cond)
        cam = build_camera(cc["camera"])
        defs = load_defects(cc)
        for lens in a.lenses:
            path = os.path.join(a.out, cond, "sim_0.png" if lens == cfg["lens"].get("default_scenario", "none") else f"sim_0_{lens}.png")
            if not os.path.exists(path):
                continue
            img = iio.imread(path)
            if img.ndim == 3:
                img = img[..., 0]
            img = img.astype(float) / 255.0
            for d in defs:
                u, v = project_mm(cam, np.array([d.center3_mm]))[0]
                if d.type.startswith("scratch") or d.type == "burr":
                    true_len, true_wid = float(d.length_mm), float(d.width_mm)
                    half = int(max(true_len, 1.0) / cam.mm_per_px / 2) + 12
                else:
                    true_len = true_wid = float(d.size_mm)
                    half = int(true_len / cam.mm_per_px / 2) + 12
                r = measure(img, u, v, half)
                if r is None:
                    continue
                rec = {"condition": cond, "lens": lens, "prim": d.prim, "zone": d.zone, "type": d.type, "size_mm": d.size_mm,
                       "mm_per_px": cam.mm_per_px, "true_len_mm": true_len, "true_wid_mm": true_wid,
                       "true_eq_diam_mm": float(np.sqrt(true_len * true_wid)) if not is_round(d.type) else true_len, "found": r["found"], "peak_cnr": r["peak_cnr"]}
                if r["found"]:
                    s = cam.mm_per_px
                    rec.update({"meas_eq_diam_mm": r["eq_diam_px"] * s, "meas_len_mm": max(r["bbox_w_px"], r["bbox_h_px"]) * s,
                                "meas_wid_mm": min(r["bbox_w_px"], r["bbox_h_px"]) * s, "area_px": r["area_px"]})
                    rec["eq_diam_err_pct"] = 100.0 * (rec["meas_eq_diam_mm"] - rec["true_eq_diam_mm"]) / rec["true_eq_diam_mm"]
                    rec["len_err_pct"] = 100.0 * (rec["meas_len_mm"] - true_len) / true_len
                    rec["wid_err_mm"] = rec["meas_wid_mm"] - true_wid
                rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "sizing.csv"), index=False)
    f = df[df.found == True]  # noqa: E712
    print(f"[sizing] {len(df)} measurements, {len(f)} found; conditions: {sorted(df.condition.unique())}")
    if len(f):
        # figure: sizing error against the defect's true size in pixels, per lens scenario
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from analysis.plot_results import style_axes, COLORS, INK, INK2, MUTED, SURFACE
        f = f.assign(true_px=f.true_eq_diam_mm / f.mm_per_px)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=SURFACE)
        for ax, (fam, label) in zip(axes, [(("pore_shallow", "pore_deep", "dent_shallow", "dent_deep"), "round defects: equivalent diameter"),
                                           (("scratch", "scratch_l1", "scratch_l2", "scratch_l3", "burr"), "linear defects: length")]):
            style_axes(ax)
            g = f[f.type.isin(fam)]
            col = "eq_diam_err_pct" if "round" in label else "len_err_pct"
            for lens, mk in (("none", "o"), ("good", "s")):
                gg = g[g.lens == lens]
                if gg.empty:
                    continue
                bins = [0, 1.5, 2.5, 4, 6, 10, 20, 60]
                gg = gg.assign(b=pd.cut(gg.true_px, bins))
                st = gg.groupby("b", observed=True)[col].agg(["median", lambda x: np.percentile(np.abs(x), 90)])
                xs = [b.mid for b in st.index]
                ax.plot(xs, st["median"], marker=mk, color=COLORS["machined"], linewidth=1.8, label=f"median error, {lens} lens")
                ax.plot(xs, st["<lambda_0>"], marker=mk, linestyle="--", color=COLORS["cast"], linewidth=1.5, label=f"90th pct |error|, {lens} lens")
            ax.axhline(0, color=MUTED, linewidth=1)
            ax.axhline(10, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
            ax.set_xscale("log")
            ax.set_xlabel("true size, pixels across", color=INK2, fontsize=9)
            ax.set_ylabel("measurement error, %", color=INK2, fontsize=9)
            ax.set_title(label, loc="left", color=INK, fontsize=10)
            ax.set_ylim(-60, 160)
        axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2)
        fig.suptitle("How well the size of a defect can be measured from the image, against its size in pixels (dashed: ±10 %)", color=INK, fontsize=11, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
        fig.savefig(os.path.join(a.out, "figures", "sizing_error.png"), dpi=160, facecolor=SURFACE)
        plt.close(fig)
        pd.set_option("display.width", 200)
        g = f.groupby(["condition", "lens", "type"]).agg(n=("prim", "size"), med_eq_err=("eq_diam_err_pct", "median"), p90_abs_eq_err=("eq_diam_err_pct", lambda x: np.percentile(np.abs(x), 90)),
                                                        med_len_err=("len_err_pct", "median")).round(1)
        print(g.to_string())


if __name__ == "__main__":
    main()
