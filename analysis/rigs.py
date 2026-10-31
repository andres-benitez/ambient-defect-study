"""Experiment 5: lighting rigs as reported conditions, compared with constant ambient light on the
test plate. Also combines the four single-direction renders into a "best of four" result (the CNR a
sequential four-segment ring light would give, taking the best direction per defect).

  python analysis/rigs.py --config config/experiment.yaml

Writes out/rigs.csv, out/rigs_summary.csv and out/figures/rigs_*.png.
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from ads.config import load_config  # noqa: E402
from ads.layout import ROW_ORDER  # noqa: E402
from analysis.plot_results import style_axes, COLORS, INK, INK2, MUTED, GRID, SURFACE, SIZE_LABEL  # noqa: E402

RIG_LABEL = {"ambient_nadir_012": "ambient ceiling (baseline)", "rig_dome": "uniform dome (softest)", "rig_ring60": "ring light, 60° elevation",
             "rig_ring30": "ring light, 30° elevation", "best_of_4": "4-direction sequential, 30° (best of four)", "control_darkfield": "single light, 15° (dark-field control)"}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--config", default=os.path.join(REPO_ROOT, "config", "experiment.yaml"))
    a.add_argument("--out", default=os.path.join(REPO_ROOT, "out"))
    args = a.parse_args()
    cfg = load_config(args.config)
    thr = cfg["threshold"]
    m = pd.read_csv(os.path.join(args.out, "metrics.csv"))
    m = m[(m.frame == 0) & (m.lens == cfg["lens"].get("default_scenario", "none"))]
    dirs = [c for c in ("rig_dir_az0", "rig_dir_az90", "rig_dir_az180", "rig_dir_az270") if c in set(m.condition)]
    base = ["ambient_nadir_012", "rig_dome", "rig_ring60", "rig_ring30", "control_darkfield"]
    present = [c for c in base if c in set(m.condition)]
    df = m[m.condition.isin(present)].copy()
    if dirs:
        d4 = m[m.condition.isin(dirs)].copy()
        best = d4.sort_values("cnr", ascending=False).groupby("prim", as_index=False).first()
        best["condition"] = "best_of_4"
        df = pd.concat([df, best], ignore_index=True)
    df["detectable"] = (df.cnr >= float(thr["cnr"])) & (df.area_px >= int(thr["min_area_px"]))
    df.to_csv(os.path.join(args.out, "rigs.csv"), index=False)
    # summary: min detectable per rig/zone/type and median CNR ratio to ambient
    amb = df[df.condition == "ambient_nadir_012"].set_index("prim")
    rows = []
    for cond, g in df.groupby("condition"):
        for (z, t), gg in g.groupby(["zone", "type"]):
            det = gg[gg.detectable]
            ratio = (gg.set_index("prim").cnr / amb.cnr.reindex(gg.prim)).replace([np.inf, -np.inf], np.nan)
            rows.append({"condition": cond, "label": RIG_LABEL.get(cond, cond), "zone": z, "type": t,
                         "min_detectable_mm": float(det.size_mm.min()) if len(det) else np.nan, "n_detectable": int(len(det)),
                         "median_cnr_ratio_vs_ambient": float(np.nanmedian(ratio.values)) if len(ratio) else np.nan,
                         "saturated_bg": bool(gg.saturated_bg.any()) if "saturated_bg" in gg else False})
    s = pd.DataFrame(rows)
    s.to_csv(os.path.join(args.out, "rigs_summary.csv"), index=False)
    # figure: smallest detectable per rig (rows) and type (columns), cast and machined
    order = [c for c in ("ambient_nadir_012", "rig_dome", "rig_ring60", "rig_ring30", "best_of_4", "control_darkfield") if c in set(s.condition)]
    types = [t for t in ROW_ORDER if t in set(s.type)]
    fig, axes = plt.subplots(1, len(types), figsize=(3.4 * len(types), 4.6), facecolor=SURFACE, sharey=True)
    axes = np.atleast_1d(axes)
    y = np.arange(len(order))
    for ax, t in zip(axes, types):
        style_axes(ax)
        for zi, z in enumerate(("cast", "machined")):
            vals = [s[(s.condition == c) & (s.zone == z) & (s.type == t)].min_detectable_mm.iloc[0] if len(s[(s.condition == c) & (s.zone == z) & (s.type == t)]) else np.nan for c in order]
            ax.barh(y + (zi - 0.5) * 0.38, [v if np.isfinite(v) else 0 for v in vals], height=0.36, color=COLORS[z], label=z)
            for yy, v in zip(y + (zi - 0.5) * 0.38, vals):
                ax.text((v if np.isfinite(v) else 0) + 0.01, yy, f"{v:g} mm" if np.isfinite(v) else "none", va="center", fontsize=7, color=INK2)
        ax.set_title(f"{t.replace('_', ' ')} ({SIZE_LABEL.get(t, 'size')})", loc="left", color=INK, fontsize=10)
        ax.set_xlabel("smallest detectable, mm", color=INK2, fontsize=9)
        ax.set_xlim(0, max(1.2, float(np.nanmax(s[s.type == t].min_detectable_mm.fillna(0))) * 1.6))
    axes[0].set_yticks(y, [RIG_LABEL.get(c, c) for c in order], fontsize=8)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    fig.suptitle("Smallest detectable size per lighting rig (test plate, 5 MP class at 0.12 mm/px)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(args.out, "figures", "rigs_min_detectable.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)
    # figure: CNR ratio vs ambient per rig, by type (median over sizes with footprint)
    fig, ax = plt.subplots(figsize=(10, 4.4), facecolor=SURFACE)
    style_axes(ax)
    x = np.arange(len(types))
    for i, c in enumerate([c for c in order if c != "ambient_nadir_012"]):
        vals = [s[(s.condition == c) & (s.type == t)].median_cnr_ratio_vs_ambient.median() for t in types]
        ax.bar(x + (i - 2) * 0.15, vals, width=0.14, label=RIG_LABEL.get(c, c), color=plt.cm.viridis(i / 5))
    ax.axhline(1, color=MUTED, linewidth=1)
    ax.set_yscale("log")
    ax.set_xticks(x, [t.replace("_", " ") for t in types])
    ax.set_ylabel("CNR relative to the ambient ceiling (median)", color=INK2, fontsize=9)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    ax.set_title("What each rig buys relative to ambient light", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "figures", "rigs_cnr_ratio.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)
    pd.set_option("display.width", 220)
    print(s.pivot_table(index=["type", "zone"], columns="condition", values="min_detectable_mm", aggfunc="first").to_string())


if __name__ == "__main__":
    main()
