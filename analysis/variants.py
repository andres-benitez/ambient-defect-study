"""Experiments 6, 8 and 9: compare variant conditions of the test plate against the ambient
baseline (tool marks, camera pose, a soft shadow). Per condition: smallest detectable size per type
and zone, median CNR ratio against the baseline (defects with a footprint), and counts.

  python analysis/variants.py --config config/experiment.yaml

Writes out/variants.csv and out/figures/variants_cnr_ratio.png.
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
from analysis.plot_results import style_axes, COLORS, INK, INK2, MUTED, SURFACE  # noqa: E402

GROUPS = {
    "tool marks": {"aniso_along": "anisotropic machined face, marks along the scratches", "aniso_across": "anisotropic machined face, marks across the scratches"},
    "camera pose": {"pose_tilt15": "camera tilted 15°", "ambient_oblique_012": "camera tilted 30°", "pose_near": "working distance −10 % (0.108 mm/px)", "pose_far": "working distance +10 % (0.132 mm/px)"},
    "shadow": {"shadow_012": "soft shadow from a 0.6 m fixture 1 m above the part"},
}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--config", default=os.path.join(REPO_ROOT, "config", "experiment.yaml"))
    a.add_argument("--out", default=os.path.join(REPO_ROOT, "out"))
    a.add_argument("--baseline", default="ambient_nadir_012")
    args = a.parse_args()
    cfg = load_config(args.config)
    thr = cfg["threshold"]
    m = pd.read_csv(os.path.join(args.out, "metrics.csv"))
    m = m[(m.frame == 0) & (m.lens == cfg["lens"].get("default_scenario", "none"))]
    m = m.assign(detectable=(m.cnr >= float(thr["cnr"])) & (m.area_px >= int(thr["min_area_px"])))
    base = m[m.condition == args.baseline].set_index("prim")
    rows = []
    for group, conds in GROUPS.items():
        for cond, label in conds.items():
            g = m[m.condition == cond]
            if g.empty:
                continue
            for (z, t), gg in g.groupby(["zone", "type"]):
                det = gg[gg.detectable]
                gi = gg.set_index("prim")
                common = gi.index.intersection(base.index)
                ratio = (gi.loc[common].cnr / base.loc[common].cnr).replace([np.inf, -np.inf], np.nan)
                fp = gi.loc[common].area_px > 0
                rows.append({"group": group, "condition": cond, "label": label, "zone": z, "type": t,
                             "min_detectable_mm": float(det.size_mm.min()) if len(det) else np.nan, "n_detectable": int(len(det)), "n_sizes": int(len(gg)),
                             "median_cnr_ratio": float(np.nanmedian(ratio[fp].values)) if fp.any() else np.nan,
                             "baseline_min_detectable_mm": float(base[(base.zone == z) & (base.type == t) & base.detectable].size_mm.min()) if len(base) else np.nan})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "variants.csv"), index=False)
    if df.empty:
        print("[variants] no variant conditions yet")
        return
    types = [t for t in ROW_ORDER if t in set(df.type)]
    conds = [c for g in GROUPS.values() for c in g if c in set(df.condition)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE, sharey=True)
    for ax, zone in zip(axes, ("machined", "cast")):
        style_axes(ax)
        x = np.arange(len(types))
        for i, c in enumerate(conds):
            vals = [df[(df.condition == c) & (df.zone == zone) & (df.type == t)].median_cnr_ratio.median() for t in types]
            ax.bar(x + (i - (len(conds) - 1) / 2) * 0.8 / len(conds), vals, width=0.8 / len(conds), label=GROUPS[[g for g in GROUPS if c in GROUPS[g]][0]][c], color=plt.cm.viridis(i / max(len(conds) - 1, 1)))
        ax.axhline(1, color=MUTED, linewidth=1)
        ax.set_xticks(x, [t.replace("_", " ") for t in types], fontsize=8)
        ax.set_title(f"{zone} face", loc="left", color=INK, fontsize=10)
    axes[0].set_ylabel("CNR relative to the ambient baseline (median over sizes)", color=INK2, fontsize=9)
    axes[1].legend(frameon=False, fontsize=7, labelcolor=INK2)
    fig.suptitle("Tool marks, camera pose and a soft shadow: contrast relative to the baseline view", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(args.out, "figures", "variants_cnr_ratio.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)
    pd.set_option("display.width", 220)
    print(df.pivot_table(index=["type", "zone"], columns="condition", values="min_detectable_mm", aggfunc="first").to_string())
    print(df.groupby("condition").median_cnr_ratio.median().round(2).to_string())


if __name__ == "__main__":
    main()
