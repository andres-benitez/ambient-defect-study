"""Experiments 1 and 2 on the extended test plate: dents and the scratch envelope.

  python analysis/ext_analysis.py --config config/experiment.yaml

Writes out/figures/dents.png (CNR against dent diameter for two depths, both surfaces, two pixel
scales), out/figures/scratch_envelope.png (CNR grid of scratch length x width with the client's
2 x 1 mm envelope marked) and out/ext_summary.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from ads.config import load_config  # noqa: E402
from analysis.plot_results import style_axes, COLORS, INK, INK2, MUTED, GRID, SURFACE  # noqa: E402

SCRATCH_LEN = {"scratch_l1": 1.0, "scratch_l2": 2.0, "scratch_l3": 3.0, "scratch": 5.0}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--config", default=os.path.join(REPO_ROOT, "config", "experiment.yaml"))
    a.add_argument("--out", default=os.path.join(REPO_ROOT, "out"))
    args = a.parse_args()
    cfg = load_config(args.config)
    thr = float(cfg["threshold"]["cnr"])
    min_area = int(cfg["threshold"]["min_area_px"])
    m = pd.read_csv(os.path.join(args.out, "metrics.csv"))
    m = m[(m.frame == 0) & (m.lens == cfg["lens"].get("default_scenario", "none")) & (m.condition.isin(["ext_nadir_012", "ext_nadir_005", "ext_ring60_012", "ext_ring30_012", "ext_dir_012"]))]
    if m.empty:
        raise SystemExit("no ext_* conditions in metrics.csv")
    m = m.assign(detectable=(m.cnr >= thr) & (m.area_px >= min_area))
    summary = {}
    # ---- dents
    d = m[m.type.isin(["dent_shallow", "dent_deep"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=SURFACE, sharey=True)
    for ax, cond in zip(axes, ["ext_nadir_012", "ext_nadir_005"]):
        style_axes(ax)
        g0 = d[d.condition == cond]
        if g0.empty:
            continue
        for zone in ("cast", "machined"):
            for typ, ls in (("dent_shallow", "-"), ("dent_deep", "--")):
                g = g0[(g0.zone == zone) & (g0.type == typ)].sort_values("size_mm")
                ax.plot(g.size_mm, g.cnr, ls, marker="o", color=COLORS[zone], linewidth=2, markersize=5,
                        label=f"{zone}, {'30 µm/mm' if typ == 'dent_shallow' else '100 µm/mm'} deep")
        ax.axhline(thr, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
        ax.set_xscale("log")
        ax.set_xticks([0.5, 1, 1.5, 2, 3, 4], ["0.5", "1", "1.5", "2", "3", "4"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("dent diameter, mm", color=INK2, fontsize=9)
        ax.set_title(f"{float(g0.mm_per_px.iloc[0])*1000:.0f} µm per pixel", loc="left", color=INK, fontsize=10)
        for typ in ("dent_shallow", "dent_deep"):
            for zone in ("cast", "machined"):
                g = g0[(g0.zone == zone) & (g0.type == typ) & g0.detectable]
                summary[f"{cond}/{typ}/{zone}/min_detectable_mm"] = float(g.size_mm.min()) if len(g) else None
    axes[0].set_ylabel("CNR", color=INK2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle("Dents under constant ambient light: shallow spherical depressions (depth 3 % and 10 % of the diameter)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(args.out, "figures", "dents.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)
    # ---- dents under lighting rigs: the same plate under a 30° ring and a single 15° light
    rig_labels = {"ext_nadir_012": "ambient ceiling", "ext_ring60_012": "ring light, 60° elevation", "ext_ring30_012": "ring light, 30° elevation", "ext_dir_012": "single light, 15° (dark field)"}
    rig_conds = [c for c in rig_labels if c in set(m.condition)]
    if len(rig_conds) > 1:
        dr = m[m.type.isin(["dent_shallow", "dent_deep"]) & m.condition.isin(rig_conds)]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), facecolor=SURFACE, sharey=True)
        for ax, zone in zip(axes, ("machined", "cast")):
            style_axes(ax)
            for i, cond in enumerate(rig_conds):
                for typ, ls in (("dent_shallow", "-"), ("dent_deep", "--")):
                    g = dr[(dr.condition == cond) & (dr.zone == zone) & (dr.type == typ)].sort_values("size_mm")
                    if g.empty:
                        continue
                    ax.plot(g.size_mm, np.maximum(g.cnr, 0.1), ls, marker="o", color=plt.cm.viridis(i / max(len(rig_conds) - 1, 1)), linewidth=2, markersize=5,
                            label=f"{rig_labels[cond]}, {'3 %' if typ == 'dent_shallow' else '10 %'} deep")
            ax.axhline(thr, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks([0.5, 1, 1.5, 2, 3, 4], ["0.5", "1", "1.5", "2", "3", "4"])
            ax.set_yticks([0.3, 1, 3, 10, 30, 100], ["0.3", "1", "3", "10", "30", "100"])
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_xlabel("dent diameter, mm", color=INK2, fontsize=9)
            ax.set_title(f"{zone} face, 120 µm per pixel", loc="left", color=INK, fontsize=10)
            for cond in rig_conds:
                for typ in ("dent_shallow", "dent_deep"):
                    g = dr[(dr.condition == cond) & (dr.zone == zone) & (dr.type == typ) & dr.detectable]
                    summary[f"{cond}/{typ}/{zone}/min_detectable_mm"] = float(g.size_mm.min()) if len(g) else None
            amb = dr[(dr.condition == "ext_nadir_012") & (dr.zone == zone)].set_index("prim")
            for cond in rig_conds:
                g = dr[(dr.condition == cond) & (dr.zone == zone) & (dr.area_px > 0)].set_index("prim")
                ratio = (g.cnr / amb.cnr.reindex(g.index)).replace([np.inf, -np.inf], np.nan)
                summary[f"{cond}/dents/{zone}/median_cnr_ratio_vs_ambient"] = float(np.nanmedian(ratio.values)) if len(ratio) and np.isfinite(ratio).any() else None
                summary[f"{cond}/dents/{zone}/n_detectable"] = int(dr[(dr.condition == cond) & (dr.zone == zone) & dr.detectable].shape[0])
        axes[0].set_ylabel("CNR (log scale)", color=INK2, fontsize=9)
        handles, labels_ = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels_, frameon=False, fontsize=8, labelcolor=INK2, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 0.0))
        fig.suptitle("Dents under the ambient ceiling and three rigs: a 60° ring, a 30° ring and a single 15° light (solid: depth 3 % of the diameter, dashed: 10 %)", color=INK, fontsize=11, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0.12, 1, 0.93))
        fig.savefig(os.path.join(args.out, "figures", "dents_rigs.png"), dpi=160, facecolor=SURFACE)
        plt.close(fig)
        summary["dent_rig_conditions"] = rig_conds
    # ---- scratch envelope: grid length (rows) x width (cols), CNR as colour/number, per zone and pixel scale
    sc = m[m.type.isin(SCRATCH_LEN)].assign(length_mm=lambda x: x.type.map(SCRATCH_LEN))
    widths = sorted(sc.size_mm.unique())
    lengths = sorted(SCRATCH_LEN.values())
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), facecolor=SURFACE)
    for i, cond in enumerate(["ext_nadir_012", "ext_nadir_005"]):
        for j, zone in enumerate(("machined", "cast")):
            ax = axes[j, i]
            g = sc[(sc.condition == cond) & (sc.zone == zone)]
            grid = np.full((len(lengths), len(widths)), np.nan)
            det = np.zeros_like(grid, bool)
            for r in g.itertuples():
                li, wi = lengths.index(r.length_mm), widths.index(r.size_mm)
                grid[li, wi] = r.cnr
                det[li, wi] = r.detectable
            im = ax.imshow(np.log10(np.maximum(grid, 0.1)), cmap="viridis", vmin=-1, vmax=2, aspect="auto", origin="lower")
            mmpx_ = float(g.mm_per_px.iloc[0]) if len(g) else 1.0
            for li in range(len(lengths)):
                for wi in range(len(widths)):
                    v = grid[li, wi]
                    if np.isfinite(v):
                        sub = widths[wi] < mmpx_
                        ax.text(wi, li, (f"{v:.0f}" if v >= 10 else f"{v:.1f}") + ("*" if sub else ""), ha="center", va="center", fontsize=9,
                                color="white" if v < 10 else "black", fontweight="bold" if det[li, wi] else "normal")
            # the client's envelope: length >= 2 mm and (either reading) width >= 1 mm
            ax.axhline(lengths.index(2.0) - 0.5, color="#eb6834", linewidth=2)
            ax.axvline(widths.index(1.0) - 0.5, color="#eb6834", linewidth=2, linestyle="--")
            ax.set_xticks(range(len(widths)), [f"{w:g}" for w in widths])
            ax.set_yticks(range(len(lengths)), [f"{l:g}" for l in lengths])
            ax.set_xlabel("scratch width, mm", color=INK2, fontsize=9)
            ax.set_ylabel("scratch length, mm", color=INK2, fontsize=9)
            mmpx = float(g.mm_per_px.iloc[0]) * 1000 if len(g) else 0
            ax.set_title(f"{zone} face, {mmpx:.0f} µm per pixel (bold = detectable)", loc="left", color=INK, fontsize=10)
            for r in g.itertuples():
                summary[f"{cond}/{zone}/L{r.length_mm:g}_W{r.size_mm:g}"] = {"cnr": round(float(r.cnr), 2), "area_px": int(r.area_px), "detectable": bool(r.detectable)}
    fig.suptitle("Scratch envelope: CNR for every length x width (solid line: 2 mm length; dashed: 1 mm width, the client's envelope; * = narrower than one pixel)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(args.out, "figures", "scratch_envelope.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)
    with open(os.path.join(args.out, "ext_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    pd.set_option("display.width", 220)
    print("dents, smallest detectable diameter:")
    print({k: v for k, v in summary.items() if "dent" in k})
    print(sc.pivot_table(index=["condition", "zone", "length_mm"], columns="size_mm", values="cnr").round(1).to_string())


if __name__ == "__main__":
    main()
