"""Experiment 3: one view per face. Aperture and focus-plane sweep on the whole block.

  python analysis/single_view.py --config config/experiment.yaml

For every whole-part condition (asset kind = part): per-defect CNR against the defect's depth
below the reference (flange) plane, grouped by f-number and focus setting; depth-of-field bands;
exposure time per aperture; and, when a depth map was captured, the realised pixel size over the
part and the fraction of the part within the 0.2 mm/px clause. Writes out/single_view.csv,
out/figures/single_view_*.png and out/single_view.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from ads.config import load_config, resolve_condition  # noqa: E402
from ads.layout import load_defects  # noqa: E402
from ads.camera import build_camera  # noqa: E402
from analysis.optics import depth_of_field_mm, airy_diameter_um  # noqa: E402
from analysis.photometry import exposure_ms  # noqa: E402
from analysis.plot_results import style_axes, COLORS, INK, INK2, MUTED, GRID, SURFACE  # noqa: E402


def pixel_scale_map(cond_dir: str, cc: dict):
    """Realised mm/px per pixel from the captured depth (distance to image plane)."""
    p = os.path.join(cond_dir, "renders", "depth_0.npy")
    if not os.path.exists(p):
        return None
    depth = np.load(p).astype(float)  # metres
    cam = build_camera(cc["camera"])
    pitch_mm = float(cc["camera"].get("pixel_um", cam.sensor_width_mm * 1000 / cam.res_x)) / 1000.0
    scale = depth * 1000.0 * pitch_mm / cam.focal_length_mm  # mm per pixel at that depth
    scale[~np.isfinite(depth) | (depth <= 0) | (depth > 10)] = np.nan
    return scale


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--config", default=os.path.join(REPO_ROOT, "config", "experiment.yaml"))
    a.add_argument("--out", default=os.path.join(REPO_ROOT, "out"))
    args = a.parse_args()
    cfg = load_config(args.config)
    m = pd.read_csv(os.path.join(args.out, "metrics.csv"))
    m = m[(m.frame == 0) & (m.lens == cfg["lens"].get("default_scenario", "none"))]
    part_conds = [c for c, s in cfg["conditions"].items() if (s or {}).get("asset", {}).get("kind") == "part" and c in set(m.condition)]
    if not part_conds:
        raise SystemExit("no whole-part conditions in metrics.csv")
    rows, summary = [], {}
    for cond in part_conds:
        cc = resolve_condition(cfg, cond)
        cam = build_camera(cc["camera"])
        N = float(cc["camera"].get("f_number", 8))
        focus = float(cc["camera"].get("focus_offset_mm", 0.0))
        pitch_um = float(cc["camera"].get("pixel_um", 2.74))
        defs = {d.prim: d for d in load_defects(cc)}
        mm = m[m.condition == cond]
        for r in mm.itertuples():
            d = defs[r.prim]
            rows.append({"condition": cond, "camera": cc["camera"]["model"], "mp": cc["camera"]["model_info"]["mp"], "f_number": N, "focus_offset_mm": focus,
                         "mm_per_px": cam.mm_per_px, "prim": r.prim, "zone": r.zone, "type": r.type, "size_mm": r.size_mm,
                         "depth_below_plane_mm": -d.center3_mm[2], "defocus_mm": abs(-d.center3_mm[2] - focus), "cnr": r.cnr, "area_px": r.area_px,
                         "detectable": r.detectable})
        dof1 = depth_of_field_mm(N, pitch_um, cam.mm_per_px, coc_px=1.0)
        dof2 = depth_of_field_mm(N, pitch_um, cam.mm_per_px, coc_px=2.0)
        det = mm.detectable.sum()
        entry = {"camera": cc["camera"]["model"], "f_number": N, "focus_offset_mm": focus, "mm_per_px": cam.mm_per_px,
                 "dof_1px_mm": dof1, "dof_2px_mm": dof2, "airy_px": airy_diameter_um(N, 550) / pitch_um,
                 "exposure_ms_1000lux": exposure_ms(1000.0, cc, f_number=N)["exposure_ms"], "exposure_ms_300lux": exposure_ms(300.0, cc, f_number=N)["exposure_ms"],
                 "detectable": int(det), "detectable_machined": int(mm[mm.zone == "machined"].detectable.sum()), "detectable_cast": int(mm[mm.zone == "cast"].detectable.sum()),
                 "spp": int(mm.spp.iloc[0])}
        scale = pixel_scale_map(os.path.join(args.out, cond), cc)
        if scale is not None:
            sem = np.load(os.path.join(args.out, cond, "renders", "zone_0.npy"))
            with open(os.path.join(args.out, cond, "renders", "id_map.json")) as fh:
                idm = json.load(fh)
            part_px = np.isin(sem, idm["class_to_ids"].get("zone_cast", []) + idm["class_to_ids"].get("zone_machined", []) + idm["class_to_ids"].get("defect", []))
            vals = scale[part_px & np.isfinite(scale)]
            entry.update({"scale_min": float(vals.min()), "scale_max": float(vals.max()), "scale_median": float(np.median(vals)),
                          "frac_under_0p2": float((vals <= 0.2).mean()), "frac_under_0p1": float((vals <= 0.1).mean())})
            fig, ax = plt.subplots(figsize=(8, 6.2), facecolor=SURFACE)
            shown = np.where(part_px, scale, np.nan)
            im = ax.imshow(shown, cmap="viridis", vmin=np.nanpercentile(shown, 1), vmax=np.nanpercentile(shown, 99))
            cs = ax.contour(np.where(part_px, scale, np.nan), levels=[0.2], colors=["#eb6834"], linewidths=1.5)
            ax.clabel(cs, fmt="0.2 mm/px", fontsize=8)
            plt.colorbar(im, ax=ax, label="realised pixel size on the part, mm/px")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"Pixel size over the part, {cc['camera']['model']} at {cam.mm_per_px*1000:.0f} µm/px on the flange; "
                         f"{entry['frac_under_0p2']*100:.0f} % of the visible part under 0.2 mm/px", loc="left", color=INK, fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(args.out, "figures", f"pixel_scale_{cond}.png"), dpi=150, facecolor=SURFACE)
            plt.close(fig)
        summary[cond] = entry
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "single_view.csv"), index=False)
    with open(os.path.join(args.out, "single_view.json"), "w") as fh:
        json.dump(summary, fh, indent=1)

    # ---- figure: CNR against depth below the flange, one panel per camera, one line per f-number
    for cam_model in sorted(df.camera.unique()):
        sub = df[df.camera == cam_model]
        fnums = sorted(sub.f_number.unique())
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), facecolor=SURFACE, sharey=True)
        for ax, zone in zip(axes, ("machined", "cast")):
            style_axes(ax)
            for i, N in enumerate(fnums):
                for focus in sorted(sub.focus_offset_mm.unique()):
                    g = sub[(sub.f_number == N) & (sub.focus_offset_mm == focus) & (sub.zone == zone) & (sub.area_px > 0)]
                    if g.empty:
                        continue
                    # bin by defocus distance
                    bins = np.array([0, 10, 25, 50, 90, 150])
                    g = g.assign(bin=pd.cut(g.defocus_mm, bins, include_lowest=True))
                    med = g.groupby("bin", observed=True).cnr.median()
                    centers = [b.mid for b in med.index]
                    style = "-" if focus > 0 else "--"
                    ax.plot(centers, med.values, style, marker="o", markersize=5, linewidth=1.8, color=plt.cm.viridis(i / max(len(fnums) - 1, 1)),
                            label=f"f/{N:g}, focus {'mid-depth' if focus > 0 else 'on flange'}")
            ax.axhline(3, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
            ax.set_xlabel("distance of the defect from the focus plane, mm", color=INK2, fontsize=9)
            ax.set_title(f"{zone} defects (median CNR of those with a footprint)", loc="left", color=INK, fontsize=10)
        axes[0].set_ylabel("CNR", color=INK2, fontsize=9)
        axes[1].legend(frameon=False, fontsize=8, labelcolor=INK2)
        fig.suptitle(f"One view of the whole block, {cam_model}: contrast against defocus for each aperture", color=INK, fontsize=11, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(os.path.join(args.out, "figures", f"single_view_{cam_model.split('-')[0]}.png"), dpi=160, facecolor=SURFACE)
        plt.close(fig)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(summary).T[["f_number", "focus_offset_mm", "dof_1px_mm", "dof_2px_mm", "airy_px", "exposure_ms_1000lux", "detectable", "detectable_machined", "detectable_cast"]].to_string())


if __name__ == "__main__":
    main()
