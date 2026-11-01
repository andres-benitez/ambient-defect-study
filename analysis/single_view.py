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
                         "detectable": r.detectable, "cnr_1exp": getattr(r, "cnr_1exp", r.cnr), "detectable_1exp": getattr(r, "detectable_1exp", r.detectable)})
        dof1 = depth_of_field_mm(N, pitch_um, cam.mm_per_px, coc_px=1.0)
        dof2 = depth_of_field_mm(N, pitch_um, cam.mm_per_px, coc_px=2.0)
        det = mm.detectable.sum()
        entry = {"camera": cc["camera"]["model"], "f_number": N, "focus_offset_mm": focus, "mm_per_px": cam.mm_per_px,
                 "dof_1px_mm": dof1, "dof_2px_mm": dof2, "airy_px": airy_diameter_um(N, 550) / pitch_um,
                 "exposure_ms_1000lux": exposure_ms(1000.0, cc, f_number=N)["exposure_ms"], "exposure_ms_300lux": exposure_ms(300.0, cc, f_number=N)["exposure_ms"],
                 "detectable": int(det), "detectable_machined": int(mm[mm.zone == "machined"].detectable.sum()), "detectable_cast": int(mm[mm.zone == "cast"].detectable.sum()),
                 "spp": int(mm.spp.iloc[0])}
        # exposure zones: on the block the bays sit several stops darker than the flange under the
        # ceiling light, so compute_cnr measured the cast defects on a second exposure; record the
        # cost of a single exposure next to it
        chk_path = os.path.join(args.out, cond, "analysis_checks.json")
        ez = (json.load(open(chk_path)).get("exposure_zones", {}) if os.path.exists(chk_path) else {})
        two = bool(ez.get("two_exposures", False))
        ratio = float(ez["gain_cast"]) / float(ez["gain_machined"]) if two else 1.0
        det1 = mm.detectable_1exp.fillna(mm.detectable) if "detectable_1exp" in mm.columns else mm.detectable
        entry.update({"two_exposures": two, "cast_exposure_ratio": ratio, "cast_stops": float(np.log2(ratio)),
                      "cast_bg_level_1exp": ez.get("cast_bg_level_at_machined_gain"),
                      "exposure_ms_1000lux_cast": entry["exposure_ms_1000lux"] * ratio, "exposure_ms_300lux_cast": entry["exposure_ms_300lux"] * ratio,
                      "detectable_cast_1exp": int(det1[mm.zone == "cast"].sum())})
        _c = mm[(mm.zone == "cast") & (mm.area_px > 0)]
        _c1 = (_c.cnr_1exp.fillna(_c.cnr) if "cnr_1exp" in _c.columns else _c.cnr)
        entry.update({"cast_cnr_median": float(_c.cnr.median()) if len(_c) else float("nan"),
                      "cast_cnr_median_1exp": float(_c1.median()) if len(_c) else float("nan")})
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
            ax.set_title(f"Pixel size over the part, {cc['camera']['model'].split('-')[0]} at {cam.mm_per_px*1000:.0f} µm/px on the flange\n"
                         f"{entry['frac_under_0p2']*100:.0f} % of the visible part under 0.2 mm/px", loc="left", color=INK, fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(args.out, "figures", f"pixel_scale_{cond}.png"), dpi=150, facecolor=SURFACE)
            plt.close(fig)
        summary[cond] = entry
    df = pd.DataFrame(rows)
    df["cnr_1exp"] = df.cnr_1exp.fillna(df.cnr)
    df["detectable_1exp"] = df.detectable_1exp.fillna(df.detectable).astype(bool)
    df.to_csv(os.path.join(args.out, "single_view.csv"), index=False)
    with open(os.path.join(args.out, "single_view.json"), "w") as fh:
        json.dump(summary, fh, indent=1)

    # ---- figure per camera: (left) every defect's contrast at the narrowest against the widest aperture,
    # coloured by its depth below the flange; (right) detectable counts against the f-number
    for cam_model in sorted(df.camera.unique()):
        sub = df[(df.camera == cam_model) & (df.focus_offset_mm > 0)]
        if sub.empty:
            continue
        fn = sorted(sub.f_number.unique())
        lo, hi = fn[0], fn[-1]
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), facecolor=SURFACE)
        ax = axes[0]
        style_axes(ax)
        a_ = sub[(sub.f_number == lo) & (sub.area_px > 0)].set_index("prim")
        b_ = sub[(sub.f_number == hi) & (sub.area_px > 0)].set_index("prim")
        common = a_.index.intersection(b_.index)
        if len(common) and lo != hi:
            sc = ax.scatter(a_.loc[common].cnr, b_.loc[common].cnr, c=a_.loc[common].depth_below_plane_mm, cmap="viridis", s=30, edgecolor="white", linewidth=0.5)
            vmax = float(max(a_.loc[common].cnr.max(), b_.loc[common].cnr.max())) * 1.05
            ax.plot([0, vmax], [0, vmax], color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
            ax.axhline(3, color=MUTED, linewidth=0.8, linestyle=(0, (2, 3)))
            ax.axvline(3, color=MUTED, linewidth=0.8, linestyle=(0, (2, 3)))
            ax.set_xlim(0, vmax)
            ax.set_ylim(0, vmax)
            cb = plt.colorbar(sc, ax=ax)
            cb.set_label("depth below the flange, mm", color=INK2, fontsize=9)
            ax.set_xlabel(f"CNR at f/{lo:g}, the widest aperture tested", color=INK2, fontsize=9)
            ax.set_ylabel(f"CNR at f/{hi:g}, the narrowest", color=INK2, fontsize=9)
            ax.set_title("One dot per defect; on the diagonal the aperture changed nothing", loc="left", color=INK, fontsize=10)
        ax = axes[1]
        style_axes(ax)
        counts = {z: [int(sub[(sub.f_number == N) & (sub.zone == z)].detectable.sum()) for N in fn] for z in ("machined", "cast")}
        x = np.arange(len(fn))
        w = 0.38
        ax.bar(x - w / 2, counts["machined"], w, color=COLORS["machined"], label="machined flange (of 24)")
        ax.bar(x + w / 2, counts["cast"], w, color=COLORS["cast"], label="cast bays (of 24, second exposure)")
        for xi in range(len(fn)):
            ax.text(xi - w / 2, counts["machined"][xi] + 0.3, str(counts["machined"][xi]), ha="center", fontsize=8, color=INK2)
            ax.text(xi + w / 2, counts["cast"][xi] + 0.3, str(counts["cast"][xi]), ha="center", fontsize=8, color=INK2)
        ax.set_xticks(x)
        ax.set_xticklabels([f"f/{N:g}" for N in fn], color=INK2)
        ax.set_ylim(0, 27)
        ax.set_ylabel("defects detectable", color=INK2, fontsize=9)
        ax.set_title("Detectable defects against the aperture, focus 60 mm below the flange", loc="left", color=INK, fontsize=10)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper right")
        fig.suptitle(f"One view of the whole block, {cam_model.split('-')[0]}: what the aperture changes", color=INK, fontsize=11, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(os.path.join(args.out, "figures", f"single_view_{cam_model.split('-')[0]}.png"), dpi=160, facecolor=SURFACE)
        plt.close(fig)
    # ---- figure: cast defects detectable with one exposure (set for the flange) against two
    two_conds = [c for c, e in summary.items() if e.get("two_exposures")]
    if two_conds:
        pick = {}
        for c in two_conds:  # one condition per camera, preferring the focus-on-flange condition at the default aperture
            e = summary[c]
            if e["camera"] not in pick or (e["focus_offset_mm"] == 0 and summary[pick[e["camera"]]]["focus_offset_mm"] != 0):
                pick[e["camera"]] = c
        types = [t for t in ("pore_shallow", "pore_deep", "scratch", "burr") if t in set(df.type)]
        labels = {"pore_shallow": "shallow pore", "pore_deep": "deep pore", "scratch": "scratch", "burr": "burr", "dent_shallow": "shallow dent", "dent_deep": "deep dent"}
        fig, axes = plt.subplots(1, len(pick), figsize=(6.4 * len(pick), 4.6), facecolor=SURFACE, squeeze=False)
        for ax, (cam_model, c) in zip(axes[0], sorted(pick.items(), reverse=True)):
            style_axes(ax)
            sub = df[(df.condition == c) & (df.zone == "cast")]
            one = [int(sub[sub.type == t].detectable_1exp.sum()) for t in types]
            both = [int(sub[sub.type == t].detectable.sum()) for t in types]
            n_sizes = max(int((sub.type == t).sum()) for t in types)
            x = np.arange(len(types))
            w = 0.38
            ax.bar(x - w / 2, one, w, color=MUTED, label="one exposure, set for the flange")
            ax.bar(x + w / 2, both, w, color=COLORS["cast"], label=f"second exposure for the bays, {summary[c]['cast_exposure_ratio']:.0f}× longer")
            for xi, (o, b) in enumerate(zip(one, both)):
                ax.text(xi - w / 2, o + 0.08, str(o), ha="center", va="bottom", fontsize=8, color=INK2)
                ax.text(xi + w / 2, b + 0.08, str(b), ha="center", va="bottom", fontsize=8, color=INK2)
            ax.set_xticks(x)
            ax.set_xticklabels([labels.get(t, t) for t in types], color=INK2, fontsize=9)
            ax.set_ylim(0, n_sizes + 0.8)
            ax.set_ylabel(f"cast defects detectable (of {n_sizes} sizes)", color=INK2, fontsize=9)
            ax.set_title(f"{cam_model.split('-')[0]}, f/{summary[c]['f_number']:g}: the bays sit {abs(summary[c]['cast_stops']):.1f} stops darker than the flange",
                         loc="left", color=INK, fontsize=10)
            ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
        fig.suptitle("One view of the whole block: what a single exposure costs in the cast bays", color=INK, fontsize=11, x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        fig.savefig(os.path.join(args.out, "figures", "single_view_exposure.png"), dpi=160, facecolor=SURFACE)
        plt.close(fig)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(summary).T[["f_number", "focus_offset_mm", "dof_1px_mm", "dof_2px_mm", "airy_px", "exposure_ms_1000lux", "detectable", "detectable_machined", "detectable_cast", "detectable_cast_1exp", "cast_stops"]].to_string())


if __name__ == "__main__":
    main()
