"""Stage 4b: metrics.csv -> summary.csv, figures/, RESULTS.md

  python analysis/plot_results.py --config config/experiment.yaml

Reported conditions are the non-control ones present in metrics.csv. Every number in RESULTS.md
comes from a row (or a median over frames of rows) in metrics.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from ads.config import load_config, resolve_condition, config_json  # noqa: E402
from ads.layout import build_layout, ROW_ORDER  # noqa: E402
from ads.camera import build_camera, project_mm  # noqa: E402

# Palette (validated default, see dataviz reference): cast = slot 1 blue, machined = slot 2 orange
COLORS = {"cast": "#2a78d6", "machined": "#eb6834"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
FAMILY_TITLES = {"pore": "Pores (opening diameter)", "scratch": "Scratches (width)", "burr": "Burrs (height)"}
SIZE_LABEL = {"pore_shallow": "diameter", "pore_deep": "diameter", "scratch": "width", "burr": "height"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "config", "experiment.yaml"))
    ap.add_argument("--metrics", default=os.path.join(REPO_ROOT, "out", "metrics.csv"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "out"))
    ap.add_argument("--conditions", nargs="*", default=None, help="conditions to report (default: all non-control)")
    return ap.parse_args()


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def summarise(df: pd.DataFrame, thr: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["condition", "zone", "type", "prim", "size_mm", "depth_mm", "row", "col"]
    agg = df.groupby(keys, dropna=False).agg(
        frames=("frame", "count"), area_px=("area_px", "median"), mean_in=("mean_in", "median"),
        mean_bg=("mean_bg", "median"), std_bg=("std_bg", "median"), cnr=("cnr", "median"), cnr_min=("cnr", "min"),
        cnr_max=("cnr", "max"), cnr_peak=("cnr_peak", "median"), weber=("weber", "median"),
        mask_empty=("mask_empty", "max"), spp=("spp", "first"), mm_per_px=("mm_per_px", "first"),
    ).reset_index()
    agg["detectable"] = (agg["cnr"] >= float(thr["cnr"])) & (agg["area_px"] >= int(thr["min_area_px"]))
    agg["marginal"] = (agg["cnr"] >= float(thr.get("marginal_cnr", 2.0))) & ~agg["detectable"]
    rows = []
    for (cond, zone, typ), g in agg.groupby(["condition", "zone", "type"]):
        g = g.sort_values("size_mm")
        det = g[g["detectable"]]
        rows.append({"condition": cond, "zone": zone, "type": typ,
                     "min_detectable_mm": float(det["size_mm"].min()) if len(det) else np.nan,
                     "min_detectable": f"{det['size_mm'].min():g}" if len(det) else "none",
                     "largest_tested_mm": float(g["size_mm"].max()), "n_detectable": int(len(det)), "n_sizes": int(len(g)),
                     "cnr_at_largest": float(g.iloc[-1]["cnr"])})
    return agg, pd.DataFrame(rows)


def plot_cnr_vs_size(agg: pd.DataFrame, thr: dict, path: str, cond: str):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), facecolor=SURFACE)
    panels = [("pore", ["pore_shallow", "pore_deep"]), ("scratch", ["scratch"]), ("burr", ["burr"])]
    styles = {"pore_shallow": "-", "pore_deep": "--", "scratch": "-", "burr": "-"}
    ymax = max(float(np.nanmax(agg["cnr"])) * 1.15, float(thr["cnr"]) * 2)
    for ax, (fam, types) in zip(axes, panels):
        style_axes(ax)
        for typ in types:
            for zone in ("cast", "machined"):
                g = agg[(agg["type"] == typ) & (agg["zone"] == zone)].sort_values("size_mm")
                if g.empty:
                    continue
                label = f"{zone}" + (f", {typ.split('_')[1]}" if fam == "pore" else "")
                ax.plot(g["size_mm"], g["cnr"], styles[typ], color=COLORS[zone], linewidth=2, label=label, zorder=3)
                filled = g[~g["mask_empty"].astype(bool)]
                hollow = g[g["mask_empty"].astype(bool)]
                ax.plot(filled["size_mm"], filled["cnr"], "o", color=COLORS[zone], markersize=6, markeredgecolor=SURFACE, markeredgewidth=1, zorder=4)
                ax.plot(hollow["size_mm"], hollow["cnr"], "o", markerfacecolor=SURFACE, markeredgecolor=COLORS[zone], markersize=6, markeredgewidth=1.5, zorder=4)
        ax.axhline(float(thr["cnr"]), color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
        ax.text(0.98, float(thr["cnr"]) - 0.012 * ymax, f"detection threshold CNR = {thr['cnr']:g}", color=INK2, fontsize=8,
                va="top", ha="right", transform=ax.get_yaxis_transform())
        ax.set_xscale("log")
        ax.set_ylim(0, ymax)
        ax.set_title(FAMILY_TITLES[fam], color=INK, fontsize=11, loc="left")
        ax.set_xlabel("size, mm (log)", color=INK2, fontsize=9)
        ax.legend(frameon=False, fontsize=8, loc="upper left", labelcolor=INK2)
    axes[0].set_ylabel("contrast-to-noise ratio (median over frames)", color=INK2, fontsize=9)
    fig.suptitle(f"CNR vs defect size under constant ambient light - {cond}  (hollow markers: defect below one pixel, no mask)",
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def plot_min_detectable(summ: pd.DataFrame, cfg: dict, path: str, cond: str):
    types = [t for t in ROW_ORDER if t in set(summ["type"])]
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=SURFACE)
    style_axes(ax)
    x = np.arange(len(types))
    w = 0.36
    for i, zone in enumerate(("cast", "machined")):
        vals, tops = [], []
        for t in types:
            r = summ[(summ["type"] == t) & (summ["zone"] == zone)]
            v = float(r["min_detectable_mm"].iloc[0]) if len(r) else np.nan
            vals.append(v)
            tops.append(float(r["largest_tested_mm"].iloc[0]) if len(r) else np.nan)
        xs = x + (i - 0.5) * (w + 0.02)
        bars = ax.bar(xs, [v if np.isfinite(v) else 0 for v in vals], width=w, color=COLORS[zone], label=zone, zorder=3)
        for xi, v, top in zip(xs, vals, tops):
            if np.isfinite(v):
                ax.text(xi, v * 1.05, f"{v:g} mm", ha="center", va="bottom", fontsize=8, color=INK2)
            else:
                ax.text(xi, 0.03, f"none\n(up to {top:g} mm)", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_yscale("log")
    ax.set_ylim(0.02, 5)
    ax.set_xticks(x, [f"{t.replace('_', ' ')}\n({SIZE_LABEL[t]})" for t in types])
    ax.set_ylabel("smallest detectable size, mm (log)", color=INK2, fontsize=9)
    ax.set_title(f"Smallest detectable defect size - {cond}", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def contact_sheet(cfg: dict, cond_dir: str, agg: pd.DataFrame, path: str, cond: str):
    import imageio.v3 as iio

    sim = iio.imread(os.path.join(cond_dir, "sim_0.png"))
    if sim.ndim == 3:
        sim = sim[..., 0]
    cam = build_camera(cfg["camera"])
    half_mm = float(cfg["analysis"].get("contact_crop_mm", 3.0))
    zoom = int(cfg["analysis"].get("contact_zoom", 8))
    half_px = max(int(round(half_mm / cam.mm_per_px)), 4)
    layout = build_layout(cfg)
    zones = list(cfg["coupon"]["zones"])
    types = [t for t in ROW_ORDER if t in cfg["defects"]]
    ncol = max(d.col for d in layout) + 1
    tile = 2 * half_px
    fig_w = (ncol * len(zones)) * 1.35 + 1.2
    fig_h = len(types) * 1.35 + 1.0
    fig, axes = plt.subplots(len(types), ncol * len(zones), figsize=(fig_w, fig_h), facecolor=SURFACE)
    axes = np.atleast_2d(axes)
    by_prim = agg.set_index("prim") if "prim" in agg.columns else None
    for d in layout:
        u, v = project_mm(cam, np.array([[d.center_mm[0], d.center_mm[1], 0.0]]))[0]
        cu, cv = int(round(u)), int(round(v))
        crop = sim[max(cv - half_px, 0):cv + half_px, max(cu - half_px, 0):cu + half_px]
        zi = zones.index(d.zone)
        ax = axes[d.row, zi * ncol + d.col]
        ax.imshow(np.kron(crop, np.ones((zoom, zoom), dtype=crop.dtype)), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        cnr = float(by_prim.loc[d.prim, "cnr"]) if by_prim is not None and d.prim in by_prim.index else np.nan
        det = bool(by_prim.loc[d.prim, "detectable"]) if by_prim is not None and d.prim in by_prim.index else False
        ax.set_title(f"{d.size_mm:g} mm  CNR {cnr:.1f}{' *' if det else ''}", fontsize=6.5, color=INK if det else INK2, pad=2)
        if d.col == 0 and zi == 0:
            ax.set_ylabel(d.type.replace("_", "\n"), fontsize=8, color=INK2)
    for zi, z in enumerate(zones):
        fig.text((zi + 0.5) / len(zones), 0.985, f"{z} zone", ha="center", va="top", color=INK, fontsize=11)
    fig.suptitle(f"Simulated 8-bit camera image, {zoom}x crops ({2*half_mm:g} mm wide) - {cond}; * = detectable (CNR >= {cfg['threshold']['cnr']:g}, area >= {cfg['threshold']['min_area_px']} px)",
                 y=0.015, fontsize=8, color=INK2)
    fig.tight_layout(rect=(0.02, 0.04, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def write_results_md(cfg_full: dict, cond: str, summ: pd.DataFrame, agg: pd.DataFrame, df_cond: pd.DataFrame, manifest: dict,
                     checks: dict, control_summary: pd.DataFrame | None, path: str, figures_rel: dict):
    thr = cfg_full["threshold"]
    cfg_c = resolve_condition(cfg_full, cond)
    lines = [f"# Results - ambient-light defect detectability ({cond})", "",
             f"Generated {date.today().isoformat()} from `out/metrics.csv` ({len(df_cond)} rows: {df_cond['frame'].nunique()} frames x {df_cond['prim'].nunique()} defects). "
             f"Detection rule: CNR >= {thr['cnr']:g} and area >= {thr['min_area_px']} px, medians over frames.", "",
             "## Smallest detectable size per defect type and surface", "",
             "| Defect type | Size measure | Cast surface | Machined surface |", "|---|---|---|---|"]
    for t in [t for t in ROW_ORDER if t in set(summ["type"])]:
        cells = []
        for z in ("cast", "machined"):
            r = summ[(summ["type"] == t) & (summ["zone"] == z)]
            if len(r) and r["min_detectable"].iloc[0] != "none":
                cells.append(f"{r['min_detectable'].iloc[0]} mm")
            else:
                top = r["largest_tested_mm"].iloc[0] if len(r) else np.nan
                cells.append(f"none up to {top:g} mm")
        lines.append(f"| {t.replace('_', ' ')} | {SIZE_LABEL[t]} | {cells[0]} | {cells[1]} |")
    lines += ["", "## Figures", "",
              f"![CNR vs size]({figures_rel['cnr_vs_size']})", "", f"![Smallest detectable]({figures_rel['min_detectable']})", "",
              f"![Contact sheet]({figures_rel['contact_sheet']})", "", f"![Mask overlay]({figures_rel['mask_overlay']})", "",
              "## Findings", ""]
    # five sentences of numbers
    n_det = int(agg["detectable"].sum())
    n_tot = len(agg)
    n_marg = int(agg["marginal"].sum())
    n_sub = int(agg["mask_empty"].astype(bool).sum())
    per_zone = {z: int(agg[(agg["zone"] == z)]["detectable"].sum()) for z in ("cast", "machined")}
    largest = agg.sort_values("size_mm").groupby(["zone", "type"]).tail(1)
    largest_txt = "; ".join(f"{r.zone} {r.type.replace('_', ' ')} {r.size_mm:g} mm CNR {r.cnr:.1f}" for r in largest.itertuples())
    best = agg.sort_values("cnr", ascending=False).iloc[0]
    rn = checks.get("render_noise", {})
    ex = checks.get("exposure", {})
    lines.append(f"1. {n_det} of {n_tot} defect sizes are detectable ({per_zone['cast']} on the cast surface, {per_zone['machined']} on the machined surface); {n_marg} are marginal (2 <= CNR < {thr['cnr']:g}); {n_sub} are smaller than one pixel at {cfg_c['camera']['mm_per_px']} mm/px and have no mask.")
    lines.append(f"2. The largest size of each family gives: {largest_txt}.")
    lines.append(f"3. The highest CNR is {best.cnr:.1f} for {best.zone} {best.type.replace('_', ' ')} at {best.size_mm:g} mm (area {best.area_px:.0f} px, Weber contrast {best.weber:+.3f}).")
    lines.append(f"4. The simulated machined background sits at {ex.get('machined_bg_mean', float('nan')):.3f} of full scale with {ex.get('saturated_frac', float('nan'))*100:.3f}% saturated pixels, and the background noise std on the simulated image is "
                 f"{agg[agg.zone=='machined']['std_bg'].median():.4f} (machined) and {agg[agg.zone=='cast']['std_bg'].median():.4f} (cast) of full scale.")
    lines.append(f"5. Residual render noise is {rn.get('ratio', float('nan')):.2f}x the sensor noise at {manifest.get('spp')} spp ({'frames are bit-identical re-renders' if manifest.get('frames_identical') else 'frames differ'}), so the medians over frames average the sensor-noise realisations only.")
    if control_summary is not None and len(control_summary):
        ctrl_name = control_summary["condition"].iloc[0]
        pc = {}
        try:
            with open(os.path.join(os.path.dirname(path), ctrl_name, "analysis_checks.json")) as fh:
                pc = json.load(fh).get("positive_control", {})
        except OSError:
            pass
        lines += ["", f"## Positive control (`{ctrl_name}`, pipeline check, not a reported result)", "",
                  f"Rule: {pc.get('rule', 'n/a')}. Result: **{'PASS' if pc.get('pass') else 'FAIL'}**. Exposure is solved on the machined zone, so the cast zone saturates under the raking light and its rows carry `saturated_bg`.", "",
                  "| Machined defect | area px | CNR | peak CNR | pass |", "|---|---|---|---|---|"]
        for r in pc.get("rows", []):
            lines.append(f"| {r['prim']} | {r['area_px']} | {r['cnr']:.1f} | {r['cnr_peak']:.1f} | {'yes' if r['pass'] else 'no'} |")
    lines += ["", "## Capture and checks", "",
              f"- Working distance {manifest['camera']['distance_mm']:.1f} mm, {manifest['camera']['res_x']}x{manifest['camera']['res_y']} px, {manifest['camera']['mm_per_px']} mm/px, view {manifest['camera']['view']}.",
              f"- Lighting: {manifest['lighting_mode']}, solved intensities {json.dumps(manifest['solved_intensity'])}.",
              f"- Path tracing {manifest['spp']} spp ({manifest['subframes']} subframes), denoiser {manifest['render_settings'].get('denoiser')}, bounces {manifest['render_settings'].get('max_bounces')}/{manifest['render_settings'].get('max_specular_bounces')}; Isaac Sim {manifest.get('isaac_sim_version')}.",
              f"- Sensor gain {ex.get('gain', float('nan')):.4f}; render-noise check: {rn.get('std', float('nan')):.5f} vs sensor {rn.get('sensor_std', float('nan')):.5f} (ratio {rn.get('ratio', float('nan')):.2f}, pass = {rn.get('pass')}).",
              f"- Masks: {checks.get('masks', {}).get('n_prims_in_id_map')} of {checks.get('masks', {}).get('n_defects')} defect prims have pixels; max centroid offset {checks.get('masks', {}).get('max_centroid_offset_px', float('nan')):.2f} px over {checks.get('masks', {}).get('n_aligned_checked')} defects with area >= 9 px.",
              "", "## Config used", "", "```json", config_json(cfg_c), "```", ""]
    with open(path, "w") as fh:
        fh.write("\n".join(lines))


def main():
    args = parse_args()
    cfg = load_config(args.config)
    thr = cfg["threshold"]
    df = pd.read_csv(args.metrics)
    agg_all, summ_all = summarise(df, thr)
    os.makedirs(args.out, exist_ok=True)
    agg_all.to_csv(os.path.join(args.out, "summary_per_defect.csv"), index=False)
    summ_all.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    figdir = os.path.join(args.out, "figures")
    os.makedirs(figdir, exist_ok=True)
    conditions = args.conditions or [c for c in df["condition"].unique() if not bool(df[df.condition == c]["control"].iloc[0])]
    controls = [c for c in df["condition"].unique() if bool(df[df.condition == c]["control"].iloc[0])]
    default_cond = next((c for c, s in cfg["conditions"].items() if s and s.get("default")), conditions[0])
    for cond in conditions:
        agg = agg_all[agg_all["condition"] == cond]
        summ = summ_all[summ_all["condition"] == cond]
        suffix = "" if cond == default_cond else f"_{cond}"
        cond_dir = os.path.join(args.out, cond)
        figs = {"cnr_vs_size": f"cnr_vs_size{suffix}.png", "min_detectable": f"min_detectable{suffix}.png",
                "contact_sheet": f"contact_sheet{suffix}.png", "mask_overlay": f"mask_overlay{suffix}.png"}
        plot_cnr_vs_size(agg, thr, os.path.join(figdir, figs["cnr_vs_size"]), cond)
        plot_min_detectable(summ, cfg, os.path.join(figdir, figs["min_detectable"]), cond)
        contact_sheet(cfg if cond == default_cond else resolve_condition(cfg, cond), cond_dir, agg, os.path.join(figdir, figs["contact_sheet"]), cond)
        import shutil

        shutil.copyfile(os.path.join(cond_dir, "mask_overlay.png"), os.path.join(figdir, figs["mask_overlay"]))
        with open(os.path.join(cond_dir, "frame_manifest.json")) as fh:
            manifest = json.load(fh)
        with open(os.path.join(cond_dir, "analysis_checks.json")) as fh:
            checks = json.load(fh)
        ctrl = summ_all[summ_all["condition"].isin(controls)] if controls else None
        write_results_md(cfg, cond, summ, agg, df[df.condition == cond], manifest, checks, ctrl,
                         os.path.join(args.out, f"RESULTS{suffix}.md"), {k: f"figures/{v}" for k, v in figs.items()})
        print(f"[plot_results] {cond}: figures -> {figdir}, RESULTS{suffix}.md")
        print(summ[["zone", "type", "min_detectable", "n_detectable", "n_sizes"]].to_string(index=False))


if __name__ == "__main__":
    main()
