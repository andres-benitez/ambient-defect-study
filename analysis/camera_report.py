"""Camera comparison: figures and tables for the cam_* conditions (one shot of the whole face).

  python analysis/camera_report.py --config config/experiment.yaml

Reads out/metrics.csv (all lens scenarios), writes out/figures/camera_*.png, out/camera_summary.csv,
out/camera_table.json (photometry, optics, prices) and out/figures/rolling_shutter_*.png for the
rolling-shutter candidate.
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
from ads.layout import ROW_ORDER  # noqa: E402
from analysis.optics import scenario_psf  # noqa: E402
from analysis.photometry import exposure_ms  # noqa: E402
from analysis.rolling_shutter import row_gains, banding_amplitude, apply_banding  # noqa: E402
from analysis.plot_results import style_axes, COLORS, INK, INK2, MUTED, GRID, SURFACE, SIZE_LABEL  # noqa: E402

from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator  # noqa: E402


def mm_log_axis(ax, values):
    """Log y axis with plain millimetre labels at the tested sizes."""
    vals = sorted(set(float(v) for v in values if np.isfinite(v)))
    ax.yaxis.set_major_locator(FixedLocator(vals))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g} mm"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())


LENS_COLORS = {"none": "#898781", "ideal": "#2a78d6", "good": "#1baf7a", "budget": "#eb6834"}
LENS_LABEL = {"none": "no optics (pixel sampling only)", "ideal": "diffraction-limited lens, f/8",
              "good": "good lens (2 um spot), f/8", "budget": "budget lens (5 um spot), f/8"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "config", "experiment.yaml"))
    ap.add_argument("--metrics", default=os.path.join(REPO_ROOT, "out", "metrics.csv"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "out"))
    return ap.parse_args()


def min_detectable(df: pd.DataFrame, thr: dict) -> pd.DataFrame:
    """Per (condition, lens, zone, type): smallest detectable size, medians over frames."""
    df = df.copy()
    # footprint across the defect in pixels: pore diameter, scratch width, burr base width
    df["footprint_px"] = np.where(df["type"].str.startswith(("pore", "dent")), df["size_mm"], df["width_mm"].fillna(df["size_mm"])) / df["mm_per_px"]
    g = df.groupby(["condition", "lens", "zone", "type", "size_mm"]).agg(cnr=("cnr", "median"), area_px=("area_px", "median"),
                                                                        footprint_px=("footprint_px", "first")).reset_index()
    g["det"] = (g["cnr"] >= float(thr["cnr"])) & (g["area_px"] >= int(thr["min_area_px"]))
    rows = []
    for (c, l, z, t), gg in g.groupby(["condition", "lens", "zone", "type"]):
        d = gg[gg["det"]]
        smallest = float(d["size_mm"].min()) if len(d) else np.nan
        fp = float(d.sort_values("size_mm")["footprint_px"].iloc[0]) if len(d) else np.nan
        rows.append({"condition": c, "lens": l, "zone": z, "type": t, "min_detectable_mm": smallest,
                     "footprint_px": fp, "at_pixel_limit": bool(np.isfinite(fp) and fp < 1.0),
                     "n_detectable": int(len(d)), "n_sizes": int(len(gg)), "largest_mm": float(gg["size_mm"].max())})
    return pd.DataFrame(rows)


def camera_table(cfg: dict, cams: list, df: pd.DataFrame) -> list:
    out = []
    for cond in cams:
        c = resolve_condition(cfg, cond)
        info = c["camera"]["model_info"]
        rows = df[df.condition == cond]
        psf_good = scenario_psf(c, "good")
        psf_budget = scenario_psf(c, "budget")
        entry = {
            "condition": cond, "model": c["camera"]["model"], "sensor": info["sensor"], "mp": info["mp"],
            "resolution": c["camera"]["resolution"], "pixel_um": info["pixel_um"], "shutter": info["shutter"],
            "price_eur": info["price_eur"], "mm_per_px": c["camera"]["mm_per_px"], "focal_length_mm": c["camera"]["focal_length_mm"],
            "sensor_width_mm": c["camera"]["sensor_width_mm"], "working_distance_mm": float(c["framing"]["working_distance_mm"]),
            "full_well_e": info["full_well_e"], "read_noise_e": info["read_noise_e"], "qe": info["qe"],
            "psf_sigma_px_good": psf_good["psf_sigma_px"], "psf_sigma_px_budget": psf_budget["psf_sigma_px"],
            "airy_diameter_um_f8": psf_good["airy_diameter_um"], "airy_diameter_px_f8": psf_good["airy_diameter_um"] / info["pixel_um"],
            "depth_of_field_mm_f8": psf_good["depth_of_field_mm"],
            "spp": int(rows["spp"].iloc[0]) if len(rows) else None,
            "exposure_ms": {str(lux): exposure_ms(float(lux), c)["exposure_ms"] for lux in cfg["photometry"]["illuminance_lux"]},
            "exposure_ms_f_sweep_1000lux": {f"{f:g}": exposure_ms(1000.0, c, f_number=float(f))["exposure_ms"] for f in cfg["lens"]["f_sweep"]},
        }
        out.append(entry)
    return out


def fig_min_detectable_by_camera(md: pd.DataFrame, table: list, path: str, lens_list=("none", "good", "budget")):
    cams = [t["condition"] for t in table]
    labels = [f"{t['model'].split('-')[0]}\n{t['mp']:g} MP\n{t['mm_per_px']*1000:.0f} µm/px\n€{t['price_eur']}" for t in table]
    types = [t for t in ROW_ORDER if t in set(md["type"])]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), facecolor=SURFACE, sharex=True)
    x = np.arange(len(cams))
    markers = {"none": "o", "good": "s", "budget": "^"}
    for ax, typ in zip(axes.ravel(), types):
        style_axes(ax)
        for zone in ("cast", "machined"):
            for lens in lens_list:
                vals, limit = [], []
                for c in cams:
                    r = md[(md.condition == c) & (md.lens == lens) & (md.zone == zone) & (md.type == typ)]
                    vals.append(float(r["min_detectable_mm"].iloc[0]) if len(r) else np.nan)
                    limit.append(bool(r["at_pixel_limit"].iloc[0]) if len(r) else False)
                off = -0.12 if zone == "cast" else 0.12
                xs_ = x + off
                ax.plot(xs_, vals, linestyle="-" if lens == "none" else "--", linewidth=1.5 if lens == "none" else 1.0,
                        color=COLORS[zone], alpha=1.0 if lens != "budget" else 0.7, label=f"{zone}, {LENS_LABEL[lens].split(',')[0]}")
                solid = [not l for l in limit]
                ax.plot(xs_[solid], np.array(vals)[solid], marker=markers[lens], markersize=7, linestyle="none", color=COLORS[zone], alpha=1.0 if lens != "budget" else 0.7)
                hollow = [l for l in limit]
                if any(hollow):
                    ax.plot(xs_[hollow], np.array(vals)[hollow], marker=markers[lens], markersize=8, linestyle="none",
                            markerfacecolor=SURFACE, markeredgecolor=COLORS[zone], markeredgewidth=1.6)
        ax.set_yscale("log")
        mm_log_axis(ax, md[md.type == typ]["min_detectable_mm"].dropna().tolist() + [md[md.type == typ]["largest_mm"].max()])
        ax.set_title(f"{typ.replace('_', ' ')} ({SIZE_LABEL[typ]})", loc="left", color=INK, fontsize=11)
        ax.set_ylabel("smallest detectable", color=INK2, fontsize=9)
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_xlim(-0.6, len(cams) - 0.4)
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2, labelcolor=INK2, loc="upper right")
    fig.suptitle("Smallest detectable size per camera (one shot of a 300 mm face, constant ambient light)\nhollow marker: the defect is narrower than one pixel there, a result at the pixel limit", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_lens_effect(df: pd.DataFrame, cond: str, path: str, title: str):
    """CNR vs size for one camera, one line per lens scenario (machined and cast panels)."""
    d = df[(df.condition == cond)].groupby(["lens", "zone", "type", "size_mm"]).agg(cnr=("cnr", "median"), area=("area_px", "median")).reset_index()
    fams = [("pore_shallow", "Shallow pores"), ("scratch", "Scratches"), ("burr", "Burrs")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), facecolor=SURFACE)
    for row, zone in enumerate(("machined", "cast")):
        for col, (typ, name) in enumerate(fams):
            ax = axes[row, col]
            style_axes(ax)
            for lens in ("none", "ideal", "good", "budget"):
                g = d[(d.lens == lens) & (d.zone == zone) & (d.type == typ)].sort_values("size_mm")
                if g.empty:
                    continue
                ax.plot(g.size_mm, g.cnr, "-o", color=LENS_COLORS[lens], linewidth=2, markersize=5, label=LENS_LABEL[lens])
            ax.axhline(3, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
            ax.set_xscale("log")
            ax.set_title(f"{name}, {zone}", loc="left", color=INK, fontsize=10)
            if col == 0:
                ax.set_ylabel("CNR", color=INK2, fontsize=9)
            if row == 1:
                ax.set_xlabel("size, mm (log)", color=INK2, fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=8, labelcolor=INK2, loc="lower right")
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def fig_f_sweep(df: pd.DataFrame, cond: str, cfg_c: dict, path: str, title: str):
    """CNR of a few reference defects vs f-number (good lens), with depth of field and exposure."""
    sweep = sorted({float(l.split("good_f")[1]) for l in df[df.condition == cond].lens.unique() if str(l).startswith("good_f")})
    if not sweep:
        return False
    refs = [("machined", "pore_shallow", 0.5), ("machined", "scratch", 0.05), ("machined", "burr", 0.2), ("cast", "pore_deep", 0.5)]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.3), facecolor=SURFACE)
    style_axes(ax)
    style_axes(ax2)
    styles = ["-", "--", "-.", ":"]
    for (zone, typ, size), ls in zip(refs, styles):
        ys = []
        for f in sweep:
            r = df[(df.condition == cond) & (df.lens == f"good_f{f:g}") & (df.zone == zone) & (df.type == typ) & (np.isclose(df.size_mm, size))]
            ys.append(float(r.cnr.median()) if len(r) else np.nan)
        ax.plot(sweep, ys, ls, marker="o", color=COLORS[zone], linewidth=2, markersize=5, label=f"{zone} {typ.replace('_', ' ')} {size:g} mm")
    ax.axhline(3, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.set_xscale("log")
    ax.set_xticks(sweep, [f"f/{f:g}" for f in sweep])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("lens aperture (f-number)", color=INK2, fontsize=9)
    ax.set_ylabel("CNR (good lens)", color=INK2, fontsize=9)
    ax.set_title("Contrast against aperture", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    dof = [scenario_psf(cfg_c, "good", f_number_override=f)["depth_of_field_mm"] for f in sweep]
    exp_ = [exposure_ms(1000.0, cfg_c, f_number=f)["exposure_ms"] for f in sweep]
    exp3 = [exposure_ms(300.0, cfg_c, f_number=f)["exposure_ms"] for f in sweep]
    ax2.plot(sweep, dof, "-o", color="#6250d6", linewidth=2, markersize=5, label="depth of field, mm (1 px blur)")
    ax2.plot(sweep, exp_, "-s", color="#eda100", linewidth=2, markersize=5, label="exposure, ms at 1000 lux")
    ax2.plot(sweep, exp3, "--s", color="#eda100", linewidth=1.5, markersize=4, label="exposure, ms at 300 lux")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xticks(sweep, [f"f/{f:g}" for f in sweep])
    ax2.xaxis.set_minor_locator(NullLocator())
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax2.yaxis.set_minor_formatter(NullFormatter())
    ax2.set_xlabel("lens aperture (f-number)", color=INK2, fontsize=9)
    ax2.set_title("What the aperture costs: focus depth and exposure time", loc="left", color=INK, fontsize=11)
    ax2.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return True


def rolling_shutter_panel(cfg_c: dict, cond: str, out_dir: str, path_img: str, path_curve: str) -> dict:
    """Banding illustration on the rolling-shutter camera's simulated image + CNR impact."""
    from analysis.io_utils import load_hdr
    from analysis.sensor_model import luminance, simulate, to_uint8, solve_gain
    from analysis.metrics import measure_defect, dilate
    from scipy import ndimage
    from ads.layout import load_defects
    from ads.camera import build_camera, project_mm

    rs = cfg_c["rolling_shutter"]
    info = cfg_c["camera"]["model_info"]
    readout = float(info.get("readout_ms", 22))
    renders = os.path.join(out_dir, cond, "renders")
    hdr = load_hdr(renders, 0)
    inst = np.load(os.path.join(renders, "mask_0.npy"))
    sem = np.load(os.path.join(renders, "zone_0.npy"))
    with open(os.path.join(renders, "id_map.json")) as fh:
        idm = json.load(fh)
    zone_m = np.isin(sem, idm["class_to_ids"]["zone_machined"])
    defect_cls = np.isin(sem, idm["class_to_ids"]["defect"])
    bg = zone_m & ~ndimage.binary_dilation(defect_cls, iterations=4)
    Y = luminance(hdr)
    gain = solve_gain(Y, bg, float(cfg_c["sensor"]["target_bg_level"]))
    exposures = sorted({float(rs["exposure_ms"]), 2.5, 5.0, 10.0})
    rng_seed = int(cfg_c["seed"])
    sims = {"none": simulate(Y, gain, cfg_c["sensor"], np.random.default_rng(rng_seed * 1000))}
    for ex in exposures:
        g = row_gains(Y.shape[0], readout, ex, float(rs["flicker_hz"]), float(rs["modulation"]))
        sims[f"{ex:g}"] = simulate(apply_banding(Y, g), gain, cfg_c["sensor"], np.random.default_rng(rng_seed * 1000))
    layout = {d.prim: d for d in load_defects(cfg_c)}
    cam = build_camera(cfg_c["camera"])
    masks = {p: np.isin(inst, idm["prim_to_ids"].get(p, [])) for p in layout}
    alld = np.zeros(inst.shape, bool)
    for m in masks.values():
        alld |= dilate(m, 2)
    zones = {}
    for z in ("cast", "machined"):
        own = sum((idm["prim_to_ids"].get(p, []) for p, d in layout.items() if d.zone == z), [])
        zones[z] = np.isin(sem, idm["class_to_ids"][f"zone_{z}"]) | (defect_cls & np.isin(inst, own))
    results = {}
    for key, sim in sims.items():
        cnrs = {}
        for p, d in layout.items():
            if masks[p].sum() < 3:
                continue
            uv = project_mm(cam, np.array([d.center3_mm]))[0]
            r = measure_defect(sim, masks[p], zones[d.zone], alld & ~dilate(masks[p], 2), cfg_c["analysis"], cfg_c["threshold"], fallback_center_uv=uv)
            cnrs[p] = r["cnr"]
        vals = np.array(list(cnrs.values()), dtype=float)
        results[key] = {"median_cnr": float(np.nanmedian(vals)), "n_detectable": int(np.sum(vals >= 3)), "n": int(len(vals))}
    ys, xs = np.where(zone_m)
    y0, y1, x0, x1 = max(ys.min() - 20, 0), ys.max() + 20, max(xs.min() - 10, 0), xs.max() + 10
    fig, axes = plt.subplots(1, len(sims), figsize=(3.2 * len(sims), 4.6), facecolor=SURFACE)
    for ax, (key, sim) in zip(axes, sims.items()):
        crop = to_uint8(sim[y0:y1, x0:x1])
        ax.imshow(crop, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        amp = 0 if key == "none" else banding_amplitude(float(key), float(rs["flicker_hz"]), float(rs["modulation"]))
        ttl = "no flicker (daylight)" if key == "none" else f"{key} ms exposure\nbands ±{amp*100:.0f}%"
        ax.set_title(ttl, fontsize=9, color=INK)
        ax.text(0.02, 0.02, f"median CNR {results[key]['median_cnr']:.1f}\n{results[key]['n_detectable']}/{results[key]['n']} detectable",
                transform=ax.transAxes, fontsize=8, color="white", va="bottom", bbox=dict(facecolor="black", alpha=0.5, linewidth=0))
    fig.suptitle(f"Rolling shutter under {rs['flicker_hz']:g} Hz flicker ({rs['modulation']*100:.0f}% modulation), {info['sensor']} readout {readout:g} ms: machined half",
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path_img, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    ex = np.linspace(0.5, 40, 400)
    amp = [banding_amplitude(e, float(rs["flicker_hz"]), float(rs["modulation"])) * 100 for e in ex]
    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor=SURFACE)
    style_axes(ax)
    ax.plot(ex, amp, color="#6250d6", linewidth=2)
    for k in (1, 2, 3, 4):
        ax.axvline(1000.0 / rs["flicker_hz"] * k, color=GRID, linewidth=1)
    ax.set_xlabel("exposure time, ms", color=INK2, fontsize=9)
    ax.set_ylabel("band amplitude, % of signal", color=INK2, fontsize=9)
    ax.set_title(f"Banding amplitude against exposure under {rs['flicker_hz']:g} Hz flicker; zero at whole flicker periods", loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path_curve, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    results["exposures_ms"] = exposures
    results["readout_ms"] = readout
    return results


def main():
    a = parse_args()
    cfg = load_config(a.config)
    thr = cfg["threshold"]
    df = pd.read_csv(a.metrics)
    # camera comparison = the coupon at the 300 mm framing; whole-part conditions (asset kind part) are reported separately
    cams = [c for c, s in cfg["conditions"].items() if (s or {}).get("camera", {}).get("model") and c in set(df.condition)
            and (s or {}).get("asset", {}).get("kind", "coupon") != "part"]
    cams.sort(key=lambda c: cfg["cameras"][cfg["conditions"][c]["camera"]["model"]]["mp"])
    if not cams:
        raise SystemExit("no cam_* conditions in metrics.csv")
    figdir = os.path.join(a.out, "figures")
    os.makedirs(figdir, exist_ok=True)
    md = min_detectable(df[df.condition.isin(cams)], thr)
    md.to_csv(os.path.join(a.out, "camera_summary.csv"), index=False)
    table = camera_table(cfg, cams, df)
    fig_min_detectable_by_camera(md, table, os.path.join(figdir, "camera_min_detectable.png"))
    for c in [c for c in cams if c in ("cam_a2A2448", "cam_a2A3840", "cam_a2A5320")]:
        mi = resolve_condition(cfg, c)["camera"]["model_info"]
        fig_lens_effect(df, c, os.path.join(figdir, f"camera_lens_effect_{c}.png"), f"How the lens changes contrast: {mi['sensor']} ({mi['mp']:g} MP, {mi['pixel_um']} µm pixels)")
    sweep_done = {}
    for c in cams:
        if any(str(l).startswith("good_f") for l in df[df.condition == c].lens.unique()):
            cc = resolve_condition(cfg, c)
            sweep_done[c] = fig_f_sweep(df, c, cc, os.path.join(figdir, f"camera_f_sweep_{c}.png"), f"Aperture trade-off for {cc['camera']['model']} ({cc['camera']['model_info']['sensor']})")
    rs_results = {}
    for c in cams:
        cc = resolve_condition(cfg, c)
        if cc["camera"]["model_info"].get("shutter") == "rolling" and os.path.exists(os.path.join(a.out, c, "renders", "hdr_0.exr")):
            rs_results[c] = rolling_shutter_panel(cc, c, a.out, os.path.join(figdir, f"rolling_shutter_{c}.png"), os.path.join(figdir, "rolling_shutter_amplitude.png"))
    with open(os.path.join(a.out, "camera_table.json"), "w") as fh:
        json.dump({"cameras": table, "rolling_shutter": rs_results, "f_sweep_figures": sweep_done}, fh, indent=1, default=float)
    print("[camera_report] cameras:", ", ".join(cams))
    piv = md[md.lens.isin(["none", "good", "budget"])].pivot_table(index=["type", "zone"], columns=["condition", "lens"], values="min_detectable_mm", aggfunc="first")
    pd.set_option("display.width", 250)
    print(piv.to_string())
    for t in table:
        print(f"  {t['model']:17s} {t['mp']:5g} MP  {t['mm_per_px']*1000:5.0f} µm/px  f={t['focal_length_mm']:4.1f} mm  σ_good={t['psf_sigma_px_good']:.2f} px  Airy={t['airy_diameter_px_f8']:.1f} px  DoF={t['depth_of_field_mm_f8']:.0f} mm  t_exp={t['exposure_ms']['300']:.1f}/{t['exposure_ms']['1000']:.1f} ms  €{t['price_eur']}")


if __name__ == "__main__":
    main()
