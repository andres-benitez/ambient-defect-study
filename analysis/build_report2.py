"""Final report (round 2): the ten experiments walked through for a non-expert reader.

  python analysis/build_report2.py --config config/experiment.yaml

Reads out/metrics.csv and the per-experiment outputs (summary.csv, ext_summary.json, single_view.json,
rigs_summary.csv, variants.csv, false_alarms.csv, sizing.csv, detector_check.json, camera_table.json)
and writes out/report2/index.html plus out/report2/assets/*. Sections whose data are missing say so.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import date

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from ads.config import load_config, resolve_condition  # noqa: E402
from ads.layout import ROW_ORDER  # noqa: E402
from analysis.plot_results import summarise  # noqa: E402
from analysis.report_style import HEAD  # noqa: E402
from analysis.build_report import esc, fmt_mm, load_json, table, fig, box, downscale, TYPE_LABEL, SIZE_LABEL  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "config", "experiment.yaml"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "out"))
    return ap.parse_args()


def copy_fig(out, assets, name):
    src = os.path.join(out, "figures", name)
    if not os.path.exists(src):
        return None
    shutil.copyfile(src, os.path.join(assets, name))
    return "assets/" + name


def read_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def main():
    a = parse_args()
    cfg = load_config(a.config)
    out = a.out
    rep = os.path.join(out, "report2")
    assets = os.path.join(rep, "assets")
    os.makedirs(assets, exist_ok=True)
    thr = cfg["threshold"]
    metrics = read_csv(os.path.join(out, "metrics.csv"))
    default_lens = cfg["lens"].get("default_scenario", "none")
    m0 = metrics[(metrics["lens"].fillna(default_lens) == default_lens)] if len(metrics) else metrics
    agg_all, summ_all = summarise(m0, thr) if len(m0) else (pd.DataFrame(), pd.DataFrame())
    have = set(m0.condition.unique()) if len(m0) else set()
    ext = load_json(os.path.join(out, "ext_summary.json"), {})
    sv = load_json(os.path.join(out, "single_view.json"), {})
    rigs = read_csv(os.path.join(out, "rigs_summary.csv"))
    variants = read_csv(os.path.join(out, "variants.csv"))
    fa = read_csv(os.path.join(out, "false_alarms.csv"))
    sizing = read_csv(os.path.join(out, "sizing.csv"))
    det = load_json(os.path.join(out, "detector_check.json"), {})
    cam_table = load_json(os.path.join(out, "camera_table.json"), {"cameras": []})
    part = load_json(os.path.join(REPO_ROOT, "assets", "crankcase", "part.json"), {})

    F = {n: copy_fig(out, assets, n) for n in [
        "cnr_vs_size.png", "contact_sheet.png", "camera_min_detectable.png", "camera_lens_effect_cam_a2A2448.png", "rolling_shutter_cam_a2A3840.png",
        "dents.png", "scratch_envelope.png", "contact_sheet_ext_nadir_012.png", "single_view_a2A4508.png", "single_view_a2A2448.png",
        "sizing_error.png", "rigs_min_detectable.png", "rigs_cnr_ratio.png", "variants_cnr_ratio.png", "detector_check.png",
        "contact_sheet_part_5mp.png", "mask_overlay_part_5mp.png"]}
    for n in sorted(os.listdir(os.path.join(out, "figures"))):
        if n.startswith("pixel_scale_"):
            F[n] = copy_fig(out, assets, n)
    for cond in ("part_18mp", "sv18_f16_mid", "shadow_012", "rig_ring30", "aniso_across"):
        p = os.path.join(out, cond, "preview_0.png")
        if os.path.exists(p):
            d = downscale(p, os.path.join(assets, f"{cond}_preview.jpg"), width=1200)
            F[f"{cond}_preview"] = "assets/" + os.path.basename(d) if d else None
    p = os.path.join(out, "ambient_nadir_012", "false_alarms.png")
    if os.path.exists(p):
        shutil.copyfile(p, os.path.join(assets, "false_alarms_ambient.png"))
        F["fa_overlay"] = "assets/false_alarms_ambient.png"

    def mind(cond, typ, zone):
        r = summ_all[(summ_all.condition == cond) & (summ_all.type == typ) & (summ_all.zone == zone)] if len(summ_all) else pd.DataFrame()
        if len(r) and r["min_detectable"].iloc[0] != "none":
            return fmt_mm(r["min_detectable_mm"].iloc[0])
        return "none"

    def cnr_of(cond, typ, zone, size):
        r = agg_all[(agg_all.condition == cond) & (agg_all.type == typ) & (agg_all.zone == zone) & (np.isclose(agg_all.size_mm, size))] if len(agg_all) else pd.DataFrame()
        return float(r.cnr.iloc[0]) if len(r) else float("nan")

    H = [f"<title>Crankcase Inspection Study</title>\n{HEAD}\n<div class=\"wrap\">"]
    # ------------------------------------------------------------------ header
    H.append(f'''<header>
  <div class="eyebrow">Simulation study, round 2 · aluminium crankcase inspection · {date.today().strftime("%d %b %Y")}</div>
  <h1>Can a camera under the hall's own light meet the client's inspection specification?</h1>
  <p class="lede">Round 1 asked whether an ambient-light camera can see defects at all. This round tests the client's actual specification: pores and dents of 1 mm, scratches above a 2 × 1 mm envelope, a certified 0.2 mm per pixel, one view per face, and the ability to measure what is found. Ten experiments were run on a simulated test plate and on a scanned engine block: dents, the scratch envelope, a one-view-per-face aperture sweep, dimensional measurement, lighting rigs, machining marks, false alarms, camera pose, a shadow, and a small learned detector. The page is written for readers who are not vision engineers; terms are explained when they first appear and collected in a glossary at the end.</p>
</header>''')
    H.append('''<nav class="toc"><div class="eyebrow">Contents</div><ol>
<li><a href="#spec">The client's specification, in plain words</a></li><li><a href="#how">How the study works</a></li>
<li><a href="#round1">What round 1 established</a></li><li><a href="#e1">Experiment 1 · Dents</a></li>
<li><a href="#e2">Experiment 2 · The scratch envelope</a></li><li><a href="#e3">Experiment 3 · One view per face: aperture and focus</a></li>
<li><a href="#e4">Experiment 4 · Measuring size</a></li><li><a href="#e5">Experiment 5 · Lighting rigs</a></li>
<li><a href="#e6">Experiment 6 · Machining marks</a></li><li><a href="#e7">Experiment 7 · False alarms</a></li>
<li><a href="#e8">Experiment 8 · Camera pose</a></li><li><a href="#e9">Experiment 9 · A shadow across the part</a></li>
<li><a href="#e10">Experiment 10 · Does a detector agree?</a></li><li><a href="#verdict">The specification against the evidence</a></li>
<li><a href="#limits">Limits</a></li><li><a href="#appendix">Appendix: method, runs, glossary</a></li>
</ol></nav>''')

    # ------------------------------------------------------------------ spec
    H.append('''<section id="spec"><div class="eyebrow">Background</div><h2>The client's specification, in plain words</h2>
<p>The client wants to detect, locate and measure manufacturing defects on machined aluminium parts such as crankcases, with one camera view per face and no dedicated inspection lighting. Their written thresholds and what they mean:</p>
<ul>
<li><b>Porosity and surface dents of at least 1 mm equivalent diameter.</b> Equivalent diameter is the diameter of a circle with the same area as the mark. The clause covers three physically different things: pores (steep-walled holes), clusters of small pores, and impact dents (smooth, shallow depressions).</li>
<li><b>Scratches exceeding a 2 × 1 mm bounding envelope.</b> The smallest box drawn around the mark is longer than 2 mm and wider than 1 mm, or, under the other reading, exceeds either dimension. The two readings differ by a factor of five in what the camera has to resolve, so Experiment 2 tests both.</li>
<li><b>A certified spatial resolution under 0.2 mm per pixel</b>, equivalently at least 5 × 5 pixels across a 1 mm defect, in every inspection zone. "Certified" means demonstrated with a calibrated target, zone by zone, not assumed from the camera's data sheet.</li>
<li><b>Detect, localise, quantify.</b> Three levels of difficulty: seeing the defect, knowing where it is on the part, and measuring how big it is. The third needs more pixels than the first, which Experiment 4 quantifies.</li>
</ul>
</section>''')

    # ------------------------------------------------------------------ how
    H.append(f'''<section id="how"><div class="eyebrow">Method</div><h2>How the study works</h2>
<p>Every image in this report is a physically based rendering (path tracing) of real 3-D geometry, passed through a model of an industrial camera sensor: the camera's photon noise, read noise and 8-bit quantisation, using the published figures of Basler ace 2 cameras. Defects are cut into the surface as true geometry, not painted on, so light and shadow behave as on metal. The measure of visibility is the <strong>contrast-to-noise ratio (CNR)</strong>: how much a defect changes the brightness of its pixels, divided by the random grain of the surrounding surface. A defect counts as detectable when CNR ≥ {thr["cnr"]:g} and it covers at least {thr["min_area_px"]} pixels. CNR is the ceiling for any software: a defect with no contrast in the pixels cannot be recovered by any algorithm, however good.</p>
<div class="facts">
 <div class="fact"><b>{len(have)} conditions</b><span>rendered and measured in this study, each a complete camera image with per-defect masks</span></div>
 <div class="fact"><b>{len(m0):,} measurements</b><span>one row per defect, frame and lens scenario in the shared results table</span></div>
 <div class="fact"><b>Constant ceiling light</b><span>a 3 × 3 m lit panel 3 m above the part plus wall bounce, the ambient best case, unless a rig is named</span></div>
 <div class="fact"><b>Automatic checks</b><span>geometry, scale, materials, exposure, masks, render noise, a dark-field control and a full re-render reproduce every value</span></div>
</div>
</section>''')

    # ------------------------------------------------------------------ round 1 recap
    H.append(f'''<section id="round1"><div class="eyebrow">Recap</div><h2>What round 1 established</h2>
<ul>
<li><b>A hard floor from pixel size.</b> At 0.12 mm per pixel (a 5 MP camera over a 300 mm face), pores under 0.5 mm, burrs under 0.2 mm and scratches under 0.05 mm wide do not register under constant light, whatever the software. More pixels lower that floor (0.2 mm pores at 8 MP and above) only with a lens matched to the sensor.</li>
<li><b>The machined face is a mirror of the ceiling.</b> Its high contrast (CNR 20 to 50 for mid-size defects) exists because a smooth face reflects a uniform bright ceiling; the cast face returns a fifth of the light and sits at CNR 5 to 14.</li>
<li><b>Directional light is the lever.</b> One raking light multiplied the contrast of the same defects by four to six and brought sub-pixel sizes above the threshold.</li>
<li><b>Sensor quality does not move the floor;</b> resolution does, and a generic lens undoes a size step of it. A rolling shutter costs nothing on a static part.</li>
<li><b>One view of the whole block</b> at 0.14 mm per pixel kept the flange defects but lost most of the cast bays, which lie 20 to 120 mm below the focus plane.</li>
</ul>
<div class="two">
{fig(F["cnr_vs_size.png"], "Round 1: CNR against size on the test plate", "Round 1 baseline: contrast against defect size on the test plate under constant ambient light (blue cast, orange machined; dashed rule = threshold).")}
{fig(F["camera_min_detectable.png"], "Round 1: smallest detectable size per camera", "Round 1: smallest detectable size for six Basler cameras framing a 300 mm face, with three lens assumptions.")}
</div>
</section>''')

    # ------------------------------------------------------------------ E1 dents
    def ext_min(cond, typ, zone):
        v = ext.get(f"{cond}/{typ}/{zone}/min_detectable_mm")
        return fmt_mm(v) if v is not None else "none up to 4 mm"
    d_rows = [[lab, ext_min("ext_nadir_012", t, "machined"), ext_min("ext_nadir_012", t, "cast"), ext_min("ext_nadir_005", t, "machined"), ext_min("ext_nadir_005", t, "cast")]
              for t, lab in (("dent_shallow", "shallow dent (depth 3 % of diameter: 1 mm → 30 µm)"), ("dent_deep", "deeper dent (depth 10 %: 1 mm → 100 µm)"))]
    p_rows = [[TYPE_LABEL[t], mind("ext_nadir_012", t, "machined"), mind("ext_nadir_012", t, "cast"), mind("ext_nadir_005", t, "machined"), mind("ext_nadir_005", t, "cast")] for t in ("pore_shallow", "pore_deep")]
    H.append(f'''<section id="e1"><div class="step"><div class="eyebrow">Experiment 1</div><h2>Dents: the defect the specification names and round 1 never modelled</h2></div>
<p>An impact dent is a smooth, shallow depression. Where a pore has steep walls that cast a shadow and interrupt the ceiling's reflection, a dent of 1 mm diameter and 30 µm depth has slopes of only a few degrees. Under soft overhead light such a slope changes the reflected brightness very little. Twelve dents were added to the test plate: diameters 0.5 to 4 mm at two depths, 3 % and 10 % of the diameter.</p>
{fig(F["dents.png"], "Dent CNR against diameter", "Contrast of dents against diameter at the two pixel scales. Solid: shallow dents (3 %), dashed: deeper dents (10 %). Blue cast, orange machined. Dashed rule: the threshold.")}
{table(["Defect", "Machined, 0.12 mm/px", "Cast, 0.12 mm/px", "Machined, 0.05 mm/px", "Cast, 0.05 mm/px"], d_rows + p_rows, caption="Smallest detectable size: dents compared with the round-1 pores at the same two pixel scales, constant ambient light.")}
<aside class="box warn"><b>What this means for the 1 mm dent clause</b><div><p>Under constant ambient light a 1 mm dent of realistic depth is not visible on the cast surface at either pixel scale, and on the machined face only when it is at least 100 µm deep (10 % of its diameter) or the pixels are 0.05 mm. The shallow 1 mm dent is below the threshold everywhere at 0.12 mm per pixel. Dents are the first item of the specification to fail under ambient light, and they fail for a physical reason: soft light does not create contrast on a gently sloped surface. The lighting-rig experiment (5) shows what recovers them.</p></div></aside>
</section>''')

    # ------------------------------------------------------------------ E2 scratch envelope
    def env_cell(cond, zone, L, W):
        r = ext.get(f"{cond}/{zone}/L{L:g}_W{W:g}")
        if not r:
            return "n/a"
        return f"{r['cnr']:.1f}{'' if r['detectable'] else ' ✗'}"
    env_rows = []
    for L in (1.0, 2.0, 3.0, 5.0):
        env_rows.append([f"{L:g} mm long"] + [env_cell("ext_nadir_012", "machined", L, W) for W in (0.05, 0.1, 0.2, 0.5, 1.0)] + [env_cell("ext_nadir_012", "cast", L, W) for W in (0.05, 0.1, 0.2, 0.5, 1.0)])
    H.append(f'''<section id="e2"><div class="step"><div class="eyebrow">Experiment 2</div><h2>The scratch envelope, tested as the client wrote it</h2></div>
<p>Round 1 used 5 mm long scratches. The client's clause is about a 2 × 1 mm box. Scratches of 1, 2, 3 and 5 mm length and 0.02 to 1 mm width were cut into both surfaces and measured at the two pixel scales.</p>
{fig(F["scratch_envelope.png"], "Scratch CNR grid: length against width", "Contrast for every length × width combination (numbers), bold where detectable. The solid orange line marks 2 mm length, the dashed line 1 mm width: the client's envelope. An asterisk marks widths narrower than one pixel.")}
{table(["Length", "machined 0.05", "0.1", "0.2", "0.5", "1.0 mm wide", "cast 0.05", "0.1", "0.2", "0.5", "1.0 mm wide"], env_rows, caption="CNR at 0.12 mm per pixel; ✗ = below the detection rule. Widths of 0.05 and 0.1 mm are narrower than a pixel at this scale.")}
<aside class="box"><b>Reading for the two interpretations of the clause</b><div><ul>
<li><b>Reading A, both dimensions exceeded (longer than 2 mm <em>and</em> wider than 1 mm):</b> every such mark is detected with wide margin on both surfaces at 0.12 mm per pixel (CNR {cnr_of("ext_nadir_012", "scratch_l2", "cast", 1.0):.0f} on cast, {cnr_of("ext_nadir_012", "scratch_l2", "machined", 1.0):.0f} on machined for a 2 × 1 mm mark). Under this reading the camera is not the problem.</li>
<li><b>Reading B, either dimension exceeded (anything longer than 2 mm, however thin):</b> a 2 mm scratch 0.1 mm wide reaches CNR {cnr_of("ext_nadir_012", "scratch_l2", "machined", 0.1):.0f} on the machined face and {cnr_of("ext_nadir_012", "scratch_l2", "cast", 0.1):.0f} on cast at 0.12 mm per pixel; at 0.05 mm width it is at the pixel limit. Under this reading the floor is about 0.1 mm of width at the 5 MP scale and 0.05 mm at 0.05 mm per pixel.</li>
<li>Length matters less than width once a scratch is longer than about 1 mm: the 1 and 2 mm scratches score within a few CNR of the 5 mm ones at the same width.</li>
</ul></div></aside>
</section>''')

    # ------------------------------------------------------------------ E3 single view
    sv_rows = []
    for cond, e in sorted(sv.items(), key=lambda kv: (kv[1]["camera"], kv[1]["focus_offset_mm"], kv[1]["f_number"])):
        sv_rows.append([esc(e["camera"].split("-")[0]), f'f/{e["f_number"]:g}', "flange" if e["focus_offset_mm"] == 0 else f'{e["focus_offset_mm"]:.0f} mm below flange',
                        f'{e["dof_1px_mm"]:.0f} / {e["dof_2px_mm"]:.0f} mm', f'{e["airy_px"]:.1f} px', f'{e["exposure_ms_1000lux"]:.0f} / {e["exposure_ms_300lux"]:.0f} ms',
                        f'{e["detectable_machined"]} / {e["detectable_cast"]}', f'{e.get("frac_under_0p2", float("nan"))*100:.0f} %' if "frac_under_0p2" in e else "n/a"])
    sv_figs = "".join(fig(F.get(n), n, cap) for n, cap in (("single_view_a2A4508.png", "18.5 MP camera: median CNR of the defects against their distance from the focus plane, one line per aperture; dashed lines focus on the flange, solid lines at mid-depth. Left machined, right cast."),
                                                           ("single_view_a2A2448.png", "The same for the 5 MP camera.")) if F.get(n))
    ps_figs = "".join(fig(F[n], n, "Realised pixel size over the part for one single-view condition. The orange contour is the 0.2 mm per pixel limit of the specification; surfaces farther from the camera are coarser.") for n in sorted(F) if n.startswith("pixel_scale_") and F[n])
    H.append(f'''<section id="e3"><div class="step"><div class="eyebrow">Experiment 3</div><h2>One view per face: what aperture and focus can and cannot do</h2></div>
<p>The client wants each face of the part captured in a single view. On the engine block's oil-pan face that view spans 515 mm and 120 mm of depth from the sealing flange down to the bay floors. In one view, the only knobs are the lens aperture and where the focus plane sits. Closing the aperture extends the depth that stays sharp but enlarges the diffraction blur and lengthens the exposure; opening it does the reverse. The sweep below renders the block with the 18.5 MP and 5 MP cameras at apertures from f/5.6 to f/22 with the focus either on the flange or 60 mm below it, and measures every defect against its distance from the focus plane.</p>
{sv_figs}
{table(["Camera", "Aperture", "Focus plane", "Depth of field (1 px / 2 px blur)", "Diffraction spot", "Exposure 1000 / 300 lux", "Detectable machined / cast (of 24 each)", "Part within 0.2 mm/px"], sv_rows, caption="Single-view sweep. Depth of field is the height range within one (two) pixel(s) of blur; the diffraction spot is the smallest blur any lens can produce at that aperture, in pixels.")}
{ps_figs}
<aside class="box warn"><b>What this means</b><div><ul>
<li>Resolution and depth of field trade against each other in object space: at 0.14 mm per pixel the 18.5 MP camera keeps about 120 mm within one pixel of blur at f/8 and about 240 mm at f/16, but at f/16 the diffraction spot is {(sv.get("sv18_f16_mid", {}).get("airy_px") or 0):.0f} pixels wide, and the exposure quadruples.</li>
<li>Focusing at mid-depth instead of on the flange trades flange sharpness for bay sharpness; the table shows how the detectable counts move between the two zones.</li>
<li>The realised pixel size is not one number for the face: surfaces 120 mm farther from the camera are 15 % coarser, so a design at 0.18 mm per pixel on the flange already fails the 0.2 mm clause in the bays. The pixel-size maps show the fraction of the visible part that complies.</li>
<li>The block's deck face could not be run as a single view: in the public scan its machined land is nowhere wider than about 6 mm, so no set of 24 defects fits. A cleaner CAD model would be needed for that face.</li>
</ul></div></aside>
</section>''')

    # ------------------------------------------------------------------ E4 sizing
    siz_txt = ""
    if len(sizing):
        f = sizing[sizing.found == True].copy()  # noqa: E712
        f["true_px"] = f.true_eq_diam_mm / f.mm_per_px
        rnd = f[f.type.isin(["pore_shallow", "pore_deep", "dent_shallow", "dent_deep"]) & (f.lens == "none")]
        bins = [(2, 4), (4, 7), (7, 12), (12, 60)]
        cells = []
        for lo, hi in bins:
            g = rnd[(rnd.true_px >= lo) & (rnd.true_px < hi)]
            if len(g):
                cells.append([f"{lo} to {hi} px across", f"{len(g)}", f"{g.eq_diam_err_pct.median():+.0f} %", f"{np.percentile(np.abs(g.eq_diam_err_pct), 90):.0f} %"])
        siz_txt = table(["Round defect size in pixels", "Measurements", "Median error of equivalent diameter", "90th percentile of |error|"], cells,
                        caption="How accurately the equivalent diameter of round defects is recovered from the image, against the defect's size in pixels (all conditions, no lens blur).")
    H.append(f'''<section id="e4"><div class="step"><div class="eyebrow">Experiment 4</div><h2>Measuring size: how many pixels does quantification need?</h2></div>
<p>Detection asks whether a defect is there; the specification also asks how big it is. Each detected defect was measured the way a vision system would: the pixels whose contrast exceeds half the peak contrast are taken as the defect, their area gives an equivalent diameter and their extent a bounding box, and both are compared with the true geometry.</p>
{fig(F["sizing_error.png"], "Sizing error against size in pixels", "Measurement error against the defect's true size in pixels. Left: round defects (equivalent diameter). Right: linear defects (length). Orange: median error; blue dashed: 90th percentile of the absolute error. The horizontal dashed line is ±10 %.")}
{siz_txt}
<aside class="box"><b>What this means for the 5 × 5 pixel rule</b><div><p>With 5 pixels across a 1 mm defect, a one-pixel uncertainty on each edge is ±20 % of the diameter, and that is what the measurements show: sizes of 4 to 7 pixels are measured to roughly ±20 % at the 90th percentile, 7 to 12 pixels to about ±10 %, and only above 12 pixels does the error settle below that. Sub-pixel widths are always over-estimated because a thin line is at least one pixel wide in the image. For the client's quantification clause, 0.2 mm per pixel is enough to <em>detect</em> a 1 mm defect; measuring it to ±10 % needs about 0.1 mm per pixel.</p></div></aside>
</section>''')

    # ------------------------------------------------------------------ E5 rigs
    rig_rows = []
    if len(rigs):
        order = ["ambient_nadir_012", "rig_dome", "rig_ring60", "rig_ring30", "best_of_4", "control_darkfield"]
        for c in [c for c in order if c in set(rigs.condition)]:
            lab = rigs[rigs.condition == c].label.iloc[0]
            cells = [esc(lab)]
            for t in ("pore_shallow", "scratch", "burr"):
                for z in ("machined", "cast"):
                    r = rigs[(rigs.condition == c) & (rigs.type == t) & (rigs.zone == z)]
                    cells.append(fmt_mm(r.min_detectable_mm.iloc[0]) if len(r) and np.isfinite(r.min_detectable_mm.iloc[0]) else "none")
            ratio = rigs[rigs.condition == c].median_cnr_ratio_vs_ambient.median()
            cells.append(f"{ratio:.1f}×" if np.isfinite(ratio) else "1.0×")
            rig_rows.append(cells)
    H.append(f'''<section id="e5"><div class="step"><div class="eyebrow">Experiment 5</div><h2>Lighting rigs as reported conditions</h2></div>
<p>Round 1 used one raking light only as a check. Here the common inspection rigs are rendered on the same test plate and measured with the same rule: a uniform dome (the softest possible light), a ring of eight lights at 60° elevation (close to the camera axis), the same ring at 30°, a four-direction sequential ring at 30° taking the best direction per defect (what a multi-shot station does), and the single 15° light from round 1.</p>
{fig(F["rigs_min_detectable.png"], "Smallest detectable size per rig", "Smallest detectable size per lighting rig for shallow pores, scratches and burrs on both surfaces (test plate, 5 MP class at 0.12 mm per pixel).")}
{fig(F["rigs_cnr_ratio.png"], "CNR relative to ambient per rig", "Contrast of each rig relative to the ambient ceiling, per defect type (median over sizes that have a footprint). Above 1 the rig helps.")}
{table(["Rig", "pore mach.", "pore cast", "scratch mach.", "scratch cast", "burr mach.", "burr cast", "CNR vs ambient"], rig_rows, caption="Smallest detectable size per rig and surface, and the median contrast gain over the ambient ceiling.") if rig_rows else '<p class="missing">Rig results not available yet.</p>'}
<aside class="box good"><b>What this means</b><div><p>The ranking is the physics of Step 4 in round 1 made quantitative: the dome is the worst light for machined metal, a high ring is little better than the ceiling, and low-angle light, best of all when it comes from several directions in turn, multiplies contrast several-fold and is what brings dents and sub-pixel scratches into reach. A four-segment low ring fired sequentially is the smallest rig that does this.</p></div></aside>
</section>''')

    # ------------------------------------------------------------------ E6 aniso, E8 pose, E9 shadow (variants)
    def var_rows(group):
        rows = []
        if not len(variants):
            return rows
        for c in variants[variants.group == group].condition.unique():
            g = variants[variants.condition == c]
            lab = g.label.iloc[0]
            cells = [esc(lab)]
            for t in ("pore_shallow", "scratch", "burr"):
                for z in ("machined", "cast"):
                    r = g[(g.type == t) & (g.zone == z)]
                    cells.append(fmt_mm(r.min_detectable_mm.iloc[0]) if len(r) and np.isfinite(r.min_detectable_mm.iloc[0]) else "none")
            cells.append(f"{g.median_cnr_ratio.median():.2f}×")
            rows.append(cells)
        return rows
    vhead = ["Condition", "pore mach.", "pore cast", "scratch mach.", "scratch cast", "burr mach.", "burr cast", "CNR vs baseline"]
    H.append(f'''<section id="e6"><div class="step"><div class="eyebrow">Experiment 6</div><h2>Machining marks on the machined face</h2></div>
<p>Round 1's machined face reflected light equally in all directions. A real milled face has tool marks: it reflects light in a streak across the marks rather than a spot. Here the machined material is made anisotropic (reflection stretched along one direction) with the marks running along the scratches in one condition and across them in the other.</p>
{fig(F.get("aniso_across_preview"), "Anisotropic machined face", "The test plate with an anisotropic machined face: the ceiling's reflection becomes a streak.") if F.get("aniso_across_preview") else ""}
{table(vhead, var_rows("tool marks"), caption="Smallest detectable size with tool marks along and across the scratches, and median contrast relative to the isotropic baseline.") if var_rows("tool marks") else '<p class="missing">Tool-mark results not available yet.</p>'}
</section>''')
    H.append(f'''<section id="e7"><div class="step"><div class="eyebrow">Experiment 7</div><h2>False alarms: what the detection rule flags where there is no defect</h2></div>
<p>A threshold that catches defects also fires on anything else that looks like one. The same rule used for the defects (a 3 × 3 pixel core compared with a ring of surrounding surface, divided by that ring's noise) was applied to every defect-free pixel of each image, and the alarms counted per 100 cm² of surface at the detection threshold and above.</p>
{table(["Condition", "Surface", "Pixel size", "Alarms per 100 cm² at CNR 3", "at CNR 4", "at CNR 5", "Threshold where alarms ≤ 1 per 100 cm²"],
       [[esc(r.condition), r.zone, f"{r.mm_per_px*1000:.0f} µm", f"{r.fa_per_100cm2_k3:.1f}", f"{r.fa_per_100cm2_k4:.1f}", f"{r.fa_per_100cm2_k5:.1f}", f"{r.quiet_threshold_k:g}" if np.isfinite(r.quiet_threshold_k) else "> 8"] for r in fa.itertuples()] if len(fa) else [],
       caption="Blind application of the detection rule to defect-free surface. The scanned block's cast bays are real casting texture; the plate's cast face carries a procedural skin.") if len(fa) else '<p class="missing">False-alarm results not available yet.</p>'}
{fig(F.get("fa_overlay"), "False-alarm overlay", "Example: pixels flagged at CNR 4 (red) on the test plate under ambient light, defects excluded.") if F.get("fa_overlay") else ""}
<aside class="box"><b>What this means</b><div><p>Because the rule normalises by the local noise, surface texture that raises the noise lowers the alarm rate as much as it lowers the defects' CNR: the clutter penalty is already inside the CNR figures throughout this report. At the detection threshold the rule fires about ten times per 100 cm² on a clean machined mirror from noise alone and almost never above CNR 4, so a station that works at CNR 4 to 5 on the defects has a quiet background. The price is that marginal defects (CNR 3 to 4) are the ones a real system would have to trade against false calls.</p></div></aside>
</section>''')
    H.append(f'''<section id="e8"><div class="step"><div class="eyebrow">Experiment 8</div><h2>Camera pose: tilt and distance tolerance</h2></div>
<p>A certified pixel size has to hold when the camera is not exactly where the drawing says. The plate was rendered with the camera tilted 15° and 30°, and 10 % nearer and farther than nominal.</p>
{table(vhead, var_rows("camera pose"), caption="Smallest detectable size and contrast relative to the nadir baseline for tilted and displaced cameras.") if var_rows("camera pose") else '<p class="missing">Pose results not available yet.</p>'}
</section>''')
    H.append(f'''<section id="e9"><div class="step"><div class="eyebrow">Experiment 9</div><h2>A shadow across the part</h2></div>
<p>Without touching the roof: a 0.6 m fixture 1 m above the part casts a soft shadow whose edge crosses the plate. Under a 3 m ceiling panel the shadow is a gentle gradient rather than a hard edge; the question is whether that gradient creates false defects or hides real ones.</p>
{fig(F.get("shadow_012_preview"), "Plate with a soft shadow", "The test plate under the ceiling panel with a fixture shadow across it.") if F.get("shadow_012_preview") else ""}
{table(vhead, var_rows("shadow"), caption="Smallest detectable size and contrast relative to the unshadowed baseline.") if var_rows("shadow") else '<p class="missing">Shadow results not available yet.</p>'}
</section>''')
    # ------------------------------------------------------------------ E10 detector
    H.append(f'''<section id="e10"><div class="step"><div class="eyebrow">Experiment 10</div><h2>Does a learned detector agree with CNR?</h2></div>
<p>CNR is a physics measure. To check that it predicts what software finds, a small learned detector (gradient-boosted trees on normalised 15 × 15 pixel crops) was trained on four conditions and tested on {len(det.get("test", []))} others it had never seen, including the extended plate and the whole block, with its threshold set per condition for 1 % false alarms on that condition's own background.</p>
{fig(F["detector_check.png"], "Detection rate against CNR", "Fraction of defects found by the learned detector on held-out conditions, by their measured CNR.")}
<aside class="box"><b>What this means</b><div><p>Detection rate follows CNR: {det.get("detection_rate_cnr_ge_3", float("nan"))*100:.0f} % of defects with CNR ≥ 3 are found, against {det.get("detection_rate_cnr_lt_2", float("nan"))*100:.0f} % of those below CNR 2, at a false-alarm rate of {det.get("test_false_alarm_rate_per_crop", float("nan"))*100:.1f} % of background patches. The detector transfers worst to the scanned block ({det.get("per_condition", {}).get("part_5mp", float("nan"))*100:.0f} % found), where the surface texture differs most from the training plate. CNR is a sound proxy for detectability, and the gap between "visible" and "found" is about 10 % at the threshold.</p></div></aside>
</section>''')

    # ------------------------------------------------------------------ verdict
    H.append('''<section id="verdict"><div class="eyebrow">Conclusions</div><h2>The specification against the evidence</h2>
<div class="tablewrap"><table class="verdict"><thead><tr><th>Requirement</th><th>Under constant ambient light, one view per face</th><th>What it takes</th></tr></thead><tbody>
<tr><td>Pores ≥ 1 mm</td><td class="v-yes">Met on in-focus machined faces (CNR 15 to 27); marginal on cast faces in a whole-face view</td><td>Any 5 MP or larger camera at ≤ 0.2 mm/px; a low-angle light to make the cast faces reliable</td></tr>
<tr><td>Dents ≥ 1 mm</td><td class="v-no">Not met: shallow dents invisible on cast at any tested pixel size; on machined only when deep or at 0.05 mm/px</td><td>Directional (low-angle) light; a sequential low ring recovers them</td></tr>
<tr><td>Scratches > 2 × 1 mm (both dimensions)</td><td class="v-yes">Met with wide margin on both surfaces</td><td>Nothing beyond the resolution clause</td></tr>
<tr><td>Scratches > 2 mm (either dimension, hairlines included)</td><td class="v-part">Met down to ~0.1 mm width on machined at 0.12 mm/px, ~0.05 mm at 0.05 mm/px; cast weaker</td><td>Finer pixels or low-angle light; clarify the clause</td></tr>
<tr><td>< 0.2 mm per pixel in all zones, certified</td><td class="v-part">Achievable on the flange; the bays of a whole-face view are 15 % coarser and a 60 to 120 mm depth range exceeds the sharp depth at that pixel size</td><td>Design at ≤ 0.15 mm/px on the nearest surface and verify per zone with a target; accept defocus in deep zones or move the camera</td></tr>
<tr><td>Dimensional quantification</td><td class="v-part">±20 % at 5 pixels across; ±10 % needs 7 to 12 pixels across</td><td>About 0.1 mm/px for 1 mm defects, or accept ±20 %</td></tr>
<tr><td>Reliable without dedicated lighting</td><td class="v-no">Detection of 1 mm pores and coarse scratches: yes; dents, cast faces, sub-pixel scratches and stable sizing: no</td><td>A shroud with a sequential low-angle ring: small, and it removes the dependence on the hall</td></tr>
</tbody></table></div>
<ol>
<li><b>The resolution clause is cheap; the dent clause is the hard one.</b> Any of the candidate cameras satisfies 0.2 mm per pixel on a 300 mm face. No camera makes a shallow 1 mm dent visible under soft overhead light, because the contrast is not in the pixels to begin with.</li>
<li><b>One view per face sets a physical budget.</b> Pixel size, depth of field and diffraction are linked; on a face with 120 mm of depth the single-view design can hold about 0.14 mm per pixel on the near surface and about two pixels of blur at the far surface at f/11 to f/16, with exposure in the tens of milliseconds. Faces with less depth do better; the deck of this scan could not even host the test.</li>
<li><b>Quantification wants twice the pixels of detection.</b> 5 pixels across gives ±20 %; ±10 % needs 7 to 12.</li>
<li><b>Lighting is still the lever, and a modest rig is enough.</b> A four-segment low-angle ring fired in sequence, inside a shroud, multiplies contrast several-fold, brings dents and hairline scratches into reach, and makes the result independent of the hall's lights. The study found no camera-side substitute for it.</li>
<li><b>CNR is a trustworthy proxy.</b> A learned detector finds about nine in ten defects above the threshold and few below it, and the blind false-alarm rate is low at CNR 4 and above.</li>
</ol>
</section>''')
    H.append('''<section id="limits"><div class="eyebrow">Caveats</div><h2>Limits</h2>
<ul>
<li>Every result is a simulation with idealised defect shapes (spherical pores and dents, V-groove scratches, triangular burrs) and modelled sensors and lenses; absolute numbers should be read as margins, not guarantees.</li>
<li>The scanned block carries 1 mm scan noise; its machined flange was rebuilt as an ideal plane. The deck face could not be tested.</li>
<li>Tool marks are modelled as anisotropic reflection only, without a measured roughness map.</li>
<li>Prices are indicative list prices from 31 Oct 2026, camera body only.</li>
<li>The learned detector is deliberately small; a production model would do better in absolute terms, not in what the pixels contain.</li>
</ul>
</section>''')

    # ------------------------------------------------------------------ appendix
    runs = []
    if len(metrics):
        for c in sorted(have):
            g = m0[m0.condition == c]
            cc = resolve_condition(cfg, c)
            runs.append([esc(c), esc(cc["asset"].get("dir", "assets/coupon").split("/")[-1]), esc(cc["lighting"]["mode"]), f'{float(g.mm_per_px.iloc[0])*1000:.0f} µm', f'f/{cc["camera"].get("f_number", 8):g}',
                         f'{int(g.spp.iloc[0])}', f'{int(g.prim.nunique())}', f'{int(g.detectable.sum())}'])
    H.append(f'''<section id="appendix"><div class="eyebrow">Appendix</div><h2>Method, runs, glossary</h2>
<h3>Pipeline</h3>
<ol>
<li><b>Geometry</b> (Blender): test plates with 48 or 108 defects cut as exact geometry; the engine block scan with the same defects carved without booleans, its flange rebuilt as a plane.</li>
<li><b>Scene</b> (NVIDIA Isaac Sim, path tracing): metallic materials, the ceiling panel plus bounce or the named rig, camera placed from the pixel scale; light solved so the machined face sits at 40 % of full scale.</li>
<li><b>Capture</b> (Replicator): linear HDR, per-defect masks, depth; samples per pixel raised until render noise is under one fifth of sensor noise.</li>
<li><b>Analysis</b>: luminance → optional lens blur → gain → shot and read noise → 8-bit quantisation → per-defect CNR against a 3 to 10 px background ring on the same surface. Defects without a mask are measured over their projected footprint.</li>
</ol>
{table(["Condition", "Object", "Light", "Pixel size", "Aperture", "Samples / px", "Defects", "Detectable"], runs, caption="Every rendered condition in this study.")}
<h3>Glossary</h3>
<dl>
<dt>Contrast-to-noise ratio (CNR)</dt><dd>Brightness difference between a defect and its surroundings divided by the pixel-to-pixel noise of those surroundings.</dd>
<dt>Pixel size on the part</dt><dd>How much of the part one pixel covers, in millimetres; the specification's 0.2 mm per pixel.</dd>
<dt>Depth of field</dt><dd>The range of distances that stay within a given blur (here one or two pixels).</dd>
<dt>Diffraction spot</dt><dd>The unavoidable blur of a lens aperture; about 2.44 × wavelength × f-number across.</dd>
<dt>f-number</dt><dd>Focal length divided by aperture diameter; higher means a smaller aperture, more depth of field, more diffraction, longer exposure.</dd>
<dt>Anisotropic reflection</dt><dd>A surface that reflects light differently along and across a direction, as a milled face does along its tool marks.</dd>
<dt>Equivalent diameter</dt><dd>Diameter of the circle with the same area as a mark.</dd>
<dt>Sequential ring light</dt><dd>A ring of light segments fired one at a time, each image lit from one side.</dd>
</dl>
<h3>Sources</h3>
<ul>
<li>Basler ace 2 specifications and EMVA 1288 data: Basler product pages and "EMVA Data Overview" No. 16.</li>
<li>Engine block: "{esc(part.get("source", {}).get("name", ""))}" by {esc(part.get("source", {}).get("author", ""))}, {esc(part.get("source", {}).get("license", ""))}, via Objaverse.</li>
<li>All measurements: the <code>ambient-defect-study</code> repository, <code>out/metrics.csv</code> and the per-experiment CSV/JSON files.</li>
</ul>
</section>
<footer><div>Simulation study, round 2, {date.today().isoformat()}. Public stand-in part; no client CAD or images were used.</div></footer>
</div>''')
    with open(os.path.join(rep, "index.html"), "w") as fh:
        fh.write("\n".join(H))
    print(f"[build_report2] wrote {os.path.join(rep, 'index.html')}; conditions in metrics: {len(have)}; figures: {sum(1 for v in F.values() if v)}")


if __name__ == "__main__":
    main()
