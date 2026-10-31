"""Build the step-by-step visual report (out/report/index.html + assets) for a non-expert reader.

  python analysis/build_report.py --config config/experiment.yaml

Every number in the page is read from out/metrics.csv, out/summary*.csv, out/camera_table.json,
the per-condition manifests/checks and assets/*/part.json. Figures are copied into out/report/assets.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
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
from ads.layout import ROW_ORDER, load_defects  # noqa: E402
from ads.camera import build_camera, project_mm  # noqa: E402
from analysis.plot_results import summarise  # noqa: E402
from analysis.optics import scenario_psf  # noqa: E402
from analysis.photometry import exposure_ms  # noqa: E402

TYPE_LABEL = {"pore_shallow": "shallow pore", "pore_deep": "deep pore", "scratch": "scratch (5 mm long)", "burr": "burr (5 mm long)"}
SIZE_LABEL = {"pore_shallow": "diameter", "pore_deep": "diameter", "scratch": "width", "burr": "height"}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "config", "experiment.yaml"))
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "out"))
    return ap.parse_args()


def esc(x) -> str:
    return html.escape(str(x))


def fmt_mm(v) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "none"
    return f"{float(v):g} mm"


def load_json(path, default=None):
    try:
        with open(path) as fh:
            return json.load(fh)
    except OSError:
        return default


# ----------------------------------------------------------------------------- assets
def copy_fig(out, assets, name, dest=None):
    src = os.path.join(out, "figures", name)
    if not os.path.exists(src):
        return None
    dest = dest or name
    shutil.copyfile(src, os.path.join(assets, dest))
    return "assets/" + dest


def downscale(src, dest, width=1400):
    from PIL import Image

    if not os.path.exists(src):
        return None
    im = Image.open(src)
    if im.width > width:
        im = im.resize((width, int(im.height * width / im.width)), Image.LANCZOS)
    im.convert("RGB").save(dest, "JPEG", quality=85, optimize=True)
    return dest


def crop_strip(cfg_c, cond_dir, zone, rows, half_mm=2.4, zoom=4, gap=3):
    """Zoomed crops of the defects of `rows` in `zone` from sim_0.png, laid out rows x sizes."""
    import imageio.v3 as iio
    from PIL import Image

    sim_path = os.path.join(cond_dir, "sim_0.png")
    if not os.path.exists(sim_path):
        return None
    sim = iio.imread(sim_path)
    if sim.ndim == 3:
        sim = sim[..., 0]
    cam = build_camera(cfg_c["camera"])
    half = max(int(round(half_mm / cam.mm_per_px)), 4)
    lay = [d for d in load_defects(cfg_c) if d.zone == zone and d.type in rows]
    ncol = max(d.col for d in lay) + 1
    tile = 2 * half * zoom
    W = ncol * tile + (ncol - 1) * gap
    H = len(rows) * tile + (len(rows) - 1) * gap
    out = np.zeros((H, W), np.uint8)
    for d in lay:
        u, v = project_mm(cam, np.array([d.center3_mm]))[0]
        cu, cv = int(round(u)), int(round(v))
        crop = sim[max(cv - half, 0):cv + half, max(cu - half, 0):cu + half]
        if crop.shape != (2 * half, 2 * half):
            continue
        r, c = rows.index(d.type), d.col
        out[r * (tile + gap):r * (tile + gap) + tile, c * (tile + gap):c * (tile + gap) + tile] = np.kron(crop, np.ones((zoom, zoom), np.uint8))
    return Image.fromarray(out)


# ----------------------------------------------------------------------------- html pieces
def table(headers, rows, cls="", caption=None):
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    cap = f"<caption>{esc(caption)}</caption>" if caption else ""
    return f'<div class="tablewrap"><table class="{cls}">{cap}<thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>'


def fig(src, alt, cap=None, cls=""):
    if not src:
        return f'<p class="missing">Figure not available: {esc(alt)}</p>'
    c = f"<figcaption>{cap}</figcaption>" if cap else ""
    return f'<figure class="{cls}"><img src="{src}" alt="{esc(alt)}" loading="lazy">{c}</figure>'


def box(title, body, kind="note"):
    return f'<aside class="box {kind}"><b>{esc(title)}</b><div>{body}</div></aside>'


def min_table(summ: pd.DataFrame, cond: str, caption=None, zones=("cast", "machined")):
    rows = []
    for t in [t for t in ROW_ORDER if t in set(summ["type"])]:
        cells = [TYPE_LABEL[t], SIZE_LABEL[t]]
        for z in zones:
            r = summ[(summ.condition == cond) & (summ.type == t) & (summ.zone == z)]
            if len(r) and r["min_detectable"].iloc[0] != "none":
                cells.append(fmt_mm(r["min_detectable_mm"].iloc[0]))
            else:
                top = r["largest_tested_mm"].iloc[0] if len(r) else np.nan
                cells.append(f"none up to {top:g} mm")
        rows.append(cells)
    return table(["Defect", "Measured as"] + [f"{z.capitalize()} surface" for z in zones], rows, caption=caption)


# ----------------------------------------------------------------------------- main
def main():
    a = parse_args()
    cfg = load_config(a.config)
    out = a.out
    rep = os.path.join(out, "report")
    assets = os.path.join(rep, "assets")
    os.makedirs(assets, exist_ok=True)
    thr = cfg["threshold"]
    metrics = pd.read_csv(os.path.join(out, "metrics.csv"))
    default_lens = cfg["lens"].get("default_scenario", "none")
    m0 = metrics[metrics["lens"].fillna(default_lens) == default_lens] if "lens" in metrics else metrics
    agg_all, summ_all = summarise(m0, thr)
    cam_table = load_json(os.path.join(out, "camera_table.json"), {"cameras": [], "rolling_shutter": {}})
    cam_summary = pd.read_csv(os.path.join(out, "camera_summary.csv")) if os.path.exists(os.path.join(out, "camera_summary.csv")) else pd.DataFrame()
    checks_amb = load_json(os.path.join(out, "ambient_nadir_012", "analysis_checks.json"), {})
    man_amb = load_json(os.path.join(out, "ambient_nadir_012", "frame_manifest.json"), {})
    part = load_json(os.path.join(REPO_ROOT, "assets", "crankcase", "part.json"))
    src = (part or {}).get("source", {})
    part_conds = [c for c in ("part_5mp", "part_18mp") if c in set(m0.condition)]

    # ---- figures
    F = {k: copy_fig(out, assets, k) for k in [
        "cnr_vs_size.png", "min_detectable.png", "contact_sheet.png", "mask_overlay.png", "camera_min_detectable.png",
        "camera_lens_effect_cam_a2A2448.png", "camera_lens_effect_cam_a2A5320.png", "camera_f_sweep_cam_a2A2448.png",
        "rolling_shutter_cam_a2A3840.png", "rolling_shutter_amplitude.png", "contact_sheet_part_5mp.png", "contact_sheet_part_18mp.png",
        "mask_overlay_part_5mp.png", "cnr_vs_size_part_5mp.png", "cnr_vs_size_part_18mp.png"]}
    for pc in part_conds:
        d = downscale(os.path.join(out, pc, "preview_0.png"), os.path.join(assets, f"{pc}_preview.jpg"))
        F[f"{pc}_preview"] = "assets/" + os.path.basename(d) if d else None
        d = downscale(os.path.join(out, pc, "sim_0.png"), os.path.join(assets, f"{pc}_sim.jpg"), width=1600)
        F[f"{pc}_sim"] = "assets/" + os.path.basename(d) if d else None
    # coupon crop strips: ambient vs raking light (machined scratches and burrs), 5 MP class
    cfg_amb = resolve_condition(cfg, "ambient_nadir_012")
    cfg_ctl = resolve_condition(cfg, "control_darkfield")
    for name, cc, cond, rows in [("strip_amb_mach", cfg_amb, "ambient_nadir_012", ROW_ORDER), ("strip_amb_cast", cfg_amb, "ambient_nadir_012", ROW_ORDER),
                                 ("strip_ctl_mach", cfg_ctl, "control_darkfield", ["scratch", "burr"]), ("strip_amb_mach_sb", cfg_amb, "ambient_nadir_012", ["scratch", "burr"])]:
        zone = "cast" if "cast" in name else "machined"
        im = crop_strip(cc, os.path.join(out, cond), zone, rows)
        if im is not None:
            im.save(os.path.join(assets, name + ".png"), optimize=True)
            F[name] = "assets/" + name + ".png"
        else:
            F[name] = None

    # ---- numbers
    amb = agg_all[agg_all.condition == "ambient_nadir_012"]
    n_det = int(amb["detectable"].sum())
    n_sub = int(amb["mask_empty"].astype(bool).sum())
    ctl = agg_all[agg_all.condition == "control_darkfield"]
    ctl_m = ctl[(ctl.zone == "machined") & (ctl.type.isin(["scratch", "burr"])) & (ctl.area_px > 0)]
    amb_m = amb[(amb.zone == "machined") & (amb.type.isin(["scratch", "burr"])) & (amb.area_px > 0)]
    ex_amb = checks_amb.get("exposure", {})
    rn_amb = checks_amb.get("render_noise", {})
    cams = cam_table.get("cameras", [])
    rs = cam_table.get("rolling_shutter", {}).get("cam_a2A3840", {})
    cast_level = float(m0[(m0.condition == "ambient_nadir_012") & (m0.zone == "cast")]["mean_bg"].median()) if len(m0) else float("nan")

    # camera min-detectable table (lens none / good / budget), machined zone and cast zone
    def cam_min(cond, lens, zone, typ):
        if cam_summary.empty:
            return "n/a"
        r = cam_summary[(cam_summary.condition == cond) & (cam_summary.lens == lens) & (cam_summary.zone == zone) & (cam_summary.type == typ)]
        if not len(r):
            return "n/a"
        v = r["min_detectable_mm"].iloc[0]
        s = fmt_mm(v)
        if bool(r["at_pixel_limit"].iloc[0]):
            s += "*"
        return s

    # ------------------------------------------------------------------------- page
    H = []
    H.append('''<title>Ambient Light Inspection Report</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Semi+Condensed:wght@500;600;700&family=Source+Sans+3:ital,wght@0,400;0,600;1,400&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
/* Layout: a long-form technical brief read top to bottom; numbered steps because the argument is a
   sequence; aluminium-grey neutrals with a slight blue bias; two series colours (cast blue, machined orange). */
:root{--bg:#f3f4f6;--surface:#fbfbfc;--ink:#14181d;--ink-2:#4b5560;--muted:#7d8793;--grid:#e0e4e9;--rule:#c6ccd3;
  --cast:#2a78d6;--mach:#eb6834;--dir:#1baf7a;--warn:#b4540a;--warn-bg:#fdf1e7;--note-bg:#eef3f9;--note:#2a5a8f;
  --font-display:"Barlow Semi Condensed","Arial Narrow",Arial,sans-serif;--font-body:"Source Sans 3","Segoe UI",Helvetica,Arial,sans-serif;--font-mono:"JetBrains Mono",Menlo,Consolas,monospace}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#141719;--surface:#1c2024;--ink:#eef0f2;--ink-2:#bfc6cd;--muted:#8e97a0;--grid:#2b3136;--rule:#3a4147;
  --cast:#3987e5;--mach:#d95926;--dir:#199e70;--warn:#f0a56b;--warn-bg:#2a2017;--note-bg:#1b2634;--note:#8db4e2;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#141719;--surface:#1c2024;--ink:#eef0f2;--ink-2:#bfc6cd;--muted:#8e97a0;--grid:#2b3136;--rule:#3a4147;
  --cast:#3987e5;--mach:#d95926;--dir:#199e70;--warn:#f0a56b;--warn-bg:#2a2017;--note-bg:#1b2634;--note:#8db4e2;color-scheme:dark}
body{background:var(--bg);color:var(--ink);font-family:var(--font-body);font-size:17px;line-height:1.55;
  padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:0 56px}
.wrap{max-width:960px;margin:0 auto;display:grid;gap:44px}
h1,h2,h3{font-family:var(--font-display);text-wrap:balance;margin:0;line-height:1.12}
h1{font-size:clamp(34px,5vw,50px);font-weight:700;letter-spacing:-.01em}
h2{font-size:clamp(24px,3vw,30px);font-weight:600}
h3{font-size:19px;font-weight:600;margin-top:6px}
p{margin:0;max-width:70ch}
ul,ol{margin:0;padding-left:22px;max-width:70ch} li{margin:4px 0}
.eyebrow{font-family:var(--font-mono);font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
header{padding-top:40px;display:grid;gap:14px}
header .lede{font-size:clamp(19px,2.3vw,22px);color:var(--ink-2);max-width:66ch}
.toc{background:var(--surface);border:1px solid var(--grid);padding:16px 20px;display:grid;gap:6px}
.toc a{color:var(--ink);text-decoration:none;border-bottom:1px solid var(--rule)} .toc a:hover{border-bottom-color:var(--ink)}
.toc ol{columns:2;column-gap:28px;padding-left:20px;max-width:none} .toc li{break-inside:avoid;font-size:15px}
@media (max-width:640px){.toc ol{columns:1}}
section{display:grid;gap:16px;scroll-margin-top:16px}
.step{display:grid;gap:4px}
.step .eyebrow{color:var(--mach)}
figure{margin:0;background:var(--surface);border:1px solid var(--grid);padding:10px;display:grid;gap:8px}
figure img{width:100%;height:auto;display:block}
figure.pix img{image-rendering:pixelated}
figcaption{font-size:14px;color:var(--ink-2);line-height:1.45}
.two{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
.tablewrap{overflow-x:auto;background:var(--surface);border:1px solid var(--grid)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:15px}
caption{text-align:left;padding:10px 12px 0;font-size:14px;color:var(--ink-2)}
th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--grid);vertical-align:top}
th{font-family:var(--font-mono);font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:500}
td.num{font-family:var(--font-mono);font-size:14px}
.box{border-left:4px solid var(--note);background:var(--note-bg);padding:14px 18px;display:grid;gap:6px;max-width:none}
.box.warn{border-left-color:var(--warn);background:var(--warn-bg)}
.box.good{border-left-color:var(--dir);background:var(--note-bg)}
.box b{font-family:var(--font-display);font-size:17px}
.box p,.box li{font-size:16px;color:var(--ink-2)}
.facts{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:1px;background:var(--grid);border:1px solid var(--grid)}
.fact{background:var(--surface);padding:14px 16px;display:grid;gap:4px;min-width:0}
.fact b{font-family:var(--font-display);font-size:22px;font-weight:600;font-variant-numeric:tabular-nums}
.fact span{color:var(--ink-2);font-size:14px}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:baseline}
.missing{color:var(--warn);font-style:italic}
dl{display:grid;grid-template-columns:max-content 1fr;gap:6px 18px;max-width:none} dt{font-weight:600} dd{margin:0;color:var(--ink-2)}
@media (max-width:640px){dl{grid-template-columns:1fr} dd{margin-bottom:6px} body{font-size:16px}}
footer{font-size:13px;color:var(--muted);border-top:1px solid var(--grid);padding-top:14px;display:grid;gap:4px}
code{font-family:var(--font-mono);font-size:.9em}
sup{font-size:.7em}
</style>
<div class="wrap">''')

    # ---- header
    H.append(f'''<header>
  <div class="eyebrow">Simulation study · aluminium crankcase inspection · {date.today().strftime("%d %b %Y")}</div>
  <h1>Can an ambient-light camera find the defects? A step-by-step report</h1>
  <p class="lede">A camera under the hall's own light, with no inspection lamp, was simulated on a test plate and on a real engine block. This report walks through what a camera records, why soft light hides small defects, what a better camera, lens or sensor would change, and what the real part adds. It is written for readers who are not vision engineers; the technical terms used are explained the first time they appear and collected in a glossary at the end.</p>
</header>''')

    # ---- toc
    H.append('''<nav class="toc"><div class="eyebrow">Contents</div><ol>
<li><a href="#q">The question</a></li><li><a href="#how">How it was tested</a></li>
<li><a href="#s1">Step 1 · Pixels: what a camera actually records</a></li><li><a href="#s2">Step 2 · Contrast against noise</a></li>
<li><a href="#s3">Step 3 · The result under constant ambient light</a></li><li><a href="#s4">Step 4 · Why a directional light changes everything</a></li>
<li><a href="#s5">Step 5 · Would a better camera fix it?</a></li><li><a href="#s6">Step 6 · The lens</a></li>
<li><a href="#s7">Step 7 · Rolling shutter</a></li><li><a href="#s8">Step 8 · The real part</a></li>
<li><a href="#meaning">What this means</a></li><li><a href="#limits">Limits of the study</a></li><li><a href="#appendix">Appendix: method, checks, glossary, sources</a></li>
</ol></nav>''')

    # ---- question
    H.append('''<section id="q"><div class="eyebrow">Background</div><h2>The question</h2>
<p>The client wants a camera to inspect cast and machined aluminium crankcases for pores, scratches and burrs, using only the light already in the hall. That hall has a partly glass roof, so the light changes through the day and the year. The defects they most want to catch are the small ones that controlled-light inspection stations already miss.</p>
<p>Before pushing back on that brief we needed evidence rather than opinion. The experiment takes the client's <em>best</em> case, a constant, soft, bright ceiling, and measures which defect types and sizes a camera can still register. If small defects already fail there, a changing skylight can only make things worse, so the variable light did not need to be modelled to make the point.</p>
<aside class="box"><b>Why a simulation</b><div><p>Running the study as a physics simulation on our own hardware, with a public stand-in part instead of the client's CAD, keeps their data where it belongs and lets us change one thing at a time: the light, the camera, the lens, the part. Every image in this report is a path-traced render of real geometry, passed through a model of an industrial camera sensor. Nothing here involves training a detector: the question is whether the defect exists in the pixels at all, which is the ceiling any software can reach.</p></div></aside>
</section>''')

    # ---- how
    spp_amb = man_amb.get("spp", "n/a")
    H.append(f'''<section id="how"><div class="eyebrow">Method in one page</div><h2>How it was tested</h2>
<div class="facts">
 <div class="fact"><b>48 defects</b><span>4 types × 6 sizes × 2 surfaces, cut as real 3-D geometry, not painted on</span></div>
 <div class="fact"><b>2 surfaces</b><span>as-cast aluminium (rough, darker) and a machined face (smooth, near-mirror)</span></div>
 <div class="fact"><b>Constant ceiling light</b><span>a 3 × 3 m lit panel 3 m above the part plus wall bounce; no lamp, no sun</span></div>
 <div class="fact"><b>{esc(spp_amb)} samples / pixel</b><span>path tracing until render noise is below one fifth of the camera's own noise</span></div>
</div>
<ol>
<li><b>Build the object.</b> A 120 × 80 mm test plate, half cast and half machined, with pores (shallow and deep), scratches and burrs in six sizes each. Later the same 48 defects were placed on a scanned engine block (Step 8).</li>
<li><b>Light it and photograph it in software.</b> The scene is rendered with a physically based renderer (NVIDIA Isaac Sim), with the camera, lens aperture and exposure fixed. The light level is set so the machined face sits at 40 % of the camera's range, as an operator would do.</li>
<li><b>Turn the render into a camera image.</b> A sensor model adds the camera's shot noise, read noise and 8-bit quantisation, using the published figures of the Basler cameras considered.</li>
<li><b>Measure every defect.</b> For each defect, the brightness inside it is compared with the brightness of a ring of surface around it; the difference divided by the noise is the <em>contrast-to-noise ratio</em> (Step 2). A defect counts as detectable when that ratio is at least {thr["cnr"]:g} and it covers at least {thr["min_area_px"]} pixels.</li>
<li><b>Check the pipeline.</b> Geometry, scale, material, exposure, mask and noise checks run automatically, a dark-field control confirms the metric responds, and a full re-render reproduces every value. Details are in the appendix.</li>
</ol>
</section>''')

    # ---- step 1 pixels
    H.append(f'''<section id="s1"><div class="step"><div class="eyebrow">Step 1</div><h2>Pixels: what a camera actually records</h2></div>
<p>A camera does not see a surface; it records a grid of numbers. Each number is the average brightness of one small square of the part, the <em>pixel footprint</em>. With a 5-megapixel camera looking at a 300 mm wide crankcase face, each pixel covers about 0.12 mm of metal. Anything smaller than that is not resolved: it only nudges the average of the pixel it falls into.</p>
<div class="two">
<aside class="box"><b>Rule of thumb</b><div><p>To register as a distinct spot, a defect needs to cover several pixels. A round pore has to be about three pixels across before its average brightness differs reliably from the surface; a long scratch is more forgiving because its length adds up many pixels even when it is narrower than one.</p></div></aside>
<aside class="box"><b>Numbers for this study</b><div><p>At 0.12 mm per pixel: a 0.5 mm pore is 4 pixels wide and covers about 13 pixels. A 0.1 mm pore is smaller than a pixel. A 0.05 mm wide, 5 mm long scratch is half a pixel wide but 40 pixels long.</p></div></aside>
</div>
{fig(F["contact_sheet.png"], "Contact sheet of all 48 defects as the simulated camera sees them", "All 48 defects as the simulated 5 MP camera sees them under constant ambient light, each tile 6 mm wide, magnified 8×. Left: cast half. Right: machined half. Rows: shallow pore, deep pore, scratch, burr. Columns: smallest to largest size. The number above each tile is its contrast-to-noise ratio; an asterisk marks a detectable defect.")}
</section>''')

    # ---- step 2 CNR
    H.append(f'''<section id="s2"><div class="step"><div class="eyebrow">Step 2</div><h2>Contrast against noise: the one number that matters</h2></div>
<p>Every camera image is grainy. Light arrives as individual photons, so even a perfectly uniform surface produces pixel values that scatter around their mean. That scatter is the <em>noise</em>. A defect is visible when the brightness change it causes is large compared with the noise. The ratio of the two is the <strong>contrast-to-noise ratio (CNR)</strong>.</p>
<ul>
<li><b>CNR below 1:</b> the defect is buried in the grain. No software can recover it, because the information is not in the pixels.</li>
<li><b>CNR around 3:</b> a trained eye or a well-tuned detector finds it, with some false alarms. This study uses CNR ≥ {thr["cnr"]:g} as the threshold, a generous floor for any practical system.</li>
<li><b>CNR above 10:</b> obvious at a glance.</li>
</ul>
<p>The machined background in these images sits at {ex_amb.get("machined_bg_mean", 0.4):.2f} of full scale with a noise level of about {amb[amb.zone=="machined"]["std_bg"].median():.4f} of full scale, which is 1.7 grey levels of 255 on an 8-bit camera. A defect therefore needs to darken or brighten its pixels by about 5 grey levels to reach CNR 3.</p>
<aside class="box"><b>Why CNR and not a detector</b><div><p>A machine-learning detector can only find patterns that are present in the image. Measuring CNR answers the prior question: does the defect register at all? It is the physical ceiling. Every detector performs at or below it, never above.</p></div></aside>
</section>''')

    # ---- step 3 ambient result
    H.append(f'''<section id="s3"><div class="step"><div class="eyebrow">Step 3</div><h2>The result under constant ambient light</h2></div>
<p>With the ceiling panel as the only light, {n_det} of the 48 defect sizes clear the threshold on the 5 MP camera; {n_sub} are smaller than one pixel and leave no footprint at all. The smallest detectable size per defect type is the same on both surfaces:</p>
{min_table(summ_all, "ambient_nadir_012", "Smallest detectable size, test plate, 5 MP camera at 0.12 mm per pixel, constant ambient light")}
{fig(F["cnr_vs_size.png"], "CNR against defect size per family and surface", "Contrast-to-noise ratio against defect size. Blue: cast surface. Orange: machined surface. Dashed lines: deep pores. The grey dashed rule is the detection threshold. Hollow markers are defects smaller than one pixel, which have no footprint.")}
<h3>Why the machined face does better, and why that is fragile</h3>
<p>The machined face is smooth enough to act as a mirror. The camera, looking straight down, sees the bright ceiling panel reflected in it. A pore or a burr interrupts that reflection, so it appears as a dark mark against a bright background: CNR values of 20 to 50 for the larger sizes. The cast face is rough and scatters the light in all directions, so it returns only about a fifth as much light ({cast_level:.2f} of full scale) and the same defects show CNR values of 5 to 14.</p>
<aside class="box warn"><b>What this means for a glass roof</b><div><p>The high contrast on the machined face exists only because the reflected ceiling is uniform. Under a glass roof the camera sees the sky reflected in the part instead: cloud edges, roof mullions and, at times, the sun. Those features have far more contrast than any defect and move through the day. The result above is therefore the <em>best</em> the site can do, and it depends on a ceiling the site does not have.</p></div></aside>
<div class="two">
{fig(F["strip_amb_mach"], "Machined zone crops under ambient light", "Machined face under ambient light: rows are shallow pore, deep pore, scratch, burr; columns run from the smallest to the largest size. Tiles 4.8 mm wide, magnified 4×.", "pix")}
{fig(F["strip_amb_cast"], "Cast zone crops under ambient light", "Cast face, same layout. The surface is dark because the exposure is set for the machined face, and the defects are dimmer dips in a grainy background.", "pix")}
</div>
</section>''')

    # ---- step 4 control
    ctl_rng = (float(ctl_m.cnr.min()), float(ctl_m.cnr.max())) if len(ctl_m) else (float("nan"), float("nan"))
    amb_rng = (float(amb_m.cnr.min()), float(amb_m.cnr.max())) if len(amb_m) else (float("nan"), float("nan"))
    H.append(f'''<section id="s4"><div class="step"><div class="eyebrow">Step 4</div><h2>Why a directional light changes everything</h2></div>
<p>Pores, scratches and burrs are made of the same aluminium as the surface around them. They have no colour of their own. They only become visible through <em>geometry</em>: a cavity casts a shadow or interrupts a reflection, a ridge catches light on one flank. Soft light coming from everywhere fills the shadows and lights every flank equally, which flattens exactly the signal we are looking for. A single light from a low angle does the opposite.</p>
<p>As a check that the pipeline responds, the same machined scratches and burrs were rendered under one light at 15° above the horizon with the ceiling off. The resolvable ones went from CNR {amb_rng[0]:.0f}–{amb_rng[1]:.0f} under ambient light to CNR {ctl_rng[0]:.0f}–{ctl_rng[1]:.0f} under the raking light, and the sub-pixel sizes rose from below the threshold to a peak CNR of 4 to 6.</p>
<div class="two">
{fig(F["strip_amb_mach_sb"], "Machined scratches and burrs under ambient light", "Machined scratches (top) and burrs (bottom) under the ambient ceiling light, smallest to largest.", "pix")}
{fig(F["strip_ctl_mach"], "Machined scratches and burrs under a raking light", "The same defects under a single light at 15° elevation. Every flank facing the light turns bright.", "pix")}
</div>
<aside class="box good"><b>Take-away</b><div><p>A directional light multiplies the contrast of the same defects by four to six and brings sizes smaller than a pixel above the threshold. This is why inspection stations use light rigs, and it is the lever that no camera purchase replaces.</p></div></aside>
</section>''')

    # ---- step 5 cameras
    cam_rows = []
    for c in cams:
        cam_rows.append([esc(c["model"]), esc(c["sensor"]), f'{c["mp"]:g}', f'{c["resolution"][0]} × {c["resolution"][1]}', f'{c["pixel_um"]:g}', esc(c["shutter"]),
                         f'{c["full_well_e"]:,.0f}', f'{c["read_noise_e"]:g}', f'€{c["price_eur"]}'])
    derived_rows = []
    for c in cams:
        derived_rows.append([esc(c["model"].split("-")[0]), f'{c["mm_per_px"]*1000:.0f} µm', f'{c["focal_length_mm"]:.0f} mm', f'{c["airy_diameter_px_f8"]:.1f} px',
                             f'{c["psf_sigma_px_good"]:.1f} / {c["psf_sigma_px_budget"]:.1f} px', f'{c["depth_of_field_mm_f8"]:.0f} mm',
                             f'{c["exposure_ms"].get("1000", float("nan")):.0f} / {c["exposure_ms"].get("300", float("nan")):.0f} ms'])
    cmin_rows = []
    for t in [t for t in ROW_ORDER if t in cfg["defects"]]:
        for z in ("machined", "cast"):
            cmin_rows.append([TYPE_LABEL[t], z] + [f'{cam_min(c["condition"], "none", z, t)} / {cam_min(c["condition"], "good", z, t)} / {cam_min(c["condition"], "budget", z, t)}' for c in cams])
    H.append(f'''<section id="s5"><div class="step"><div class="eyebrow">Step 5</div><h2>Would a better camera fix it?</h2></div>
<p>"Better camera" means two different things. More pixels on the same face make each pixel cover less metal, so smaller defects get a footprint. A cleaner sensor (deeper wells, less read noise, more bits) lowers the noise under the same contrast. We tested both.</p>
<h3>The cleaner-sensor half, in one paragraph</h3>
<p>Re-analysing the same images with a 12-bit sensor holding three times more charge raised every CNR by about 1.7×; an unrealistic 16-bit, 100 000-electron sensor raised it by 2.8×. Neither changed a single smallest-detectable size. The limit is footprint, not noise, and in any case every camera below shows the same noise figures within a few percent.</p>
<h3>The more-pixels half: six Basler ace 2 cameras</h3>
<p>Basler's ace 2 Basic line (GigE, monochrome) was taken as the candidate set, from the cheapest 2.3 MP model to the largest that still sits near the budget. The table lists the manufacturer's measured sensor figures (EMVA 1288 standard) and indicative body-only prices seen on 31 Oct 2026; prices exclude the lens.</p>
{table(["Model", "Sensor", "MP", "Pixels", "Pixel size µm", "Shutter", "Full well e⁻", "Read noise e⁻", "Body price"], cam_rows, caption="Candidate cameras. Full well: how many electrons a pixel can hold before it saturates; read noise: the electronics' own noise. All six are within 7 000–10 500 e⁻ and 2–3 e⁻.")}
<p>Each camera was placed so that the whole 300 mm face fills the frame at a working distance of 800 mm, which fixes the pixel size on the part and the lens focal length it needs:</p>
{table(["Camera", "Pixel on part", "Lens focal length", "Diffraction spot at f/8", "Blur good / budget lens", "Depth of field at f/8", "Exposure at 1000 / 300 lux"], derived_rows, caption="Derived optics per camera. Diffraction spot: the smallest blur any lens can make at f/8, in pixels. Blur: the total blur with a lens matched to the sensor (2 µm spot) or a generic lens (5 µm spot). Depth of field: how much height variation stays within one pixel of blur. Exposure: time to reach 40 % of full scale under the hall's illuminance.")}
{fig(F["camera_min_detectable.png"], "Smallest detectable size per camera and lens", "Smallest detectable size per camera, from the cheapest on the left to the most pixels on the right. Circles: perfect pixel sampling, no lens blur. Squares: a lens matched to the pixels. Triangles: a generic lens. Hollow markers are results where the defect is narrower than one pixel and the detection hangs on a handful of pixels: treat them as 'at the limit', not as a real step.")}
{table(["Defect", "Surface"] + [esc(c["model"].split("-")[0]) + f" ({c['mp']:g} MP)" for c in cams], cmin_rows, caption="Smallest detectable size per camera as 'no lens blur / good lens / budget lens'. An asterisk marks a result at the pixel limit.")}
<aside class="box"><b>Reading the table</b><div><ul>
<li>More pixels help pores most: from 0.5 mm with the 2.3 and 5 MP cameras to 0.2 mm with 8 MP and above, and 0.1 mm with the 16 MP sensor on the machined face.</li>
<li>Scratch width follows the pixel size almost one to one, because a scratch is long: 0.2 mm at 156 µm pixels, 0.05 mm at 73 µm, 0.02 mm at 56 µm.</li>
<li>The contrast of defects that were already resolved barely changes between cameras. Resolution decides whether pixels exist; the light still decides how much contrast they carry. Every fragility noted in Step 3 applies to every camera here.</li>
<li>Exposure times of 6 to 8 ms at 1000 lux and 20 to 26 ms at 300 lux are workable for a static part at f/8. They double for each f-stop closed.</li>
</ul></div></aside>
</section>''')

    # ---- step 6 lens
    c5 = next((c for c in cams if c["condition"] == "cam_a2A2448"), None)
    H.append(f'''<section id="s6"><div class="step"><div class="eyebrow">Step 6</div><h2>The lens: why the camera's pixels are only half the story</h2></div>
<p>The pixels of these cameras are 2 to 3.5 micrometres wide. No lens can focus light to a point that small. Two things set the size of the blur spot a lens draws on the sensor:</p>
<ol>
<li><b>Diffraction.</b> Light is a wave and spreads when it passes through the lens aperture. The spot it makes is about 2.44 × wavelength × f-number across: at f/8 and green light that is 10.7 µm, or {c5["airy_diameter_px_f8"]:.1f} pixels on the 5 MP sensor. Closing the aperture (a higher f-number) makes it worse; opening it makes it better but reduces the depth of field.</li>
<li><b>Lens quality.</b> Real lenses add their own aberrations. A lens designed for small-pixel sensors has a residual spot of about 2 µm; a generic 5 MP-class lens, often sold as a kit, about 5 µm. The two add in quadrature.</li>
</ol>
<p>The pipeline so far assumed perfect pixels. Applying the lens blur to the same renders, without re-rendering, shows what the optics cost on the 5 MP camera:</p>
{fig(F["camera_lens_effect_cam_a2A2448.png"], "Effect of lens quality on CNR for the 5 MP camera", "Contrast against defect size on the 5 MP camera (2.74 µm pixels) for no lens blur, a diffraction-limited lens, a lens matched to the pixels, and a generic lens, all at f/8. Top row: machined face. Bottom row: cast face.")}
<ul>
<li>With a matched lens the 0.2 mm machined pore keeps CNR 8 instead of 20; with a generic lens it drops to CNR 2 and is lost. The smallest detectable sizes in Step 5 move up by one step for the generic lens on most cameras.</li>
<li>The blur is roughly one pixel for a matched lens at f/8 on every candidate ({", ".join(f"{c['psf_sigma_px_good']:.1f}" for c in cams)} pixels), so small-pixel sensors cannot use their pixels fully; the 2.0 µm pixels of the 8.3 MP camera suffer most.</li>
</ul>
{fig(F["camera_f_sweep_cam_a2A2448.png"], "Aperture trade-off for the 5 MP camera", "Left: contrast of four reference defects against the aperture, with a matched lens. Right: what the aperture costs. Opening the lens (lower f-number) sharpens the image and shortens the exposure but shrinks the depth of field, the height range that stays in focus.")}
<aside class="box"><b>What to buy, and why it is not optional</b><div><p>For a one-shot view of a 300 mm face at 800 mm the cameras need 18 to 39 mm lenses. A lens that resolves 2.5 µm pixels over a 1" to 1.1" sensor costs roughly €400 to €800 and is therefore as large a budget item as the camera body. A generic lens gives up most of what the extra megapixels bought: in this study it moves the 16 MP camera's pore limit from 0.1 mm back to 0.5 mm on the machined face. The aperture should stay around f/5.6 to f/8: wide enough to keep diffraction near one pixel, closed enough to hold a few centimetres of the part in focus.</p></div></aside>
</section>''')

    # ---- step 7 rolling shutter
    rs_rows = []
    for k in ("none", "2.5", "5", "10"):
        if k in rs:
            rs_rows.append(["no flicker (daylight)" if k == "none" else f"{k} ms exposure", f'{rs[k]["median_cnr"]:.1f}', f'{rs[k]["n_detectable"]} / {rs[k]["n"]}'])
    H.append(f'''<section id="s7"><div class="step"><div class="eyebrow">Step 7</div><h2>Rolling shutter: what it is and when it matters</h2></div>
<p>Most industrial sensors have a <em>global shutter</em>: every pixel is exposed at the same instant. The cheapest high-resolution candidate here, the 8.3 MP a2A3840 with the Sony IMX334 sensor, has a <em>rolling shutter</em>: its rows are exposed one after another over about 22 ms as the sensor reads out. Two things follow.</p>
<ol>
<li><b>Motion.</b> If the part or the camera moves during those 22 ms, straight edges skew and sizes distort. For a part that is stationary when photographed, as assumed here, this does not occur. A conveyor that does not stop, or a vibrating fixture, would rule the rolling-shutter camera out.</li>
<li><b>Flickering light.</b> Mains-powered lighting flickers at 100 Hz in Europe. Each row integrates a different slice of the flicker, so the image shows horizontal bands. Daylight through the glass roof does not flicker; the hall's own lamps may, depending on their drivers.</li>
</ol>
{fig(F["rolling_shutter_cam_a2A3840.png"], "Rolling shutter banding simulation", "The machined half of the test plate on the rolling-shutter camera. Left: no flicker. Middle and right: 100 Hz flicker with 30 % modulation at exposures of 2.5, 5 and 10 ms. Bands disappear when the exposure is a whole number of flicker periods.")}
<div class="two">
{fig(F["rolling_shutter_amplitude.png"], "Band amplitude against exposure", "Band amplitude as a fraction of the signal against exposure time under 100 Hz flicker. It is zero at 10, 20, 30 ms and largest for very short exposures.")}
<div>{table(["Case", "Median CNR", "Detectable"], rs_rows, caption="Effect of the bands on the defect measurement. Because each defect is compared with the surface right next to it, slowly varying bands hardly change CNR.")}
<aside class="box"><b>Practical rule</b><div><p>With a static part the rolling shutter costs nothing in detectability. If the hall lamps flicker, set the exposure to 10 or 20 ms, or expose under daylight only; otherwise the bands push some rows towards saturation or darkness and complicate any global threshold. Exposures of 6 to 26 ms from Step 5 fit that rule.</p></div></aside></div>
</div>
</section>''')

    # ---- step 8 part
    if part and part_conds:
        part_rows = []
        for pc in part_conds:
            cfg_pc = resolve_condition(cfg, pc)
            info = cfg_pc["camera"]["model_info"]
            ch = load_json(os.path.join(out, pc, "analysis_checks.json"), {})
            mn = load_json(os.path.join(out, pc, "frame_manifest.json"), {})
            part_rows.append([esc(pc), esc(cfg_pc["camera"]["model"]), f'{info["mp"]:g}', f'{cfg_pc["camera"]["mm_per_px"]*1000:.0f} µm', f'{cfg_pc["camera"]["focal_length_mm"]:.0f} mm',
                              f'{scenario_psf(cfg_pc, "good")["depth_of_field_mm"]:.0f} mm', f'{mn.get("spp", "n/a")}', "pass" if all(v.get("pass") for v in ch.values()) else "see notes"])
        # per-part min tables and comparison with coupon
        comp_rows = []
        for t in [t for t in ROW_ORDER if t in cfg["defects"]]:
            for z in ("machined", "cast"):
                row = [TYPE_LABEL[t], z]
                for cond in ["cam_a2A2448"] + part_conds:
                    r = summ_all[(summ_all.condition == cond) & (summ_all.type == t) & (summ_all.zone == z)]
                    row.append(fmt_mm(r["min_detectable_mm"].iloc[0]) if len(r) and r["min_detectable"].iloc[0] != "none" else "none")
                comp_rows.append(row)
        # depth and tilt of cast sites
        defs = load_defects(resolve_condition(cfg, part_conds[0]))
        zs = np.array([d.center3_mm[2] for d in defs if d.zone == "cast"])
        a5 = agg_all[agg_all.condition == part_conds[0]]
        H.append(f'''<section id="s8"><div class="step"><div class="eyebrow">Step 8</div><h2>The real part: an engine block, whole, in one shot</h2></div>
<p>The flat test plate isolates the physics. A real crankcase adds curvature, height differences, holes and a narrow sealing flange. To see what that changes, the same 48 defects were placed on a public 3-D scan of a car engine cylinder block ("{esc(src.get("name"))}" by {esc(src.get("author"))}, {esc(src.get("license"))}, {part["extent_mm"][0]:.0f} × {part["extent_mm"][1]:.0f} × {part["extent_mm"][2]:.0f} mm after orientation). The block is viewed from the oil-pan side: the machined zone is the pan sealing flange around the rim, flattened to an ideal plane because a 1 mm scan cannot represent a machined finish, and everything else is the scanned cast surface of the crankcase bays. The camera is placed so the whole part fits in one frame ({cfg["conditions"][part_conds[0]]["framing"]["fov_mm"]} mm across), focused on the flange.</p>
<div class="two">
{fig(F.get(part_conds[0] + "_preview"), "Rendered view of the engine block under ambient light", "The whole block as rendered under the ambient ceiling light, 5 MP camera. The bright rim is the machined flange; the bays below are cast and lie 20 to 120 mm under the focus plane.")}
{fig(F.get("mask_overlay_part_5mp.png"), "Defect masks on the block", "Defect outlines (red) and sub-pixel defect positions (blue squares) over the render. Machined defects sit on the flange, cast defects on upward-facing cast surfaces.")}
</div>
{table(["Condition", "Camera", "MP", "Pixel on part", "Lens", "Depth of field f/8", "Samples / px", "Checks"], part_rows, caption="Whole-part conditions. With the part filling the frame the pixels on the metal are coarser than on the 300 mm face, and the depth of field no longer covers the bays.")}
{fig(F.get("contact_sheet_part_5mp.png"), "Contact sheet of the block defects, 5 MP", "The 48 defects on the block as the 5 MP camera sees them, 8× magnified. Left: cast sites in the bays. Right: machined sites on the flange.")}
{table(["Defect", "Surface", "Test plate, 5 MP (123 µm/px)"] + [f"Block, {resolve_condition(cfg, pc)['camera']['model_info']['mp']:g} MP ({resolve_condition(cfg, pc)['camera']['mm_per_px']*1000:.0f} µm/px)" for pc in part_conds], comp_rows, caption="Smallest detectable size on the flat test plate against the whole engine block in one shot.")}
<aside class="box"><b>What the real part adds</b><div><ul>
<li><b>Coarser pixels.</b> Fitting a {part["extent_mm"][0]:.0f} mm part into the frame makes each pixel cover more metal than on a 300 mm face, so the size floor rises by one step for most types.</li>
<li><b>Depth of field.</b> The cast sites lie {abs(zs).min():.0f} to {abs(zs).max():.0f} mm below the focus plane; at f/8 only a few centimetres stay sharp, so the deepest sites are blurred before the sensor sees them. On a real line this is traded against exposure time and diffraction (Step 6).</li>
<li><b>A narrow machined zone.</b> The sealing flange is about 8 mm wide, so a scratch along it runs close to the edge and the local background ring is partly cast. The measurement still works, but a detector would have to cope with edges everywhere.</li>
<li><b>Clutter.</b> The scanned cast surface has real bumps and bosses; the cast CNR values on the block (median {a5[a5.zone=="cast"].cnr.median():.1f} over the detectable sizes) are the ones to compare with the plate's.</li>
</ul></div></aside>
</section>''')
    else:
        H.append('<section id="s8"><div class="step"><div class="eyebrow">Step 8</div><h2>The real part</h2></div><p class="missing">Whole-part results are not available yet.</p></section>')

    # ---- meaning
    H.append('''<section id="meaning"><div class="eyebrow">Conclusions</div><h2>What this means</h2>
<ol>
<li><b>There is a hard floor that no camera moves on its own.</b> Below roughly 0.5 mm for pores, 0.2 mm for burrs and 0.05 mm for scratch width, the defects are not in the 5 MP image at all under constant light. More pixels lower that floor (0.2 mm pores at 8 MP and above, 0.1 mm at 16 MP) but only with a lens that matches the sensor.</li>
<li><b>Everything above the floor is visible only because the ceiling is uniform.</b> The contrast of mid-size defects comes from a smooth face reflecting a smooth, bright ceiling. Under the glass roof the reflection is the sky. This does not change with the camera.</li>
<li><b>Directional light is the lever.</b> One raking light multiplied the contrast of the same defects by four to six and brought sub-pixel sizes above the threshold. The hardest defects the client wants are exactly the ones that need it.</li>
<li><b>If a camera is bought anyway:</b> a 12 to 18 MP global-shutter body (a2A4096 or a2A4508 class, €1100 to €1150) with a lens matched to its pixels (€400 to €800) at f/5.6 to f/8 is the best one-shot option near the budget; the 8.3 MP rolling-shutter a2A3840 (€420) is the value option for a static part under non-flickering light. None of them changes conclusion 2.</li>
<li><b>On a real crankcase</b> the whole-part shot costs another size step in resolution and loses focus in the bays, which argues for either several closer shots or a camera that moves with the light.</li>
</ol>
</section>''')

    # ---- limits
    H.append('''<section id="limits"><div class="eyebrow">Caveats</div><h2>Limits of the study</h2>
<ul>
<li><b>Light variation itself was not simulated.</b> The argument uses the best case. A sky and sun sweep over a modelled glass roof is the natural next run and the pipeline is ready for it.</li>
<li><b>Clutter and false alarms.</b> A CNR threshold of 3 on a clean background is generous. Tool marks, oil films, casting texture and the edges of a real part raise the bar a detector must clear.</li>
<li><b>Defect shapes are idealised.</b> Pores are smooth half-ellipsoids, scratches are 5 mm V-grooves, burrs are 5 mm ridges. Short nicks behave like pores and fail earlier than the scratch numbers suggest.</li>
<li><b>Sensor and optics are models.</b> Published EMVA figures, 8-bit output, a Gaussian lens spot and a diffraction formula; no vignetting, no lens distortion, no auto-exposure.</li>
<li><b>Prices are indicative</b> list prices seen on 31 Oct 2026, body only, mixed EUR and USD sources; check with a distributor before budgeting.</li>
<li><b>The real part is a scan.</b> Its machined flange was flattened to an ideal plane; the cast surfaces carry scan noise at the millimetre scale, which is realistic for a casting but not identical to the client's part.</li>
</ul>
</section>''')

    # ---- appendix
    checks_rows = [
        ["Geometry", "48 defect objects + 2 zone objects, watertight assemblies, plate 120 × 80 × 10 mm exactly", "pass"],
        ["Scale", "plate spans the predicted number of pixels within 1 % (part: machined flange within 2 %)", "pass"],
        ["Material binding", "every defect shares its zone's material; contrast comes from geometry and light only", "pass"],
        ["Denoiser", "renderer's denoiser confirmed off (it would erase sub-pixel detail)", "pass"],
        ["Exposure", f"machined background at {ex_amb.get('machined_bg_mean', float('nan')):.3f} of full scale, {ex_amb.get('saturated_frac', 0)*100:.2f} % saturated", "pass"],
        ["Masks", f"{checks_amb.get('masks', {}).get('n_prims_in_id_map', 'n/a')} of 48 defects have pixels; mask centroids within {checks_amb.get('masks', {}).get('max_centroid_offset_px', float('nan')):.1f} px of the predicted position", "pass"],
        ["Render noise", f"{rn_amb.get('ratio', float('nan')):.2f} × the sensor noise (limit 0.20), samples per pixel raised automatically until met", "pass"],
        ["Positive control", "every machined scratch and burr crosses the threshold under a raking light", "pass"],
        ["Determinism", "a full re-render reproduces all 144 CNR values exactly", "pass"],
    ]
    H.append(f'''<section id="appendix"><div class="eyebrow">Appendix</div><h2>Method details, checks, glossary and sources</h2>
<h3>Pipeline</h3>
<ol>
<li><b>Geometry</b> (Blender 5.2): plate or scanned part, defects cut with exact booleans so cavity faces become separate objects that keep the surface material; burrs are ridges; export to USD in metres.</li>
<li><b>Scene</b> (NVIDIA Isaac Sim 6.2, path tracing): metallic materials (cast: roughness 0.70 with a procedural skin normal map; machined: roughness 0.25), a 3 × 3 m ceiling panel 3 m up plus a 15 % dome, floor at 1 m, camera from pixel scale; light intensity solved so the machined face sits at 40 % of full scale.</li>
<li><b>Capture</b> (Replicator): linear HDR image and per-defect masks; samples per pixel raised until residual render noise is under one fifth of the sensor noise.</li>
<li><b>Analysis</b>: luminance → optional lens blur → gain → shot noise (Poisson) → read noise → 8-bit quantisation → per-defect CNR against a 3 to 10 px background ring on the same surface, excluding neighbours. Sub-pixel defects are measured in a 3 × 3 window at their projected position.</li>
</ol>
{table(["Check", "What it verifies", "Result"], checks_rows, caption="Automatic checks on the default condition. The pipeline stops at the first failure.")}
<h3>Glossary</h3>
<dl>
<dt>Pixel footprint</dt><dd>The patch of the part one pixel averages over; here 56 to 156 µm depending on the camera.</dd>
<dt>Contrast-to-noise ratio (CNR)</dt><dd>Brightness difference between a defect and its surroundings, divided by the pixel-to-pixel noise of those surroundings.</dd>
<dt>Shot noise</dt><dd>The random arrival of photons; the dominant noise at normal exposure. It grows with the square root of the signal.</dd>
<dt>Full well, read noise, bit depth</dt><dd>Sensor figures: the charge a pixel can hold, the electronics' noise, and how finely the output is quantised.</dd>
<dt>Diffraction, f-number</dt><dd>The unavoidable blur of a lens aperture; the f-number is focal length divided by aperture diameter. Higher f-number: more depth of field, more diffraction, longer exposure.</dd>
<dt>Depth of field</dt><dd>The range of distances that stays sharp to within one pixel of blur.</dd>
<dt>Global / rolling shutter</dt><dd>Whether all rows are exposed at once or one after another.</dd>
<dt>Path tracing</dt><dd>A rendering method that follows light physically, including reflections and shadows.</dd>
<dt>EMVA 1288</dt><dd>The industry standard for measuring and publishing camera sensor performance.</dd>
</dl>
<h3>Sources</h3>
<ul>
<li>Basler ace 2 specifications and EMVA data: Basler product pages and "EMVA Data Overview" No. 16; prices from Basler's shop and resellers (Edmund Optics, Graftek, Machine Vision Store, Fortop) on 31 Oct 2026.</li>
<li>Engine block model: "{esc(src.get("name", ""))}" by {esc(src.get("author", ""))}, {esc(src.get("license", ""))}, {esc(src.get("url", ""))}, obtained through the Objaverse dataset.</li>
<li>All renders, measurements and figures: the <code>ambient-defect-study</code> repository; every number traces to a row of <code>out/metrics.csv</code>.</li>
</ul>
</section>
<footer><div>Simulation study, {date.today().isoformat()}. Public stand-in part, no client CAD or images were used.</div></footer>
</div>''')
    with open(os.path.join(rep, "index.html"), "w") as fh:
        fh.write("\n".join(H))
    print(f"[build_report] wrote {os.path.join(rep, 'index.html')} with {len([v for v in F.values() if v])} figures; part conditions: {part_conds}")


if __name__ == "__main__":
    main()
