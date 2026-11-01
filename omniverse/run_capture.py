"""Stage 3: Replicator capture -> out/<condition>/renders/{hdr_*.npy, mask_*.npy, zone_*.npy, id_map.json}

  omniverse/isaacsim_docker.sh omniverse/run_capture.py --config config/experiment.yaml --condition ambient_nadir_012

Opens scene.usd from Stage 2, labels prims, attaches HdrColor + instance/semantic segmentation
annotators to one render product, renders `frames_per_condition` converged frames and writes the
linear HDR images (float32 .npy; run_all.py converts them to EXR on the host) plus per-defect masks.
If render.auto_spp is on, the render-noise check (machined-zone residual std vs sensor noise) is
run on frame 0 and spp is raised until it passes, up to render.max_spp.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from ads.config import load_config  # noqa: E402
from ads.layout import load_defects, load_part_info, asset_paths  # noqa: E402
from omniverse import common  # noqa: E402



def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment.yaml")
    ap.add_argument("--condition", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--frames", type=int, default=None)
    ap.add_argument("--spp", type=int, default=None, help="override starting spp")
    ap.add_argument("--no-auto-spp", action="store_true")
    ap.add_argument("--keep-failed", action="store_true", help="write the frames even when the render-noise check fails (diagnostics only)")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.condition)
    cond = cfg["condition"]["name"]
    out_dir = os.path.abspath(args.out or os.path.join("out", cond))
    renders = common.ensure_dir(os.path.join(out_dir, "renders"))
    scene_path = os.path.join(out_dir, "scene.usd")
    if not os.path.exists(scene_path):
        raise RuntimeError(f"{scene_path} missing: run omniverse/build_scene.py first")
    import json

    with open(os.path.join(out_dir, "scene_manifest.json")) as fh:
        scene_manifest = json.load(fh)
    W, H = (int(v) for v in cfg["camera"]["resolution"])
    layout = load_defects(cfg)
    ZONE_PLATE = asset_paths(cfg)["zone_prims"]
    part = load_part_info(cfg)
    n_frames = int(args.frames or cfg["render"]["frames_per_condition"])
    t_start = time.time()

    app = common.start_app(cfg, W, H)
    import numpy as np  # noqa: E402
    import omni.usd  # noqa: E402
    import omni.replicator.core as rep  # noqa: E402
    from isaacsim.core.utils.semantics import add_labels  # noqa: E402
    from analysis.sensor_model import luminance, expected_noise_std  # noqa: E402
    from analysis.render_noise import flat_patch_std  # noqa: E402

    ctx = omni.usd.get_context()
    if not ctx.open_stage(scene_path):
        raise RuntimeError(f"could not open {scene_path}")
    stage = ctx.get_stage()
    for _ in range(10):
        app.update()

    # semantics: every D_* prim is class=defect (instance = prim), plates are zones, floor is floor
    for d in layout:
        p = stage.GetPrimAtPath(f"/World/Coupon/{d.prim}")
        if not p.IsValid():
            raise RuntimeError(f"defect prim {d.prim} missing in scene")
        add_labels(p, labels=["defect"], instance_name="class")
    for z in cfg["coupon"]["zones"]:
        add_labels(stage.GetPrimAtPath(f"/World/Coupon/{ZONE_PLATE[z]}"), labels=[f"zone_{z}"], instance_name="class")
    add_labels(stage.GetPrimAtPath("/World/Floor"), labels=["floor"], instance_name="class")

    spp = int(args.spp or cfg["render"]["spp"])
    settings = common.apply_render_settings(cfg, spp_override=spp)
    common.check_denoiser_off()
    rp = rep.create.render_product("/World/Camera", (W, H), name="capture")
    ann = common.Annotators(rep, rp, with_ldr=False)
    for _ in range(3):
        app.update()

    sensor = cfg["sensor"]
    frac = float(cfg["checks"]["render_noise_frac"])
    # the capture aims below the analysis limit so compute_cnr's re-check (same estimator, slightly
    # different background mask) cannot land on the boundary and fail at 0.21 after a 0.20 pass here
    target = frac * float(cfg["render"].get("noise_margin", 0.9))
    auto = bool(cfg["render"].get("auto_spp", True)) and not args.no_auto_spp
    max_spp = int(cfg["render"].get("max_spp", 16384))
    per = int(cfg["render"].get("spp_per_subframe", 32))

    def render_noise(frame):
        classes = common.semantic_class_ids(frame["sem_info"])
        zone = common.mask_from_ids(frame["sem"], classes.get("zone_machined", []))
        defects = common.mask_from_ids(frame["sem"], classes.get("defect", []))
        from scipy import ndimage

        bg = zone & ~ndimage.binary_dilation(defects, iterations=6)
        Y = luminance(frame["hdr"])
        est = flat_patch_std(Y, bg, seed=int(cfg["seed"]))
        est["sensor_std"] = expected_noise_std(est["level"], sensor)
        est["ratio"] = est["std"] / est["sensor_std"]
        est["pass"] = est["ratio"] <= target
        return est

    # ---- frame 0 with spp escalation
    noise_history = []
    while True:
        settings = common.apply_render_settings(cfg, spp_override=spp)
        sub = common.subframes_for(cfg, spp)
        frame0 = ann.render(sub)
        est = render_noise(frame0)
        est.update({"spp": spp, "subframes": sub, "wall_s": frame0["wall_s"]})
        noise_history.append(est)
        print(f"[run_capture] frame 0 @ {spp} spp ({sub} subframes, {frame0['wall_s']:.1f}s): render std {est['std']:.5f} vs sensor {est['sensor_std']:.5f} -> ratio {est['ratio']:.2f} ({'pass' if est['pass'] else 'fail'})")
        if est["pass"] or not auto or spp >= max_spp:
            break
        need = spp * (est["ratio"] / target) ** 2 * 1.15
        spp = int(min(max_spp, max(spp * 2, per * math.ceil(need / per))))
    if not est["pass"]:
        if not args.keep_failed:
            raise RuntimeError(f"render noise check failed at {spp} spp: ratio {est['ratio']:.2f} > {target:.3f}")
        print(f"[run_capture] WARNING: render noise check failed (ratio {est['ratio']:.2f}); writing frames anyway (--keep-failed)")

    frames = [frame0]
    for f in range(1, n_frames):
        fr = ann.render(common.subframes_for(cfg, spp))
        frames.append(fr)
        print(f"[run_capture] frame {f} rendered in {fr['wall_s']:.1f}s")

    # ---- frame-to-frame comparison (the RTX path tracer has no exposed seed; frames may be identical)
    diffs = [float(np.abs(frames[i]["hdr"][..., :3] - frames[0]["hdr"][..., :3]).max()) for i in range(1, len(frames))]
    identical = all(d == 0.0 for d in diffs)
    if len(frames) > 1 and not identical:
        Ys = np.stack([luminance(f["hdr"]) for f in frames])
        classes = common.semantic_class_ids(frames[0]["sem_info"])
        zone = common.mask_from_ids(frames[0]["sem"], classes.get("zone_machined", []))
        f2f = float(Ys.std(axis=0, ddof=1)[zone].mean())
    else:
        f2f = 0.0

    # ---- write
    inst_map = frames[0]["inst_info"]
    for f, fr in enumerate(frames):
        common.save_npy(os.path.join(renders, f"hdr_{f}.npy"), fr["hdr"][..., :3].astype(np.float32))
        common.save_npy(os.path.join(renders, f"mask_{f}.npy"), fr["inst"].astype(np.uint32))
        common.save_npy(os.path.join(renders, f"zone_{f}.npy"), fr["sem"].astype(np.uint32))
        if fr.get("depth") is not None and f == 0:
            common.save_npy(os.path.join(renders, "depth_0.npy"), fr["depth"].astype(np.float32))
        if fr["inst_info"].get("idToLabels") != inst_map.get("idToLabels"):
            common.write_json(os.path.join(renders, f"id_map_{f}.json"), {"instance": fr["inst_info"], "semantic": fr["sem_info"]})
    common.write_json(os.path.join(renders, "id_map.json"), {"instance": inst_map, "semantic": frames[0]["sem_info"],
                                                             "prim_to_ids": common.instance_prim_ids(inst_map),
                                                             "class_to_ids": common.semantic_class_ids(frames[0]["sem_info"])})
    manifest = {
        "condition": cond,
        "control": cfg["condition"]["control"],
        "camera": scene_manifest["camera"],
        "working_distance_mm": scene_manifest["camera"]["distance_mm"],
        "lighting_mode": scene_manifest["lighting_mode"],
        "solved_intensity": scene_manifest["solved_intensity"],
        "spp": spp,
        "spp_per_subframe": per,
        "subframes": common.subframes_for(cfg, spp),
        "frames": n_frames,
        "seed": cfg["seed"],
        "render_settings": settings,
        "render_noise": noise_history[-1],
        "render_noise_history": noise_history,
        "frames_identical": identical,
        "frame_max_abs_diff": diffs,
        "frame_to_frame_std_machined": f2f,
        "isaac_sim_version": common.isaac_version(),
        "wall_time_s": time.time() - t_start,
        "render_wall_s": [fr["wall_s"] for fr in frames],
        "files": {"hdr": "hdr_<f>.npy (float32 HxWx3 linear RGB)", "mask": "mask_<f>.npy (uint32 instance id)",
                  "zone": "zone_<f>.npy (uint32 semantic id)", "id_map": "id_map.json"},
    }
    common.write_json(os.path.join(out_dir, "frame_manifest.json"), manifest)
    print(f"[run_capture] wrote {n_frames} frames to {renders} ({manifest['wall_time_s']:.0f}s total; frames identical: {identical})")
    app.close()


if __name__ == "__main__":
    main()
