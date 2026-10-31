#!/usr/bin/env python3
"""Orchestrate the four stages. Stops at the first failed check.

  python run_all.py --config config/experiment.yaml                 # default condition + positive control + report
  python run_all.py --condition ambient_nadir_005                   # one extra condition, then re-plot
  python run_all.py --all                                           # every condition that can run
  python run_all.py --verify                                        # re-render the default condition and compare cnr
  python run_all.py --skip-geometry --skip-capture                  # analysis only

Stages: 1 Blender coupon (+checks)  2 Isaac Sim scene (+scale/material/denoiser checks)
        3 Replicator capture (+render-noise check)  4 analysis (+exposure/mask checks), figures, RESULTS.md
Isaac Sim runs in Docker via omniverse/isaacsim_docker.sh; Blender runs on the host.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
from ads.config import load_config, default_condition  # noqa: E402

BLENDER = os.environ.get("BLENDER", "blender")
ISAAC = os.path.join(REPO_ROOT, "omniverse", "isaacsim_docker.sh")


def run(cmd, label):
    t0 = time.time()
    print(f"\n=== {label}\n$ {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, cwd=REPO_ROOT)
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"!!! {label} failed (exit {res.returncode}) after {dt:.0f}s", flush=True)
        sys.exit(res.returncode)
    print(f"=== {label} done in {dt:.0f}s", flush=True)
    return dt


def run_isaac(script, cfg_path, condition, label, extra=()):
    return run([ISAAC, script, "--config", cfg_path, "--condition", condition, *extra], label)


def stage_geometry(cfg_path):
    run([BLENDER, "-b", "-P", "blender/make_coupon.py", "--", "--config", cfg_path, "--out", "assets/coupon/"], "Stage 1: Blender coupon")
    run([BLENDER, "-b", "-P", "blender/checks.py", "--", "--config", cfg_path, "--assets", "assets/coupon/"], "Stage 1: geometry checks")
    run([sys.executable, "analysis/gen_textures.py", "--config", cfg_path], "Stage 1b: cast normal map")


def stage_capture(cfg_path, condition, out_dir=None):
    extra = ["--out", out_dir] if out_dir else []
    run_isaac("omniverse/build_scene.py", cfg_path, condition, f"Stage 2: scene ({condition})", extra)
    run_isaac("omniverse/run_capture.py", cfg_path, condition, f"Stage 3: capture ({condition})", extra)


def stage_analysis(cfg_path, condition, out_dir=None, metrics=None):
    cmd = [sys.executable, "analysis/compute_cnr.py", "--config", cfg_path, "--condition", condition]
    if out_dir:
        cmd += ["--out", out_dir]
    if metrics:
        cmd += ["--metrics", metrics]
    run(cmd, f"Stage 4: metrics ({condition})")


def verify(cfg_path, cfg):
    """Re-render and re-analyse the default condition into out/_verify and compare every cnr."""
    import pandas as pd

    cond = default_condition(cfg)
    vdir_rel = os.path.join("out", "_verify", cond)  # repo-relative: the container sees the repo at /workspace
    vdir = os.path.join(REPO_ROOT, vdir_rel)
    for d in (os.path.join(REPO_ROOT, "out", "_verify"), vdir):  # the container (uid 1234) must write here
        os.makedirs(d, exist_ok=True)
        os.chmod(d, 0o777)
    vmetrics = os.path.join(REPO_ROOT, "out", "_verify", "metrics.csv")
    if os.path.exists(vmetrics):
        os.remove(vmetrics)
    stage_capture(cfg_path, cond, out_dir=vdir_rel)
    stage_analysis(cfg_path, cond, out_dir=vdir_rel, metrics=vmetrics)
    a = pd.read_csv(os.path.join(REPO_ROOT, "out", "metrics.csv"))
    a = a[a.condition == cond]
    b = pd.read_csv(vmetrics)
    m = a.merge(b, on=["prim", "frame"], suffixes=("_a", "_b"))
    tol = float(cfg["checks"]["determinism_tol_frac"])
    rel = (m["cnr_a"] - m["cnr_b"]).abs() / m["cnr_a"].abs().clip(lower=1e-9)
    worst = m.loc[rel.idxmax()]
    report = {"condition": cond, "rows_compared": int(len(m)), "max_rel_diff": float(rel.max()), "tolerance": tol,
              "pass": bool((rel <= tol).all()), "worst": {"prim": worst["prim"], "frame": int(worst["frame"]), "cnr_a": float(worst["cnr_a"]), "cnr_b": float(worst["cnr_b"])},
              "hdr_identical": None}
    try:
        import numpy as np
        from analysis.io_utils import load_hdr

        h0 = load_hdr(os.path.join(REPO_ROOT, "out", cond, "renders"), 0)
        h1 = load_hdr(os.path.join(vdir, "renders"), 0)
        report["hdr_identical"] = bool(np.array_equal(h0, h1))
        report["hdr_max_abs_diff"] = float(np.abs(h0 - h1).max())
    except Exception as exc:  # noqa: BLE001
        report["hdr_compare_error"] = str(exc)
    with open(os.path.join(REPO_ROOT, "out", "_verify", "determinism.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps(report, indent=1))
    if not report["pass"]:
        print("!!! determinism check failed")
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment.yaml")
    ap.add_argument("--condition", default=None, help="run this condition (default: the config's default condition)")
    ap.add_argument("--all", action="store_true", help="run every runnable condition")
    ap.add_argument("--skip-geometry", action="store_true")
    ap.add_argument("--skip-capture", action="store_true")
    ap.add_argument("--skip-control", action="store_true")
    ap.add_argument("--verify", action="store_true", help="determinism check: re-render + re-analyse the default condition and compare")
    a = ap.parse_args()
    cfg_path = a.config
    cfg = load_config(cfg_path)
    t0 = time.time()
    if a.verify:
        verify(cfg_path, cfg)
        return
    if not a.skip_geometry:
        stage_geometry(cfg_path)
    default = default_condition(cfg)
    if a.all:
        conds = [c for c, s in cfg["conditions"].items() if not (s or {}).get("control")]
        # the HDRI condition needs a site HDRI
        conds = [c for c in conds if not (cfg["conditions"][c] or {}).get("lighting", {}).get("mode") == "hdri" or cfg["lighting"]["hdri"].get("path")]
    else:
        conds = [a.condition or default]
    controls = [] if a.skip_control else [c for c, s in cfg["conditions"].items() if (s or {}).get("control")]
    for cond in conds + controls:
        if not a.skip_capture:
            stage_capture(cfg_path, cond)
        stage_analysis(cfg_path, cond)
    run([sys.executable, "analysis/plot_results.py", "--config", cfg_path], "Stage 4: figures + RESULTS.md")
    print(f"\nAll stages done in {(time.time() - t0)/60:.1f} min. See out/RESULTS.md")


if __name__ == "__main__":
    main()
