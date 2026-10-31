"""Experiment 10: does CNR predict what a learned detector finds?

  python analysis/detector_check.py --config config/experiment.yaml

A small learned detector (gradient-boosted trees on normalised 15 x 15 crops plus a few contrast
features) is trained on crops from the training conditions (defect-centred positives with +-2 px
jitter, random background negatives) and tested on held-out conditions. Per defect the detector
score is compared with the measured CNR; the detection rate per CNR bin is the result. Writes
out/detector_check.csv, out/detector_check.json and out/figures/detector_check.png.
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
from ads.camera import build_camera, project_mm  # noqa: E402
from analysis.plot_results import style_axes, COLORS, INK, INK2, MUTED, SURFACE  # noqa: E402

K = 7  # half-size of the crop (15 x 15)


def crops_for(cond: str, cfg: dict, out: str, rng, lens: str = "none", n_neg_per_pos: int = 3):
    import imageio.v3 as iio

    cc = resolve_condition(cfg, cond)
    cam = build_camera(cc["camera"])
    path = os.path.join(out, cond, "sim_0.png" if lens == "none" else f"sim_0_{lens}.png")
    if not os.path.exists(path):
        return None
    img = iio.imread(path)
    if img.ndim == 3:
        img = img[..., 0]
    img = img.astype(np.float32) / 255.0
    sem = np.load(os.path.join(out, cond, "renders", "zone_0.npy"))
    with open(os.path.join(out, cond, "renders", "id_map.json")) as fh:
        idm = json.load(fh)
    inst = np.load(os.path.join(out, cond, "renders", "mask_0.npy"))
    defs = load_defects(cc)
    m = pd.read_csv(os.path.join(out, "metrics.csv"))
    m = m[(m.condition == cond) & (m.frame == 0) & (m.lens == lens)].set_index("prim")
    X, y, meta = [], [], []
    defect_px = np.isin(inst, sum((idm["prim_to_ids"].get(d.prim, []) for d in defs), []))
    for d in defs:
        u, v = project_mm(cam, np.array([d.center3_mm]))[0]
        cu, cv = int(u), int(v)
        for dx, dy in ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (2, 1), (-2, -1)):
            c = img[cv + dy - K:cv + dy + K + 1, cu + dx - K:cu + dx + K + 1]
            if c.shape != (2 * K + 1, 2 * K + 1):
                continue
            X.append(c)
            y.append(1)
            meta.append({"condition": cond, "prim": d.prim, "zone": d.zone, "type": d.type, "size_mm": d.size_mm,
                         "cnr": float(m.loc[d.prim, "cnr"]) if d.prim in m.index else np.nan, "jitter": dx != 0 or dy != 0})
    # negatives: random positions inside the zones, away from any defect
    zones = np.isin(sem, idm["class_to_ids"].get("zone_cast", []) + idm["class_to_ids"].get("zone_machined", []))
    from scipy import ndimage

    ok = ndimage.binary_erosion(zones, iterations=K + 2) & ~ndimage.binary_dilation(defect_px, iterations=K + 4)
    for d in defs:  # also keep clear of maskless defects
        u, v = project_mm(cam, np.array([d.center3_mm]))[0]
        ok[int(v) - K - 4:int(v) + K + 5, int(u) - K - 4:int(u) + K + 5] = False
    ys, xs = np.where(ok)
    n_neg = n_neg_per_pos * sum(1 for _ in y)
    idx = rng.choice(len(ys), size=min(n_neg, len(ys)), replace=False)
    zone_cast = np.isin(sem, idm["class_to_ids"].get("zone_cast", []))
    for i in idx:
        c = img[ys[i] - K:ys[i] + K + 1, xs[i] - K:xs[i] + K + 1]
        X.append(c)
        y.append(0)
        meta.append({"condition": cond, "prim": None, "zone": "cast" if zone_cast[ys[i], xs[i]] else "machined", "type": "background", "size_mm": np.nan, "cnr": np.nan, "jitter": False})
    return np.array(X), np.array(y), pd.DataFrame(meta)


def features(X: np.ndarray) -> np.ndarray:
    """Per-crop normalisation (subtract the median, divide by a robust spread) plus summary features."""
    med = np.median(X.reshape(len(X), -1), axis=1)[:, None, None]
    q1, q3 = np.percentile(X.reshape(len(X), -1), [25, 75], axis=1)
    spread = ((q3 - q1) / 1.349)[:, None, None] + 1e-4
    Z = (X - med) / spread
    c = Z[:, K - 1:K + 2, K - 1:K + 2].mean(axis=(1, 2))
    ring = np.concatenate([Z[:, 0:2, :].reshape(len(Z), -1), Z[:, -2:, :].reshape(len(Z), -1)], axis=1).std(axis=1)
    return np.concatenate([Z.reshape(len(Z), -1), c[:, None], np.abs(c)[:, None], ring[:, None], Z.min(axis=(1, 2))[:, None], Z.max(axis=(1, 2))[:, None]], axis=1)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--config", default=os.path.join(REPO_ROOT, "config", "experiment.yaml"))
    a.add_argument("--out", default=os.path.join(REPO_ROOT, "out"))
    a.add_argument("--train", nargs="*", default=["ambient_nadir_012", "cam_a2A2448", "cam_a2A3840", "cam_a2A1920"])
    a.add_argument("--test", nargs="*", default=["ambient_nadir_005", "cam_a2A4096", "cam_a2A4508", "cam_a2A5320", "ext_nadir_012", "part_5mp", "part_18mp"])
    args = a.parse_args()
    cfg = load_config(args.config)
    rng = np.random.default_rng(int(cfg["seed"]))
    from sklearn.ensemble import HistGradientBoostingClassifier

    def gather(conds):
        Xs, ys, metas = [], [], []
        for c in conds:
            r = crops_for(c, cfg, args.out, rng)
            if r is None:
                continue
            Xs.append(r[0]); ys.append(r[1]); metas.append(r[2])
        return np.concatenate(Xs), np.concatenate(ys), pd.concat(metas, ignore_index=True)

    Xtr, ytr, mtr = gather(args.train)
    Xte, yte, mte = gather([c for c in args.test if os.path.exists(os.path.join(args.out, c, "sim_0.png"))])
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, random_state=int(cfg["seed"]))
    clf.fit(features(Xtr), ytr)
    # operating point: per test condition, the score threshold that gives 1 % false alarms on that
    # condition's own background crops (the calibration step any deployed station performs)
    p_te = clf.predict_proba(features(Xte))[:, 1]
    mte = mte.assign(score=p_te, label=yte)
    thr_by_cond = {}
    for c, g in mte.groupby("condition"):
        neg_scores = np.sort(g[g.label == 0].score.values)
        thr_by_cond[c] = float(neg_scores[int(0.99 * len(neg_scores)) - 1]) if len(neg_scores) else 0.5
    mte["threshold"] = mte.condition.map(thr_by_cond)
    mte["detected"] = mte.score >= mte.threshold
    thr = thr_by_cond
    pos = mte[(mte.label == 1) & (~mte.jitter)]
    neg = mte[mte.label == 0]
    fa_rate = float(neg.detected.mean())
    bins = [0, 1, 2, 3, 4, 6, 10, 20, 1000]
    pos = pos.assign(cnr_bin=pd.cut(pos.cnr.fillna(0), bins, include_lowest=True))
    rate = pos.groupby("cnr_bin", observed=True).agg(n=("detected", "size"), rate=("detected", "mean")).reset_index()
    rate["cnr_bin"] = rate["cnr_bin"].astype(str)
    out = {"train": args.train, "test": sorted(mte.condition.unique()), "n_train": int(len(ytr)), "n_test_pos": int(len(pos)), "n_test_neg": int(len(neg)),
           "score_threshold_per_condition": thr, "test_false_alarm_rate_per_crop": fa_rate,
           "detection_rate_cnr_ge_3": float(pos[pos.cnr >= 3].detected.mean()) if (pos.cnr >= 3).any() else None,
           "detection_rate_cnr_lt_2": float(pos[pos.cnr < 2].detected.mean()) if (pos.cnr < 2).any() else None,
           "per_bin": rate.to_dict("records"), "per_condition": pos.groupby("condition").detected.mean().round(3).to_dict()}
    mte.to_csv(os.path.join(args.out, "detector_check.csv"), index=False)
    with open(os.path.join(args.out, "detector_check.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURFACE)
    style_axes(ax)
    x = np.arange(len(rate))
    ax.bar(x, rate["rate"] * 100, color=COLORS["machined"], width=0.6)
    for xi, (r, n) in enumerate(zip(rate["rate"], rate["n"])):
        ax.text(xi, r * 100 + 2, f"n={n}", ha="center", fontsize=8, color=INK2)
    ax.set_xticks(x, rate["cnr_bin"], fontsize=8)
    ax.set_ylim(0, 110)
    ax.set_xlabel("measured CNR of the defect", color=INK2, fontsize=9)
    ax.set_ylabel("found by the learned detector, %", color=INK2, fontsize=9)
    ax.set_title(f"Detection rate against CNR on held-out conditions, detector calibrated to 1 % false alarms per condition", loc="left", color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "figures", "detector_check.png"), dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(json.dumps({k: v for k, v in out.items() if k != "per_bin"}, indent=1))
    print(rate.to_string(index=False))


if __name__ == "__main__":
    main()
