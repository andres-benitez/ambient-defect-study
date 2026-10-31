"""Residual Monte-Carlo noise estimate for a linear render (shared by capture and analysis).

The machined zone has a uniform material and smooth illumination, so the detrended per-pixel
standard deviation inside small defect-free patches is the render noise. Reported in the same
units as the sensor model input (linear radiance, background ~ target level).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def flat_patch_std(Y: np.ndarray, bg_mask: np.ndarray, patch: int = 24, max_patches: int = 64, seed: int = 0) -> dict:
    """Median detrended std over up to `max_patches` patches fully inside bg_mask."""
    rng = np.random.default_rng(seed)
    eroded = ndimage.binary_erosion(bg_mask, structure=np.ones((patch + 2, patch + 2), bool))
    ys, xs = np.where(eroded)
    if len(ys) == 0:
        raise ValueError("no flat background patch fits inside the mask")
    idx = rng.choice(len(ys), size=min(max_patches, len(ys)), replace=False)
    yy, xx = np.mgrid[:patch, :patch]
    A = np.c_[np.ones(patch * patch), yy.ravel(), xx.ravel()]
    stds, means = [], []
    for i in idx:
        r0, c0 = ys[i] - patch // 2, xs[i] - patch // 2
        p = Y[r0 : r0 + patch, c0 : c0 + patch].astype(np.float64)
        coef = np.linalg.lstsq(A, p.ravel(), rcond=None)[0]
        stds.append((p.ravel() - A @ coef).std(ddof=3))
        means.append(p.mean())
    return {"std": float(np.median(stds)), "level": float(np.median(means)), "n_patches": int(len(idx)), "patch_px": patch}
