"""Per-defect contrast metrics on a simulated camera image.

measure_defect() implements the table in the plan: area_px, mean_in, mean_bg, std_bg, cnr,
cnr_peak, weber, detectable, marginal, mask_empty. The background is an annulus from
`bg_annulus_px[0]` to `bg_annulus_px[1]` outside the (dilated) defect mask, restricted to the
defect's zone and excluding every other defect's dilated mask.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask.astype(bool)
    return ndimage.binary_dilation(mask, structure=np.ones((3, 3), bool), iterations=int(px))


def annulus(mask: np.ndarray, r_in: int, r_out: int) -> np.ndarray:
    """Pixels whose Chebyshev distance to the mask is in (r_in, r_out]."""
    inner = dilate(mask, r_in)
    outer = dilate(mask, r_out)
    return outer & ~inner


def fallback_window(shape, center_uv, size_px: int) -> np.ndarray:
    """size x size window centred on the pixel containing continuous coordinate (u, v)."""
    h, w = shape[:2]
    cu, cv = int(np.floor(center_uv[0])), int(np.floor(center_uv[1]))
    r = size_px // 2
    m = np.zeros((h, w), bool)
    u0, u1 = max(cu - r, 0), min(cu + r + 1, w)
    v0, v1 = max(cv - r, 0), min(cv + r + 1, h)
    if u1 > u0 and v1 > v0:
        m[v0:v1, u0:u1] = True
    return m


def measure_defect(img: np.ndarray, mask: np.ndarray, zone_mask: np.ndarray, exclude_mask: np.ndarray,
                   cfg_analysis: dict, cfg_threshold: dict, fallback_center_uv=None) -> dict:
    """img: simulated image in [0,1]; mask: this defect's pixels; zone_mask: pixels of the defect's
    zone; exclude_mask: union of every *other* defect's dilated mask. If `mask` is empty, a
    fallback window around `fallback_center_uv` is used and mask_empty is set."""
    r_in, r_out = (int(v) for v in cfg_analysis["bg_annulus_px"])
    dil = int(cfg_analysis.get("mask_dilate_px", 1))
    mask = mask.astype(bool)
    area = int(mask.sum())
    mask_empty = area == 0
    region = mask
    if mask_empty:
        if fallback_center_uv is None:
            raise ValueError("empty mask and no fallback centre")
        region = fallback_window(img.shape, fallback_center_uv, int(cfg_analysis.get("fallback_window_px", 3)))
    # work on a crop around the region: the annulus never reaches further than r_out + dil pixels
    ys, xs = np.where(region)
    margin = r_out + dil + 2
    y0, y1 = max(int(ys.min()) - margin, 0), min(int(ys.max()) + margin + 1, img.shape[0])
    x0, x1 = max(int(xs.min()) - margin, 0), min(int(xs.max()) + margin + 1, img.shape[1])
    sl = (slice(y0, y1), slice(x0, x1))
    img, region, zone_mask, exclude_mask = img[sl], region[sl], zone_mask[sl], exclude_mask[sl]
    core = dilate(region, dil)
    bg = annulus(core, r_in, r_out) & zone_mask.astype(bool) & ~exclude_mask.astype(bool)
    n_bg = int(bg.sum())
    if n_bg < 20:
        raise ValueError(f"only {n_bg} background pixels available")
    vals_in = img[region].astype(np.float64)
    vals_bg = img[bg].astype(np.float64)
    mean_in, mean_bg = float(vals_in.mean()), float(vals_bg.mean())
    std_bg = float(vals_bg.std(ddof=1))
    saturated_bg = bool(std_bg < 1e-9)  # a clipped (all-equal) background has no noise estimate
    if saturated_bg:
        cnr, cnr_peak = float("nan"), float("nan")
    else:
        cnr = abs(mean_in - mean_bg) / std_bg
        cnr_peak = float(np.max(np.abs(vals_in - mean_bg)) / std_bg)
    weber = (mean_in - mean_bg) / mean_bg if mean_bg > 0 else float("nan")
    thr = float(cfg_threshold["cnr"])
    min_area = int(cfg_threshold["min_area_px"])
    marginal_thr = float(cfg_threshold.get("marginal_cnr", 2.0))
    return {
        "area_px": area,
        "n_bg_px": n_bg,
        "mean_in": mean_in,
        "mean_bg": mean_bg,
        "std_bg": std_bg,
        "cnr": float(cnr),
        "cnr_peak": cnr_peak,
        "weber": float(weber),
        "detectable": bool(cnr >= thr and area >= min_area),
        "marginal": bool(marginal_thr <= cnr < thr),
        "mask_empty": bool(mask_empty),
        "saturated_bg": saturated_bg,
    }
