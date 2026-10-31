"""Procedural cast-skin normal map (tileable, 16-bit PNG).

  python analysis/gen_textures.py --config config/experiment.yaml --out assets/coupon/cast_normal.png

A tileable fractal height field is synthesised in the Fourier domain (periodic by construction),
scaled so that the RMS surface slope equals `textures.cast_normal.slope_deg`, and converted to a
tangent-space normal map. One UV tile spans `coupon.uv_tile_mm`, so the physical texel size is
uv_tile_mm / resolution (about 10 um at the defaults).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from ads.config import load_config  # noqa: E402


def fractal_height(n: int, base_period_px: float, octaves: int, lacunarity: float, gain: float, rng) -> np.ndarray:
    """Sum of band-limited periodic noise octaves. Each octave is white noise filtered to a
    band around spatial frequency n/period; amplitude falls by `gain` per octave."""
    fy = np.fft.fftfreq(n)[:, None] * n  # cycles per tile
    fx = np.fft.fftfreq(n)[None, :] * n
    f = np.hypot(fx, fy)
    h = np.zeros((n, n), dtype=np.float64)
    amp = 1.0
    period = base_period_px
    for _ in range(octaves):
        f0 = n / period
        band = np.exp(-0.5 * ((np.log(f + 1e-9) - np.log(f0)) / 0.45) ** 2)  # log-normal band
        band[0, 0] = 0.0
        spec = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) * band
        oct_h = np.fft.ifft2(spec).real
        oct_h /= oct_h.std() + 1e-12
        h += amp * oct_h
        amp *= gain
        period /= lacunarity
    # mild "pebbling": soften peaks so bumps look like solidified skin rather than symmetric noise
    h = np.tanh(1.2 * h / h.std())
    h -= h.mean()
    h /= h.std()
    return h


def height_to_normal(h: np.ndarray, texel_mm: float, slope_rms_deg: float):
    """Scale the height field so RMS slope = slope_rms_deg, return (normal map [0,1], height_mm)."""
    gy, gx = np.gradient(h)  # per texel, periodic edges differ slightly: use roll-based central diff
    gx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) / 2.0
    gy = (np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)) / 2.0
    slope_rms = np.sqrt(np.mean(gx**2 + gy**2))  # per texel, unit height
    target = np.tan(np.deg2rad(slope_rms_deg))
    scale = target / slope_rms  # height units per texel -> height in texels
    height_mm = h * scale * texel_mm
    dx, dy = gx * scale, gy * scale  # dimensionless slopes
    nrm = np.stack([-dx, -dy, np.ones_like(dx)], axis=-1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
    # OpenGL-style tangent space: +Y up in texture space. Image rows run top->bottom, so flip y.
    rgb = np.empty_like(nrm)
    rgb[..., 0] = nrm[..., 0] * 0.5 + 0.5
    rgb[..., 1] = -nrm[..., 1] * 0.5 + 0.5
    rgb[..., 2] = nrm[..., 2]
    stats = {
        "slope_rms_deg": float(np.rad2deg(np.arctan(np.sqrt(np.mean(dx**2 + dy**2))))),
        "slope_p99_deg": float(np.rad2deg(np.arctan(np.percentile(np.sqrt(dx**2 + dy**2), 99)))),
        "height_rms_um": float(height_mm.std() * 1000),
        "height_pp_um": float((height_mm.max() - height_mm.min()) * 1000),
    }
    return rgb, stats


def write_png16(path: str, img: np.ndarray) -> None:
    """Minimal 16-bit RGB PNG writer (Pillow cannot write 16-bit RGB)."""
    import struct
    import zlib

    h, w, c = img.shape
    assert c == 3 and img.dtype == np.uint16
    raw = b"".join(b"\x00" + img[y].astype(">u2").tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0)
    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "config", "experiment.yaml"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    t = cfg["textures"]["cast_normal"]
    n = int(t["resolution"])
    out = a.out or os.path.join(REPO_ROOT, cfg["materials"]["cast"]["normal_map"])
    rng = np.random.default_rng(int(cfg["seed"]) + 7)
    h = fractal_height(n, float(t["base_period_px"]), int(t["octaves"]), float(t["lacunarity"]), float(t["gain"]), rng)
    texel_mm = float(cfg["coupon"]["uv_tile_mm"]) / n
    rgb, stats = height_to_normal(h, texel_mm, float(t["slope_deg"]))
    img16 = np.clip(np.round(rgb * 65535.0), 0, 65535).astype(np.uint16)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_png16(out, img16)
    stats.update({"resolution": n, "texel_um": texel_mm * 1000, "tile_mm": cfg["coupon"]["uv_tile_mm"], "file": os.path.relpath(out, REPO_ROOT)})
    with open(os.path.splitext(out)[0] + "_stats.json", "w") as fh:
        json.dump(stats, fh, indent=1)
    print("[gen_textures]", json.dumps(stats))


if __name__ == "__main__":
    main()
