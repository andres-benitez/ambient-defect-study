"""Simple industrial mono camera: luminance -> gain -> shot + read noise -> N-bit quantisation.

All images are float arrays in [0, 1] of full scale once the gain is applied. The gain is solved
once per condition so that the machined-zone background mean equals `sensor.target_bg_level`.
"""
from __future__ import annotations

import numpy as np

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def luminance(rgb: np.ndarray) -> np.ndarray:
    """Y = 0.2126 R + 0.7152 G + 0.0722 B (mono camera). Accepts HxWx3 or HxWx4."""
    rgb = np.asarray(rgb, dtype=np.float64)
    return rgb[..., :3] @ LUMA


def solve_gain(Y: np.ndarray, bg_mask: np.ndarray, target_bg_level: float) -> float:
    m = float(np.mean(Y[bg_mask]))
    if not np.isfinite(m) or m <= 0:
        raise ValueError(f"background mean {m} is not positive; cannot solve gain")
    return float(target_bg_level) / m


def simulate(Y: np.ndarray, gain: float, sensor: dict, rng: np.random.Generator) -> np.ndarray:
    """Apply gain, shot noise (Poisson on electrons), Gaussian read noise and quantisation."""
    full_well = float(sensor["full_well_e"])
    read_noise = float(sensor["read_noise_e"])
    bits = int(sensor["bit_depth"])
    lin = np.clip(np.asarray(Y, dtype=np.float64) * gain, 0.0, None)
    electrons = lin * full_well
    # Poisson for moderate counts; Gaussian approximation above 1e6 e- (identical to 5 decimals)
    noisy = np.where(electrons < 1e6, rng.poisson(np.minimum(electrons, 1e6)).astype(np.float64),
                     electrons + rng.standard_normal(electrons.shape) * np.sqrt(np.maximum(electrons, 0)))
    noisy = noisy + rng.standard_normal(electrons.shape) * read_noise
    levels = (1 << bits) - 1
    out = np.clip(noisy / full_well, 0.0, 1.0)
    return np.round(out * levels) / levels


def expected_noise_std(level: float, sensor: dict) -> float:
    """Analytic sensor noise std (fraction of full scale) at a given signal level in [0,1]."""
    full_well = float(sensor["full_well_e"])
    read_noise = float(sensor["read_noise_e"])
    bits = int(sensor["bit_depth"])
    shot = np.sqrt(max(level, 0.0) * full_well) / full_well
    read = read_noise / full_well
    quant = (1.0 / ((1 << bits) - 1)) / np.sqrt(12.0)
    return float(np.sqrt(shot**2 + read**2 + quant**2))


def to_uint8(img01: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.asarray(img01) * 255.0), 0, 255).astype(np.uint8)
