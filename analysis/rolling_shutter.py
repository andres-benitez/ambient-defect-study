"""Rolling-shutter banding under flickering light.

Rows are exposed one after another over the frame readout time. If the illumination flickers,
each row averages a different slice of the flicker and the image shows horizontal bands. With a
static part there is no geometric distortion; banding is the only rolling-shutter effect that
matters here, and it vanishes under daylight (no flicker) or when the exposure is a whole number
of flicker periods.
"""
from __future__ import annotations

import numpy as np


def row_gains(n_rows: int, readout_ms: float, exposure_ms: float, flicker_hz: float, modulation: float, phase: float = 0.0) -> np.ndarray:
    """Relative exposure of each row: mean of 1 + m sin(2 pi f t + phase) over that row's exposure."""
    if modulation <= 0 or flicker_hz <= 0:
        return np.ones(n_rows)
    t0 = np.arange(n_rows) * (readout_ms / 1000.0) / max(n_rows - 1, 1)
    T = exposure_ms / 1000.0
    w = 2.0 * np.pi * flicker_hz
    if T <= 0:
        return 1.0 + modulation * np.sin(w * t0 + phase)
    return 1.0 + modulation * (np.cos(w * t0 + phase) - np.cos(w * (t0 + T) + phase)) / (w * T)


def banding_amplitude(exposure_ms: float, flicker_hz: float, modulation: float) -> float:
    """Peak relative band amplitude for a given exposure: m * |sinc(f T)| (sinc in the sin(x)/x sense)."""
    x = np.pi * flicker_hz * exposure_ms / 1000.0
    return float(modulation * abs(np.sin(x) / x)) if x > 0 else float(modulation)


def apply_banding(img: np.ndarray, gains: np.ndarray) -> np.ndarray:
    g = gains.reshape(-1, *([1] * (img.ndim - 1)))
    return img * g
