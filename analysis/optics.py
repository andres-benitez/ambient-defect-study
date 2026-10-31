"""Lens model applied in analysis: diffraction plus a lens-quality blur, as a Gaussian PSF.

The path tracer samples each pixel's footprint with many jittered rays, so the render already
integrates over the pixel aperture. The optics are added afterwards as a convolution of the linear
image (convolutions commute, so the order does not matter apart from sub-pixel aliasing).

PSF width: an Airy pattern at f-number N and wavelength lambda has FWHM ~ 1.03 lambda N, which a
Gaussian matches with sigma ~ 0.44 lambda N. Lens aberrations are modelled as an RMS spot radius
`blur_um` added in quadrature. The result is converted to pixels with the pixel pitch and to object
space with the magnification.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

AIRY_TO_GAUSS = 0.44


def psf_sigma_um(f_number: float, wavelength_nm: float, blur_um: float | None) -> float:
    diff = AIRY_TO_GAUSS * (wavelength_nm / 1000.0) * float(f_number)
    return math.sqrt(diff**2 + float(blur_um or 0.0) ** 2)


def airy_diameter_um(f_number: float, wavelength_nm: float) -> float:
    """Diameter of the Airy disc (first dark ring)."""
    return 2.44 * (wavelength_nm / 1000.0) * float(f_number)


def depth_of_field_mm(f_number: float, pixel_um: float, mm_per_px: float, coc_px: float = 1.0) -> float:
    """Total depth of field for a circle of confusion of `coc_px` pixels: 2 N c / m^2 with
    magnification m = pixel_um / (mm_per_px * 1000)."""
    m = pixel_um / (mm_per_px * 1000.0)
    c_mm = coc_px * pixel_um / 1000.0
    return 2.0 * float(f_number) * c_mm / (m * m)


def apply_psf(img: np.ndarray, sigma_px: float) -> np.ndarray:
    """Gaussian blur of an HxW or HxWxC linear image; sigma in pixels. sigma < 0.05 px is a no-op."""
    if sigma_px < 0.05:
        return img
    if img.ndim == 2:
        return ndimage.gaussian_filter(img, sigma_px, mode="nearest")
    return np.stack([ndimage.gaussian_filter(img[..., c], sigma_px, mode="nearest") for c in range(img.shape[-1])], axis=-1)


def scenario_psf(cfg: dict, scenario: str, f_number_override: float | None = None) -> dict:
    """Resolve a lens scenario of the config into sigma in um, px and object mm."""
    lens = cfg["lens"]
    sc = lens["scenarios"][scenario]
    cam = cfg["camera"]
    pixel_um = float(cam.get("pixel_um") or (float(cam["sensor_width_mm"]) * 1000.0 / int(cam["resolution"][0])))
    N = float(f_number_override if f_number_override is not None else sc.get("f_number", cam.get("f_number", 8)))
    blur = sc.get("blur_um")
    if blur is None and scenario != "none":
        blur = 0.0
    if scenario == "none" or (blur is None):
        sigma_um = 0.0
    else:
        sigma_um = psf_sigma_um(N, float(lens.get("wavelength_nm", 550)), blur)
    sigma_px = sigma_um / pixel_um
    return {
        "lens": scenario,
        "f_number": N,
        "blur_um": None if scenario == "none" else float(blur or 0.0),
        "psf_sigma_um": sigma_um,
        "psf_sigma_px": sigma_px,
        "psf_sigma_object_mm": sigma_px * float(cam["mm_per_px"]),
        "airy_diameter_um": airy_diameter_um(N, float(lens.get("wavelength_nm", 550))),
        "pixel_um": pixel_um,
        "depth_of_field_mm": depth_of_field_mm(N, pixel_um, float(cam["mm_per_px"])),
    }
