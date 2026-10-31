"""Exposure-time estimate: how long the camera must expose under a given hall illuminance to put
the machined face at the target level. All inputs are explicit assumptions from the config.

Chain: illuminance on the part E (lux) -> luminance of the lit ceiling L_c = E / Omega_proj
-> luminance of the machined face L = rho * L_c (it mirrors the ceiling) -> sensor illuminance
E_s = pi * T * L / (4 N^2 (1+m)^2) -> photons per pixel per second -> electrons with QE -> time to
reach target_level * full_well.
"""
from __future__ import annotations

import math

H_C = 6.62607e-34 * 2.99792e8  # J m


def photons_per_lux_m2_s(lumens_per_watt: float, wavelength_nm: float = 550.0) -> float:
    watts_per_m2_per_lux = 1.0 / lumens_per_watt
    e_photon = H_C / (wavelength_nm * 1e-9)
    return watts_per_m2_per_lux / e_photon


def exposure_ms(illuminance_lux: float, cfg: dict, f_number: float | None = None) -> dict:
    ph = cfg["photometry"]
    cam, sensor, lens = cfg["camera"], cfg["sensor"], cfg["lens"]
    N = float(f_number if f_number is not None else cam.get("f_number", 8))
    pixel_um = float(cam.get("pixel_um") or (float(cam["sensor_width_mm"]) * 1000.0 / int(cam["resolution"][0])))
    m = pixel_um / (float(cam["mm_per_px"]) * 1000.0)  # magnification
    L_ceiling = illuminance_lux / float(ph["ceiling_solid_angle_sr"])
    L_mach = float(ph["machined_reflectance"]) * L_ceiling
    E_sensor = math.pi * float(lens.get("transmission", 0.9)) * L_mach / (4.0 * N * N * (1.0 + m) ** 2)
    flux = photons_per_lux_m2_s(float(ph["lumens_per_watt"]), float(lens.get("wavelength_nm", 550)))
    pixel_area = (pixel_um * 1e-6) ** 2
    electrons_per_s = E_sensor * flux * pixel_area * float(sensor.get("qe", 0.6))
    target_e = float(sensor["target_bg_level"]) * float(sensor["full_well_e"])
    t_s = target_e / electrons_per_s
    return {
        "illuminance_lux": illuminance_lux, "f_number": N, "magnification": m,
        "ceiling_luminance_cd_m2": L_ceiling, "machined_luminance_cd_m2": L_mach,
        "sensor_illuminance_lux": E_sensor, "electrons_per_s": electrons_per_s,
        "target_electrons": target_e, "exposure_ms": t_s * 1000.0,
    }
