"""Unit tests for the metric code (run: python -m pytest tests/ or python tests/test_cnr.py)."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from analysis.metrics import measure_defect, fallback_window  # noqa: E402
from analysis.sensor_model import simulate, expected_noise_std, luminance, solve_gain  # noqa: E402
from ads.camera import build_camera, project_mm  # noqa: E402

ANALYSIS = {"bg_annulus_px": [3, 10], "mask_dilate_px": 1, "fallback_window_px": 3}
THRESH = {"cnr": 3.0, "min_area_px": 3, "marginal_cnr": 2.0}


def test_synthetic_disc_cnr_matches_analytic():
    rng = np.random.default_rng(0)
    h, w = 400, 400
    sigma, bg, delta = 0.01, 0.40, 0.08  # analytic CNR = 8
    img = bg + rng.standard_normal((h, w)) * sigma
    yy, xx = np.mgrid[:h, :w]
    disc = (xx - 200) ** 2 + (yy - 200) ** 2 <= 20**2
    img[disc] += delta
    zone = np.ones((h, w), bool)
    r = measure_defect(img, disc, zone, np.zeros((h, w), bool), ANALYSIS, THRESH)
    assert abs(r["cnr"] - delta / sigma) / (delta / sigma) < 0.05, r
    assert r["detectable"] and not r["marginal"] and not r["mask_empty"]
    assert abs(r["weber"] - delta / bg) / (delta / bg) < 0.05
    assert r["area_px"] == int(disc.sum())
    assert r["cnr_peak"] >= r["cnr"]


def test_background_excludes_other_defects_and_zone():
    rng = np.random.default_rng(1)
    h, w = 200, 200
    img = 0.4 + rng.standard_normal((h, w)) * 0.01
    yy, xx = np.mgrid[:h, :w]
    a = (xx - 100) ** 2 + (yy - 100) ** 2 <= 5**2
    b = (xx - 112) ** 2 + (yy - 100) ** 2 <= 4**2  # neighbour inside a's annulus
    img[b] = 1.0  # a very bright neighbour; must not leak into a's background
    zone = np.ones((h, w), bool)
    r_excl = measure_defect(img, a, zone, b, ANALYSIS, THRESH)
    r_noexcl = measure_defect(img, a, zone, np.zeros((h, w), bool), ANALYSIS, THRESH)
    assert r_excl["std_bg"] < 0.02 and r_noexcl["std_bg"] > 0.05
    # zone restriction: a half-plane zone halves the annulus
    half = xx < 100
    r_half = measure_defect(img, a, half, b, ANALYSIS, THRESH)
    assert 0.35 < r_half["n_bg_px"] / r_excl["n_bg_px"] < 0.65


def test_empty_mask_fallback_hits_the_right_window():
    h, w = 64, 64
    img = np.full((h, w), 0.4)
    img[30:33, 40:43] = 0.6  # 3x3 bright window centred at pixel (41, 31)
    rng = np.random.default_rng(2)
    img += rng.standard_normal((h, w)) * 0.005
    empty = np.zeros((h, w), bool)
    zone = np.ones((h, w), bool)
    win = fallback_window((h, w), (41.4, 31.7), 3)
    assert win.sum() == 9 and win[31, 41] and win[30, 40] and win[32, 42] and not win[29, 41]
    r = measure_defect(img, empty, zone, empty, ANALYSIS, THRESH, fallback_center_uv=(41.4, 31.7))
    assert r["mask_empty"] and r["area_px"] == 0 and not r["detectable"]  # area 0 < min_area
    assert r["cnr"] > 20, r
    # window elsewhere sees no contrast
    r2 = measure_defect(img, empty, zone, empty, ANALYSIS, THRESH, fallback_center_uv=(10.2, 10.9))
    assert r2["cnr"] < 2


def test_sensor_noise_matches_analytic():
    sensor = {"full_well_e": 10000, "read_noise_e": 3, "bit_depth": 8}
    rng = np.random.default_rng(3)
    Y = np.full((512, 512), 0.40)
    sim = simulate(Y, 1.0, sensor, rng)
    assert abs(sim.mean() - 0.40) < 0.002
    assert abs(sim.std() - expected_noise_std(0.40, sensor)) / expected_noise_std(0.40, sensor) < 0.05
    assert set(np.unique(np.round(sim * 255))) <= set(range(256))


def test_luminance_and_gain():
    rgb = np.zeros((4, 4, 4)); rgb[..., :3] = [0.2, 0.2, 0.2]; rgb[..., 3] = 1.0
    Y = luminance(rgb)
    assert np.allclose(Y, 0.2)
    g = solve_gain(Y, np.ones((4, 4), bool), 0.4)
    assert abs(g - 2.0) < 1e-9


def test_camera_projection_nadir_pixel_scale():
    cam = build_camera({"mm_per_px": 0.12, "resolution": [2448, 2048], "focal_length_mm": 25, "sensor_width_mm": 8.4, "view": "nadir"})
    assert abs(cam.distance_mm - 874.2857) < 0.01
    uv = project_mm(cam, np.array([[0, 0, 0], [60, 0, 0], [-60, 0, 0], [0, 40, 0], [12, -24, 0]]))
    assert np.allclose(uv[0], [1224, 1024])
    assert abs((uv[1, 0] - uv[2, 0]) - 1000.0) < 1e-6  # 120 mm plate spans 1000 px
    assert uv[3, 1] < uv[0, 1]  # +y world is up in the image (smaller v)
    assert np.allclose(uv[4], [1224 + 100, 1024 + 200])


def test_camera_projection_oblique_keeps_centre():
    cam = build_camera({"mm_per_px": 0.12, "resolution": [2448, 2048], "focal_length_mm": 25, "sensor_width_mm": 8.4, "view": "oblique30", "oblique_tilt_deg": 30})
    uv = project_mm(cam, np.array([[0, 0, 0], [60, 0, 0], [0, 40, 0]]))
    assert np.allclose(uv[0], [1224, 1024])
    assert abs((uv[1, 0] - 1224) - 500) < 1e-6  # x unaffected at the centre row
    assert uv[2, 1] < 1024 and (1024 - uv[2, 1]) < 333.4  # foreshortened: cos30 * 333.3 px


if __name__ == "__main__":
    import pytest

    sys.exit(pytest.main([__file__, "-q"]))


def test_saturated_background_gives_nan_cnr():
    img = np.ones((64, 64))  # fully clipped
    yy, xx = np.mgrid[:64, :64]
    disc = (xx - 32) ** 2 + (yy - 32) ** 2 <= 4**2
    img[disc] = 0.9
    r = measure_defect(img, disc, np.ones((64, 64), bool), np.zeros((64, 64), bool), ANALYSIS, THRESH)
    assert r["saturated_bg"] and np.isnan(r["cnr"]) and not r["detectable"]


def test_optics_psf_and_dof():
    from analysis.optics import psf_sigma_um, airy_diameter_um, depth_of_field_mm, apply_psf
    assert abs(airy_diameter_um(8, 550) - 10.74) < 0.05
    assert abs(psf_sigma_um(8, 550, 0.0) - 1.936) < 0.01
    assert abs(psf_sigma_um(8, 550, 2.0) - (1.936**2 + 4) ** 0.5) < 1e-6
    # 2.74 um pixels at 0.1225 mm/px, f/8: m = 0.02237, c = 2.74 um -> DoF = 2*8*0.00274/m^2 ~ 87.6 mm
    assert abs(depth_of_field_mm(8, 2.74, 0.1225) - 87.6) < 1.0
    img = np.zeros((32, 32)); img[16, 16] = 1.0
    b = apply_psf(img, 1.5)
    assert abs(b.sum() - 1.0) < 1e-6 and b[16, 16] < 0.1


def test_photometry_exposure_reasonable():
    from analysis.photometry import exposure_ms
    cfg = {"photometry": {"ceiling_solid_angle_sr": 0.78, "machined_reflectance": 0.85, "lumens_per_watt": 250},
           "camera": {"f_number": 8, "pixel_um": 2.74, "mm_per_px": 0.1225, "resolution": [2448, 2048], "sensor_width_mm": 6.7},
           "sensor": {"qe": 0.66, "target_bg_level": 0.4, "full_well_e": 9900}, "lens": {"transmission": 0.9, "wavelength_nm": 550}}
    r = exposure_ms(500, cfg)
    assert 5 < r["exposure_ms"] < 40, r  # a few tens of ms at 500 lux and f/8
    assert abs(exposure_ms(1000, cfg)["exposure_ms"] * 2 - r["exposure_ms"]) < 1e-6


def test_rolling_shutter_banding():
    from analysis.rolling_shutter import row_gains, banding_amplitude
    g = row_gains(1000, 22.0, 10.0, 100.0, 0.3)
    assert g.shape == (1000,) and abs(g.mean() - 1.0) < 0.02
    assert banding_amplitude(10.0, 100.0, 0.3) < 1e-9  # exposure = one flicker period: no bands
    assert abs(banding_amplitude(5.0, 100.0, 0.3) - 0.3 * 2 / np.pi) < 1e-6  # half a period: 64% of m
