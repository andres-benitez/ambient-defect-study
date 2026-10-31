"""Pinhole camera model shared by the scene builder (to place the camera) and the analysis
(to project sub-pixel defects to image coordinates).

Conventions follow USD: the camera looks along its local -Z, +Y is image up, +X image right.
Image pixel (u, v): u to the right, v downwards, continuous coordinates with pixel i spanning [i, i+1).
World: coupon centre at the origin, top face z = 0, +x right, +y up (towards image top at nadir).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np


@dataclass
class CameraModel:
    focal_length_mm: float
    sensor_width_mm: float
    res_x: int
    res_y: int
    mm_per_px: float
    view: str
    tilt_deg: float
    distance_mm: float
    position_m: tuple  # world position (m)
    rotation_xyz_deg: tuple  # Euler XYZ (deg) to orient the camera (USD xformOp:rotateXYZ)

    @property
    def sensor_height_mm(self) -> float:
        return self.sensor_width_mm * self.res_y / self.res_x

    @property
    def f_px(self) -> float:
        return self.focal_length_mm / (self.sensor_width_mm / self.res_x)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["position_m"] = list(self.position_m)
        d["rotation_xyz_deg"] = list(self.rotation_xyz_deg)
        d["f_px"] = self.f_px
        d["sensor_height_mm"] = self.sensor_height_mm
        return d


def working_distance_mm(cfg_cam: dict) -> float:
    """distance = focal_length_mm * resolution_x * mm_per_px / sensor_width_mm"""
    return float(cfg_cam["focal_length_mm"]) * int(cfg_cam["resolution"][0]) * float(cfg_cam["mm_per_px"]) / float(cfg_cam["sensor_width_mm"])


def rot_x(deg: float) -> np.ndarray:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def build_camera(cfg_cam: dict) -> CameraModel:
    D = working_distance_mm(cfg_cam)
    view = str(cfg_cam.get("view", "nadir"))
    tilt = float(cfg_cam.get("oblique_tilt_deg", 30)) if view.startswith("oblique") else 0.0
    # identity camera looks along -z from (0,0,D). Rotating by tilt about +x moves the camera to
    # (0, -D sin t, D cos t), still looking at the origin, with image-up tilted towards +z.
    R = rot_x(tilt)
    pos = R @ np.array([0.0, 0.0, D]) / 1000.0
    return CameraModel(
        focal_length_mm=float(cfg_cam["focal_length_mm"]),
        sensor_width_mm=float(cfg_cam["sensor_width_mm"]),
        res_x=int(cfg_cam["resolution"][0]),
        res_y=int(cfg_cam["resolution"][1]),
        mm_per_px=float(cfg_cam["mm_per_px"]),
        view=view,
        tilt_deg=tilt,
        distance_mm=D,
        position_m=tuple(float(v) for v in pos),
        rotation_xyz_deg=(tilt, 0.0, 0.0),
    )


def camera_rotation(cam: CameraModel) -> np.ndarray:
    return rot_x(cam.tilt_deg)


def project_mm(cam: CameraModel, pts_mm: np.ndarray) -> np.ndarray:
    """World points (N,3) in mm -> continuous pixel coordinates (N,2) (u right, v down)."""
    pts = np.atleast_2d(np.asarray(pts_mm, dtype=float))
    R = camera_rotation(cam)
    P = np.asarray(cam.position_m) * 1000.0
    xc = (pts - P) @ R  # R^T (X - P): rows of R are world axes expressed... R is orthonormal so X@R = R^T X
    depth = -xc[:, 2]
    u = cam.res_x / 2.0 + cam.f_px * xc[:, 0] / depth
    v = cam.res_y / 2.0 - cam.f_px * xc[:, 1] / depth
    return np.stack([u, v], axis=1)


def expected_plate_width_px(cam: CameraModel, plate_w_mm: float) -> float:
    return plate_w_mm / cam.mm_per_px
