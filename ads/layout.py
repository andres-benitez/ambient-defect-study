"""Coupon layout: where every defect sits, how big it is, and what it is called.

Shared by the Blender generator (to build geometry), the capture stage (to sanity check)
and the analysis (to project sub-pixel defects into the image).
Coordinates are in mm, origin at the coupon centre, +x to the right, +y up, top face at z = 0.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List

ROW_ORDER = ["pore_shallow", "pore_deep", "scratch", "burr"]
ZONE_SHORT = {"cast": "cast", "machined": "mach"}


@dataclass
class Defect:
    prim: str
    zone: str
    type: str
    size_mm: float
    depth_mm: float
    length_mm: float | None
    width_mm: float | None
    center_mm: List[float]
    row: int
    col: int

    def to_record(self) -> dict:
        d = asdict(self)
        d["center_mm"] = [round(self.center_mm[0], 6), round(self.center_mm[1], 6)]
        return d


def defect_name(zone: str, dtype: str, size_mm: float) -> str:
    size_um = int(round(size_mm * 1000))
    return f"D_{ZONE_SHORT[zone]}_{dtype}_{size_um:04d}"


def size_list(dcfg: dict, dtype: str) -> list:
    if dtype.startswith("pore"):
        return list(dcfg[dtype]["diam_mm"])
    if dtype == "scratch":
        return list(dcfg[dtype]["width_mm"])
    if dtype == "burr":
        return list(dcfg[dtype]["height_mm"])
    raise KeyError(dtype)


def zone_extent_x(cfg: dict, zone: str) -> tuple:
    """(xmin, xmax) of a zone in coupon mm coordinates."""
    plate_w = float(cfg["coupon"]["plate_mm"][0])
    zones = list(cfg["coupon"]["zones"])
    n = len(zones)
    w = plate_w / n
    i = zones.index(zone)
    xmin = -plate_w / 2 + i * w
    return xmin, xmin + w


def build_layout(cfg: dict) -> List[Defect]:
    """4 rows (defect families) x N sizes per zone, centred on the zone, pitch from config."""
    coupon = cfg["coupon"]
    dcfg = cfg["defects"]
    pitch = float(coupon["pitch_mm"])
    plate_h = float(coupon["plate_mm"][1])
    rows = [r for r in ROW_ORDER if r in dcfg]
    out: List[Defect] = []
    for zone in coupon["zones"]:
        xmin, xmax = zone_extent_x(cfg, zone)
        xc = 0.5 * (xmin + xmax)
        for r, dtype in enumerate(rows):
            sizes = size_list(dcfg, dtype)
            ncol = len(sizes)
            # grid centred on the zone: column c at xc + (c - (ncol-1)/2) * pitch
            # row r at +y for row 0 (top), descending
            for c, s in enumerate(sizes):
                x = xc + (c - (ncol - 1) / 2.0) * pitch
                y = ((len(rows) - 1) / 2.0 - r) * pitch
                margin = float(coupon.get("margin_mm", 5))
                if not (xmin + margin - 1e-9 <= x <= xmax - margin + 1e-9) or abs(y) > plate_h / 2 - margin + 1e-9:
                    raise ValueError(f"defect {dtype} size {s} at ({x},{y}) violates the {margin} mm margin")
                spec = dcfg[dtype]
                if dtype.startswith("pore"):
                    depth = float(spec["depth_ratio"]) * s
                    length, width = None, None
                elif dtype == "scratch":
                    depth = float(spec["depth_ratio"]) * s
                    length, width = float(spec["length_mm"]), s
                else:  # burr: size is height, "depth" is the height above the surface (positive)
                    depth = s
                    length, width = float(spec["length_mm"]), float(spec["width_ratio"]) * s
                out.append(
                    Defect(
                        prim=defect_name(zone, dtype, s),
                        zone=zone,
                        type=dtype,
                        size_mm=float(s),
                        depth_mm=float(depth),
                        length_mm=length,
                        width_mm=width,
                        center_mm=[x, y],
                        row=r,
                        col=c,
                    )
                )
    return out


if __name__ == "__main__":
    import json, os, sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ads.config import load_config

    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config/experiment.yaml")
    for d in build_layout(cfg):
        print(json.dumps(d.to_record()))
