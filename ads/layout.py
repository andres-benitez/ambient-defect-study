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
        d["center_mm"] = [round(float(v), 6) for v in self.center_mm]
        return d

    @property
    def center3_mm(self) -> List[float]:
        """3-D centre (z = 0 for coupon defects, whose records carry only x, y)."""
        c = list(self.center_mm)
        return c + [0.0] * (3 - len(c)) if len(c) < 3 else c[:3]

    @classmethod
    def from_record(cls, r: dict) -> "Defect":
        return cls(prim=r["prim"], zone=r["zone"], type=r["type"], size_mm=float(r["size_mm"]), depth_mm=float(r["depth_mm"]),
                   length_mm=None if r.get("length_mm") is None else float(r["length_mm"]),
                   width_mm=None if r.get("width_mm") is None else float(r["width_mm"]),
                   center_mm=[float(v) for v in r["center_mm"]], row=int(r.get("row", 0)), col=int(r.get("col", 0)))


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


def asset_paths(cfg: dict) -> dict:
    """Absolute paths of the asset files named in cfg['asset'] (defaults to the coupon)."""
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    a = cfg.get("asset", {}) or {}
    d = os.path.join(root, a.get("dir", "assets/coupon"))
    return {"dir": d, "usd": os.path.join(d, a.get("usd", "coupon.usd")), "defects": os.path.join(d, a.get("defects", "defects.json")),
            "part_json": os.path.join(d, a.get("part_json", "part.json")),
            "zone_prims": dict(a.get("zone_prims", {"cast": "Coupon_Cast", "machined": "Coupon_Machined"})), "kind": a.get("kind", "coupon")}


def load_defects(cfg: dict) -> List[Defect]:
    """Defects of the configured asset from its defects.json; falls back to the coupon layout."""
    import json, os

    path = asset_paths(cfg)["defects"]
    if os.path.exists(path):
        with open(path) as fh:
            return [Defect.from_record(r) for r in json.load(fh)]
    return build_layout(cfg)


def load_part_info(cfg: dict) -> dict:
    """part.json of the asset (extent, zone prims, source); synthesised for a coupon without one."""
    import json, os

    paths = asset_paths(cfg)
    if os.path.exists(paths["part_json"]):
        with open(paths["part_json"]) as fh:
            return json.load(fh)
    w, h, t = (float(v) for v in cfg["coupon"]["plate_mm"])
    return {"kind": "coupon", "extent_mm": [w, h, t], "bbox_min_mm": [-w / 2, -h / 2, -t], "bbox_max_mm": [w / 2, h / 2, 0.0],
            "zone_prims": paths["zone_prims"], "reference_plane_z_mm": 0.0}


if __name__ == "__main__":
    import json, os, sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ads.config import load_config

    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else "config/experiment.yaml")
    for d in build_layout(cfg):
        print(json.dumps(d.to_record()))
