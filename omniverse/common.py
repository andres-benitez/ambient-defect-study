"""Shared Omniverse helpers for build_scene.py and run_capture.py (run inside Isaac Sim's Python)."""
from __future__ import annotations

import json
import math
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PT_SETTINGS = {
    "rendermode": "/rtx/rendermode",
    "spp": "/rtx/pathtracing/spp",
    "total_spp": "/rtx/pathtracing/totalSpp",
    "clamp_spp": "/rtx/pathtracing/clampSpp",
    "denoiser": "/rtx/pathtracing/optixDenoiser/enabled",
    "max_bounces": "/rtx/pathtracing/maxBounces",
    "max_specular_bounces": "/rtx/pathtracing/maxSpecularAndTransmissionBounces",
    "texture_streaming": "/rtx-transient/resourcemanager/enableTextureStreaming",
    "mdl_texture_streaming": "/rtx/hydra/mdlMaterialStreaming",
}


def start_app(cfg: dict, width: int, height: int):
    from isaacsim import SimulationApp

    r = cfg["render"]
    app = SimulationApp(
        {
            "headless": True,
            "renderer": "PathTracing",
            "width": int(width),
            "height": int(height),
            "denoiser": bool(r.get("denoiser", False)),
            "samples_per_pixel_per_frame": int(r.get("spp_per_subframe", 32)),
            "max_bounces": int(r.get("max_bounces", 8)),
            "max_specular_transmission_bounces": int(r.get("max_specular_bounces", 8)),
            "anti_aliasing": 0,
        }
    )
    return app


def apply_render_settings(cfg: dict, spp_override: int | None = None) -> dict:
    """Path tracing, converged spp via accumulation, denoiser off, bounces from config."""
    import carb

    r = cfg["render"]
    s = carb.settings.get_settings()
    total = int(spp_override or r["spp"])
    per = int(r.get("spp_per_subframe", 32))
    per = min(per, total)
    s.set(PT_SETTINGS["rendermode"], "PathTracing")
    s.set(PT_SETTINGS["spp"], per)
    s.set(PT_SETTINGS["total_spp"], total)
    s.set(PT_SETTINGS["clamp_spp"], total)
    s.set(PT_SETTINGS["denoiser"], bool(r.get("denoiser", False)))
    s.set(PT_SETTINGS["max_bounces"], int(r.get("max_bounces", 8)))
    s.set(PT_SETTINGS["max_specular_bounces"], int(r.get("max_specular_bounces", 8)))
    s.set(PT_SETTINGS["texture_streaming"], False)
    s.set("/rtx/post/aa/op", 0)
    s.set("/rtx/post/tonemap/op", 0)  # irrelevant for HdrColor but keeps LdrColor previews simple
    s.set("/rtx/sceneDb/ambientLightIntensity", 0.0)  # nothing else emits light
    s.set("/rtx/indirectDiffuse/enabled", True)
    return read_render_settings()


def read_render_settings() -> dict:
    import carb

    s = carb.settings.get_settings()
    return {k: s.get(v) for k, v in PT_SETTINGS.items()}


def subframes_for(cfg: dict, spp_override: int | None = None) -> int:
    total = int(spp_override or cfg["render"]["spp"])
    per = min(int(cfg["render"].get("spp_per_subframe", 32)), total)
    return max(1, int(math.ceil(total / per)))


def check_denoiser_off() -> None:
    import carb

    v = carb.settings.get_settings().get(PT_SETTINGS["denoiser"])
    if v not in (False, 0):
        raise RuntimeError(f"denoiser check failed: {PT_SETTINGS['denoiser']} reads back {v!r}")


def ensure_dir(path: str) -> str:
    """mkdir -p with mode 777 on every level created (the container runs as uid 1234, the host
    user owns the repo; both must be able to write)."""
    parts = []
    p = os.path.abspath(path)
    while not os.path.isdir(p):
        parts.append(p)
        p = os.path.dirname(p)
    for d in reversed(parts):
        os.mkdir(d)
        try:
            os.chmod(d, 0o777)
        except OSError:
            pass
    return path


def write_json(path: str, obj) -> None:
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=str)
    try:
        os.chmod(path, 0o666)
    except OSError:
        pass


def save_npy(path: str, arr) -> None:
    import numpy as np

    np.save(path, arr)
    try:
        os.chmod(path, 0o666)
    except OSError:
        pass


def save_png(path: str, arr) -> bool:
    try:
        from PIL import Image

        Image.fromarray(arr).save(path)
        os.chmod(path, 0o666)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[common] PNG write skipped ({exc})")
        return False


def isaac_version() -> str:
    try:
        with open("/isaac-sim/VERSION") as fh:
            return fh.read().strip()
    except OSError:
        return "unknown"


class Annotators:
    """HdrColor + instance/semantic segmentation (+ LdrColor) attached to one render product."""

    def __init__(self, rep, render_product, with_ldr: bool = True):
        self.rep = rep
        self.rp = render_product
        self.hdr = rep.AnnotatorRegistry.get_annotator("HdrColor")
        self.inst = rep.AnnotatorRegistry.get_annotator("instance_segmentation", init_params={"colorize": False})
        self.sem = rep.AnnotatorRegistry.get_annotator("semantic_segmentation", init_params={"colorize": False})
        self.ldr = rep.AnnotatorRegistry.get_annotator("LdrColor") if with_ldr else None
        self.depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        for a in (self.hdr, self.inst, self.sem, self.ldr, self.depth):
            if a is not None:
                a.attach(render_product)

    def render(self, rt_subframes: int) -> dict:
        import numpy as np

        t0 = time.time()
        self.rep.orchestrator.step(rt_subframes=int(rt_subframes), pause_timeline=False)
        self.rep.orchestrator.wait_until_complete()
        hdr = np.asarray(self.hdr.get_data(), dtype=np.float32)
        inst = self.inst.get_data()
        sem = self.sem.get_data()
        out = {
            "hdr": hdr,
            "inst": np.asarray(inst["data"]),
            "inst_info": {k: _jsonable(v) for k, v in inst["info"].items()},
            "sem": np.asarray(sem["data"]),
            "sem_info": {k: _jsonable(v) for k, v in sem["info"].items()},
            "wall_s": time.time() - t0,
        }
        if self.ldr is not None:
            out["ldr"] = np.asarray(self.ldr.get_data())
        try:
            out["depth"] = np.asarray(self.depth.get_data(), dtype=np.float32)
        except Exception:  # noqa: BLE001
            out["depth"] = None
        return out

    def detach(self):
        for a in (self.hdr, self.inst, self.sem, self.ldr, self.depth):
            if a is not None:
                try:
                    a.detach(self.rp)
                except Exception:  # noqa: BLE001
                    pass


def _jsonable(v):
    try:
        json.dumps(v)
        return v
    except TypeError:
        if isinstance(v, dict):
            return {str(k): _jsonable(x) for k, x in v.items()}
        if hasattr(v, "tolist"):
            return v.tolist()
        return str(v)


def semantic_class_ids(sem_info: dict) -> dict:
    """{'defect': [ids...], 'zone_cast': [...], ...} from the semantic annotator's idToLabels."""
    out: dict = {}
    for k, v in sem_info.get("idToLabels", {}).items():
        label = v.get("class") if isinstance(v, dict) else str(v)
        out.setdefault(label, []).append(int(k))
    return out


def instance_prim_ids(inst_info: dict) -> dict:
    """{prim_name: [ids...]} from the instance annotator's idToLabels (values are prim paths)."""
    out: dict = {}
    for k, v in inst_info.get("idToLabels", {}).items():
        path = str(v)
        name = path.rstrip("/").split("/")[-1]
        out.setdefault(name, []).append(int(k))
    return out


def mask_from_ids(arr, ids):
    import numpy as np

    m = np.zeros(arr.shape[:2], bool)
    for i in ids:
        m |= arr == i
    return m
