"""Reading renders (EXR or npy), writing EXR, and preview images."""
from __future__ import annotations

import os

import numpy as np


def write_exr(path: str, rgb: np.ndarray) -> None:
    import OpenEXR

    rgb = np.ascontiguousarray(rgb.astype(np.float32))
    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    channels = {"RGB": rgb}
    with OpenEXR.File(header, channels) as f:
        f.write(path)


def read_exr(path: str) -> np.ndarray:
    import OpenEXR

    with OpenEXR.File(path) as f:
        ch = f.channels()
        if "RGB" in ch:
            return np.asarray(ch["RGB"].pixels, dtype=np.float32)
        if "RGBA" in ch:
            return np.asarray(ch["RGBA"].pixels, dtype=np.float32)[..., :3]
        return np.stack([np.asarray(ch[c].pixels, dtype=np.float32) for c in ("R", "G", "B")], axis=-1)


def load_hdr(renders_dir: str, frame: int) -> np.ndarray:
    exr = os.path.join(renders_dir, f"hdr_{frame}.exr")
    npy = os.path.join(renders_dir, f"hdr_{frame}.npy")
    if os.path.exists(exr):
        return read_exr(exr)
    if os.path.exists(npy):
        return np.load(npy)[..., :3].astype(np.float32)
    raise FileNotFoundError(f"no hdr_{frame}.exr/.npy in {renders_dir}")


def convert_npy_renders_to_exr(renders_dir: str, remove_npy: bool = True) -> list:
    """hdr_<f>.npy -> hdr_<f>.exr (ZIP). Returns the list of EXR files written."""
    out = []
    for name in sorted(os.listdir(renders_dir)):
        if name.startswith("hdr_") and name.endswith(".npy"):
            arr = np.load(os.path.join(renders_dir, name))[..., :3]
            exr = os.path.join(renders_dir, name[:-4] + ".exr")
            write_exr(exr, arr)
            out.append(exr)
            if remove_npy:
                os.remove(os.path.join(renders_dir, name))
    return out


def tonemap_preview(rgb: np.ndarray, bg_level: float = 0.40, bg_mask=None) -> np.ndarray:
    """Simple display transform: scale so the background sits at mid-grey, then sRGB-ish gamma."""
    Y = rgb[..., :3].mean(axis=-1)
    ref = float(Y[bg_mask].mean()) if bg_mask is not None and bg_mask.any() else float(np.percentile(Y, 90))
    lin = np.clip(rgb[..., :3] * (bg_level / max(ref, 1e-9)), 0, 1)
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    return (np.clip(srgb, 0, 1) * 255).astype(np.uint8)


def write_png(path: str, img: np.ndarray) -> None:
    import imageio.v3 as iio

    iio.imwrite(path, img)
