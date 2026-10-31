# Ambient-light defect detectability study

Physics measurement of which defect types and sizes on an aluminium crankcase surface stay
detectable under constant, realistic ambient light with no inspection light rig. The pipeline
renders a flat test coupon (cast half + machined half, 48 defects as real geometry) with a path
tracer, pushes the linear image through a simple industrial-camera sensor model, and measures
the contrast-to-noise ratio (CNR) of every defect against its own local background. The output is
`out/RESULTS.md`: a CNR-vs-size chart per defect family, the smallest detectable size per defect
type and surface, and a contact sheet.

It is not a model-training exercise. CNR is the ceiling any detector can reach.

## Requirements

| Tool | Version used | Role |
|---|---|---|
| Blender | 5.2.2 LTS (`blender` on PATH, or `BLENDER=/path/to/blender`) | Stage 1: coupon geometry + USD export |
| Isaac Sim | 6.2.0, Docker image `nvcr.io/nvidia/isaac-sim:6.2.0` | Stages 2-3: scene, lighting, path tracing, Replicator masks |
| Docker + NVIDIA Container Toolkit | `docker run --gpus all` must work | runs Isaac Sim headless |
| Python | 3.10+ with `requirements.txt` | analysis, orchestration |
| GPU | one RTX (developed on an RTX 5080, 16 GB) | rendering |

Install:

```bash
# Blender: download the 5.2 LTS tarball from blender.org, unpack, put `blender` on PATH.
# Isaac Sim: docker login nvcr.io, then
docker pull nvcr.io/nvidia/isaac-sim:6.2.0
# Python deps (host)
pip install -r requirements.txt
python -m pytest tests/           # metric unit tests, no renders needed
```

The Isaac Sim container is started by `omniverse/isaacsim_docker.sh`; it mounts this repo at
`/workspace` and keeps shader/extension caches in `~/.cache/ambient-defect-study/isaacsim`
(override with `ISAACSIM_CACHE`, image with `ISAACSIM_IMAGE`). If you have a native Isaac Sim
install, replace the script body with `/path/to/isaac-sim/python.sh "$@"`.

## Run

```bash
python run_all.py --config config/experiment.yaml
```

This runs, stopping at the first failed check:

1. `blender/make_coupon.py` + `blender/checks.py` -> `assets/coupon/coupon.usd`, `defects.json`; `analysis/gen_textures.py` -> `cast_normal.png`
2. `omniverse/build_scene.py` -> `out/<condition>/scene.usd`, light intensity solved, scale / material-binding / denoiser checks
3. `omniverse/run_capture.py` -> `out/<condition>/renders/hdr_*.exr`, `mask_*.npy`, `zone_*.npy`, `id_map.json`, `frame_manifest.json`; render-noise check (spp is raised automatically until it passes)
4. `analysis/compute_cnr.py` -> `out/metrics.csv` (one row per defect per frame), `sim_*.png`, `mask_overlay.png`; exposure and mask checks
5. the positive control `control_darkfield` (steps 2-4 with a single 15 degree raking light) -> `out/control_darkfield/`
6. `analysis/plot_results.py` -> `out/summary.csv`, `out/figures/*.png`, `out/RESULTS.md`

Useful flags: `--condition <name>` runs one extra condition, `--all` runs the matrix,
`--skip-geometry` / `--skip-capture` re-run only the later stages, `--verify` re-renders the
default condition into `out/_verify/` and checks every `cnr` reproduces within
`checks.determinism_tol_frac`.

Each stage is also a standalone CLI, for example:

```bash
blender -b -P blender/make_coupon.py -- --config config/experiment.yaml --out assets/coupon/
omniverse/isaacsim_docker.sh omniverse/build_scene.py --config config/experiment.yaml --condition ambient_nadir_012
omniverse/isaacsim_docker.sh omniverse/run_capture.py --config config/experiment.yaml --condition ambient_nadir_012
python analysis/compute_cnr.py --config config/experiment.yaml --condition ambient_nadir_012
python analysis/plot_results.py --config config/experiment.yaml
```

## What comes out

```
assets/coupon/        coupon.usd, coupon.blend, defects.json, cast_normal.png (+ _stats.json)
out/<condition>/      scene.usd, scene_manifest.json, frame_manifest.json, analysis_checks.json,
                      renders/ (hdr_<f>.exr linear RGB, mask_<f>.npy instance ids, zone_<f>.npy semantic ids, id_map.json),
                      sim_<f>.png (simulated 8-bit camera image), preview_<f>.png, mask_overlay.png
out/metrics.csv       one row per defect per frame per condition (all columns in the plan + camera/lighting values)
out/summary.csv       smallest detectable size per (condition, zone, type)
out/summary_per_defect.csv   medians over frames per defect
out/figures/          cnr_vs_size.png, min_detectable.png, contact_sheet.png, mask_overlay.png
out/RESULTS.md        table, figures, five sentences of findings, exact config
```

## Configuration

Everything lives in `config/experiment.yaml`; no code edits are needed to change a parameter.
`conditions:` holds named overrides that are deep-merged onto the base config. To add a
condition, add an entry:

```yaml
conditions:
  ambient_nadir_005:
    camera: {mm_per_px: 0.05}
  ring_light_012:                 # example extension
    lighting: {mode: directional, directional: {elevation_deg: 60, azimuth_deg: 0}}
```

and run `python run_all.py --condition ring_light_012`. `default: true` marks the condition run
when none is given; `control: true` keeps a condition out of the figures (used for the
dark-field positive control). `lighting.mode: hdri` needs `lighting.hdri.path` set to a site HDRI.

## Implementation notes and deviations from the plan

- **Pore geometry.** The plan's "sphere of diameter d, centre at z = d/2 - depth" buries the sphere
  for depth_ratio 1.5. Pores are therefore built as the lower half of an ellipsoid centred on the
  surface: opening diameter = `size_mm`, depth = `depth_ratio x size_mm` (0.5 is exactly a
  hemisphere, as the plan says).
- **Burrs** are rooted `coupon.burr_root_mm` (10 um) into the plate so no face is coplanar with
  the top surface.
- **Manifold check.** Separated cavity shells are open by construction, so the check is that each
  zone assembly (plate + its cavities, vertices merged) is watertight, plus every burr on its own.
- **Blender units.** Blender's USD exporter ignores the scene unit scale, so the coupon is modelled
  at 1 unit = 1 mm and the geometry is scaled by 0.001 before export (`metersPerUnit = 1`).
- **Frames.** With a static scene the RTX path tracer keeps its accumulation buffer between
  Replicator steps, and it exposes no sampling seed, so frames 1..N-1 are bit-identical copies of
  frame 0 (they cost ~10 s each; `frame_manifest.json` records `frames_identical`). Frame-to-frame
  differences therefore cannot measure render noise. Instead the render-noise check measures the
  detrended per-pixel std in flat machined-zone patches of the linear render against the sensor noise
  at that level, and `run_capture.py` raises spp until the ratio is below `checks.render_noise_frac`
  (`render.auto_spp`, capped at `render.max_spp`). At the defaults this lands at ~5.4k spp
  (about 6 min for the capture of one condition on an RTX 5080; the whole default run is ~12 min). Medians over frames average the sensor-noise
  realisations (seeded per frame), not render noise.
- **HDR output.** The container writes `hdr_<f>.npy`; `compute_cnr.py` converts to `hdr_<f>.exr`
  (ZIP) on the host, where OpenEXR is available. `preview_<f>.png` is a host-side display transform
  of the HDR image rather than Replicator's LdrColor.
- **Light intensity.** The panel and dome are scaled together so the machined-zone mean linear
  radiance equals `lighting.solve_target` with sensor gain 1.0; `compute_cnr.py` re-solves the gain
  from frame 0 (it comes out at ~1.0) so the two stages stay independent.
- **Dome intensity** is `dome_fraction x panel intensity` on the USD `intensity` attribute of both lights.
- **Mask check.** Sub-pixel defects produce no instance id at all; the check requires every defect
  with a footprint >= 2 px to be in `id_map.json`, and every defect with area >= 9 px to have its mask
  centroid within 2 px of the projected `center_mm` (this validates the projection used for the
  empty-mask fallback).

## Layout

```
run_all.py                   orchestrates the stages
config/experiment.yaml       every parameter
ads/                         shared: config loader, coupon layout, camera model
blender/make_coupon.py       Stage 1 (Blender Python)
blender/checks.py            geometry checks
omniverse/isaacsim_docker.sh container launcher
omniverse/common.py          render settings, annotators, helpers (Isaac Sim Python)
omniverse/build_scene.py     Stage 2
omniverse/run_capture.py     Stage 3
analysis/gen_textures.py     cast-skin normal map
analysis/sensor_model.py     gain, shot/read noise, quantisation
analysis/metrics.py          per-defect CNR measurement
analysis/render_noise.py     residual render-noise estimate
analysis/io_utils.py         EXR / PNG helpers
analysis/compute_cnr.py      Stage 4a
analysis/plot_results.py     Stage 4b
tests/test_cnr.py            unit tests
```
