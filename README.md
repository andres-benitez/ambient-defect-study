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
| Git LFS | any recent `git-lfs` | holds `assets/crankcase/crankcase.usd` (136 MB); without it a clone gets a 134-byte pointer file |

Install:

```bash
# Clone (Git LFS must be installed first so the crankcase USD is downloaded)
git lfs install
git clone https://github.com/andres-benitez/ambient-defect-study.git
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

## Camera comparison, optics, exposure and rolling shutter

`config/experiment.yaml` carries a `cameras:` catalogue (Basler ace 2 Basic GigE candidates with their
EMVA 1288 figures and indicative prices). A condition that sets `camera.model` derives resolution,
sensor width, pixel scale and focal length from the catalogue and the `framing:` block (one shot of a
`fov_mm` wide face at `working_distance_mm`), and takes full well / read noise / QE from the sensor.
Run them with `python run_all.py --condition cam_a2A2448` (and the other `cam_*` conditions), then
`python analysis/camera_report.py` for `out/figures/camera_*.png`, `out/camera_summary.csv` and
`out/camera_table.json`.

- **Lens (`lens:`).** Applied in analysis, no re-render: a Gaussian point-spread function with
  sigma = sqrt((0.44 lambda N)^2 + blur_um^2) where the first term is diffraction at f-number N and
  `blur_um` is the lens's RMS spot (0 = diffraction-limited, 2 um = a lens matched to small pixels,
  5 um = a generic 5 MP-class lens). `compute_cnr.py --lens all` writes one row set per scenario
  (`lens` column); `--f-sweep` adds the "good" lens at every f-number in `lens.f_sweep`. Depth of
  field (1 px circle of confusion) is reported per f-number.
- **Exposure time (`photometry:`).** From hall illuminance, the ceiling's solid angle, the machined
  face's reflectance, the f-number, pixel area and QE: the time needed to put the machined face at
  the target level. Assumptions, not measurements.
- **Rolling shutter (`rolling_shutter:`).** For rolling-shutter candidates, horizontal banding under
  flickering light is simulated by scaling rows with the flicker averaged over each row's exposure
  window; `camera_report.py` writes the image panel, the CNR impact and the amplitude-vs-exposure curve.
  Static parts have no geometric rolling-shutter distortion.

## Real part (engine block scan)

`blender/make_part.py` puts the same 48 defects on a real part: `part.source` (a glTF/STL/OBJ mesh;
the default is the CC BY "Engine Cylinder Block" scan by 3dscantech from Sketchfab via Objaverse,
stored under `assets/crankcase/source/`, not committed). It stitches the mesh, finds the largest
planar region that can face the camera, rotates the part so that region is horizontal at z = 0,
rebuilds that region as a clean plane from a raster of its footprint (the machined zone; a scan
cannot represent a machined finish and its flattened triangles overlap), keeps everything else as
scanned cast geometry, chooses defect sites by ray casting (locally flat, camera-facing, clear of
edges and holes), and carves pores and scratches **without booleans**: the scan faces under each
footprint are deleted, the exact cavity surface is built as its own object, and a collar bridges its
rim to the hole (shared rim coordinates, no cracks). Booleans are unreliable on an open scan shell
(no inside volume), which is why the coupon keeps them and the part does not. Burrs are ridges. Output: `assets/crankcase/crankcase.usd`,
`defects.json` (3-D centres), `part.json` (extent, zone prims, source and licence). Conditions
`part_5mp` and `part_18mp` render the whole part in one shot (`framing.fov_mm: 620`) with the
client's 5 MP class and the best in-budget candidate. `asset:` in the config (or a condition) selects
coupon or part.

Scan-specific details worth knowing: the glTF importer rescales geometry into the scene's unit
system, so the generator imports before switching the scene to millimetres and checks the imported
extent against `part.expected_extent_mm`; the scan's custom per-vertex normals are discarded
(otherwise a flattened face still shades as bumpy) and the machined zone is shaded flat, the cast
zone smooth by angle; the machined zone is every face within `plane_band_mm` of the fitted plane
whose normal is within `fill_tol_deg` of vertical (a wider tolerance than the one used to find the
plane, so the notches a noisy scan leaves inside a flange are filled), then rasterised at
`part.replanarize.cell_mm`, morphologically closed, grown by one cell and lifted 20 um so the plane
overlaps the cast boundary instead of leaving gaps; the scale check compares the machined zone's
pixel width with `machined_extent_mm`, because the rest of the part lies below the reference plane
and is foreshortened; `check_part.py` replaces `checks.py`.

## Round 2: ten further experiments

All round-2 conditions are in `config/experiment.yaml` under `conditions:` and run with
`python run_all.py --condition <name> --skip-geometry` once the assets exist (`assets/coupon_ext` from
`blender/make_coupon.py --config ... --condition ext_nadir_012`, the part from `make_part.py`).
The analyses below read `out/metrics.csv` and write to `out/` and `out/figures/`; `analysis/build_report2.py`
assembles `out/report2/index.html`.

| Experiment | Conditions | Analysis |
|---|---|---|
| 1 Dents, 2 scratch envelope | `ext_nadir_012`, `ext_nadir_005` (120 x 100 mm plate, 108 defects incl. dents and 1/2/3 mm long scratches) | `analysis/ext_analysis.py` |
| 1b Dents under lighting rigs | `ext_ring60_012` (60 deg ring), `ext_ring30_012` (30 deg ring), `ext_dir_012` (single 15 deg light) on the same plate | `analysis/ext_analysis.py` (`dents_rigs.png`) |
| 3 One view per face: aperture, focus and exposure | `sv18_f56_mid` ... `sv18_f22_mid` (18 MP), `sv5_f8_mid`, `sv5_f16_mid` (5 MP); whole block, focus 60 mm below the flange; `part_5mp`, `part_18mp` give the focus-on-flange rows | `analysis/single_view.py` |
| 4 Sizing | every coupon condition | `analysis/sizing.py` |
| 5 Lighting rigs | `rig_ring60`, `rig_ring30`, `rig_dome`, `rig_dir_az0/90/180/270` | `analysis/rigs.py` |
| 6 Machining marks | `aniso_along`, `aniso_across` | `analysis/variants.py` |
| 7 False alarms | every condition | `analysis/false_alarms.py` |
| 8 Camera pose | `pose_tilt15`, `pose_near`, `pose_far` | `analysis/variants.py` |
| 9 Shadow | `shadow_012` | `analysis/variants.py` |
| 10 Learned detector | every coupon condition | `analysis/detector_check.py` |

Round-2 notes:

- **Depth of field in the renderer (fixed 2026-11-01).** The RTX path tracer sizes the lens aperture
  as focalLength / fStop in tenths of a scene unit; this stage is in metres, so the f-number as
  authored gave an aperture 100x too large (f/8 on a 25 mm lens was a 31 cm aperture). Flat-coupon
  renders with the focus on the plate were barely affected, but everything off the focus plane (the
  cast bays of the engine block, a tilted plate) was blurred by tens of pixels and the defocus
  sampling left a fixed-pattern floor no sample count removes. `build_scene.py` now scales the USD
  fStop by `0.1 / (metersPerUnit / 1000)` and records `fstop_usd` in the manifest; the calibration
  against a pinhole render is in `out/_diag/` (fStop 80 -> 3.0 px disc, as predicted for 100x). All
  whole-part and single-view conditions were re-rendered after the fix (see the round-2 report for
  which coupon conditions were re-rendered and why).
- **Two exposures for directional light.** Under a 30 deg lamp the mirror-like machined zone reflects
  the beam away from the camera, so the exposure that puts its background at 40 % of full scale
  saturates the diffuse casting (3-4 stops brighter). `compute_cnr.py` then measures cast defects on
  a second frame exposed for the cast zone and records `exposure_zone` per row (`exposure_zones`
  in `analysis_checks.json`). Ambient conditions never trigger this. The same logic fires on the whole engine block under the ambient ceiling light: the cast bays sit about 3.2 stops darker than the flange (they see only a cone of the ceiling and the casting is darker), so the cast sites of the part and single-view conditions are measured on a second exposure about 9x longer. `metrics.csv` carries the single-exposure result next to it (`cnr_1exp`, `detectable_1exp`; equal to `cnr`/`detectable` where one exposure serves both zones), and `single_view.py` reports both counts and the bay exposure time.
- **Render-noise margin.** `run_capture.py` aims at 0.9 x `checks.render_noise_frac` so the analysis
  re-check (same estimator, a slightly different background mask) cannot fail on the boundary;
  `--keep-failed` writes frames for diagnostics even when the check fails.
- **Maskless long scratches** are measured over their projected footprint (`footprint_region`)
  instead of a 3 x 3 window.
- **Deck face.** The scan's cylinder-deck face has a machined land under 6 mm wide between bores, too
  narrow for the defect layout, so only the oil-pan flange is used as a machined zone.

## Layout

```
run_all.py                   orchestrates the stages
config/experiment.yaml       every parameter
ads/                         shared: config loader, coupon layout, camera model
blender/make_coupon.py       Stage 1 (Blender Python)
blender/checks.py            geometry checks
blender/make_part.py         Stage 1 for a real part (scan or CAD mesh)
omniverse/isaacsim_docker.sh container launcher
omniverse/common.py          render settings, annotators, helpers (Isaac Sim Python)
omniverse/build_scene.py     Stage 2
omniverse/run_capture.py     Stage 3
analysis/gen_textures.py     cast-skin normal map
analysis/sensor_model.py     gain, shot/read noise, quantisation
analysis/metrics.py          per-defect CNR measurement
analysis/render_noise.py     residual render-noise estimate
analysis/io_utils.py         EXR / PNG helpers
analysis/optics.py           lens PSF (diffraction + spot), depth of field
analysis/photometry.py       exposure-time estimate
analysis/rolling_shutter.py  flicker banding model
analysis/camera_report.py    camera comparison figures and tables
analysis/compute_cnr.py      Stage 4a
analysis/plot_results.py     Stage 4b
analysis/ext_analysis.py     dents and scratch envelope (round 2)
analysis/single_view.py      one view per face: aperture/focus sweep
analysis/sizing.py           size measurement error vs pixels
analysis/rigs.py             lighting rigs vs ambient
analysis/variants.py         tool marks, camera pose, shadow
analysis/false_alarms.py     false-alarm density of the detection rule
analysis/detector_check.py   learned detector vs CNR
analysis/build_report.py     round-1 report (out/report)
analysis/build_report2.py    round-2 report (out/report2)
tests/test_cnr.py            unit tests
```
