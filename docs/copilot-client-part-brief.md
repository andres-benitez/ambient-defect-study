# Brief for GitHub Copilot: single-view experiment on the client's crankcase CAD

You are working in a clone of `ambient-defect-study` on a GCP VM with an NVIDIA RTX PRO 6000, through
VS Code Remote-SSH (your terminal runs on the VM). Read `README.md` first (sections "Requirements",
"Run", "Real part", "Round 2"), then follow this brief. Work autonomously; ask the user only for the
items listed under "Ask the user for".

## What the repo does

Synthetic study of defect detectability under ambient hall lighting. Blender builds a test object with
known defects on a cast zone and a machined zone; Isaac Sim path-traces it under a physically based
light rig with a real camera model; a sensor model adds photon and read noise and 8-bit quantisation;
`analysis/compute_cnr.py` measures each defect's contrast-to-noise ratio (CNR). Detectable means
CNR >= 3 over >= 3 pixels. Everything is driven by `config/experiment.yaml`; a *condition* is a named
deep-merge override of that file, run with `python run_all.py --condition <name>`.

The defect set is 48 defects, 24 per zone: shallow and deep pores (0.05 to 2 mm), 5 mm scratches
(0.01 to 0.5 mm wide) and 5 mm burrs (0.05 to 1 mm high), six sizes each. They cover the client's
classes (pores >= 1 mm, scratches larger than 2 x 1 mm).

It has run on a flat coupon and on a public engine-block scan (`assets/crankcase`, kept in Git LFS).
The client has now provided their real crankcase as a STEP file. The question to answer:

> With one camera view per face under the hall's ambient light, which defect classes are detectable
> on the client's actual geometry at 5 MP and at 18 MP, at what pixel scale, and what exposure does
> each face need?

That is experiment 3 of round 2 ("one view per face: aperture, focus and exposure",
`analysis/single_view.py`), re-run on the client part.

## Baseline to compare against (public scan, round-2 report)

- 18 MP `a2A4508-6gmBAS`, 620 mm field: 0.138 mm/px on the machined flange; 30 to 31 of 48 defects
  detectable (14 of 24 cast with a second exposure, 13 with one).
- 5 MP `a2A2448-23gmBAS`, same field: 0.253 mm/px, so it misses the client's 0.2 mm/px clause for a
  620 mm face; 19 to 20 of 48 detectable.
- The cast bays sit about 3.2 stops darker than the flange under the ceiling light, so the analysis
  measures cast defects on a second exposure about 9x longer (`cnr_1exp` / `detectable_1exp` keep the
  one-exposure result; `exposure_zones` in `out/<condition>/analysis_checks.json`).
- Aperture f/5.6 to f/22 and the focus position changed nothing; light, not focus, limits the bays.

## Rules

1. **Client confidentiality.** The STEP file and everything derived from it (STL, .blend, .usd,
   `defects.json`, `part.json`, renders, `out/`) must never be committed, pushed or uploaded anywhere.
   Before creating any of them, add `assets/client_part*/` to `.gitignore` (`out/` is already
   ignored). Do not paste geometry or file contents into chat.
2. Do not modify or delete `assets/crankcase/` or the existing conditions. Add new ones.
3. Never run `run_all.py --all` (36 conditions, many hours). Run only the conditions in this brief,
   one at a time. Only ONE Isaac Sim container may run at a time (two break the semantic annotator).
   Never `docker kill` a render; let it finish or use `docker stop`.
4. Use absolute paths in commands. Never ask the user to paste tokens or keys into chat; ask them to
   run the login command themselves.
5. If a stage fails, read its log, fix the cause and re-run that stage. Report what failed and what
   you changed. Do not claim a step succeeded without showing its output.

## Ask the user for (only these)

- The path of the STEP file on the VM. They upload it (scp, or drag-drop into the VS Code explorer)
  to `assets/client_part/source/`.
- Which face(s) of the part to inspect, if the automatic choice (the largest planar face) is not the
  one they mean. Show the candidates (step 3) before building.
- NGC login (`docker login nvcr.io` needs their NGC API key) and GitHub auth for this private repo,
  if either is missing.

## Step 0: VM environment

Check: `nvidia-smi` shows the RTX PRO 6000 with a current production driver;
`docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi` works (NVIDIA Container
Toolkit); at least 60 GB free disk (Isaac Sim image about 20 GB, plus caches and renders);
Python 3.10 to 3.12.

```bash
# private repo: authenticate first (gh auth login); skip the 136 MB LFS asset, it is the public scan
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/andres-benitez/ambient-defect-study.git
cd ambient-defect-study
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt cadquery
python -m pytest tests/              # must pass; no GPU needed
# Blender 5.2 LTS: download the Linux tarball from blender.org, unpack, symlink `blender` into PATH.
# A headless VM usually needs these libraries for Blender:
sudo apt-get install -y libxi6 libxxf86vm1 libxfixes3 libxrender1 libgl1 libxkbcommon0 libsm6 libice6
blender -b --version
docker login nvcr.io                 # username $oauthtoken, password = NGC API key (typed by the user)
docker pull nvcr.io/nvidia/isaac-sim:6.2.0
```

`omniverse/isaacsim_docker.sh` starts the container (runs as uid 1234, repo mounted at `/workspace`,
caches in `~/.cache/ambient-defect-study/isaacsim`). The first start compiles shaders for 10 to 30
minutes; later starts take about a minute.

## Step 1: STEP to STL

Blender cannot import STEP. Convert with CadQuery (OpenCascade), tessellated finely enough that flat
faces stay flat at 0.2 mm/px:

```python
# tools/step_to_stl.py  (new file; contains no client data, fine to commit)
import sys
import cadquery as cq

wp = cq.importers.importStep(sys.argv[1])
vals = wp.vals()
shape = vals[0] if len(vals) == 1 else cq.Compound.makeCompound(vals)
bb = shape.BoundingBox()
print(f"extent mm: {bb.xlen:.1f} x {bb.ylen:.1f} x {bb.zlen:.1f}, solids: {len(vals)}")
shape.exportStl(sys.argv[2], tolerance=0.02, angularTolerance=0.1, ascii=False, relative=False)
```

Run: `python tools/step_to_stl.py /abs/path/client.stp /abs/path/ambient-defect-study/assets/client_part/source/client.stl`.
Check the printed extent: a crankcase is a few hundred mm per side. If it prints about 0.3 to 0.5, the
file is in metres; then set `part.source_units_per_mm: 0.001` in the conditions below (the generator
also checks the imported extent against `part.expected_extent_mm`, 100 to 2000 mm). A triangle count
in the low millions is fine (the public scan had 2.4 million). If the `exportStl` signature differs,
check `help(shape.exportStl)`. If CadQuery will not install, use `pip install gmsh`
(`gmsh.model.occ.importShapes`, `Mesh.MeshSizeMax` about 1.0, `Mesh.MeshSizeFromCurvature` about 60,
`gmsh.model.mesh.generate(2)`, `gmsh.write("client.stl")`), or `freecadcmd` if FreeCAD is present.

## Step 2: Configuration

Leave the top-level `part:` block (public scan) untouched and add conditions that override `asset`,
`part`, `camera` and `framing` for the client part. `make_part.py` orients the chosen face towards the
camera and treats it as the machined zone; `part.machined.prefer_axis` selects the face
(`+x -x +y -y +z -z`, in the CAD file's own frame). One asset directory per face. Template, written
here for a face called `top` on `+z`; repeat per face:

```yaml
  # ---- client part (CAD, confidential: assets/client_part* is git-ignored)
  client_top_18mp:
    asset: {kind: part, dir: assets/client_part_top, usd: crankcase.usd, zone_prims: {cast: Part_Cast, machined: Part_Machined}}
    part:
      source: assets/client_part/source/client.stl
      source_name: "client crankcase (STEP)"
      source_author: client
      source_url: null
      source_license: "client confidential, do not distribute"
      source_units_per_mm: 1.0
      machined: {prefer_axis: "+z"}
    camera: {model: a2A4508-6gmBAS, f_number: 8}
    framing: {fov_mm: FOV}
    render: {frames_per_condition: 1}
  client_top_5mp:        # same block, camera: {model: a2A2448-23gmBAS, f_number: 8}
  client_top_18mp_f16:   # same block as client_top_18mp, camera: {model: a2A4508-6gmBAS, f_number: 16}
```

`FOV` = the part's longest horizontal extent in that orientation x 1.08 margin, read from
`extent_mm` in `part.json` after step 3. Pixel scale = fov_mm / horizontal resolution (4508 px at
18 MP, 2448 px at 5 MP). The client's clause is <= 0.2 mm/px, so one 5 MP view covers at most about
490 mm and one 18 MP view about 900 mm. Run the 5 MP condition even if it misses the clause; the
report needs that number. Keep `defects:` as it is. The camera position used to choose defect sites is
(0, 0, `framing.working_distance_mm`) and does not depend on `fov_mm`, so the asset can be built
before `FOV` is known.

## Step 3: Build the part asset (per face)

```bash
mkdir -p assets/client_part_top out
blender -b -P blender/make_part.py -- --config config/experiment.yaml --condition client_top_18mp --out assets/client_part_top 2>&1 | tee out/make_part_top.log
blender -b -P blender/check_part.py -- --config config/experiment.yaml --assets assets/client_part_top --condition client_top_18mp
```

Before settling on a face, run `make_part.py` once with `prefer_axis` removed and read the
`[make_part]` log lines: it reports the planar region it picked. Show the user the candidate faces
(axis, planar area, extent) and your proposal, then build with `prefer_axis` set.

Verify in `assets/client_part_top/part.json`: `extent_mm` plausible; `machined_extent_mm` and
`machined_area_mm2` match the face the user means; `tris` reasonable. `defects.json` must have 48
records, 24 per zone, types pore_shallow, pore_deep, scratch, burr. If the generator stops with
"only N valid sites for zone ... relax part.sites", relax `part.sites.clearance_mm` or
`min_spacing_mm` inside the condition, or pick another face, and say so in the results.

Known behaviour: the pipeline was built for a noisy open scan shell. On a clean CAD mesh `flatten` and
`replanarize` are harmless (they rebuild the planar face as a 0.5 mm raster). Other faces co-planar
with the chosen one also become "machined", which is realistic for a machined deck. Defect sites are
chosen by rays from the camera, so only camera-facing surfaces receive defects, as in the real
inspection. Then set `FOV` in the three conditions from `extent_mm` and re-run `check_part.py`.

## Step 4: Render and analyse, one condition at a time

```bash
python run_all.py --config config/experiment.yaml --condition client_top_18mp     --skip-geometry --skip-control 2>&1 | tee out/run_client_top_18mp.log
python run_all.py --config config/experiment.yaml --condition client_top_5mp      --skip-geometry --skip-control 2>&1 | tee out/run_client_top_5mp.log
python run_all.py --config config/experiment.yaml --condition client_top_18mp_f16 --skip-geometry --skip-control 2>&1 | tee out/run_client_top_18mp_f16.log
```

`--skip-geometry` because the geometry stage builds the coupon, not the part; `--skip-control`
because the positive and negative controls are coupon conditions that were validated in rounds 1 and
2. Each run: `build_scene.py` (solves the light so the machined background sits at 40 % of full scale,
writes `out/<condition>/scene_manifest.json`), `run_capture.py` (path tracing with automatic sample
escalation until render noise is below the sensor noise; about 17 min for an 18 MP whole-part frame
on an RTX 5080, expect less here; writes `frame_manifest.json`, `id_map.json`), `compute_cnr.py`
(appends to `out/metrics.csv`; the whole-part cast zone will trigger the second exposure), then
`plot_results.py`. If `plot_results.py` fails at the very end because only part conditions exist in
`metrics.csv`, that is not fatal: the metrics are already written. Note it.

Failure modes: the exposure check fails (`analysis/compute_cnr.py --no-checks` reports instead of
failing, but understand the cause first); the render-noise check does not converge (inspect the
manifests; `omniverse/run_capture.py --keep-failed` writes the frames anyway); the container cannot
see the GPU (container toolkit). Out-of-memory is unlikely with 96 GB.

## Step 5: Results

```bash
python analysis/single_view.py --config config/experiment.yaml
```

Writes `out/single_view.json`, `out/figures/single_view_a2A4508.png`, `single_view_a2A2448.png`,
`single_view_exposure.png` and `pixel_scale_<condition>.png`. Then write `out/CLIENT_RESULTS.md`
(git-ignored; the user will move it) with, per face and camera:

- mm/px on the machined face and in the deepest cast region (`pixel_scale_*` figures; `mm_per_px`
  in `metrics.csv`), and whether the 0.2 mm/px clause is met
- detectable machined / cast counts of 24, with one exposure and with the second exposure
  (`detectable`, `detectable_1exp`), median CNR per zone, smallest detectable size per type and zone
- `cast_stops` (how much darker the cast zone is than the machined face) and the exposure times at
  1000 and 300 lux (`exposure_ms_*` keys in `single_view.json`)
- whether f/8 vs f/16 changed anything
- two or three sentences comparing with the public-scan baseline above

Copy the figures next to it and show the user the key table in chat.

## Step 6 (only after steps 1 to 5, and only if the user agrees): the lighting fix on the real part

Add `client_top_18mp_ring60`: the `client_top_18mp` block plus the `lighting:` and `checks:` lines of
the existing `ext_ring60_012` condition (an 8-light ring at 60 degrees elevation, exposure check not
enforced). Run it the same way and add to the results the median per-defect CNR ratio per zone
against `client_top_18mp`, computed from `out/metrics.csv` (`analysis/rigs.py` does this for the
coupon rigs; it is a few lines of pandas for the part). Round 2 found the 60 degree ring is what makes
cast faces reliable (3.4x contrast on the coupon).

## Done when

- [ ] `tests/` pass; Blender and the Isaac Sim container both run on the VM
- [ ] `assets/client_part*/` is git-ignored and `git status` shows no client-derived file
- [ ] `check_part.py` passes for every face built
- [ ] the three conditions per face have rows in `out/metrics.csv` and pass their checks, or the
      failure is explained
- [ ] `out/CLIENT_RESULTS.md` and its figures exist and the user has seen the table
- [ ] only `config/experiment.yaml`, `tools/step_to_stl.py` and `.gitignore` changed in git; commit
      them with a clear message, do not push client data
