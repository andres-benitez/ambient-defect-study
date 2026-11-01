"""Stage 2: scene, materials, ambient light, camera and render settings -> out/<condition>/scene.usd

  omniverse/isaacsim_docker.sh omniverse/build_scene.py --config config/experiment.yaml --condition ambient_nadir_012

Steps: reference coupon.usd, check its scale, add the floor, create OmniPBR (cast) and OmniSurface
(machined) materials and bind them to the plates and every defect prim, build the lights, place
the camera from the pixel scale, apply path-tracing settings (denoiser off), render a probe frame,
solve the light intensity so the machined background sits at the target level, and run the scale,
material-binding and denoiser checks. Writes scene.usd, probe.npy, probe_preview.png,
scene_manifest.json.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from ads.config import load_config  # noqa: E402
from ads.layout import load_defects, load_part_info, asset_paths  # noqa: E402
from ads.camera import build_camera  # noqa: E402
from omniverse import common  # noqa: E402



def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/experiment.yaml")
    ap.add_argument("--condition", default=None)
    ap.add_argument("--out", default=None, help="output dir (default out/<condition>)")
    ap.add_argument("--probe-spp", type=int, default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.condition)
    cond = cfg["condition"]["name"]
    out_dir = common.ensure_dir(os.path.abspath(args.out or os.path.join("out", cond)))
    W, H = (int(v) for v in cfg["camera"]["resolution"])
    cam = build_camera(cfg["camera"])
    layout = load_defects(cfg)
    ZONE_PLATE = asset_paths(cfg)["zone_prims"]
    part = load_part_info(cfg)
    t_start = time.time()

    app = common.start_app(cfg, W, H)
    import numpy as np  # noqa: E402
    import carb  # noqa: E402
    import omni.kit.commands  # noqa: E402
    import omni.usd  # noqa: E402
    import omni.replicator.core as rep  # noqa: E402
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade  # noqa: E402
    from isaacsim.core.utils.semantics import add_labels  # noqa: E402

    # create scene.usd on disk first so relative asset references resolve against its folder
    scene_path = os.path.join(out_dir, "scene.usd")
    if os.path.exists(scene_path):
        os.remove(scene_path)
    Usd.Stage.CreateNew(scene_path).Save()
    ctx = omni.usd.get_context()
    if not ctx.open_stage(scene_path):
        raise RuntimeError(f"could not open {scene_path}")
    stage = ctx.get_stage()
    stage.SetEditTarget(stage.GetRootLayer())
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    # ---- coupon reference (relative path so scene.usd works from the host too)
    coupon_abs = asset_paths(cfg)["usd"]
    coupon_rel = os.path.relpath(coupon_abs, out_dir)
    coupon_prim = stage.DefinePrim("/World/Coupon", "Xform")
    coupon_prim.GetReferences().AddReference(coupon_rel, "/Coupon")
    for _ in range(5):
        app.update()
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    plate_prims = [stage.GetPrimAtPath(f"/World/Coupon/{ZONE_PLATE[z]}") for z in cfg["coupon"]["zones"]]
    if not all(p.IsValid() for p in plate_prims):
        raise RuntimeError("coupon plates not found under /World/Coupon; is assets/coupon/coupon.usd present?")
    rng = Gf.Range3d()
    for p in plate_prims:
        rng.UnionWith(bbox_cache.ComputeWorldBound(p).ComputeAlignedRange())
    width_m = rng.GetMax()[0] - rng.GetMin()[0]
    plate_w_m = float(part["extent_mm"][0]) / 1000.0
    if abs(width_m - plate_w_m) > 1e-4:
        raise RuntimeError(f"asset width in stage is {width_m} m, want {plate_w_m} m")
    print(f"[build_scene] coupon bbox {rng.GetMin()} .. {rng.GetMax()} (width {width_m:.6f} m)")
    defect_prims = {d.prim: stage.GetPrimAtPath(f"/World/Coupon/{d.prim}") for d in layout}
    missing = [k for k, p in defect_prims.items() if not p.IsValid()]
    if missing:
        raise RuntimeError(f"defect prims missing in stage: {missing[:5]} ...")

    # ---- floor
    fl = cfg["materials"]["floor"]
    fw, fh = (float(v) for v in fl["size_m"])
    fz = -float(fl["drop_m"])
    floor = UsdGeom.Mesh.Define(stage, "/World/Floor")
    floor.CreatePointsAttr([Gf.Vec3f(-fw / 2, -fh / 2, fz), Gf.Vec3f(fw / 2, -fh / 2, fz), Gf.Vec3f(fw / 2, fh / 2, fz), Gf.Vec3f(-fw / 2, fh / 2, fz)])
    floor.CreateFaceVertexCountsAttr([4])
    floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    floor.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
    floor.CreateExtentAttr([Gf.Vec3f(-fw / 2, -fh / 2, fz), Gf.Vec3f(fw / 2, fh / 2, fz)])

    # ---- materials
    def mdl_material(path: str, mdl: str, name: str):
        ok, _ = omni.kit.commands.execute("CreateMdlMaterialPrim", mtl_url=mdl, mtl_name=name, mtl_path=path)
        if not ok:
            raise RuntimeError(f"could not create MDL material {name} at {path}")
        for _ in range(2):
            app.update()
        mat = UsdShade.Material(stage.GetPrimAtPath(path))
        shader = UsdShade.Shader(stage.GetPrimAtPath(path + "/Shader"))
        if not mat.GetPrim().IsValid() or not shader.GetPrim().IsValid():
            raise RuntimeError(f"MDL material {path} has no Shader child")
        return mat, shader

    authored_inputs = []

    def set_input(shader, name, value, vtype, colorspace=None):
        # Author directly with UsdShade on the root layer. (omni.usd.create_material_input left no
        # inputs in the saved layer in Isaac Sim 6.2, so both materials rendered with MDL defaults.)
        inp = shader.CreateInput(name, vtype)
        inp.Set(value)
        attr = inp.GetAttr()
        if colorspace:
            attr.SetColorSpace(colorspace)
        authored_inputs.append((shader.GetPath().pathString, name, value))
        return attr

    mc, mm = cfg["materials"]["cast"], cfg["materials"]["machined"]
    cast_mat, cast_sh = mdl_material("/World/Looks/Cast", "OmniPBR.mdl", "OmniPBR")
    set_input(cast_sh, "diffuse_color_constant", Gf.Vec3f(*[float(v) for v in mc["base_color"]]), Sdf.ValueTypeNames.Color3f)
    set_input(cast_sh, "metallic_constant", 1.0, Sdf.ValueTypeNames.Float)
    set_input(cast_sh, "reflection_roughness_constant", float(mc["roughness"]), Sdf.ValueTypeNames.Float)
    set_input(cast_sh, "specular_level", 0.5, Sdf.ValueTypeNames.Float)
    nm_abs = os.path.abspath(os.path.join(REPO_ROOT, mc["normal_map"]))
    if not os.path.exists(nm_abs):
        raise RuntimeError(f"normal map missing: {nm_abs} (run analysis/gen_textures.py)")
    nm_rel = os.path.relpath(nm_abs, out_dir)
    set_input(cast_sh, "normalmap_texture", Sdf.AssetPath(nm_rel), Sdf.ValueTypeNames.Asset, colorspace="raw")
    set_input(cast_sh, "bump_factor", float(mc["normal_strength"]), Sdf.ValueTypeNames.Float)
    set_input(cast_sh, "texture_scale", Gf.Vec2f(1.0, 1.0), Sdf.ValueTypeNames.Float2)

    mach_mat, mach_sh = mdl_material("/World/Looks/Machined", "OmniSurface.mdl", "OmniSurface")
    set_input(mach_sh, "diffuse_reflection_color", Gf.Vec3f(*[float(v) for v in mm["base_color"]]), Sdf.ValueTypeNames.Color3f)
    set_input(mach_sh, "diffuse_reflection_weight", 1.0, Sdf.ValueTypeNames.Float)
    set_input(mach_sh, "metalness", 1.0, Sdf.ValueTypeNames.Float)
    set_input(mach_sh, "specular_reflection_weight", 1.0, Sdf.ValueTypeNames.Float)
    set_input(mach_sh, "specular_reflection_roughness", float(mm["roughness"]), Sdf.ValueTypeNames.Float)
    set_input(mach_sh, "specular_reflection_anisotropy", float(mm.get("anisotropy", 0.0)), Sdf.ValueTypeNames.Float)
    if "anisotropy_rotation" in mm:
        set_input(mach_sh, "specular_reflection_anisotropy_rotation", float(mm["anisotropy_rotation"]), Sdf.ValueTypeNames.Float)

    floor_mat, floor_sh = mdl_material("/World/Looks/Floor", "OmniPBR.mdl", "OmniPBR")
    a = float(fl["albedo"])
    set_input(floor_sh, "diffuse_color_constant", Gf.Vec3f(a, a, a), Sdf.ValueTypeNames.Color3f)
    set_input(floor_sh, "reflection_roughness_constant", float(fl["roughness"]), Sdf.ValueTypeNames.Float)
    set_input(floor_sh, "metallic_constant", 0.0, Sdf.ValueTypeNames.Float)

    zone_mat = {"cast": cast_mat, "machined": mach_mat}

    def bind(prim, mat):
        api = UsdShade.MaterialBindingAPI.Apply(prim)
        api.UnbindDirectBinding()
        api.Bind(mat, UsdShade.Tokens.strongerThanDescendants)

    for z in cfg["coupon"]["zones"]:
        bind(stage.GetPrimAtPath(f"/World/Coupon/{ZONE_PLATE[z]}"), zone_mat[z])
    for d in layout:
        bind(defect_prims[d.prim], zone_mat[d.zone])
    bind(floor.GetPrim(), floor_mat)

    # material-binding check: every defect bound to its zone material, nothing else
    for d in layout:
        bound = UsdShade.MaterialBindingAPI(defect_prims[d.prim]).ComputeBoundMaterial()[0]
        want = zone_mat[d.zone].GetPath()
        if not bound or bound.GetPath() != want:
            raise RuntimeError(f"material check failed: {d.prim} bound to {bound.GetPath() if bound else None}, want {want}")
    for z in cfg["coupon"]["zones"]:
        bound = UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(f"/World/Coupon/{ZONE_PLATE[z]}")).ComputeBoundMaterial()[0]
        if bound.GetPath() != zone_mat[z].GetPath():
            raise RuntimeError(f"material check failed for plate {z}")
    print("[build_scene] material binding check passed (48 defects share their zone material)")

    # ---- semantics (needed for the probe masks; run_capture re-applies them)
    for d in layout:
        add_labels(defect_prims[d.prim], labels=["defect"], instance_name="class")
    for z in cfg["coupon"]["zones"]:
        add_labels(stage.GetPrimAtPath(f"/World/Coupon/{ZONE_PLATE[z]}"), labels=[f"zone_{z}"], instance_name="class")
    add_labels(floor.GetPrim(), labels=["floor"], instance_name="class")

    # ---- lights
    L = cfg["lighting"]
    mode = str(L["mode"])
    lights_scope = UsdGeom.Scope.Define(stage, "/World/Lights")
    light_prims = {}
    initial_intensity = 1000.0
    if mode == "panel":
        p = L["panel"]
        rect = UsdLux.RectLight.Define(stage, "/World/Lights/Panel")
        rect.CreateWidthAttr(float(p["size_m"][0]))
        rect.CreateHeightAttr(float(p["size_m"][1]))
        rect.CreateIntensityAttr(initial_intensity)
        rect.CreateEnableColorTemperatureAttr(True)
        rect.CreateColorTemperatureAttr(float(p["color_temp_k"]))
        rect.CreateNormalizeAttr(False)
        UsdGeom.Xformable(rect.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(0, 0, float(p["height_m"])))  # emits along -Z
        light_prims["panel"] = rect
        dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
        dome.CreateIntensityAttr(initial_intensity * float(L["dome_fraction"]))
        dome.CreateColorAttr(Gf.Vec3f(1, 1, 1))
        dome.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
        light_prims["dome"] = dome
    elif mode == "hdri":
        hp = L["hdri"]
        if not hp.get("path"):
            raise RuntimeError("lighting.mode=hdri needs lighting.hdri.path (a site HDRI); none configured")
        dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
        dome.CreateIntensityAttr(initial_intensity)
        dome.CreateTextureFileAttr(Sdf.AssetPath(os.path.relpath(os.path.abspath(hp["path"]), out_dir)))
        dome.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
        UsdGeom.Xformable(dome.GetPrim()).AddRotateZOp().Set(float(hp.get("rotation_deg", 0)))
        light_prims["dome"] = dome
    elif mode in ("directional", "ring"):
        def distant(path, el_deg, az_deg, intensity, angle):
            el, az = math.radians(el_deg), math.radians(az_deg)
            s = Gf.Vec3d(math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))  # towards the light
            x = Gf.Cross(Gf.Vec3d(0, 0, 1), s).GetNormalized()
            y = Gf.Cross(s, x).GetNormalized()
            m = Gf.Matrix4d(1.0)
            m.SetRow(0, Gf.Vec4d(x[0], x[1], x[2], 0))
            m.SetRow(1, Gf.Vec4d(y[0], y[1], y[2], 0))
            m.SetRow(2, Gf.Vec4d(s[0], s[1], s[2], 0))  # light's +Z points at the light; it emits along -Z
            dl = UsdLux.DistantLight.Define(stage, path)
            dl.CreateIntensityAttr(intensity)
            dl.CreateAngleAttr(angle)
            UsdGeom.Xformable(dl.GetPrim()).AddTransformOp().Set(m)
            return dl

        if mode == "directional":
            dp = L["directional"]
            light_prims["directional"] = distant("/World/Lights/Directional", float(dp["elevation_deg"]), float(dp["azimuth_deg"]),
                                                 initial_intensity, float(dp.get("angle_deg", 0.5)))
        else:  # ring: n distant lights around the camera axis at one elevation (a segmented ring light)
            rp_ = L["ring"]
            n = int(rp_.get("n_lights", 8))
            for k in range(n):
                light_prims[f"ring_{k}"] = distant(f"/World/Lights/Ring_{k}", float(rp_["elevation_deg"]), 360.0 * k / n,
                                                   initial_intensity / n, float(rp_.get("angle_deg", 2.0)))
    elif mode == "dome":  # uniform dome only: the softest possible light
        dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
        dome.CreateIntensityAttr(initial_intensity)
        dome.CreateColorAttr(Gf.Vec3f(1, 1, 1))
        dome.CreateTextureFormatAttr(UsdLux.Tokens.latlong)
        light_prims["dome"] = dome
    else:
        raise RuntimeError(f"unknown lighting.mode {mode!r}")
    occ = L.get("occluder")
    if occ:  # a dark plate between the ceiling and the part, casting a soft shadow across it
        ow, oh, ot = (float(v) for v in occ["size_m"])
        ox, oy, oz = (float(v) for v in occ["position_m"])
        occ_mesh = UsdGeom.Cube.Define(stage, "/World/Occluder")
        occ_mesh.CreateSizeAttr(1.0)
        xf_o = UsdGeom.Xformable(occ_mesh.GetPrim())
        xf_o.AddTranslateOp().Set(Gf.Vec3d(ox, oy, oz))
        xf_o.AddScaleOp().Set(Gf.Vec3f(ow, oh, ot))
        occ_mat, occ_sh = mdl_material("/World/Looks/Occluder", "OmniPBR.mdl", "OmniPBR")
        set_input(occ_sh, "diffuse_color_constant", Gf.Vec3f(0.2, 0.2, 0.2), Sdf.ValueTypeNames.Color3f)
        set_input(occ_sh, "reflection_roughness_constant", 0.9, Sdf.ValueTypeNames.Float)
        bind(occ_mesh.GetPrim(), occ_mat)

    # ---- camera
    usd_cam = UsdGeom.Camera.Define(stage, "/World/Camera")
    usd_cam.CreateFocalLengthAttr(cam.focal_length_mm)
    usd_cam.CreateHorizontalApertureAttr(cam.sensor_width_mm)
    usd_cam.CreateVerticalApertureAttr(cam.sensor_height_mm)
    # The RTX path tracer sizes the lens aperture as focalLength / fStop in TENTHS of a scene unit
    # (the USD camera convention: millimetres when the stage unit is a centimetre). This stage is in
    # metres, so an f-number authored as-is gave an aperture 100x too large (f/8 on a 25 mm lens became
    # a 31 cm aperture; calibrated 2026-11-01 against a pinhole render: fStop 80 -> 3 px disc as
    # predicted for 100x). The USD fStop is therefore the physical f-number scaled by 0.1 / (metersPerUnit / 1000).
    f_number = float(cfg["camera"].get("f_number", 8))
    fstop_scale = 0.1 / (UsdGeom.GetStageMetersPerUnit(stage) * 1e-3)
    fstop_usd = f_number * fstop_scale if f_number > 0 else 0.0
    usd_cam.CreateFStopAttr(fstop_usd)
    focus_offset = float(cfg["camera"].get("focus_offset_mm", 0.0))  # > 0: focus plane below the reference plane
    usd_cam.CreateFocusDistanceAttr((cam.distance_mm + focus_offset) / 1000.0)
    usd_cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    xf = UsdGeom.Xformable(usd_cam.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(*cam.position_m))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(*cam.rotation_xyz_deg))

    live_missing = [f"{path}.inputs:{name}" for path, name, _ in authored_inputs
                    if not stage.GetPrimAtPath(path).GetAttribute(f"inputs:{name}").HasAuthoredValue()]
    if live_missing:
        raise RuntimeError(f"material inputs not authored on the stage: {live_missing}")

    # ---- render settings + probe
    probe_spp = int(args.probe_spp or L.get("probe_spp", 64))
    settings = common.apply_render_settings(cfg, spp_override=probe_spp)
    common.check_denoiser_off()
    rp = rep.create.render_product("/World/Camera", (W, H), name="probe")
    ann = common.Annotators(rep, rp, with_ldr=True)
    for _ in range(3):
        app.update()

    from analysis.sensor_model import luminance  # noqa: E402

    def machined_bg_mean(frame):
        classes = common.semantic_class_ids(frame["sem_info"])
        zone = common.mask_from_ids(frame["sem"], classes.get("zone_machined", []))
        defects = common.mask_from_ids(frame["sem"], classes.get("defect", []))
        from scipy import ndimage

        bg = zone & ~ndimage.binary_dilation(defects, iterations=3)
        if bg.sum() < 1000:
            raise RuntimeError(f"machined background mask has only {bg.sum()} pixels; classes seen: {sorted(classes)}")
        Y = luminance(frame["hdr"])
        return float(Y[bg].mean()), zone, defects, classes

    sub = common.subframes_for(cfg, probe_spp)
    # the semantic labels can take a few frames to reach the annotators: retry the probe until the
    # machined zone is present instead of failing on a half-initialised frame
    frame = None
    for attempt in range(5):
        frame = ann.render(sub)
        seen = common.semantic_class_ids(frame["sem_info"])
        if "zone_machined" in seen:
            break
        print(f"[build_scene] probe attempt {attempt + 1}: classes {sorted(seen)}; waiting for semantics")
        for _ in range(10):
            app.update()
    mean0, zone_m, defects_m, classes = machined_bg_mean(frame)
    print(f"[build_scene] probe 1: machined bg mean {mean0:.5f} at intensity {initial_intensity} ({frame['wall_s']:.1f}s, {sub} subframes)")
    target = float(L.get("solve_target", cfg["sensor"]["target_bg_level"]))
    factor = target / mean0
    solved = {}
    for key, lp in light_prims.items():
        attr = lp.GetIntensityAttr()
        attr.Set(float(attr.Get()) * factor)
        solved[key] = float(attr.Get())
    for _ in range(2):
        app.update()
    frame = ann.render(sub)
    mean1, zone_m, defects_m, classes = machined_bg_mean(frame)
    print(f"[build_scene] probe 2: machined bg mean {mean1:.5f} after solving (intensities {solved})")
    if abs(mean1 - target) / target > 0.03:
        raise RuntimeError(f"intensity solve did not converge: {mean1} vs target {target}")

    # ---- scale check: width in pixels of geometry lying on the reference plane (z = 0). For the
    # coupon that is the whole plate; for a real part only the machined zone sits on the plane (the
    # rest is farther from the camera and foreshortened), so its extent is compared instead.
    if part.get("kind") == "part":
        zone_all = common.mask_from_ids(frame["sem"], classes.get("zone_machined", []))
        want_px = float(part["machined_extent_mm"][0]) / cam.mm_per_px
        tol = float(cfg["checks"].get("part_scale_tol_frac", 0.02))
    else:
        zone_all = common.mask_from_ids(frame["sem"], sum((classes.get(k, []) for k in ("zone_cast", "zone_machined", "defect")), []))
        want_px = float(part["extent_mm"][0]) / cam.mm_per_px
        tol = float(cfg["checks"]["scale_tol_frac"])
    cols = np.where(zone_all.any(axis=0))[0]
    rows = np.where(zone_all.any(axis=1))[0]
    width_px = int(cols.max() - cols.min() + 1)
    height_px = int(rows.max() - rows.min() + 1)
    if abs(width_px - want_px) / want_px > tol:
        raise RuntimeError(f"scale check failed: plate spans {width_px} px, want {want_px:.1f} +- {tol*100:.0f}%")
    print(f"[build_scene] scale check passed: plate {width_px} x {height_px} px (want {want_px:.0f} px wide)")

    # ---- save
    common.save_npy(os.path.join(out_dir, "probe_hdr.npy"), frame["hdr"])
    if "ldr" in frame:
        common.save_png(os.path.join(out_dir, "probe_preview.png"), frame["ldr"][..., :3])
    ann.detach()
    stage.GetRootLayer().Save()
    # material-input check: reopen the saved layer and confirm every configured shader input is there
    saved = Usd.Stage.Open(scene_path)
    missing_inputs = [f"{path}.inputs:{name}" for path, name, _ in authored_inputs
                      if not saved.GetPrimAtPath(path).GetAttribute(f"inputs:{name}").HasAuthoredValue()]
    if missing_inputs:
        raise RuntimeError(f"material inputs not saved to scene.usd: {missing_inputs}")
    print(f"[build_scene] material inputs check passed ({len(authored_inputs)} shader inputs authored and saved)")
    try:
        os.chmod(scene_path, 0o666)
    except OSError:
        pass
    manifest = {
        "condition": cond,
        "control": cfg["condition"]["control"],
        "scene_usd": os.path.relpath(scene_path, REPO_ROOT),
        "camera": {**cam.to_dict(), "f_number": f_number, "fstop_usd": fstop_usd, "fstop_scale": fstop_scale, "focus_offset_mm": focus_offset},
        "lighting_mode": mode,
        "solved_intensity": solved,
        "solve": {"probe_spp": probe_spp, "bg_mean_before": mean0, "bg_mean_after": mean1, "target": target, "factor": factor},
        "checks": {"scale_px": {"width_px": width_px, "height_px": height_px, "want_px": want_px, "pass": True},
                   "material_binding": True, "material_inputs": len(authored_inputs), "denoiser_off": True, "coupon_width_m": width_m},
        "render_settings_probe": settings,
        "semantic_classes": classes,
        "isaac_sim_version": common.isaac_version(),
        "wall_time_s": time.time() - t_start,
        "hdr_dtype": str(frame["hdr"].dtype), "hdr_shape": list(frame["hdr"].shape),
        "inst_dtype": str(frame["inst"].dtype), "n_instance_ids": len(frame["inst_info"].get("idToLabels", {})),
    }
    common.write_json(os.path.join(out_dir, "scene_manifest.json"), manifest)
    print(f"[build_scene] wrote {scene_path} and scene_manifest.json in {manifest['wall_time_s']:.0f}s")
    app.close()


if __name__ == "__main__":
    main()
