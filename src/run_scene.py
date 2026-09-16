"""Build and validate a small YCB tabletop scene in Isaac Sim/Isaac Lab."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=600, help="Physics steps; 0 keeps a GUI scene open.")
    parser.add_argument("--robot-usd", type=Path, help="Optional robot USD to spawn at the robot interface pose.")
    parser.add_argument("--no-image", action="store_true", help="Skip saving the RGB observation.")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    return args


ARGS = parse_args()
EXPERIENCE = PROJECT / "configs" / (
    "ycb.python.headless.rendering.kit" if ARGS.headless else "ycb.python.rendering.kit"
)
ARGS.experience = str(EXPERIENCE)
ARGS.kit_args = f"--portable-root {PROJECT}/outputs/runtime/kit"
APP = AppLauncher(ARGS).app

# Isaac/Omniverse imports must happen after the application starts.
import numpy as np
import torch
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors.camera import Camera, CameraCfg

from ycb_assets import prepare_assets


def make_static_box(path: str, size: tuple[float, float, float], position: tuple[float, float, float]) -> None:
    box = sim_utils.CuboidCfg(
        size=size,
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.92, 0.92, 0.92), roughness=0.75, metallic=0.0
        ),
    )
    box.func(path, box, translation=position)


def create_table(cfg: dict) -> None:
    sx, sy, thickness = cfg["table_size"]
    top = cfg["table_height"]
    make_static_box("/World/Table/Top", (sx, sy, thickness), (0.0, 0.0, top - thickness / 2.0))
    leg_size = 0.055
    leg_height = top - thickness
    for index, (x_sign, y_sign) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        position = (
            x_sign * (sx / 2.0 - 0.08),
            y_sign * (sy / 2.0 - 0.08),
            leg_height / 2.0,
        )
        make_static_box(f"/World/Table/Leg_{index}", (leg_size, leg_size, leg_height), position)


def create_robot_interface(robot_usd: Path | None) -> dict:
    """Create a stable interface without choosing a final robot model."""
    pose = {"position": [0.0, -0.72, 0.0], "orientation_wxyz": [1.0, 0.0, 0.0, 0.0]}
    if robot_usd is not None:
        robot_usd = robot_usd.expanduser().resolve()
        if not robot_usd.is_file():
            raise FileNotFoundError(f"Robot USD does not exist: {robot_usd}")
        spawn = sim_utils.UsdFileCfg(usd_path=str(robot_usd))
        spawn.func("/World/Robot", spawn, translation=pose["position"], orientation=pose["orientation_wxyz"])
        pose["usd"] = str(robot_usd)
        pose["prim_path"] = "/World/Robot"
    else:
        prim = prim_utils.create_prim("/World/RobotSpawn", "Xform", translation=pose["position"])
        prim.CreateAttribute("ycb:purpose", Sdf.ValueTypeNames.String).Set(
            "Spawn the selected real robot here; the +Y direction faces the table."
        )
        pose["usd"] = None
        pose["prim_path"] = "/World/RobotSpawn"
    return pose


def create_objects(cfg: dict, asset_paths: dict[str, Path], inspection: dict) -> dict[str, RigidObject]:
    objects = {}
    upright = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)  # +90 degrees about X; asset Y becomes world Z.
    for spec in cfg["objects"]:
        name = spec["name"]
        bounds = inspection[name]
        half_height = (bounds["max"][1] - bounds["min"][1]) / 2.0
        initial_z = cfg["table_height"] + half_height + 0.08
        rigid_cfg = RigidObjectCfg(
            prim_path=f"/World/YCB/{name}",
            spawn=sim_utils.UsdFileCfg(usd_path=str(asset_paths[name]), semantic_tags=[("class", name)]),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(spec["xy"][0], spec["xy"][1], initial_z), rot=upright
            ),
        )
        objects[name] = RigidObject(rigid_cfg)
    return objects


def create_camera(cfg: dict) -> Camera:
    width, height = cfg["camera_resolution"]
    return Camera(
        CameraCfg(
            prim_path="/World/ObservationCamera",
            update_period=0.0,
            width=width,
            height=height,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=28.0,
                focus_distance=2.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 10.0),
            ),
        )
    )


def save_rgb(camera: Camera, path: Path) -> None:
    rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
    if rgb.shape[-1] == 4:
        rgb = rgb[..., :3]
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path)


def physics_report(objects: dict[str, RigidObject], cfg: dict, steps: int) -> dict:
    report = {"passed": True, "steps": steps, "objects": {}}
    for name, obj in objects.items():
        position = obj.data.root_pos_w[0].detach().cpu().tolist()
        velocity = obj.data.root_vel_w[0].detach().cpu().tolist()
        speed = float(torch.linalg.vector_norm(obj.data.root_vel_w[0]).item())
        mass = float(obj.data.default_mass[0, 0].item())
        inertia = obj.data.default_inertia[0].detach().cpu().tolist()
        inertia_positive = inertia[0] > 0.0 and inertia[4] > 0.0 and inertia[8] > 0.0
        finite = bool(torch.isfinite(obj.data.root_state_w[0]).all().item())
        above_table = position[2] > cfg["table_height"] - 0.01
        stable = speed < 0.08
        passed = finite and above_table and stable and mass > 0.0 and inertia_positive
        report["objects"][name] = {
            "position_m": position,
            "velocity_mps_and_radps": velocity,
            "speed_norm": speed,
            "mass_kg": mass,
            "runtime_inertia_tensor": inertia,
            "inertia_positive": inertia_positive,
            "finite": finite,
            "above_table": above_table,
            "stable": stable,
            "passed": passed,
        }
        report["passed"] = report["passed"] and passed
    return report


def main() -> int:
    cfg = json.loads((PROJECT / "configs" / "scene.json").read_text())
    inspection = json.loads((PROJECT / "outputs" / "asset-inspection.json").read_text())
    asset_paths = prepare_assets(PROJECT, cfg["objects"])

    sim_cfg = sim_utils.SimulationCfg(dt=cfg["physics_dt"], render_interval=2, device=ARGS.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim_utils.GroundPlaneCfg(color=(0.25, 0.25, 0.25)).func(
        "/World/Ground", sim_utils.GroundPlaneCfg(color=(0.25, 0.25, 0.25))
    )
    sim_utils.DomeLightCfg(intensity=1100.0, color=(0.85, 0.88, 1.0)).func(
        "/World/DomeLight", sim_utils.DomeLightCfg(intensity=1100.0, color=(0.85, 0.88, 1.0))
    )
    sim_utils.DistantLightCfg(intensity=2200.0, color=(1.0, 0.95, 0.88), angle=0.8).func(
        "/World/KeyLight",
        sim_utils.DistantLightCfg(intensity=2200.0, color=(1.0, 0.95, 0.88), angle=0.8),
        orientation=(0.9239, 0.3827, 0.0, 0.0),
    )
    create_table(cfg)
    robot_interface = create_robot_interface(ARGS.robot_usd)
    objects = create_objects(cfg, asset_paths, inspection)
    camera = create_camera(cfg)

    stage = stage_utils.get_current_stage()
    stage.Export(str(PROJECT / "outputs" / "ycb_pick_place_scene.usd"))
    sim.reset()
    camera.set_world_poses_from_view(
        torch.tensor([cfg["camera_eye"]], device=sim.device),
        torch.tensor([cfg["camera_target"]], device=sim.device),
    )

    requested_steps = ARGS.steps
    steps = 0
    while APP.is_running() and (requested_steps == 0 or steps < requested_steps):
        # Run most finite validation steps without rendering, then render the final frames for the camera.
        render_now = requested_steps == 0 or steps >= max(requested_steps - 5, 0)
        sim.step(render=render_now)
        for obj in objects.values():
            obj.update(sim.get_physics_dt())
        if render_now:
            camera.update(sim.get_physics_dt())
        steps += 1
        if requested_steps == 0 and ARGS.headless and steps >= 600:
            break

    outputs = PROJECT / "outputs"
    if not ARGS.no_image and "rgb" in camera.data.output:
        save_rgb(camera, outputs / "scene_rgb.png")
    report = physics_report(objects, cfg, steps)
    report["robot_interface"] = robot_interface
    report["camera"] = {
        "eye": cfg["camera_eye"],
        "target": cfg["camera_target"],
        "resolution": cfg["camera_resolution"],
    }
    (outputs / "physics-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close()
