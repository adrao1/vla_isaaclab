"""Run a white-table scene with official Isaac Sim bowl and plate assets."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=600, help="Physics steps; 0 keeps a GUI scene open.")
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
from pxr import UsdGeom

import isaacsim.core.utils.prims as prim_utils
import isaacsim.core.utils.stage as stage_utils
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sensors.camera import Camera, CameraCfg
from isaaclab_assets import G1_CFG

from isaac_assets import prepare_official_assets


COLLIDERS = {
    "bowl": {"radius": 0.112, "height": 0.030},
    "plate": {"radius": 0.160, "height": 0.012},
}


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
        make_static_box(
            f"/World/Table/Leg_{index}",
            (leg_size, leg_size, leg_height),
            (
                x_sign * (sx / 2.0 - 0.08),
                y_sign * (sy / 2.0 - 0.08),
                leg_height / 2.0,
            ),
        )


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


def create_dinnerware(cfg: dict, paths: dict[str, Path]) -> dict[str, RigidObject]:
    """Reference official visuals and add simple, stable physics proxies."""
    stage = stage_utils.get_current_stage()
    objects = {}
    for spec in cfg["objects"]:
        name = spec["name"]
        root_path = f"/World/Dinnerware/{name}"
        initial_position = (spec["xy"][0], spec["xy"][1], cfg["table_height"] + 0.08)

        prim_utils.create_prim(root_path, "Xform", translation=initial_position)
        # ArchVis dinnerware is authored in centimeters; the simulation stage uses meters.
        prim_utils.create_prim(
            f"{root_path}/Visual",
            "Xform",
            usd_path=str(paths[name]),
            scale=(0.01, 0.01, 0.01),
        )

        collision = COLLIDERS[name]
        cylinder = UsdGeom.Cylinder.Define(stage, f"{root_path}/Collision")
        cylinder.CreateAxisAttr("Z")
        cylinder.CreateRadiusAttr(collision["radius"])
        cylinder.CreateHeightAttr(collision["height"])
        cylinder.AddTranslateOp().Set((0.0, 0.0, collision["height"] / 2.0))
        cylinder.MakeInvisible()

        sim_utils.define_rigid_body_properties(
            root_path,
            sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                disable_gravity=False,
                linear_damping=0.05,
                angular_damping=0.05,
                solver_position_iteration_count=8,
            ),
        )
        sim_utils.define_mass_properties(root_path, sim_utils.MassPropertiesCfg(mass=spec["mass"]))
        sim_utils.define_collision_properties(
            f"{root_path}/Collision",
            sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.002, rest_offset=0.0),
        )

        objects[name] = RigidObject(
            RigidObjectCfg(
                prim_path=root_path,
                spawn=None,
                init_state=RigidObjectCfg.InitialStateCfg(pos=initial_position),
            )
        )
    return objects


def create_robot(cfg: dict, asset_path: Path) -> Articulation:
    """Spawn a fixed-base G1 in its default standing pose."""
    spec = cfg["robot"]
    robot_cfg = G1_CFG.copy()
    robot_cfg.prim_path = "/World/Robot"
    robot_cfg.spawn.usd_path = str(asset_path)
    robot_cfg.spawn.articulation_props.fix_root_link = True
    robot_cfg.init_state.pos = tuple(spec["position"])
    robot_cfg.init_state.rot = tuple(spec["orientation_wxyz"])
    return Articulation(robot_cfg)


def robot_report(robot: Articulation) -> dict:
    movable_patterns = (
        "torso_joint",
        "shoulder",
        "elbow",
        "_zero_joint",
        "_one_joint",
        "_two_joint",
        "_three_joint",
        "_four_joint",
        "_five_joint",
        "_six_joint",
    )
    movable = [name for name in robot.joint_names if any(pattern in name for pattern in movable_patterns)]
    return {
        "model": "Unitree G1",
        "fixed_base": True,
        "root_position_m": robot.data.root_pos_w[0].detach().cpu().tolist(),
        "waist_yaw_joint": "torso_joint",
        "movable_upper_body_joints": movable,
        "all_joint_names": list(robot.joint_names),
    }


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
    cfg = json.loads((PROJECT / "configs/dinnerware_scene.json").read_text())
    asset_paths = prepare_official_assets(PROJECT, cfg["objects"] + [cfg["robot"]])

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
    robot = create_robot(cfg, asset_paths[cfg["robot"]["name"]])
    objects = create_dinnerware(cfg, asset_paths)
    camera = create_camera(cfg)

    outputs = PROJECT / "outputs"
    stage_utils.get_current_stage().Export(str(outputs / "dinnerware_g1_scene.usd"))
    sim.reset()
    camera.set_world_poses_from_view(
        torch.tensor([cfg["camera_eye"]], device=sim.device),
        torch.tensor([cfg["camera_target"]], device=sim.device),
    )

    requested_steps = ARGS.steps
    steps = 0
    while APP.is_running() and (requested_steps == 0 or steps < requested_steps):
        render_now = requested_steps == 0 or steps >= max(requested_steps - 5, 0)
        # Hold the default standing pose. The fixed root prevents falling while
        # the waist, arm and hand joints remain actuated for later commands.
        robot.set_joint_position_target(robot.data.default_joint_pos)
        robot.write_data_to_sim()
        sim.step(render=render_now)
        robot.update(sim.get_physics_dt())
        for obj in objects.values():
            obj.update(sim.get_physics_dt())
        if render_now:
            camera.update(sim.get_physics_dt())
        steps += 1
        if requested_steps == 0 and ARGS.headless and steps >= 600:
            break

    if not ARGS.no_image and "rgb" in camera.data.output:
        save_rgb(camera, outputs / "dinnerware_g1_scene_rgb.png")
    report = physics_report(objects, cfg, steps)
    report["robot"] = robot_report(robot)
    report["asset_sources"] = {
        "dinnerware": "NVIDIA Isaac Sim 4.5 ArchVis Residential Kitchen Dinnerware",
        "robot": "NVIDIA Isaac Lab v2.0.2 Unitree G1 (Robots/Unitree/G1/g1.usd)",
    }
    report["collision_note"] = "Hidden solid-cylinder proxies approximate the outer support shape."
    (outputs / "dinnerware-g1-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close()
