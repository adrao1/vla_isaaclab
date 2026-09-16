#!/usr/bin/env python3
"""Run a composed Isaac Lab scenario."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", default="World-Tabletop-v0")
    parser.add_argument("--robot", default="Robot-UnitreeG1-v0")
    parser.add_argument("--objects", default="Objects-Dinnerware-v0")
    parser.add_argument("--sensors", default="Sensors-FixedRGBD-v0")
    parser.add_argument("--task", default="Task-PickPlace-v0")
    parser.add_argument("--controller", default="Controller-RaiseLower-v0")
    parser.add_argument("--steps", type=int, default=300, help="Control steps; 0 keeps a GUI run open.")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--dataset-name", default="scenario_demo")
    parser.add_argument("--no-image", action="store_true")
    parser.add_argument("--list-components", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    return args


ARGS = parse_args()
ARGS.experience = str(
    PROJECT / "configs" / ("ycb.python.headless.rendering.kit" if ARGS.headless else "ycb.python.rendering.kit")
)
ARGS.kit_args = f"--portable-root {PROJECT}/outputs/runtime/kit"
APP = AppLauncher(ARGS).app

import numpy as np
import torch
from PIL import Image

import isaacsim.core.utils.stage as stage_utils
from isaaclab.managers import DatasetExportMode

from sim_platform import ScenarioSelection, list_components, register_defaults
from sim_platform.recording import ScenarioRecorderCfg
from sim_platform.runtime import ScenarioEnv
from sim_platform.scenario import compose_scenario


def safe_name(value: str) -> str:
    return value.replace("-v0", "").replace("-", "_").lower()


def save_rgb(env, path):
    rgb = env.scene["camera"].data.output["rgb"][0].detach().cpu().numpy()[..., :3]
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path)


def task_distance(env, bundle) -> float:
    if bundle.selection.task == "Task-Reach-v0":
        robot = env.scene["robot"]
        body_ids, _ = robot.find_bodies([bundle.robot.left_end_effector], preserve_order=True)
        palm = robot.data.body_pos_w[:, body_ids[0]] - env.scene.env_origins
        target = torch.tensor(bundle.world.reach_target, device=env.device).unsqueeze(0)
        return torch.linalg.vector_norm(palm - target, dim=-1)[0].item()
    obj = env.scene["object"]
    goal = env.scene["goal"]
    return torch.linalg.vector_norm(obj.data.root_pos_w - goal.data.root_pos_w, dim=-1)[0].item()


def main() -> int:
    register_defaults()
    if ARGS.list_components:
        print(json.dumps(list_components(), indent=2))
        return 0

    selection = ScenarioSelection(
        world=ARGS.world,
        robot=ARGS.robot,
        objects=ARGS.objects,
        sensors=ARGS.sensors,
        task=ARGS.task,
        controller=ARGS.controller,
    )
    bundle = compose_scenario(selection)
    cfg = bundle.env_cfg
    cfg.sim.device = ARGS.device
    cfg.seed = 42
    finite_steps = ARGS.steps if ARGS.steps > 0 else 1800
    cfg.episode_length_s = (finite_steps if ARGS.record else finite_steps + 1) * cfg.decimation * cfg.sim.dt

    dataset_path = PROJECT / "outputs/datasets" / f"{Path(ARGS.dataset_name).stem}.hdf5"
    if ARGS.record:
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        recorder = ScenarioRecorderCfg()
        recorder.dataset_export_dir_path = str(dataset_path.parent)
        recorder.dataset_filename = dataset_path.stem
        recorder.dataset_export_mode = DatasetExportMode.EXPORT_ALL
        cfg.recorders = recorder

    output_dir = PROJECT / "outputs/scenarios" / safe_name(selection.scenario_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[scenario] composing {selection.scenario_id}", flush=True)
    env = ScenarioEnv(cfg)
    try:
        origin = env.scene.env_origins[0]
        eye = torch.tensor([bundle.world.camera_eye], device=env.device) + origin
        target = torch.tensor([bundle.world.camera_target], device=env.device) + origin
        env.scene["camera"].set_world_poses_from_view(eye, target)
        env.reset()
        controller = bundle.controller.factory(env, bundle.robot, False)
        initial_obs_dim = int(env.obs_buf["policy"].shape[-1])

        if ARGS.record:
            handler = env.recorder_manager._dataset_file_handler
            handler.add_env_args(
                {
                    **cfg.scenario_metadata,
                    "physics_hz": round(1.0 / cfg.sim.dt),
                    "control_hz": round(1.0 / env.step_dt),
                    "camera_hz": 30,
                }
            )

        stage_utils.get_current_stage().Export(str(output_dir / "scene.usd"))
        steps = 0
        terminated_count = 0
        timed_out_count = 0
        with torch.inference_mode():
            while APP.is_running() and (ARGS.steps == 0 or steps < ARGS.steps):
                _, _, terminated, timed_out, _ = env.step(controller.compute(steps))
                terminated_count += int(terminated.sum().item())
                timed_out_count += int(timed_out.sum().item())
                steps += 1
                if ARGS.steps == 0 and ARGS.headless and steps >= finite_steps:
                    break

        if not ARGS.no_image:
            save_rgb(env, output_dir / "rgb.png")

        robot = env.scene["robot"]
        lower_ids, _ = robot.find_joints(list(bundle.robot.lower_body_joint_names), preserve_order=True)
        lower_error = torch.max(
            torch.abs(robot.data.joint_pos[:, lower_ids] - robot.data.default_joint_pos[:, lower_ids])
        ).item()
        rigid_objects = {}
        objects_valid = True
        for name, obj in env.scene.rigid_objects.items():
            speed = torch.linalg.vector_norm(obj.data.root_vel_w[0]).item()
            above_support = obj.data.root_pos_w[0, 2].item() > bundle.world.support_height - 0.03
            finite = bool(torch.isfinite(obj.data.root_state_w[0]).all().item())
            stable = speed < 0.08
            rigid_objects[name] = {
                "position_m": obj.data.root_pos_w[0].detach().cpu().tolist(),
                "speed_norm": speed,
                "finite": finite,
                "above_support": above_support,
                "stable": stable,
            }
            objects_valid = objects_valid and finite and above_support and stable

        report = {
            "passed": lower_error < 0.05 and objects_valid,
            "scenario": cfg.scenario_metadata,
            "rates_hz": {"physics": 120, "control": 30, "camera": 30},
            "steps": steps,
            "action_dimension": env.action_manager.total_action_dim,
            "observation_dimension": initial_obs_dim,
            "controlled_joints": controller.joint_names,
            "max_lower_body_default_pose_error_rad": lower_error,
            "task_distance_m": task_distance(env, bundle),
            "rigid_objects": rigid_objects,
            "terminated_count": terminated_count,
            "timed_out_count": timed_out_count,
            "recording_enabled": ARGS.record,
            "dataset": str(dataset_path) if ARGS.record else None,
        }
        (output_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        return 0 if report["passed"] else 2
    finally:
        env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close()
