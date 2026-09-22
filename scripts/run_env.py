#!/usr/bin/env python3
"""Run a registered VLA Isaac Lab environment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from isaaclab.app import AppLauncher


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="VLA-ScenePreview-YCB-G1-v0", help="Registered Gym environment ID.")
    parser.add_argument("--policy", choices=("auto", "standing", "sugar-box"), default="auto")
    parser.add_argument("--steps", type=int, default=300, help="Control steps; 0 keeps a GUI run open.")
    parser.add_argument("--preview-video", type=Path)
    parser.add_argument("--record-format", choices=("none", "hdf5", "lerobot"), default="none")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--dataset-name", default="vla_demo")
    parser.add_argument("--task-prompt")
    parser.add_argument("--list-tasks", action="store_true")
    parser.add_argument("--physics-only", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = not args.physics_only
    return args


ARGS = parse_args()
ARGS.experience = str(
    PROJECT_ROOT
    / "configs"
    / (
        "ycb.python.headless.kit"
        if ARGS.physics_only
        else ("ycb.python.headless.rendering.kit" if ARGS.headless else "ycb.python.rendering.kit")
    )
)
ARGS.kit_args = f"--portable-root {PROJECT_ROOT}/outputs/runtime/kit"
APP = AppLauncher(ARGS).app

import gymnasium as gym
import numpy as np
import torch

import vla_isaaclab  # noqa: F401  Register environments.
from isaaclab.managers import DatasetExportMode
from isaaclab_tasks.utils import parse_env_cfg

from vla_isaaclab.envs.common import LOWER_BODY_JOINT_NAMES, SUPPORT_HEIGHT
from vla_isaaclab.envs.common.managers import ACTION_TERM_NAME
from vla_isaaclab.policies import StandingPolicy, YCBSugarBoxScriptedPolicy


ENV_PREFIX = "VLA-"


def registered_tasks():
    return sorted(env_id for env_id in gym.registry if env_id.startswith(ENV_PREFIX))


def make_policy(env):
    selected = ARGS.policy
    if selected == "auto":
        selected = "sugar-box" if ARGS.task == "VLA-YCBSugarBox-G1-JointPos-v0" else "standing"
    return YCBSugarBoxScriptedPolicy(env) if selected == "sugar-box" else StandingPolicy(env)


def task_succeeded(env) -> bool:
    if "success" not in env.termination_manager.active_terms:
        return False
    return bool(env.termination_manager.get_term("success")[0].item())


def validate(env, policy, steps, success_count, dataset_path=None):
    robot = env.scene["robot"]
    lower_ids, _ = robot.find_joints(list(LOWER_BODY_JOINT_NAMES), preserve_order=True)
    lower_error = torch.max(
        torch.abs(robot.data.joint_pos[:, lower_ids] - robot.data.default_joint_pos[:, lower_ids])
    ).item()
    objects = {}
    objects_valid = True
    for name, obj in env.scene.rigid_objects.items():
        speed = torch.linalg.vector_norm(obj.data.root_vel_w[0]).item()
        finite = bool(torch.isfinite(obj.data.root_state_w[0]).all().item())
        above_support = obj.data.root_pos_w[0, 2].item() > SUPPORT_HEIGHT - 0.03
        stable = speed < 0.08
        objects[name] = {
            "position_m": obj.data.root_pos_w[0].detach().cpu().tolist(),
            "speed_norm": speed,
            "finite": finite,
            "above_support": above_support,
            "stable": stable,
        }
        objects_valid = objects_valid and finite and above_support and stable

    articulations = {}
    for name, articulation in env.scene.articulations.items():
        if name == "robot":
            continue
        finite = bool(torch.isfinite(articulation.data.joint_pos[0]).all().item())
        articulations[name] = {
            "joint_names": list(articulation.joint_names),
            "joint_positions_rad": articulation.data.joint_pos[0].detach().cpu().tolist(),
            "joint_velocities_rad_s": articulation.data.joint_vel[0].detach().cpu().tolist(),
            "finite": finite,
        }
        objects_valid = objects_valid and finite

    term = env.action_manager.get_term(ACTION_TERM_NAME)
    report = {
        "passed": lower_error < 0.05 and objects_valid,
        "task": ARGS.task,
        "steps": steps,
        "rates_hz": {"physics": 120, "control": 30, "camera": 30},
        "action_dimension": env.action_manager.total_action_dim,
        "controlled_joints": list(term._joint_names),
        "max_lower_body_default_pose_error_rad": lower_error,
        "rigid_objects": objects,
        "articulations": articulations,
        "success_count": success_count,
        "termination_terms": {
            name: bool(env.termination_manager.get_term(name)[0].item())
            for name in env.termination_manager.active_terms
        },
        "preview_video": str(ARGS.preview_video.resolve()) if ARGS.preview_video else None,
        "recording_format": ARGS.record_format,
        "dataset": str(dataset_path) if dataset_path else None,
    }
    if hasattr(policy, "diagnostics"):
        report["policy"] = policy.diagnostics()
    if ARGS.task == "VLA-YCBSugarBox-G1-JointPos-v0":
        report["passed"] = bool(
            report["passed"] and success_count > 0 and not report.get("policy", {}).get("failed", False)
        )
    return report


def main() -> int:
    if ARGS.list_tasks:
        tasks = registered_tasks()
        if not tasks:
            raise RuntimeError("No VLA Gym environments were registered")
        print(json.dumps(tasks, indent=2), flush=True)
        return 0
    if ARGS.task not in registered_tasks():
        raise ValueError(f"Unknown VLA environment {ARGS.task!r}. Available: {registered_tasks()}")
    if ARGS.preview_video and ARGS.physics_only:
        raise ValueError("--preview-video requires rendering; remove --physics-only")
    if ARGS.record_format != "none" and ARGS.physics_only:
        raise ValueError("recording requires cameras; remove --physics-only")
    if ARGS.episodes < 1:
        raise ValueError("--episodes must be at least 1")
    if ARGS.record_format != "lerobot" and ARGS.episodes != 1:
        raise ValueError("multiple episodes currently require --record-format lerobot")

    cfg = parse_env_cfg(ARGS.task, device=ARGS.device, num_envs=1)
    finite_steps = ARGS.steps if ARGS.steps > 0 else 1800
    episode_steps = finite_steps if ARGS.record_format == "hdf5" else finite_steps + 1
    cfg.episode_length_s = episode_steps * cfg.decimation * cfg.sim.dt
    if ARGS.physics_only:
        cfg.scene.camera = None
    dataset_path = None
    if ARGS.record_format == "hdf5":
        from vla_isaaclab.recording.recorder_cfg import VLARecorderCfg

        dataset_path = PROJECT_ROOT / "outputs/datasets" / f"{Path(ARGS.dataset_name).stem}.hdf5"
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        cfg.recorders = VLARecorderCfg()
        cfg.recorders.dataset_export_dir_path = str(dataset_path.parent)
        cfg.recorders.dataset_filename = dataset_path.stem
        cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL

    env = gym.make(ARGS.task, cfg=cfg).unwrapped
    video_container = None
    video_stream = None
    try:
        env.reset(seed=42)
        if ARGS.record_format == "hdf5":
            env.recorder_manager._dataset_file_handler.add_env_args(
                {
                    "environment_id": ARGS.task,
                    "task_instruction": cfg.task_instruction,
                    "rates_hz": {"physics": 120, "control": 30, "camera": 30},
                }
            )
        policy = make_policy(env)
        if ARGS.preview_video:
            import av

            path = ARGS.preview_video.resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            video_container = av.open(str(path), mode="w")
            video_stream = video_container.add_stream("libx264", rate=30)
            video_stream.width = 640
            video_stream.height = 480
            video_stream.pix_fmt = "yuv420p"

        steps = 0
        success_count = 0
        if ARGS.record_format == "lerobot":
            from vla_isaaclab.recording.frame import EnvironmentFrameAdapter
            from vla_isaaclab.recording.staging import StagingHDF5Writer

            adapter = EnvironmentFrameAdapter(env)
            task_prompt = ARGS.task_prompt or cfg.task_instruction
            writer = StagingHDF5Writer(
                root=PROJECT_ROOT / "outputs/lerobot_staging",
                dataset_name=Path(ARGS.dataset_name).stem,
                features=adapter.features,
                metadata={
                    "environment_id": ARGS.task,
                    "rates_hz": {"physics": 120, "control": 30, "camera": 30},
                    "joint_names": adapter.joint_names,
                    "environment_state_names": adapter.environment_state_names,
                    "task_prompt": task_prompt,
                    "units": {"joint_position": "rad", "joint_velocity": "rad/s", "position": "m"},
                    "isaac_sim_version": "4.5.0.0",
                    "isaac_lab_version": "v2.0.2",
                },
            )
            try:
                with torch.inference_mode():
                    for episode_index in range(ARGS.episodes):
                        if episode_index:
                            env.reset(seed=42 + episode_index)
                            policy = make_policy(env)
                        episode_succeeded = False
                        for episode_step in range(finite_steps):
                            snapshot = adapter.capture()
                            action = policy.compute(episode_step)
                            _, reward, terminated, timed_out, _ = env.step(action)
                            step_succeeded = task_succeeded(env) and not getattr(policy, "failed", False)
                            writer.add_frame(
                                adapter.complete_frame(
                                    snapshot,
                                    reward,
                                    terminated,
                                    timed_out,
                                    episode_step + 1 == finite_steps,
                                    task_prompt,
                                    42 + episode_index,
                                    step_succeeded,
                                )
                            )
                            success_count += int(step_succeeded)
                            episode_succeeded = episode_succeeded or step_succeeded
                            steps += 1
                            if getattr(policy, "failed", False) or bool(terminated[0]) or bool(timed_out[0]):
                                break
                        if ARGS.task == "VLA-YCBSugarBox-G1-JointPos-v0" and not episode_succeeded:
                            raise RuntimeError(
                                "Refusing to save a failed sugar-box demonstration: "
                                + json.dumps(policy.diagnostics())
                            )
                        writer.save_episode()
                staging_path = writer.finalize()
                conversion = subprocess.run(
                    [sys.executable, str(PROJECT_ROOT / "scripts/convert_staging_to_lerobot.py"), str(staging_path)],
                    check=False,
                )
                if conversion.returncode:
                    raise RuntimeError(f"LeRobot conversion failed with exit code {conversion.returncode}")
                dataset_path = PROJECT_ROOT / "outputs/lerobot" / Path(ARGS.dataset_name).stem
            except BaseException:
                writer.abort()
                raise
        else:
            with torch.inference_mode():
                while APP.is_running() and (ARGS.steps == 0 or steps < ARGS.steps):
                    action = policy.compute(steps)
                    _, _, terminated, timed_out, _ = env.step(action)
                    if video_stream is not None:
                        import av

                        rgb = env.scene["camera"].data.output["rgb"][0, ..., :3].detach().cpu().numpy()
                        frame = av.VideoFrame.from_ndarray(rgb.astype(np.uint8), format="rgb24")
                        for packet in video_stream.encode(frame):
                            video_container.mux(packet)
                    success_count += int(task_succeeded(env))
                    steps += 1
                    if getattr(policy, "failed", False):
                        break
                    if bool(terminated[0].item()) or bool(timed_out[0].item()):
                        break
                    if ARGS.steps == 0 and ARGS.headless and steps >= finite_steps:
                        break

        report = validate(env, policy, steps, success_count, dataset_path)
        output_dir = PROJECT_ROOT / "outputs/environments" / ARGS.task
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
        return 0 if report["passed"] else 2
    finally:
        if video_stream is not None:
            for packet in video_stream.encode():
                video_container.mux(packet)
        if video_container is not None:
            video_container.close()
        env.close()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:
        traceback.print_exc()
        raise
    finally:
        APP.close()
