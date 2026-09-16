"""Run the manager-based G1 dinnerware task and optionally record an HDF5 episode."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300, help="Control steps; 0 keeps a GUI scene open.")
    parser.add_argument("--record", action="store_true", help="Record one episode as HDF5.")
    parser.add_argument("--dataset-name", default="g1_dinnerware_demo", help="HDF5 filename without extension.")
    parser.add_argument("--demo-motion", action="store_true", help="Exercise waist and arm action channels.")
    parser.add_argument("--no-image", action="store_true", help="Do not save the final RGB image.")
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

# Isaac Lab and Omniverse imports must happen after the app starts.
import numpy as np
import torch
from PIL import Image

import isaacsim.core.utils.stage as stage_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import DatasetExportMode

from ycb_sim.controllers import StandingUpperBodyController
from ycb_sim.env_cfg import G1DinnerwareEnvCfg, SCENE_SPEC
from ycb_sim.recording import G1DatasetRecorderCfg


class G1TaskEnv(ManagerBasedRLEnv):
    """Avoid masking constructor errors when Isaac Lab cleans up a partial environment."""

    def __init__(self, cfg):
        self._construction_complete = False
        try:
            super().__init__(cfg)
        except BaseException:
            traceback.print_exc()
            raise
        self._construction_complete = True

    def __del__(self):
        if getattr(self, "_construction_complete", False):
            self.close()


def save_rgb(env: ManagerBasedRLEnv, path: Path) -> None:
    rgb = env.scene["camera"].data.output["rgb"][0].detach().cpu().numpy()[..., :3]
    Image.fromarray(rgb.astype(np.uint8), mode="RGB").save(path)


def main() -> int:
    outputs = PROJECT / "outputs"
    datasets = outputs / "datasets"
    outputs.mkdir(exist_ok=True)

    cfg = G1DinnerwareEnvCfg()
    cfg.sim.device = ARGS.device
    cfg.seed = 42
    requested_steps = ARGS.steps
    finite_steps = requested_steps if requested_steps > 0 else 1800
    episode_steps = finite_steps if ARGS.record else finite_steps + 1
    cfg.episode_length_s = episode_steps * cfg.decimation * cfg.sim.dt
    if ARGS.record:
        datasets.mkdir(parents=True, exist_ok=True)
        recorder_cfg = G1DatasetRecorderCfg()
        recorder_cfg.dataset_export_dir_path = str(datasets)
        recorder_cfg.dataset_filename = Path(ARGS.dataset_name).stem
        recorder_cfg.dataset_export_mode = DatasetExportMode.EXPORT_ALL
        cfg.recorders = recorder_cfg

    print("[managed] creating ManagerBasedRLEnv", flush=True)
    env = G1TaskEnv(cfg=cfg)
    try:
        origin = env.scene.env_origins[0]
        eye = torch.tensor([SCENE_SPEC["camera_eye"]], device=env.device) + origin
        target = torch.tensor([SCENE_SPEC["camera_target"]], device=env.device) + origin
        env.scene["camera"].set_world_poses_from_view(eye, target)
        print("[managed] environment created; resetting", flush=True)
        env.reset()
        controller = StandingUpperBodyController(env, demo_motion=ARGS.demo_motion)
        print("[managed] reset complete; starting control loop", flush=True)

        stage_utils.get_current_stage().Export(str(outputs / "managed_g1_dinnerware_scene.usd"))
        steps = 0
        terminated_count = 0
        timed_out_count = 0
        with torch.inference_mode():
            while APP.is_running() and (requested_steps == 0 or steps < requested_steps):
                action = controller.compute(steps)
                _, _, terminated, timed_out, _ = env.step(action)
                terminated_count += int(terminated.sum().item())
                timed_out_count += int(timed_out.sum().item())
                steps += 1
                if requested_steps == 0 and ARGS.headless and steps >= finite_steps:
                    break

        if not ARGS.no_image:
            save_rgb(env, outputs / "managed_g1_dinnerware_rgb.png")

        robot = env.scene["robot"]
        lower_names = [name for name in robot.joint_names if name not in controller.joint_names]
        lower_ids, _ = robot.find_joints(lower_names, preserve_order=True)
        lower_error = torch.max(
            torch.abs(robot.data.joint_pos[:, lower_ids] - robot.data.default_joint_pos[:, lower_ids])
        ).item()
        bowl = env.scene["bowl"]
        plate = env.scene["plate"]
        distance = torch.linalg.vector_norm(bowl.data.root_pos_w - plate.data.root_pos_w, dim=-1)[0].item()
        object_speeds = {
            "bowl": torch.linalg.vector_norm(bowl.data.root_vel_w[0]).item(),
            "plate": torch.linalg.vector_norm(plate.data.root_vel_w[0]).item(),
        }
        physics_valid = bool(
            torch.isfinite(bowl.data.root_state_w).all()
            and torch.isfinite(plate.data.root_state_w).all()
            and bowl.data.root_pos_w[0, 2] > 0.72
            and plate.data.root_pos_w[0, 2] > 0.72
        )
        lower_body_tolerance = 0.05
        report = {
            "passed": lower_error < lower_body_tolerance and physics_valid,
            "architecture": "Isaac Lab ManagerBasedRLEnv",
            "environment": cfg.env_name,
            "physics_hz": round(1.0 / cfg.sim.dt),
            "control_hz": round(1.0 / (cfg.sim.dt * cfg.decimation)),
            "camera_hz": 30,
            "steps": steps,
            "action_dimension": env.action_manager.total_action_dim,
            "controlled_joints": controller.joint_names,
            "held_lower_body_joints": lower_names,
            "max_lower_body_default_pose_error_rad": lower_error,
            "lower_body_hold_tolerance_rad": lower_body_tolerance,
            "bowl_to_plate_distance_m": distance,
            "object_speed_norms": object_speeds,
            "physics_state_finite_and_above_table": physics_valid,
            "terminated_count": terminated_count,
            "timed_out_count": timed_out_count,
            "recording_enabled": ARGS.record,
            "dataset": str(datasets / f"{Path(ARGS.dataset_name).stem}.hdf5") if ARGS.record else None,
        }
        (outputs / "managed-task-validation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    finally:
        APP.close()
