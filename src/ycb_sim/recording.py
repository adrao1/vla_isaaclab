"""HDF5 recorder terms for camera observations and task metrics."""

from __future__ import annotations

import torch

from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass


class CameraCalibrationRecorder(RecorderTerm):
    def record_post_reset(self, env_ids):
        camera = self._env.scene["camera"]
        return "camera/calibration", {
            "intrinsic_matrix": camera.data.intrinsic_matrices[env_ids],
            "position_world": camera.data.pos_w[env_ids],
            "orientation_world": camera.data.quat_w_world[env_ids],
        }


class CameraFrameRecorder(RecorderTerm):
    def record_post_step(self):
        output = self._env.scene["camera"].data.output
        rgb = output["rgb"][..., :3]
        depth = torch.nan_to_num(output["distance_to_image_plane"], posinf=10.0, neginf=0.0)
        return "camera/frames", {"rgb": rgb, "depth": depth}


class TaskMetricRecorder(RecorderTerm):
    def record_post_step(self):
        bowl = self._env.scene["bowl"]
        plate = self._env.scene["plate"]
        distance = torch.linalg.vector_norm(bowl.data.root_pos_w - plate.data.root_pos_w, dim=-1, keepdim=True)
        return "task", {"bowl_to_plate_distance": distance}


@configclass
class CameraCalibrationRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = CameraCalibrationRecorder


@configclass
class CameraFrameRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = CameraFrameRecorder


@configclass
class TaskMetricRecorderCfg(RecorderTermCfg):
    class_type: type[RecorderTerm] = TaskMetricRecorder


@configclass
class G1DatasetRecorderCfg(ActionStateRecorderManagerCfg):
    camera_calibration = CameraCalibrationRecorderCfg()
    camera_frames = CameraFrameRecorderCfg()
    task_metrics = TaskMetricRecorderCfg()
