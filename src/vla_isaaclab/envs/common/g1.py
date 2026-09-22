"""Unitree G1 asset configuration and joint semantics."""

from pathlib import Path

from isaaclab.assets import ArticulationCfg
from isaaclab_assets import G1_CFG


PROJECT_ROOT = Path(__file__).resolve().parents[4]
G1_USD = PROJECT_ROOT / "assets/Isaac/4.5/Isaac/IsaacLab/Robots/Unitree/G1/g1.usd"

ACTION_JOINT_NAMES = (
    "torso_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_elbow_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_elbow_roll_joint",
    "left_zero_joint",
    "left_one_joint",
    "left_two_joint",
    "left_three_joint",
    "left_four_joint",
    "left_five_joint",
    "left_six_joint",
    "right_zero_joint",
    "right_one_joint",
    "right_two_joint",
    "right_three_joint",
    "right_four_joint",
    "right_five_joint",
    "right_six_joint",
)

WAIST_JOINT_NAMES = ("torso_joint",)
LEFT_ARM_JOINT_NAMES = ACTION_JOINT_NAMES[1:6]
LEFT_HAND_JOINT_NAMES = ACTION_JOINT_NAMES[11:18]
LOWER_BODY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
)

LEFT_END_EFFECTOR = "left_palm_link"
RIGHT_END_EFFECTOR = "right_palm_link"
LEFT_HAND_OPEN_JOINT_POSITIONS = (0.0, 1.0, 0.18, 0.19, -0.12, 0.19, -0.12)
LEFT_HAND_CLOSED_JOINT_POSITIONS = (0.0, 1.02, 0.45, -0.45, -0.55, -0.32, -0.38)

LEFT_ARM_HOME_JOINT_POSITIONS = {
    "torso_joint": 0.0,
    "left_shoulder_pitch_joint": 0.255715,
    "left_shoulder_roll_joint": 0.431127,
    "left_shoulder_yaw_joint": -0.368285,
    "left_elbow_pitch_joint": -0.04442,
    "left_elbow_roll_joint": -0.333947,
}


def make_g1_cfg(
    position: tuple[float, float, float],
    orientation_wxyz: tuple[float, float, float, float] = (0.70710678, 0.0, 0.0, 0.70710678),
) -> ArticulationCfg:
    """Return the fixed-base G1 used by all environments in this project."""
    if not G1_USD.is_file():
        raise FileNotFoundError(f"Missing cached G1 asset: {G1_USD}")
    cfg = G1_CFG.copy()
    cfg.prim_path = "{ENV_REGEX_NS}/Robot"
    cfg.spawn.usd_path = str(G1_USD)
    cfg.spawn.articulation_props.fix_root_link = True
    cfg.spawn.articulation_props.solver_position_iteration_count = 16
    cfg.spawn.articulation_props.solver_velocity_iteration_count = 4
    cfg.init_state.pos = position
    cfg.init_state.rot = orientation_wxyz

    # The upstream config uses a regex shared by both elbow-pitch joints. Split
    # it before overriding only the left arm so that every joint matches once.
    default_elbow_pitch = cfg.init_state.joint_pos.pop(".*_elbow_pitch_joint")
    cfg.init_state.joint_pos["right_elbow_pitch_joint"] = default_elbow_pitch
    cfg.init_state.joint_pos.update(LEFT_ARM_HOME_JOINT_POSITIONS)
    cfg.init_state.joint_pos.update(dict(zip(LEFT_HAND_JOINT_NAMES, LEFT_HAND_OPEN_JOINT_POSITIONS)))
    return cfg
