"""Unitree G1 adapter."""

from pathlib import Path

from isaaclab_assets import G1_CFG

from ..contracts import RobotDefinition


PROJECT = Path(__file__).resolve().parents[3]
G1_USD = PROJECT / "assets/Isaac/4.5/Isaac/IsaacLab/Robots/Unitree/G1/g1.usd"

ACTION_JOINTS = (
    "torso_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "left_elbow_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_elbow_roll_joint",
    "left_zero_joint", "left_one_joint", "left_two_joint", "left_three_joint",
    "left_four_joint", "left_five_joint", "left_six_joint",
    "right_zero_joint", "right_one_joint", "right_two_joint", "right_three_joint",
    "right_four_joint", "right_five_joint", "right_six_joint",
)

LOWER_BODY_JOINTS = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "left_hip_roll_joint", "right_hip_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "left_knee_joint", "right_knee_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint", "left_ankle_roll_joint", "right_ankle_roll_joint",
)


def configure_g1(scene, world) -> None:
    if not G1_USD.is_file():
        raise FileNotFoundError(f"Missing cached G1 asset: {G1_USD}")
    cfg = G1_CFG.copy()
    cfg.prim_path = "{ENV_REGEX_NS}/Robot"
    cfg.spawn.usd_path = str(G1_USD)
    cfg.spawn.articulation_props.fix_root_link = True
    cfg.init_state.pos = world.robot_position
    cfg.init_state.rot = world.robot_orientation_wxyz
    scene.robot = cfg


UNITREE_G1 = RobotDefinition(
    component_id="Robot-UnitreeG1-v0",
    capabilities=frozenset({"fixed_base", "upper_body_joint_control", "left_end_effector", "right_end_effector"}),
    action_joint_names=ACTION_JOINTS,
    lower_body_joint_names=LOWER_BODY_JOINTS,
    left_end_effector="left_palm_link",
    right_end_effector="right_palm_link",
    configure_scene=configure_g1,
)
