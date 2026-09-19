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

WAIST_JOINTS = ("torso_joint",)
LEFT_ARM_JOINTS = ACTION_JOINTS[1:6]
RIGHT_ARM_JOINTS = ACTION_JOINTS[6:11]
LEFT_HAND_JOINTS = ACTION_JOINTS[11:18]
RIGHT_HAND_JOINTS = ACTION_JOINTS[18:25]

# Start with the left hand retracted toward the robot and above table height.
# The open fingers begin behind the box instead of above its footprint.
# The joint geometry is checked offline; this is an articulation reset pose,
# not a scripted motion through the table edge.
LEFT_ARM_HOME_JOINT_POS = {
    "torso_joint": 0.0,
    "left_shoulder_pitch_joint": 0.255715,
    "left_shoulder_roll_joint": 0.431127,
    "left_shoulder_yaw_joint": -0.368285,
    "left_elbow_pitch_joint": -0.04442,
    "left_elbow_roll_joint": -0.333947,
}

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
    cfg.spawn.articulation_props.solver_position_iteration_count = 16
    cfg.spawn.articulation_props.solver_velocity_iteration_count = 4
    cfg.init_state.pos = world.robot_position
    cfg.init_state.rot = world.robot_orientation_wxyz
    # G1_CFG initializes both elbow-pitch joints with one regex.  Split it
    # before applying the left-arm override so every joint matches once.
    default_elbow_pitch = cfg.init_state.joint_pos.pop(".*_elbow_pitch_joint")
    cfg.init_state.joint_pos["right_elbow_pitch_joint"] = default_elbow_pitch
    cfg.init_state.joint_pos.update(LEFT_ARM_HOME_JOINT_POS)
    cfg.init_state.joint_pos.update(
        dict(zip(LEFT_HAND_JOINTS, UNITREE_G1.left_hand_open_joint_positions))
    )
    scene.robot = cfg


UNITREE_G1 = RobotDefinition(
    component_id="Robot-UnitreeG1-v0",
    capabilities=frozenset({"fixed_base", "upper_body_joint_control", "left_end_effector", "right_end_effector"}),
    action_joint_names=ACTION_JOINTS,
    lower_body_joint_names=LOWER_BODY_JOINTS,
    waist_joint_names=WAIST_JOINTS,
    left_arm_joint_names=LEFT_ARM_JOINTS,
    right_arm_joint_names=RIGHT_ARM_JOINTS,
    left_hand_joint_names=LEFT_HAND_JOINTS,
    right_hand_joint_names=RIGHT_HAND_JOINTS,
    left_fingertip_body_names=("left_two_link", "left_four_link", "left_six_link"),
    right_fingertip_body_names=("right_two_link", "right_four_link", "right_six_link"),
    left_end_effector="left_palm_link",
    right_end_effector="right_palm_link",
    left_hand_open_joint_positions=(0.0, 1.0, 0.18, 0.19, -0.12, 0.19, -0.12),
    left_hand_closed_joint_positions=(0.0, 1.02, 0.45, -0.45, -0.55, -0.32, -0.38),
    configure_scene=configure_g1,
)
