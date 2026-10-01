"""X-Sim object-centric reward mechanics adapted to the G1 + left Dex3 hand.

Source-faithful (portal-cornell/X-Sim, mustard_place.py / panda_ninja.py):
  * reach  = ((1 - tanh(3 d)) + (1 - tanh(30 d))) / 2, d = |TCP - object origin|
  * + is_grasped (require_grasp)
  * waypoint = 1 - tanh(scale_k * dist_to_current) + 2 * current_idx,
    scale_k = 1 / |waypoint_k - waypoint_{k-1}| (start pose precedes waypoint 0),
    index advances when dist < goal_thresh, clamped at the last waypoint
  * static = (1 - tanh(5 |robot qvel|)) * placed * rotated
  * +1 on success; success = placed & rotated & robot static
  * grasp flag per finger: |F| >= 0.5 N and angle(F, opening dir) <= 85 deg

G1 adaptations (not X-Sim):
  * TCP = fixed point in left_hand_palm_link frame from the Dex3 closure probe
  * tripod grasp: thumb AND (index OR middle); opening directions from pad
    positions; finger force = sum of filtered forces over that finger's links
  * static joints = every robot joint except the left Dex3 hand joints

Orientation term (X-Sim rotation_reward option, off by default):
  * waypoint reward += 1 - tanh(scale * angle), scale = clamp(1/(angle between
    consecutive waypoints + 1e-6), max=10) = 10 here (all waypoints share the
    start orientation); advancing also requires angle < angle_goal_thresh
  * deviation: angle is the shortest rotation angle (uses |q1.q2|); X-Sim's
    quaternion_angle uses 2*acos(w_diff) without abs, which gives ~2*pi for q vs -q
"""

import math

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply

from ...common import LEFT_END_EFFECTOR, LEFT_HAND_JOINT_NAMES

# Midpoint between thumb pad and index/middle pads over 10-50% closure
# (rl/legacy/probe_dex_closure.py). Hand geometry only; no object information.
TCP_OFFSET_PALM = (0.089, -0.057, 0.0)

# Pad centers (distal-link geometry bbox centers) in each link's own frame.
PAD_LINKS = (
    "left_hand_thumb_2_link",
    "left_hand_index_1_link",
    "left_hand_middle_1_link",
)
PAD_LOCAL = (
    (0.0, -0.0225, 0.0),
    (0.0225, 0.0, 0.0),
    (0.0225, 0.0, 0.0),
)

FINGER_SENSORS = {
    "thumb": ("thumb_1_contact", "thumb_2_contact"),
    "index": ("index_0_contact", "index_1_contact"),
    "middle": ("middle_0_contact", "middle_1_contact"),
}

# Simplification (not demo-derived): object-center offsets from the start pose.
LIFT_WAYPOINT_OFFSETS = ((0.0, 0.0, 0.025), (0.0, 0.0, 0.05))


def _state(env: ManagerBasedRLEnv) -> dict:
    st = getattr(env, "_xsim_state", None)
    if st is None:
        robot = env.scene["robot"]
        palm_ids, _ = robot.find_bodies([LEFT_END_EFFECTOR], preserve_order=True)
        pad_ids, found = robot.find_bodies(list(PAD_LINKS), preserve_order=True)
        if list(found) != list(PAD_LINKS):
            raise RuntimeError(f"Unexpected pad bodies: {found}")
        hand_ids, _ = robot.find_joints(list(LEFT_HAND_JOINT_NAMES), preserve_order=True)
        mask = torch.ones(robot.num_joints, dtype=torch.bool, device=env.device)
        mask[torch.as_tensor(hand_ids, device=env.device)] = False
        st = {
            "idx": torch.zeros(env.num_envs, dtype=torch.long, device=env.device),
            "palm_id": palm_ids[0],
            "pad_ids": list(pad_ids),
            "static_joint_mask": mask,
            "tcp_offset": torch.tensor(TCP_OFFSET_PALM, device=env.device),
            "pad_local": torch.tensor(PAD_LOCAL, device=env.device),
            "metrics": {},
        }
        env._xsim_state = st
    return st


def reset_xsim_state(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Reset event: waypoint index back to 0 for reset environments."""
    st = _state(env)
    idx = st["idx"].clone()
    idx[env_ids] = 0
    st["idx"] = idx


def tcp_pos_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    st = _state(env)
    robot = env.scene["robot"]
    pos = robot.data.body_pos_w[:, st["palm_id"]]
    quat = robot.data.body_quat_w[:, st["palm_id"]]
    return pos + quat_apply(quat, st["tcp_offset"].expand(env.num_envs, 3))


def _pad_pos_w(env: ManagerBasedRLEnv) -> torch.Tensor:
    st = _state(env)
    robot = env.scene["robot"]
    n = env.num_envs
    pos = robot.data.body_pos_w[:, st["pad_ids"]]
    quat = robot.data.body_quat_w[:, st["pad_ids"]]
    local = st["pad_local"].unsqueeze(0).expand(n, 3, 3)
    return pos + quat_apply(quat.reshape(-1, 4), local.reshape(-1, 3)).reshape(n, 3, 3)


def _finger_force_w(env: ManagerBasedRLEnv, sensor_names) -> torch.Tensor:
    total = None
    for name in sensor_names:
        fm = env.scene[name].data.force_matrix_w
        if fm is None:
            raise RuntimeError(f"Contact sensor {name!r} has no filtered force data")
        f = fm[:, 0, 0, :]
        total = f if total is None else total + f
    return total


def xsim_grasp_flags(env: ManagerBasedRLEnv, min_force: float = 0.5,
                     max_angle_deg: float = 85.0) -> dict:
    pads = _pad_pos_w(env)
    thumb, index, middle = pads[:, 0], pads[:, 1], pads[:, 2]
    open_dir = {
        "thumb": thumb - 0.5 * (index + middle),
        "index": index - thumb,
        "middle": middle - thumb,
    }
    cos_max = math.cos(math.radians(max_angle_deg))
    out = {}
    for finger, sensors in FINGER_SENSORS.items():
        force = _finger_force_w(env, sensors)
        mag = torch.linalg.vector_norm(force, dim=-1)
        d = open_dir[finger]
        cos = (force * d).sum(-1) / (mag * torch.linalg.vector_norm(d, dim=-1) + 1e-9)
        out[f"{finger}_force"] = mag
        out[f"{finger}_cos"] = cos
        out[f"{finger}_ok"] = (mag >= min_force) & (cos >= cos_max)
    out["is_grasped"] = out["thumb_ok"] & (out["index_ok"] | out["middle_ok"])
    return out


def _waypoints(env: ManagerBasedRLEnv):
    obj = env.scene["object"]
    start = obj.data.default_root_state[:, :3] + env.scene.env_origins
    offs = torch.tensor(LIFT_WAYPOINT_OFFSETS, device=env.device)
    wps = start.unsqueeze(1) + offs.unsqueeze(0)
    prev = torch.cat((torch.zeros(1, 3, device=env.device), offs[:-1]), dim=0)
    scales = 1.0 / torch.linalg.vector_norm(offs - prev, dim=-1)
    return start, wps, scales


def _quat_angle(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    dot = (q1 * q2).sum(-1).abs().clamp(max=1.0)
    return 2.0 * torch.acos(dot)


def xsim_dense_reward(
    env: ManagerBasedRLEnv,
    goal_thresh: float = 0.01,
    success_radius: float = 0.015,
    success_angle: float = 0.3,
    static_thresh: float = 0.1,
    min_force: float = 0.5,
    max_angle_deg: float = 85.0,
    require_grasp: bool = True,
    rotation_reward: bool = False,
    angle_goal_thresh: float = 0.3,
) -> torch.Tensor:
    """Per-step X-Sim dense reward (unnormalized, as X-Sim trains with reward_mode='dense')."""
    st = _state(env)
    obj = env.scene["object"]
    robot = env.scene["robot"]
    n = env.num_envs
    ar = torch.arange(n, device=env.device)
    obj_pos = obj.data.root_pos_w
    obj_quat = obj.data.root_quat_w

    # evaluate() equivalent, from the current state
    start, wps, scales = _waypoints(env)
    start_quat = obj.data.default_root_state[:, 3:7]
    grasp = xsim_grasp_flags(env, min_force, max_angle_deg)
    placed = torch.linalg.vector_norm(wps[:, -1] - obj_pos, dim=-1) <= success_radius
    rotated = _quat_angle(start_quat, obj_quat) <= success_angle
    qvel = robot.data.joint_vel[:, st["static_joint_mask"]]
    is_static = qvel.abs().amax(dim=-1) <= static_thresh
    success = placed & rotated & is_static

    # 1. reaching (+ grasp)
    d = torch.linalg.vector_norm(obj_pos - tcp_pos_w(env), dim=-1)
    reach = ((1 - torch.tanh(3 * d)) + (1 - torch.tanh(30 * d))) / 2
    reach_total = reach + grasp["is_grasped"].float() if require_grasp else reach

    # 2. waypoint tracking + progress, then index update (X-Sim order)
    idx = st["idx"]
    dist_cur = torch.linalg.vector_norm(wps[ar, idx] - obj_pos, dim=-1)
    wp_track = 1 - torch.tanh(scales[idx] * dist_cur)
    wp_angle = _quat_angle(start_quat, obj_quat)
    angle_scale = min(1.0 / (0.0 + 1e-6), 10.0)
    angle_reward = 1 - torch.tanh(angle_scale * wp_angle)
    reached = dist_cur < goal_thresh
    if rotation_reward:
        wp_track = wp_track + angle_reward
        reached = reached & (wp_angle < angle_goal_thresh)
    else:
        angle_reward = torch.zeros_like(angle_reward)
    wp_total = wp_track + 2 * idx.float()
    st["idx"] = torch.clamp(idx + reached.long(), 0, wps.shape[1] - 1)

    # 3. static (only when placed & rotated)
    static_r = (1 - torch.tanh(5 * torch.linalg.vector_norm(qvel, dim=-1))) * placed * rotated

    total = reach_total + wp_total + static_r + success.float()

    st["metrics"] = {
        "tcp_dist": d, "reach": reach, "is_grasped": grasp["is_grasped"],
        "thumb_ok": grasp["thumb_ok"], "index_ok": grasp["index_ok"], "middle_ok": grasp["middle_ok"],
        "thumb_force": grasp["thumb_force"], "index_force": grasp["index_force"],
        "middle_force": grasp["middle_force"],
        "wp_idx_before": idx, "wp_dist": dist_cur, "wp_track": wp_track,
        "lift": obj_pos[:, 2] - start[:, 2], "placed": placed, "rotated": rotated,
        "is_static": is_static, "static_reward": static_r, "success": success, "total": total,
        "wp_angle": wp_angle, "angle_reward": angle_reward, "toppled": wp_angle > 0.7854,
    }
    return total


# ---------------- observations (X-Sim state obs, G1 adaptation noted) ----------------

def obs_tcp_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    st = _state(env)
    quat = env.scene["robot"].data.body_quat_w[:, st["palm_id"]]
    return torch.cat((tcp_pos_w(env) - env.scene.env_origins, quat), dim=-1)


def obs_object_pose(env: ManagerBasedRLEnv) -> torch.Tensor:
    obj = env.scene["object"]
    return torch.cat((obj.data.root_pos_w - env.scene.env_origins, obj.data.root_quat_w), dim=-1)


def obs_desired_goal(env: ManagerBasedRLEnv) -> torch.Tensor:
    st = _state(env)
    _, wps, _ = _waypoints(env)
    ar = torch.arange(env.num_envs, device=env.device)
    goal = wps[ar, st["idx"]] - env.scene.env_origins
    start_quat = env.scene["object"].data.default_root_state[:, 3:7]
    return torch.cat((goal, start_quat), dim=-1)


def obs_goal_position_diff(env: ManagerBasedRLEnv) -> torch.Tensor:
    st = _state(env)
    _, wps, _ = _waypoints(env)
    ar = torch.arange(env.num_envs, device=env.device)
    return wps[ar, st["idx"]] - env.scene["object"].data.root_pos_w


def obs_goal_rotation_diff(env: ManagerBasedRLEnv) -> torch.Tensor:
    obj = env.scene["object"]
    return _quat_angle(obj.data.default_root_state[:, 3:7], obj.data.root_quat_w).unsqueeze(-1)


def obs_is_grasped(env: ManagerBasedRLEnv, min_force: float = 0.5,
                   max_angle_deg: float = 85.0) -> torch.Tensor:
    return xsim_grasp_flags(env, min_force, max_angle_deg)["is_grasped"].float().unsqueeze(-1)
