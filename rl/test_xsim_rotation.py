#!/usr/bin/env python3
"""Reward-code test for the XSimLiftRot orientation term.

Places the box directly (no physics steps; robot never moves or touches it),
tilted by chosen angles, and checks the orientation reward and waypoint
advance against an independent recomputation. Diagnostic only.
"""
import argparse
import math
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = False
args.experience = str(ROOT / 'configs/ycb.python.headless.kit')
app = AppLauncher(args).app

TASK = 'VLA-YCBSugarBox-G1-XSimLiftRot-v0'
# (label, tilt_deg about world x, dz, flip quaternion sign, expected idx after, reset first)
CASES = [
    ('upright, rest',        0.0, 0.00, False, 0, True),
    ('tilt 2 deg',           2.0, 0.00, False, 0, False),
    ('tilt 5 deg',           5.0, 0.00, False, 0, False),
    ('tilt 10 deg',         10.0, 0.00, False, 0, False),
    ('tilt 90 deg (on side)', 90.0, 0.00, False, 0, False),
    ('upright, -q sign',     0.0, 0.00, True,  0, False),
    ('up 2 cm, upright',     0.0, 0.02, False, 1, True),
    ('up 2 cm, tilt 10 deg', 10.0, 0.02, False, 1, True),
    ('up 2 cm, tilt 30 deg', 30.0, 0.02, False, 0, True),
]
FAILS = []


def main():
    import torch
    import gymnasium as gym
    import vla_isaaclab  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.utils.math import quat_mul
    from vla_isaaclab.envs.ycb_sugar_box.mdp import xsim as X
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
    cfg.seed = 0
    for name in ('camera', 'cam_side', 'cam_left_high', 'cam_left_wrist', 'cam_right_wrist'):
        if hasattr(cfg.scene, name):
            setattr(cfg.scene, name, None)
    env = gym.make(TASK, cfg=cfg).unwrapped
    try:
        obj = env.scene['object']
        params = dict(env.cfg.rewards.xsim.params)
        print(f"params: rotation_reward={params['rotation_reward']} angle_goal_thresh={params['angle_goal_thresh']}")
        if not params['rotation_reward']:
            FAILS.append('rotation_reward not enabled')
        offs = torch.tensor(X.LIFT_WAYPOINT_OFFSETS, device=env.device)
        print('  label                  | idx b->a (exp) | angle deg | angle_r  exp_r  | reward   expected | result')
        for label, tilt, dz, flip, exp_idx, do_reset in CASES:
            if do_reset:
                env.reset(seed=0)
            start = obj.data.default_root_state[:, :3] + env.scene.env_origins
            q0 = obj.data.default_root_state[:, 3:7]
            h = math.radians(tilt) / 2
            q_tilt = torch.tensor([[math.cos(h), math.sin(h), 0.0, 0.0]], device=env.device)
            q = quat_mul(q_tilt, q0)
            if flip:
                q = -q
            target = start + torch.tensor([[0.0, 0.0, dz]], device=env.device)
            obj.write_root_pose_to_sim(torch.cat((target, q), dim=-1))
            obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
            obj.update(env.physics_dt)
            idx_before = int(env._xsim_state['idx'][0])
            p = obj.data.root_pos_w
            # independent recomputation
            ang = 2 * torch.acos((q0 * obj.data.root_quat_w).sum(-1).abs().clamp(max=1.0))
            exp_ar = 1 - torch.tanh(10.0 * ang)
            d = (p - X.tcp_pos_w(env)).norm(dim=-1)
            reach = ((1 - torch.tanh(3 * d)) + (1 - torch.tanh(30 * d))) / 2
            track = 1 - torch.tanh(40.0 * (start + offs[idx_before] - p).norm(dim=-1))
            placed = (start + offs[-1] - p).norm(dim=-1) <= params['success_radius']
            rotated = ang <= params['success_angle']
            qvel = env.scene['robot'].data.joint_vel[:, env._xsim_state['static_joint_mask']]
            static_r = (1 - torch.tanh(5 * qvel.norm(dim=-1))) * (placed & rotated).float()
            success = placed & rotated & (qvel.abs().amax(dim=-1) <= params['static_thresh'])
            expected = reach + track + exp_ar + 2 * idx_before + static_r + success.float()
            reward = X.xsim_dense_reward(env, **params)
            m = env._xsim_state['metrics']
            idx_after = int(env._xsim_state['idx'][0])
            ok = (idx_after == exp_idx
                  and abs(m['angle_reward'][0].item() - exp_ar[0].item()) < 1e-4
                  and abs(reward[0].item() - expected[0].item()) < 1e-4)
            if not ok:
                FAILS.append(label)
            print(f"  {label:22s} | {idx_before} -> {idx_after} ({exp_idx})   | {math.degrees(m['wp_angle'][0].item()):8.2f}  | "
                  f"{m['angle_reward'][0].item():.4f}  {exp_ar[0].item():.4f} | {reward[0].item():7.4f}  "
                  f"{expected[0].item():7.4f} | {'PASS' if ok else 'FAIL'}", flush=True)
        print('RESULT', 'ALL PASS' if not FAILS else f'FAILED: {FAILS}', flush=True)
    finally:
        env.close()


if __name__ == '__main__':
    try:
        import torch
        with torch.inference_mode():
            main()
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        app.close()
