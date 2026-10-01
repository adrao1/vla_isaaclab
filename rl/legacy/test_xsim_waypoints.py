#!/usr/bin/env python3
"""Reward-code test for XSimLift waypoint progression.

Places the box directly at chosen heights (no physics steps, robot never moves
or touches the box) and calls the X-Sim reward function, checking the waypoint
index and reward against an independent recomputation. Diagnostic only.
"""
import argparse
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

TASK = 'VLA-YCBSugarBox-G1-XSimLift-v0'
# (label, dx, dz, expected idx AFTER the call)
SEQUENCE = [
    ('rest',           0.0,  0.000, 0),
    ('up 1.0 cm',      0.0,  0.010, 0),
    ('up 2.0 cm',      0.0,  0.020, 1),   # within 1 cm of wp0 (2.5 cm) -> advance
    ('up 2.5 cm',      0.0,  0.025, 1),
    ('up 3.5 cm',      0.0,  0.035, 1),
    ('up 4.5 cm',      0.0,  0.045, 1),   # would advance, but clamped at last index (1)
    ('up 5.0 cm',      0.0,  0.050, 1),   # placed -> static + success terms
    ('back to rest',   0.0,  0.000, 1),   # X-Sim never decrements the index
]
SIDEWAYS = [
    ('rest (after reset)', 0.0,  0.000, 0),
    ('up 2 cm, 2 cm side', 0.02, 0.020, 0),  # 2.06 cm from wp0 -> no advance
]
FAILS = []


def main():
    import torch
    import gymnasium as gym
    import vla_isaaclab  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
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
        offs = torch.tensor(X.LIFT_WAYPOINT_OFFSETS, device=env.device)

        def run(title, seq):
            env.reset(seed=0)
            start = obj.data.default_root_state[:, :3] + env.scene.env_origins
            quat = obj.data.default_root_state[:, 3:7]
            print(title)
            print('  label                | idx before->after (exp) | reward   expected | placed success | readback err')
            for label, dx, dz, exp_idx in seq:
                target = start + torch.tensor([[dx, 0.0, dz]], device=env.device)
                obj.write_root_pose_to_sim(torch.cat((target, quat), dim=-1))
                obj.write_root_velocity_to_sim(torch.zeros(1, 6, device=env.device))
                obj.update(env.physics_dt)  # refresh cached data; no physics step
                err = (obj.data.root_pos_w - target).norm().item()
                idx_before = int(env._xsim_state['idx'][0])
                # independent recomputation
                p = obj.data.root_pos_w
                d = (p - X.tcp_pos_w(env)).norm(dim=-1)
                reach = ((1 - torch.tanh(3 * d)) + (1 - torch.tanh(30 * d))) / 2
                wp = start + offs[idx_before]
                track = 1 - torch.tanh(40.0 * (wp - p).norm(dim=-1))
                placed = (start + offs[-1] - p).norm(dim=-1) <= params['success_radius']
                qvel = env.scene['robot'].data.joint_vel[:, env._xsim_state['static_joint_mask']]
                static_r = (1 - torch.tanh(5 * qvel.norm(dim=-1))) * placed.float()
                success = placed & (qvel.abs().amax(dim=-1) <= params['static_thresh'])
                expected = reach + track + 2 * idx_before + static_r + success.float()
                reward = X.xsim_dense_reward(env, **params)
                idx_after = int(env._xsim_state['idx'][0])
                m = env._xsim_state['metrics']
                ok = (idx_after == exp_idx and abs(reward.item() - expected.item()) < 1e-4
                      and err < 1e-4 and not bool(m['is_grasped'][0]))
                if not ok:
                    FAILS.append(f'{title}: {label}')
                print(f"  {label:20s} | {idx_before} -> {idx_after} ({exp_idx})             | "
                      f"{reward.item():7.4f}  {expected.item():7.4f} | {int(m['placed'][0])}      "
                      f"{int(m['success'][0])}       | {err:.1e}  {'PASS' if ok else 'FAIL'}", flush=True)

        run('STRAIGHT UP', SEQUENCE)
        run('SIDEWAYS', SIDEWAYS)
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
