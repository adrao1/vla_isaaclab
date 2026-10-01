#!/usr/bin/env python3
"""Check the mustard GuidedComputed task: reward terms vs. a scripted drive to a saved grasp.

Drives the hand to saved grasp 0 with the EE controller using the same object-frame
transform as validate_grasp_candidates.py, then prints the guidance errors/rewards
(computed independently inside the env). Diagnostic only; no training.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task', default='VLA-YCBMustard-G1-GuidedComputed-v0')
parser.add_argument('--library', type=Path, default=ROOT / 'assets/grasps/mustard_bottle.json')
parser.add_argument('--grasp', type=int, default=1, help='library index to drive to')
parser.add_argument('--standoff', type=float, default=0.08)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = False
args.experience = str(ROOT / 'configs/ycb.python.headless.kit')
app = AppLauncher(args).app


def main():
    import numpy as np
    import torch
    import gymnasium as gym
    import vla_isaaclab  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.utils.math import compute_pose_error, matrix_from_quat, quat_from_matrix
    from dex_grip_controller import DexGripEEDeltaController
    from vla_isaaclab.envs.ycb_sugar_box.mdp import grasp_guidance as GG
    from vla_isaaclab.envs.ycb_sugar_box.mdp import xsim as X

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg.seed = 0
    cfg.episode_length_s = 60.0
    for name in ('camera', 'cam_side', 'cam_left_high', 'cam_left_wrist', 'cam_right_wrist'):
        if hasattr(cfg.scene, name):
            setattr(cfg.scene, name, None)
    env = gym.make(args.task, cfg=cfg).unwrapped
    dev = env.device
    lib = json.loads(args.library.read_text())
    g = lib['grasps'][args.grasp]
    print('reward terms:', list(env.reward_manager.active_terms), flush=True)
    try:
        env.reset(seed=0)
        c = DexGripEEDeltaController(env)
        obj, robot = env.scene['object'], env.scene['robot']
        path = str(args.library)

        def guide():
            d, a, i = GG.nearest_grasp_errors(env, path)
            return (d[0].item() * 100, np.degrees(a[0].item()), int(i[0]),
                    GG.guidance_reach(env, path)[0].item(), GG.guidance_orientation(env, path)[0].item())

        def show(tag):
            d, a, i, r, o = guide()
            gr = X.xsim_grasp_flags(env)['is_grasped'][0].item()
            m = env._xsim_state.get('metrics') or {}
            lift = m['lift'][0].item() * 100 if 'lift' in m else float('nan')
            print(f'{tag:10s} nearest#{i} dist {d:6.2f} cm angle {a:6.1f} deg | reach_r {r:.3f} orient_r {o:.3f} | grasp {int(gr)} lift {lift:5.2f} cm', flush=True)

        def step(pe=None, re=None, grip=-1.0, lift=0.0):
            a = torch.zeros(1, 7, device=dev)
            if pe is not None:
                a[:, 0:3] = (pe / c.position_scale).clamp(-1, 1)
                a[:, 3:6] = (0.5 * re / c.rotation_scale).clamp(-1, 1)
            a[:, 2] += lift
            a[:, 6] = grip
            return env.step(c.compute(a))

        def move_to(pos, quat, n, tol=0.008):
            for _ in range(n):
                pp = robot.data.body_pos_w[:, c.palm_body_id]
                pq = robot.data.body_quat_w[:, c.palm_body_id]
                pe, re = compute_pose_error(pp, pq, pos, quat, rot_error_type='axis_angle')
                if pe.norm().item() < tol and np.degrees(re.norm().item()) < 4.0:
                    return
                step(pe, re)

        show('start')
        oR = matrix_from_quat(obj.data.root_quat_w)[0]
        t = torch.as_tensor(g['palm_position_asset_m'], dtype=torch.float32, device=dev)
        R = torch.as_tensor(g['palm_rotation_asset_from_palm'], dtype=torch.float32, device=dev)
        pos = (oR @ t + obj.data.root_pos_w[0]).unsqueeze(0)
        Rw = oR @ R
        quat = quat_from_matrix(Rw.unsqueeze(0))
        standoff = pos + args.standoff * (-Rw[:, 0]).unsqueeze(0)
        move_to(standoff, quat, 250)
        show('standoff')
        move_to(pos, quat, 250)
        show('at grasp')
        for k in range(75):
            step(grip=2.0 * (k + 1) / 75 - 1.0)
        show('closed')
        for k in range(90):
            step(grip=1.0, lift=0.5 if k < 60 else 0.0)
        show('lifted')
        rm = env.reward_manager
        print('last-step reward terms:', {n: round(v[0].item(), 3) for n, v in zip(rm.active_terms, rm._step_reward.T)}, flush=True)
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
