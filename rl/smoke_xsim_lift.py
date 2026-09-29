#!/usr/bin/env python3
"""Smoke checks for VLA-YCBSugarBox-G1-XSimLift-v0 with the 13-D Dex controller. No training."""
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--num-envs', type=int, default=4)
parser.add_argument('--random-steps', type=int, default=90)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = False
args.experience = str(ROOT / 'configs/ycb.python.headless.kit')
app = AppLauncher(args).app

TASK = 'VLA-YCBSugarBox-G1-XSimLift-v0'
FAILS = []

def check(name, ok, detail=''):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)
    if not ok:
        FAILS.append(name)

def main():
    import torch
    import gymnasium as gym
    import vla_isaaclab  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from dex_ee_delta_controller import DexEEDeltaController
    from vla_isaaclab.envs.ycb_sugar_box.mdp import xsim as X
    from vla_isaaclab.envs.ycb_sugar_box.xsim_lift_env_cfg import XSIM_HAND_OPEN_JOINT_POS
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=args.num_envs)
    cfg.seed = 0
    for name in ('camera', 'cam_side', 'cam_left_high', 'cam_left_wrist', 'cam_right_wrist'):
        if hasattr(cfg.scene, name):
            setattr(cfg.scene, name, None)
    env = gym.make(TASK, cfg=cfg).unwrapped
    try:
        obs, _ = env.reset(seed=0)
        n = env.num_envs
        c = DexEEDeltaController(env)
        robot = env.scene['robot']
        hid = torch.as_tensor(c.hand_joint_ids, device=env.device)
        print('SETUP')
        print(f'  step_dt={env.step_dt:.5f} reward_weight={env.cfg.rewards.xsim.weight:.3f} '
              f'max_episode_steps={env.max_episode_length}')
        check('obs shape (N,33)', tuple(obs['policy'].shape) == (n, 33), str(tuple(obs['policy'].shape)))
        expect = torch.tensor([XSIM_HAND_OPEN_JOINT_POS[k] for k in c.hand_joint_names], device=env.device)
        q0 = robot.data.joint_pos[:, hid].clone()
        check('hand reset pose', torch.allclose(q0, expect.expand_as(q0), atol=1e-3),
              str([round(v, 4) for v in q0[0].tolist()]))
        soft = robot.data.soft_joint_pos_limits[0, hid]
        check('hand reset inside soft limits', bool(((q0[0] >= soft[:, 0]) & (q0[0] <= soft[:, 1])).all()))
        start, wps, scales = X._waypoints(env)
        print(f'  object start (env0, local)={[round(v, 4) for v in (start[0] - env.scene.env_origins[0]).tolist()]}')
        print(f'  waypoint offsets={[[round(v, 4) for v in w] for w in (wps[0] - start[0]).tolist()]} scales={scales.tolist()}')
        print(f'  initial TCP->object distance={[round(v, 4) for v in (start - X.tcp_pos_w(env)).norm(dim=-1).tolist()]}')

        print('ZERO ACTION (15 steps)')
        palm0 = robot.data.body_pos_w[:, c.palm_body_id].clone()
        zero = torch.zeros(n, 13, device=env.device)
        for _ in range(15):
            obs, rew, term, trunc, _ = env.step(c.compute(zero))
        dq = (robot.data.joint_pos[:, hid] - q0).abs().max().item()
        dp = (robot.data.body_pos_w[:, c.palm_body_id] - palm0).norm(dim=-1).max().item()
        check('fingers still under zero action', dq < 0.01, f'max |dq|={dq:.4f} rad')
        print(f'  palm drift under zero arm action: {dp * 1000:.2f} mm (info)')
        m = env._xsim_state['metrics']
        recon = (m['reach'] + m['is_grasped'].float() + m['wp_track'] + 2 * m['wp_idx_before'].float()
                 + m['static_reward'] + m['success'].float())
        check('env reward == X-Sim formula', torch.allclose(rew, recon, atol=1e-4),
              f'env={rew[0].item():.4f} formula={recon[0].item():.4f}')
        check('no termination at rest', not bool((term | trunc).any()))
        print(f'  at rest: tcp_dist={m["tcp_dist"][0].item():.4f} reach={m["reach"][0].item():.4f} '
              f'wp_track={m["wp_track"][0].item():.4f} lift={m["lift"][0].item():.4f} '
              f'success={bool(m["success"][0])}')
        check('unlifted box is not success', not bool(m['success'].any()))

        print(f'RANDOM 13-D ACTIONS ({args.random_steps} steps)')
        g = torch.Generator(device=env.device).manual_seed(0)
        rmin, rmax, fmax, lift_max, grasp_any, idx_max, dones = 1e9, -1e9, 0.0, -1.0, 0, 0, 0
        finite = True
        for _ in range(args.random_steps):
            a = torch.rand(n, 13, device=env.device, generator=g) * 2 - 1
            obs, rew, term, trunc, _ = env.step(c.compute(a))
            m = env._xsim_state['metrics']
            finite &= bool(torch.isfinite(obs['policy']).all() and torch.isfinite(rew).all())
            rmin, rmax = min(rmin, rew.min().item()), max(rmax, rew.max().item())
            fmax = max(fmax, max(m[k].max().item() for k in ('thumb_force', 'index_force', 'middle_force')))
            lift_max = max(lift_max, m['lift'].max().item())
            grasp_any += int(m['is_grasped'].sum().item())
            idx_max = max(idx_max, int(env._xsim_state['idx'].max().item()))
            dones += int((term | trunc).sum().item())
        check('obs and reward finite', finite)
        print(f'  reward range=[{rmin:.3f}, {rmax:.3f}] max finger force={fmax:.2f} N '
              f'max lift={lift_max * 100:.2f} cm grasp-steps={grasp_any} max wp idx={idx_max} dones={dones}')

        print('RESET')
        env._xsim_state['idx'] = torch.ones_like(env._xsim_state['idx'])
        env.reset()
        check('waypoint index reset to 0', int(env._xsim_state['idx'].max().item()) == 0)
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
