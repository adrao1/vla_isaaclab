#!/usr/bin/env python3
"""Scripted approach/close/lift to validate the X-Sim tripod grasp detector. Diagnostic only."""
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--approach-steps', type=int, default=150)
parser.add_argument('--close-steps', type=int, default=75)
parser.add_argument('--lift-steps', type=int, default=60)
parser.add_argument('--print-every', type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = False
args.experience = str(ROOT / 'configs/ycb.python.headless.kit')
app = AppLauncher(args).app

TASK = 'VLA-YCBSugarBox-G1-XSimLift-v0'
# LEFT_HAND_JOINT_NAMES order: thumb0, thumb1, thumb2, middle0, middle1, index0, index1
CLOSE = [0.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]

def main():
    import torch
    import gymnasium as gym
    import vla_isaaclab  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from dex_ee_delta_controller import DexEEDeltaController
    from vla_isaaclab.envs.ycb_sugar_box.mdp import xsim as X
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
    cfg.seed = 0
    for name in ('camera', 'cam_side', 'cam_left_high', 'cam_left_wrist', 'cam_right_wrist'):
        if hasattr(cfg.scene, name):
            setattr(cfg.scene, name, None)
    env = gym.make(TASK, cfg=cfg).unwrapped
    try:
        env.reset(seed=0)
        c = DexEEDeltaController(env)
        obj = env.scene['object']
        close = torch.tensor(CLOSE, device=env.device)
        print('phase    step | tcp_d  | thumb F/cos     index F/cos     middle F/cos   | grasp | lift cm | idx | reward')
        step = 0
        phases = (('approach', args.approach_steps), ('close', args.close_steps), ('lift', args.lift_steps))
        for phase, n_steps in phases:
            grasp_steps = 0
            for k in range(n_steps):
                a = torch.zeros(1, 13, device=env.device)
                if phase == 'approach':
                    err = obj.data.root_pos_w - X.tcp_pos_w(env)
                    a[:, 0:3] = (0.5 * err / c.translation_scale if hasattr(c, 'translation_scale')
                                 else 0.5 * err / 0.02).clamp(-1.0, 1.0)
                else:
                    a[:, 6:13] = close
                    if phase == 'lift':
                        a[:, 2] = 0.5
                obs, rew, term, trunc, _ = env.step(c.compute(a))
                step += 1
                m = env._xsim_state['metrics']
                g = X.xsim_grasp_flags(env)
                grasp_steps += int(g['is_grasped'][0])
                if k % args.print_every == 0 or k == n_steps - 1 or bool(term[0] | trunc[0]):
                    print(f"{phase:8s} {step:4d} | {m['tcp_dist'][0].item():.4f} | "
                          f"{g['thumb_force'][0].item():5.2f}/{g['thumb_cos'][0].item():+.2f}   "
                          f"{g['index_force'][0].item():5.2f}/{g['index_cos'][0].item():+.2f}   "
                          f"{g['middle_force'][0].item():5.2f}/{g['middle_cos'][0].item():+.2f}  | "
                          f"{int(g['is_grasped'][0])}     | {m['lift'][0].item() * 100:6.2f}  | "
                          f"{int(env._xsim_state['idx'][0])}   | {rew[0].item():.3f}", flush=True)
                    _xy = (obj.data.root_pos_w[0, :2] - (obj.data.default_root_state[0, :2] + env.scene.env_origins[0, :2])).norm().item()
                    _tilt = X._quat_angle(obj.data.default_root_state[:, 3:7], obj.data.root_quat_w)[0].item()
                    print(f"{'':13s} wp_dist={m['wp_dist'][0].item() * 100:.2f} cm  xy_shift={_xy * 100:.2f} cm  tilt={_tilt * 57.3:.1f} deg", flush=True)
                if bool(term[0] | trunc[0]):
                    print(f'  episode ended during {phase} (term={bool(term[0])}, trunc={bool(trunc[0])})')
                    print('DONE', flush=True)
                    return
            print(f'  -> {phase}: grasp detected on {grasp_steps}/{n_steps} steps', flush=True)
        print('NOTE: cos > +0.087 means force within 85 deg of that finger\'s opening direction.')
        print('DONE', flush=True)
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
