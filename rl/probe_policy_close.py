#!/usr/bin/env python3
"""Run a trained 13-D policy to approach, then force the fingers closed and lift.

Tests the hypothesis "the learned approach pose would grasp if the fingers
curled". Diagnostic only: the scripted close/lift never feeds into training.
"""
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--task', default='VLA-YCBSugarBox-G1-XSimLiftRot-v0')
parser.add_argument('--policy-steps', type=int, default=200)
parser.add_argument('--close-steps', type=int, default=75)
parser.add_argument('--lift-steps', type=int, default=60)
parser.add_argument('--print-every', type=int, default=10)
parser.add_argument('--close-mode', choices=('delta', 'absolute'), default='delta')
parser.add_argument('--close-fraction', type=float, default=1.0,
                    help='absolute mode: fraction of the open->closed path to target')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = False
args.experience = str(ROOT / 'configs/ycb.python.headless.kit')
app = AppLauncher(args).app

# LEFT_HAND_JOINT_NAMES order: thumb0, thumb1, thumb2, middle0, middle1, index0, index1
CLOSE = [0.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]


def main():
    import torch
    import gymnasium as gym
    import vla_isaaclab  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from dex_ee_delta_controller import DexEEDeltaController
    from ppo_components import Agent, load_agent_checkpoint
    from vla_isaaclab.envs.ycb_sugar_box.mdp import xsim as X
    from isaaclab.utils.math import quat_apply, quat_conjugate
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg.seed = 1000
    cfg.episode_length_s = 60.0  # long enough for policy + close + lift phases
    for name in ('camera', 'cam_side', 'cam_left_high', 'cam_left_wrist', 'cam_right_wrist'):
        if hasattr(cfg.scene, name):
            setattr(cfg.scene, name, None)
    env = gym.make(args.task, cfg=cfg).unwrapped
    try:
        obs, _ = env.reset(seed=1000)
        agent = Agent(obs_dim=obs['policy'].shape[-1], action_dim=13).to(env.device)
        load_agent_checkpoint(agent, args.checkpoint.resolve(), env.device)
        agent.eval()
        c = DexEEDeltaController(env)
        robot = env.scene['robot']
        hid = torch.as_tensor(c.hand_joint_ids, device=env.device)
        close = torch.tensor(CLOSE, device=env.device)
        soft = robot.data.soft_joint_pos_limits[0, hid]
        closed_q = torch.where(close > 0, soft[:, 1], torch.where(close < 0, soft[:, 0], robot.data.joint_pos[0, hid]))
        q_open = robot.data.joint_pos[0, hid].clone()
        closed_q = q_open + args.close_fraction * (closed_q - q_open)
        print('close mode:', args.close_mode, 'fraction:', args.close_fraction, ' closed target:', [round(v, 2) for v in closed_q.tolist()])
        print('phase   step | tcp_d  | thumb F/cos    index F/cos    middle F/cos  | grasp | lift cm | rot/tilt  | fingers (rad)')
        step = 0
        for phase, n_steps in (('policy', args.policy_steps), ('close', args.close_steps), ('lift', args.lift_steps)):
            grasp_steps = 0
            if phase == 'close' and args.close_mode == 'absolute':
                c._hand_targets = lambda action: closed_q.unsqueeze(0).expand(action.shape[0], -1)
            for k in range(n_steps):
                if phase == 'policy':
                    a = agent.get_action(obs['policy'], deterministic=True).clamp(-1, 1)
                else:
                    a = torch.zeros(1, 13, device=env.device)
                    a[:, 6:13] = close
                    if phase == 'lift':
                        a[:, 2] = 0.5
                obs, rew, term, trunc, _ = env.step(c.compute(a))
                step += 1
                m = env._xsim_state['metrics']
                g = X.xsim_grasp_flags(env)
                grasp_steps += int(g['is_grasped'][0])
                if k % args.print_every == 0 or k == n_steps - 1 or bool(term[0] | trunc[0]):
                    q = robot.data.joint_pos[0, hid]
                    _o = env.scene['object'].data
                    _ez = torch.tensor([[0.0, 0.0, 1.0]], device=env.device)
                    _up = quat_apply(_o.root_quat_w, quat_apply(quat_conjugate(_o.default_root_state[:, 3:7]), _ez))
                    _tilt = torch.rad2deg(torch.acos(_up[0, 2].clamp(-1.0, 1.0))).item()
                    print(f"{phase:7s} {step:4d} | {m['tcp_dist'][0].item():.4f} | "
                          f"{g['thumb_force'][0].item():5.2f}/{g['thumb_cos'][0].item():+.2f}  "
                          f"{g['index_force'][0].item():5.2f}/{g['index_cos'][0].item():+.2f}  "
                          f"{g['middle_force'][0].item():5.2f}/{g['middle_cos'][0].item():+.2f} | "
                          f"{int(g['is_grasped'][0])}     | {m['lift'][0].item() * 100:6.2f}  | "
                          f"{m['wp_angle'][0].item() * 57.2958:5.1f}/{_tilt:4.1f} | "
                          f"{' '.join(f'{v:+.2f}' for v in q.tolist())}", flush=True)
                if bool(term[0] | trunc[0]):
                    print(f'  episode ended during {phase} (term={bool(term[0])}, trunc={bool(trunc[0])})')
                    print('DONE', flush=True)
                    return
            print(f'  -> {phase}: grasp detected on {grasp_steps}/{n_steps} steps', flush=True)
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
