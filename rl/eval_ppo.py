"""Evaluate a G1 grasp checkpoint without PPO updates."""
import argparse
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--task', default='VLA-YCBSugarBox-G1-GuidedGrasp-v0')
parser.add_argument('--episodes', type=int, default=20)
parser.add_argument('--seed', type=int, default=1000)
parser.add_argument('--video', action='store_true')
parser.add_argument('--measure-lift', action='store_true', help='Disable contact-only success and measure lift-and-hold instead.')
parser.add_argument('--lift-height', type=float, default=0.05)
parser.add_argument('--hold-seconds', type=float, default=0.5)
parser.add_argument('--output', type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.measure_lift and args.task == 'VLA-YCBSugarBox-G1-Waypoint-v0':
    parser.error('Use original-task evaluation for waypoints; its success term advances the stages.')
if args.episodes < 1 or args.lift_height <= 0 or args.hold_seconds <= 0:
    parser.error('Episode count, lift height and hold time must be positive')
args.checkpoint = args.checkpoint.resolve()
if not args.checkpoint.is_file():
    parser.error('Checkpoint does not exist')
args.enable_cameras = args.video
args.experience = str(ROOT / 'configs' / ('ycb.python.headless.rendering.kit' if args.headless else 'ycb.python.rendering.kit')) if args.video else str(ROOT / 'configs/ycb.python.headless.kit')
output = args.output or ROOT / 'outputs/evaluations' / datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
output.mkdir(parents=True, exist_ok=False)
args.kit_args = f'--portable-root {output.resolve()}/kit'
app = AppLauncher(args).app

import numpy as np
import torch
import gymnasium as gym
import vla_isaaclab
from isaaclab_tasks.utils import parse_env_cfg
from ee_delta_controller import EEDeltaController
from ppo_components import Agent, RolloutVideoRecorder, load_agent_checkpoint
from vla_isaaclab.envs.ycb_sugar_box.mdp.terminations import dex3_grasp_contacts


def main():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg.seed = args.seed
    if args.measure_lift:
        # Measure the existing policy, without rewarding or training it further.
        cfg.terminations.success = None
        if hasattr(cfg.rewards, 'completion'):
            cfg.rewards.completion = None
    for name in ('camera', 'cam_side', 'cam_left_high', 'cam_left_wrist', 'cam_right_wrist'):
        if hasattr(cfg.scene, name) and not (args.video and name == 'cam_side'):
            setattr(cfg.scene, name, None)
    env = gym.make(args.task, cfg=cfg).unwrapped
    rows = []
    try:
        obs, _ = env.reset(seed=args.seed)
        agent = Agent(obs_dim=obs['policy'].shape[-1], action_dim=7).to(env.device)
        metadata = load_agent_checkpoint(agent, args.checkpoint, env.device)
        agent.eval()
        controller = EEDeltaController(env)
        hold_steps = math.ceil(args.hold_seconds / env.step_dt)
        limit = math.ceil(cfg.episode_length_s / env.step_dt)
        pending = {}
        original_reset = env._reset_idx

        def snapshot():
            obj = env.scene['object']
            return {
                'waypoint_stage': int(env.command_manager.get_term('target_pose').stage[0]) if args.task == 'VLA-YCBSugarBox-G1-Waypoint-v0' else None,
                'height': float(obj.data.root_pos_w[0, 2]),
                'contact': bool(dex3_grasp_contacts(env, min_force=0.5)['is_grasping'][0]),
                'terms': {name: bool(env.termination_manager.get_term(name)[0])
                          for name in env.termination_manager.active_terms},
            }

        def capture_before_reset(self, ids):
            if pending.get('stepping', False):
                pending['final'] = snapshot()
            return original_reset(ids)
        env._reset_idx = MethodType(capture_before_reset, env)

        report = {
            'checkpoint': str(args.checkpoint), 'checkpoint_metadata': metadata,
            'task': args.task, 'deterministic_actor': True,
            'mode': 'lift_probe' if args.measure_lift else 'original_task',
            'lift_threshold_m': args.lift_height, 'hold_steps': hold_steps,
            'control_dt': env.step_dt,
            'limitation': 'Different seeds do not create pose variation when resets are fixed.',
            'episodes': rows,
        }
        for episode in range(args.episodes):
            seed = args.seed + episode
            obs, _ = env.reset(seed=seed)
            if hasattr(env, 'grasp_success_counter'):
                env.grasp_success_counter.zero_()
            controller.reset()
            initial_height = float(env.scene['object'].data.root_pos_w[0, 2])
            max_height = 0.0
            held = 0
            max_held = 0
            lift_pass = False
            recorder = RolloutVideoRecorder(output / f'episode_{episode:03d}.mp4', round(1 / env.step_dt)) if args.video else None
            terms = {}
            try:
                for step in range(limit):
                    if recorder:
                        recorder.add_rgb(env.scene['cam_side'].data.output['rgb'][0, ..., :3].detach().cpu().numpy())
                    with torch.inference_mode():
                        action = agent.get_action(obs['policy'], deterministic=True).clamp(-1, 1)
                        command = controller.compute(action)
                        if not torch.isfinite(command).all():
                            raise RuntimeError('Non-finite controller command')
                        pending.pop('final', None)
                        pending['stepping'] = True
                        try:
                            obs, _, terminated, truncated, _ = env.step(command)
                        finally:
                            pending['stepping'] = False
                        state = pending['final'] if 'final' in pending else snapshot()
                    rise = state['height'] - initial_height
                    max_height = max(max_height, rise)
                    held = held + 1 if rise >= args.lift_height and state['contact'] else 0
                    max_held = max(max_held, held)
                    lift_pass = lift_pass or held >= hold_steps
                    terms = state['terms']
                    if bool(terminated[0]) or bool(truncated[0]) or (args.measure_lift and lift_pass):
                        break
            finally:
                if recorder:
                    recorder.close()
            row = {
                'episode': episode, 'seed': seed, 'steps': step + 1,
                'named_success': bool(terms.get('success', False)),
                'lift_hold_success': lift_pass,
                'max_rise_m': max_height, 'max_hold_seconds': max_held * env.step_dt,
                'termination_terms': terms,
                'waypoint_stage': state['waypoint_stage'],
            }
            rows.append(row)
            report['named_success_rate'] = sum(r['named_success'] for r in rows) / len(rows)
            report['lift_hold_success_rate'] = sum(r['lift_hold_success'] for r in rows) / len(rows)
            temporary = output / 'results.tmp'
            temporary.write_text(json.dumps(report, indent=2))
            temporary.replace(output / 'results.json')
            print(json.dumps(row), flush=True)
        print(f"Results: {output / 'results.json'}", flush=True)
        print(f"Named success: {report['named_success_rate']:.1%}; lift-and-hold: {report['lift_hold_success_rate']:.1%}")
    finally:
        env.close()


if __name__ == '__main__':
    try:
        with torch.inference_mode():
            main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        app.close()
