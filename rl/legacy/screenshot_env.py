#!/usr/bin/env python3
"""Reset a registered task, let it settle with zero motion, save camera PNGs. Diagnostic only."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task', required=True)
parser.add_argument('--out-dir', type=Path, required=True)
parser.add_argument('--settle-steps', type=int, default=30)
parser.add_argument('--cameras', nargs='+', default=['cam_side', 'cam_left_high'])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.experience = str(ROOT / 'configs/ycb.python.headless.rendering.kit')
app = AppLauncher(args).app


def main():
    import torch
    import gymnasium as gym
    from PIL import Image
    import vla_isaaclab  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from dex_grip_controller import DexGripEEDeltaController

    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg.seed = 0
    for name in ('camera', 'cam_side', 'cam_left_high', 'cam_left_wrist', 'cam_right_wrist'):
        if hasattr(cfg.scene, name) and name not in args.cameras:
            setattr(cfg.scene, name, None)
    env = gym.make(args.task, cfg=cfg).unwrapped
    try:
        env.reset(seed=0)
        c = DexGripEEDeltaController(env)
        obj = env.scene['object']
        p0 = obj.data.root_pos_w[0].clone()
        for _ in range(args.settle_steps):
            a = torch.zeros(1, 7, device=env.device)
            a[:, 6] = -1.0
            env.step(c.compute(a))
        p1 = obj.data.root_pos_w[0]
        print('object start pos', [round(v, 4) for v in p0.tolist()],
              'after settle', [round(v, 4) for v in p1.tolist()],
              'drift_cm', round((p1 - p0).norm().item() * 100, 3), flush=True)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        for name in args.cameras:
            rgb = env.scene[name].data.output['rgb'][0, ..., :3].detach().cpu().numpy()
            path = args.out_dir / f'{name}.png'
            Image.fromarray(rgb.astype('uint8')).save(path)
            print('saved', path, flush=True)
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
