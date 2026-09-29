#!/usr/bin/env python3
"""Read Dex3 hand gains/limits and object mass/friction; predict squeeze capability. No env.step."""
import argparse
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--lever-m', type=float, default=0.05,
                    help='rough joint-to-contact distance used only to convert torque to force')
parser.add_argument('--lead-cap-rad', type=float, default=0.35,
                    help='max target lead for the persistent-target scheme')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = False
args.experience = str(ROOT / 'configs/ycb.python.headless.kit')
app = AppLauncher(args).app

TASK = 'VLA-YCBSugarBox-G1-Grasp-v0'

def main():
    import torch
    import gymnasium as gym
    import vla_isaaclab  # noqa: F401  (registers tasks)
    from isaaclab_tasks.utils import parse_env_cfg
    from dex_ee_delta_controller import DexEEDeltaController
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
    cfg.seed = 0
    for name in ('camera', 'cam_side', 'cam_left_high', 'cam_left_wrist', 'cam_right_wrist'):
        if hasattr(cfg.scene, name):
            setattr(cfg.scene, name, None)
    env = gym.make(TASK, cfg=cfg).unwrapped
    try:
        env.reset(seed=0)
        c = DexEEDeltaController(env)
        robot = env.scene['robot']
        view = robot.root_physx_view
        ids = torch.as_tensor(c.hand_joint_ids, dtype=torch.long).cpu()
        kp = view.get_dof_stiffnesses()[0].cpu()[ids]
        kd = view.get_dof_dampings()[0].cpu()[ids]
        fmax = view.get_dof_max_forces()[0].cpu()[ids]
        print('HAND_ACTUATION (values PhysX is actually using)')
        print(f'  delta_scale={c.hand_delta_scale} rad, lead_cap={args.lead_cap_rad} rad, lever={args.lever_m} m')
        for n, a, b, f in zip(c.hand_joint_names, kp.tolist(), kd.tolist(), fmax.tolist()):
            t_re = min(a * c.hand_delta_scale, f)
            t_pe = min(a * args.lead_cap_rad, f)
            print(f'  {n:26s} kp={a:9.3f} kd={b:8.4f} max_torque={f:8.3f} | '
                  f'reanchor={t_re:7.4f} Nm (~{t_re / args.lever_m:6.2f} N) | '
                  f'persistent={t_pe:7.4f} Nm (~{t_pe / args.lever_m:6.2f} N)')
        print('RIGID_OBJECTS')
        for key, obj in env.scene.rigid_objects.items():
            mass = float(obj.root_physx_view.get_masses()[0].cpu().sum())
            mat = obj.root_physx_view.get_material_properties()[0].cpu()
            mu_s, mu_d = float(mat[:, 0].min()), float(mat[:, 1].min())
            need = mass * 9.81 / (2 * max(mu_d, 1e-6))
            print(f'  {key}: mass={mass:.4f} kg static_mu={mu_s:.3f} dynamic_mu={mu_d:.3f} '
                  f'-> two-sided pinch needs ~{need:.2f} N normal force per side (object friction only)')
        print('NOTE: force = torque / lever is rough; PhysX combines hand and object friction '
              '(default: average), so the true requirement may differ.', flush=True)
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
