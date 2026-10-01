#!/usr/bin/env python3
"""Measure reset hand geometry; validate 13-D mapping without moving the robot."""
import argparse
import json
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

def main():
    import torch
    import gymnasium as gym
    import vla_isaaclab
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.utils.math import quat_apply, quat_conjugate
    from dex_ee_delta_controller import DexEEDeltaController
    cfg = parse_env_cfg('VLA-YCBSugarBox-G1-Grasp-v0', device=args.device, num_envs=1)
    cfg.seed = 0
    for name in ('camera', 'cam_side', 'cam_left_high', 'cam_left_wrist', 'cam_right_wrist'):
        if hasattr(cfg.scene, name):
            setattr(cfg.scene, name, None)
    env = gym.make('VLA-YCBSugarBox-G1-Grasp-v0', cfg=cfg).unwrapped
    try:
        env.reset(seed=0)
        c = DexEEDeltaController(env)
        robot = env.scene['robot']
        names = ['left_hand_thumb_2_link', 'left_hand_index_1_link', 'left_hand_middle_1_link']
        ids, found = robot.find_bodies(names, preserve_order=True)
        if list(found) != names:
            raise RuntimeError(f'Unexpected fingertip bodies: {found}')
        palm = robot.data.body_pos_w[:, c.palm_body_id]
        quat = robot.data.body_quat_w[:, c.palm_body_id]
        local = quat_apply(quat_conjugate(quat.expand(3, -1)), robot.data.body_pos_w[0, ids] - palm)
        zero = torch.zeros((1, 13), device=env.device)
        current = robot.data.joint_pos[:, c.hand_joint_ids]
        _zt = c._hand_targets(zero)
        _ids = c.hand_joint_ids
        _soft = c.robot.data.soft_joint_pos_limits[:, _ids]
        _hard = getattr(c.robot.data, 'joint_pos_limits', None); _hard = (c.robot.data.joint_limits if _hard is None else _hard)[:, _ids]
        print('ZERO_ACTION_CHECK')
        print('  current     ', [round(v, 5) for v in current.reshape(-1).tolist()])
        print('  zero_target ', [round(v, 5) for v in _zt.reshape(-1).tolist()])
        print('  default     ', [round(v, 5) for v in c.robot.data.default_joint_pos[0, _ids].tolist()])
        print('  soft_lower  ', [round(v, 5) for v in _soft[0, :, 0].tolist()])
        print('  soft_upper  ', [round(v, 5) for v in _soft[0, :, 1].tolist()])
        print('  hard_lower  ', [round(v, 5) for v in _hard[0, :, 0].tolist()])
        print('  hard_upper  ', [round(v, 5) for v in _hard[0, :, 1].tolist()])
        print('  max_abs_diff', (_zt.reshape(-1) - current.reshape(-1)).abs().max().item(), flush=True)
        _oh = c.open_hand.reshape(-1, 7)[0]; _ch = c.closed_hand.reshape(-1, 7)[0]
        print('OLD_GRIPPER_CHECK')
        print('  open_hand        ', [round(v, 5) for v in _oh.tolist()])
        print('  closed_hand      ', [round(v, 5) for v in _ch.tolist()])
        print('  closed_clamped   ', [round(v, 5) for v in _ch.clamp(_soft[0, :, 0], _soft[0, :, 1]).tolist()])
        print('  closed_in_soft   ', ((_ch >= _soft[0, :, 0]) & (_ch <= _soft[0, :, 1])).tolist(), flush=True)
        limits = robot.data.soft_joint_pos_limits[:, c.hand_joint_ids]
        for i in range(7):
            for sign in (-1, 1):
                action = zero.clone()
                action[:, 6+i] = sign
                expected = current.clone()
                expected[:, i] += sign * c.hand_delta_scale
                expected = expected.clamp(limits[..., 0], limits[..., 1])
                assert torch.allclose(c._hand_targets(action), expected, atol=1e-6)
        command = c.compute(zero)
        assert command.shape == (1, 43) and torch.isfinite(command).all()
        print(json.dumps({
            'hand_joint_order': c.hand_joint_names,
            'reset_hand_positions_rad': current[0].cpu().tolist(),
            'soft_limits_rad': limits[0].cpu().tolist(),
            'distal_link_origins_in_palm_m': dict(zip(names, local.cpu().tolist())),
            'distal_origin_centroid_in_palm_m': local.mean(0).cpu().tolist(),
            'note': 'Link origins are not necessarily contact-pad centers; centroid is a candidate, not a calibrated TCP.',
            'mapping_checks': 'passed; no physics actions executed',
        }, indent=2), flush=True)
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
