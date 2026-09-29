#!/usr/bin/env python3
"""Kinematic finger-closure sweep: fingertip pad centers in palm frame. No env.step, no physics."""
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

TASK = 'VLA-YCBSugarBox-G1-Grasp-v0'
TIPS = ['left_hand_thumb_2_link', 'left_hand_index_1_link', 'left_hand_middle_1_link']
# closing direction per joint in LEFT_HAND_JOINT_NAMES order; 0 = hold at open value
CLOSE_SIGN = [0.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0]

def fmt(t):
    return '[' + ', '.join(f'{v:+.4f}' for v in t.tolist()) + ']'

def main():
    import torch
    import gymnasium as gym
    import vla_isaaclab  # noqa: F401
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.utils.math import quat_apply, quat_conjugate
    from dex_ee_delta_controller import DexEEDeltaController
    cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
    cfg.seed = 0
    for name in ('camera', 'cam_side', 'cam_left_high', 'cam_left_wrist', 'cam_right_wrist'):
        if hasattr(cfg.scene, name):
            setattr(cfg.scene, name, None)
    env = gym.make(TASK, cfg=cfg).unwrapped
    try:
        env.reset(seed=0)
        dev = env.device
        c = DexEEDeltaController(env)
        robot = env.scene['robot']
        view = robot.root_physx_view
        ids, found = robot.find_bodies(TIPS, preserve_order=True)
        if list(found) != TIPS:
            raise RuntimeError(f'Unexpected fingertip bodies: {found}')

        # 1) fingertip geometry center in each distal link's own frame (from USD)
        stage = omni.usd.get_context().get_stage()
        root = stage.GetPrimAtPath('/World/envs/env_0')
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                  [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.guide])
        centers = []
        print('DISTAL_LINK_GEOMETRY (link-local frame, m)')
        for n in TIPS:
            matches = [p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()) if p.GetName() == n]
            prims = [p for p in matches if p.HasAPI(UsdPhysics.RigidBodyAPI)]
            print(f'  {n}: name matches {[str(p.GetPath()) for p in matches]}')
            if len(prims) != 1:
                raise RuntimeError(f'{n}: {len(prims)} rigid-body prims')
            print(f'  {n}: using {prims[0].GetPath()}')
            box = cache.ComputeUntransformedBound(prims[0]).ComputeAlignedRange()
            if box.IsEmpty():
                print(f'  {n}: EMPTY bbox, falling back to link origin')
                centers.append([0.0, 0.0, 0.0])
                continue
            mn, mx = box.GetMin(), box.GetMax()
            ctr = [(mn[i] + mx[i]) / 2 for i in range(3)]
            centers.append(ctr)
            print(f'  {n}: min={[round(v, 4) for v in mn]} max={[round(v, 4) for v in mx]} center={[round(v, 4) for v in ctr]}')
        centers = torch.tensor(centers, device=dev)

        def read_pads():
            tf = view.get_link_transforms()[0].clone().to(dev)
            pos, quat = tf[:, :3], tf[:, [6, 3, 4, 5]]  # PhysX xyzw -> wxyz
            pp, pq = pos[c.palm_body_id], quat[c.palm_body_id]
            pads_w = pos[ids] + quat_apply(quat[ids], centers)
            return pos, quat, quat_apply(quat_conjugate(pq.expand(3, -1)), pads_w - pp)

        # 2) sanity: PhysX link transforms agree with Isaac Lab body data at reset
        pos, quat, _ = read_pads()
        dp = (pos[c.palm_body_id] - robot.data.body_pos_w[0, c.palm_body_id]).norm().item()
        dq = 1 - abs(float((quat[c.palm_body_id] * robot.data.body_quat_w[0, c.palm_body_id]).sum()))
        print(f'SANITY palm_pos_diff={dp:.2e} m, palm_quat_misalign={dq:.2e} (both should be ~0)')

        # 3) closure sweep
        hid = torch.as_tensor(c.hand_joint_ids, dtype=torch.long, device=dev)
        soft = robot.data.soft_joint_pos_limits[0, hid]
        sign = torch.tensor(CLOSE_SIGN, device=dev)
        open_q = torch.zeros(7, device=dev).clamp(soft[:, 0], soft[:, 1])
        closed_q = torch.where(sign > 0, soft[:, 1], torch.where(sign < 0, soft[:, 0], open_q))
        print('OPEN_Q  ', fmt(open_q))
        print('CLOSED_Q', fmt(closed_q))
        full = robot.data.joint_pos.clone()
        print('SWEEP  frac | thumb-index  thumb-middle  index-middle (m) | pad centroid in palm frame (m)')
        prev = None
        for k in range(11):
            f = k / 10
            q = open_q + f * (closed_q - open_q)
            full[0, hid] = q
            robot.write_joint_state_to_sim(full, torch.zeros_like(full))
            env.sim.forward()
            readback = view.get_dof_positions()[0].to(dev)[hid]
            _, _, pads = read_pads()
            d = lambda a, b: (pads[a] - pads[b]).norm().item()
            moved = '' if prev is None else f'  moved={(pads - prev).norm(dim=1).max().item():.4f}'
            prev = pads.clone()
            print(f'  {f:4.1f} | {d(0, 1):.4f}       {d(0, 2):.4f}        {d(1, 2):.4f}          | '
                  f'{fmt(pads.mean(0))} mid={fmt((pads[0] + 0.5 * (pads[1] + pads[2])) / 2)} width={(pads[0] - 0.5 * (pads[1] + pads[2])).norm().item():.4f}  q_err={(readback - q).abs().max().item():.1e}{moved}')
            if k in (0, 5, 10):
                for n, p in zip(TIPS, pads):
                    print(f'         {n:26s} {fmt(p)}')
        print('NOTE: kinematic only; no contacts resolved, self-collision disabled. '
              'Pad = bbox center of distal link geometry (approximation).', flush=True)
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
