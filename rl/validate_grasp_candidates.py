#!/usr/bin/env python3
"""Physics validation of geometry-computed palm grasp candidates. Diagnostic only.

For each candidate palm pose (object frame, from outputs/grasp_geometry):
  reset -> drive the real arm (EE-delta IK; --controller ee7, grip7, or straight)
  to a standoff behind the hand along its finger axis -> advance to the candidate
  palm pose -> ramp the grip closed -> lift. Nothing is teleported or attached;
  the hand only uses the normal controller. Results are MEASURED diagnostics,
  not the named task success.

Candidates run in parallel: --num-envs environments (capped at the number of
candidates). --video writes one cam_side mp4 per candidate.

Assumptions (ours, not X-Sim): fingers extend along palm +x (TCP offset is
+0.089 m in x), so the standoff is straight back along -x of the palm.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import grasp_geometry as G
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task', default='VLA-YCBSugarBox-G1-XSimLiftGrip-v0',
                    help='Gym ID whose scene is used for physics validation')
parser.add_argument('--object', default=None, choices=G.OBJECT_NAMES,
                    help='mesh to generate candidates from; default is the object already in --task')
parser.add_argument('--hand-folder', type=Path, default=ROOT / 'outputs/grasp_geometry',
                    help='object-independent hand_closure.npz lives here')
parser.add_argument('--indices', type=int, nargs='*', default=None,
                    help='candidate indices (default: all that pass the open-hand screen)')
parser.add_argument('--limit', type=int, default=None)
parser.add_argument('--margin', type=float, default=0.01,
                    help='aperture margin (m) added each side of the object span')
parser.add_argument('--offsets', type=float, nargs='+', default=[-0.25, 0.0, 0.25],
                    help='palm position offsets as fractions of each principal span')
parser.add_argument('--local-anchors', type=int, default=0,
                    help='>0: local-width proposals from N surface anchors (thin/concave features)')
parser.add_argument('--screen-mode', choices=('hull', 'points'), default='hull')
parser.add_argument('--screen-only', action='store_true',
                    help='generate + screen candidates, print counts, exit (no sim)')
parser.add_argument('--standoff', type=float, default=0.08, help='metres back along palm -x')
parser.add_argument('--move-steps', type=int, default=250)
parser.add_argument('--close-steps', type=int, default=75)
parser.add_argument('--lift-steps', type=int, default=60)
parser.add_argument('--hold-steps', type=int, default=30)
parser.add_argument('--lift-target-cm', type=float, default=5.0)
parser.add_argument('--out-dir', type=Path, default=None,
                    help='default: outputs/grasp_geometry/objects/<object>')
parser.add_argument('--controller', choices=('ee7', 'grip7', 'straight'), default='grip7',
                    help='ee7: broken thumb-only close; grip7: curl all fingers; '
                         'straight: proximal close, distal joints stay open')
parser.add_argument('--num-envs', type=int, default=16,
                    help='parallel Isaac environments; each chunk validates min(this, remaining) candidates')
parser.add_argument('--video', action='store_true',
                    help='record a separate cam_side video per candidate')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = args.video
args.experience = str(ROOT / 'configs' / ('ycb.python.headless.rendering.kit' if args.video
                                          else 'ycb.python.headless.kit'))
app = AppLauncher(args).app


def _table_task(task):
    return 'MustardPlace' not in task


def main():
    import numpy as np
    import torch
    import gymnasium as gym
    import vla_isaaclab  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg
    from isaaclab.utils.math import (compute_pose_error, matrix_from_quat, quat_apply,
                                     quat_conjugate, quat_from_matrix)
    from dex_grip_controller import DexGripEEDeltaController, STRAIGHT_CLOSE_SIGN
    from ee_delta_controller import EEDeltaController
    from ppo_components import RolloutVideoRecorder
    from vla_isaaclab.envs.ycb_sugar_box.mdp import xsim as X

    from vla_isaaclab.envs.common import SUPPORT_HEIGHT
    from vla_isaaclab.envs.ycb_sugar_box import env_cfg as sugar_cfg

    probe = parse_env_cfg(args.task, device=args.device, num_envs=1)
    usd = Path(probe.scene.object.spawn.usd_path)
    spawn_pos = tuple(float(x) for x in probe.scene.object.init_state.pos)
    spawn_rot = tuple(float(x) for x in probe.scene.object.init_state.rot)
    override_spawn = False
    if args.object is not None:
        usd = G.object_usd(ROOT, args.object)
        if args.object in G.YCB_OBJECTS and _table_task(args.task):
            orient = G.YCB_OBJECTS[args.object][1]
            points_tmp = G.load_mesh_points(usd)
            spawn_rot = G.upright_quat(orient, sugar_cfg.SUGAR_BOX_TABLE_YAW_RAD)
            rest_z = SUPPORT_HEIGHT - (points_tmp @ G.quat_to_matrix(spawn_rot).T)[:, 2].min()
            spawn_pos = (*sugar_cfg.INITIAL_XY, float(rest_z))
            override_spawn = True
    object_name = args.object or usd.stem
    args.out_dir = args.out_dir or (args.hand_folder / 'objects' / object_name)
    points = G.load_mesh_points(usd)
    hand = np.load(args.hand_folder / 'hand_closure.npz')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    oR0 = G.quat_to_matrix(spawn_rot)
    op0 = np.array(spawn_pos)
    if args.local_anchors > 0:
        cands = G.generate_local_candidates(points, hand, margin=args.margin, anchors=args.local_anchors)
        spans = points.max(0) - points.min(0)
    else:
        cands, spans = G.generate_candidates(points, hand, margin=args.margin, offsets=tuple(args.offsets))
    print(f'{args.task} / {object_name}: principal spans (cm) {np.round(spans * 100, 1).tolist()}, '
          f'{len(cands)} candidates (margin {args.margin} m)', flush=True)
    print(f'mesh {usd}', flush=True)
    print(f'spawn pos {spawn_pos} rot {spawn_rot}', flush=True)
    if args.indices is not None:
        indices = args.indices
    else:
        screen = G.screen_candidates(cands, points, hand, oR0, op0, mode=args.screen_mode)
        indices = [r['candidate_index'] for r in screen if r['passes']]
        (args.out_dir / 'screen.json').write_text(json.dumps(
            {'task': args.task, 'object': object_name, 'mesh': str(usd),
             'margin_m': args.margin, 'local_anchors': args.local_anchors, 'screen_mode': args.screen_mode,
             'offsets': args.offsets, 'spans_m': spans.tolist(),
             'spawn_pos': list(spawn_pos), 'spawn_rot': list(spawn_rot), 'rows': screen}, indent=1))
    if args.limit:
        indices = indices[:args.limit]
    print(f'Validating {len(indices)} candidates: {indices}', flush=True)
    if args.screen_only:
        return
    if not indices:
        print('No candidates to validate', flush=True)
        return
    if args.num_envs < 1:
        raise ValueError('--num-envs must be >= 1')

    n_envs = min(args.num_envs, len(indices))
    print(f'Parallel envs: {n_envs} (chunks of up to {n_envs})', flush=True)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=n_envs)
    cfg.seed = 0
    cfg.episode_length_s = 60.0
    if args.object is not None:
        cfg.scene.object.spawn.usd_path = str(usd)
    if override_spawn:
        cfg.scene.object.init_state.pos = spawn_pos
        cfg.scene.object.init_state.rot = spawn_rot
    # Isaac Lab resets terminated sub-envs mid-chunk; disable those terms so a
    # fallen object does not restart while its neighbors are still grasping.
    for name in ('object_fallen', 'invalid_state', 'success'):
        if hasattr(cfg.terminations, name):
            setattr(cfg.terminations, name, None)
    for name in ('camera', 'cam_side', 'cam_left_high', 'cam_left_wrist', 'cam_right_wrist'):
        if hasattr(cfg.scene, name) and not (args.video and name == 'cam_side'):
            setattr(cfg.scene, name, None)
    env = None
    results = []
    try:
        env = gym.make(args.task, cfg=cfg).unwrapped
        dev = env.device
        if args.controller == 'ee7':
            controller = EEDeltaController(env)
        elif args.controller == 'straight':
            controller = DexGripEEDeltaController(env, close_sign=STRAIGHT_CLOSE_SIGN)
        else:
            controller = DexGripEEDeltaController(env)
        robot = env.scene['robot']
        obj = env.scene['object']
        ez = torch.zeros(env.num_envs, 3, device=dev)
        ez[:, 2] = 1.0

        def tilt_deg():
            up = quat_apply(obj.data.root_quat_w,
                            quat_apply(quat_conjugate(obj.data.default_root_state[:, 3:7]), ez))
            return torch.rad2deg(torch.acos(up[:, 2].clamp(-1.0, 1.0)))

        def xy_shift_cm():
            d = obj.data.root_pos_w[:, :2] - (obj.data.default_root_state[:, :2] + env.scene.env_origins[:, :2])
            return d.norm(dim=-1) * 100.0

        def record_videos(videos):
            if not videos:
                return
            rgb = env.scene['cam_side'].data.output['rgb'][..., :3].detach().cpu().numpy()
            for i, rec in enumerate(videos):
                rec.add_rgb(rgb[i])

        def step(moving, pos_err=None, rot_err=None, grip=None, lift=0.0, videos=None):
            a = torch.zeros(env.num_envs, 7, device=dev)
            a[:, 6] = -1.0
            if grip is not None:
                a[moving, 6] = grip[moving] if torch.is_tensor(grip) else grip
            if pos_err is not None:
                a[moving, 0:3] = (pos_err[moving] / controller.position_scale).clamp(-1.0, 1.0)
                a[moving, 3:6] = (0.5 * rot_err[moving] / controller.rotation_scale).clamp(-1.0, 1.0)
            if lift:
                a[moving, 2] += lift
            record_videos(videos)
            _, _, term, trunc, _ = env.step(controller.compute(a))
            return term | trunc

        def move_to(target_pos, target_q, max_steps, live, videos, pos_tol=0.01, rot_tol_deg=4.0):
            reached = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
            pn = torch.zeros(env.num_envs, device=dev)
            rn = torch.zeros(env.num_envs, device=dev)
            ended = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
            for _ in range(max_steps):
                palm_p = robot.data.body_pos_w[:, controller.palm_body_id]
                palm_q = robot.data.body_quat_w[:, controller.palm_body_id]
                pe, re = compute_pose_error(palm_p, palm_q, target_pos, target_q, rot_error_type='axis_angle')
                pn = pe.norm(dim=-1)
                rn = torch.rad2deg(re.norm(dim=-1))
                reached |= (pn < pos_tol) & (rn < rot_tol_deg)
                moving = live & ~reached & ~ended
                if not moving.any():
                    break
                ended |= step(moving, pe, re, videos=videos)
            return reached, pn, rn, ended

        def chunk_targets(chunk):
            n = env.num_envs
            t_asset = torch.zeros(n, 3, device=dev)
            r_asset = torch.eye(3, device=dev).unsqueeze(0).repeat(n, 1, 1)
            for i, idx in enumerate(chunk):
                t_asset[i] = torch.as_tensor(cands[idx]['palm_position_asset_m'], dtype=torch.float32, device=dev)
                r_asset[i] = torch.as_tensor(cands[idx]['palm_rotation_asset_from_palm'], dtype=torch.float32, device=dev)
            if chunk:
                t_asset[len(chunk):] = t_asset[0]
                r_asset[len(chunk):] = r_asset[0]
            op = obj.data.root_pos_w
            oR = matrix_from_quat(obj.data.root_quat_w)
            target_p = torch.einsum('nij,nj->ni', oR, t_asset) + op
            target_R = oR @ r_asset
            target_q = quat_from_matrix(target_R)
            back = -target_R[:, :, 0]
            return target_p, target_q, target_p + args.standoff * back, oR, op

        for start in range(0, len(indices), env.num_envs):
            chunk = indices[start:start + env.num_envs]
            live = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
            live[:len(chunk)] = True
            env.reset(seed=0)
            controller.reset()
            videos = []
            if args.video:
                fps = round(1 / env.step_dt)
                videos = [
                    RolloutVideoRecorder(args.out_dir / 'videos' / f'candidate_{idx:03d}.mp4', fps)
                    for idx in chunk
                ]
            target_p, target_q, standoff_p, _, _ = chunk_targets(chunk)
            standoff_ok, pn, rn, ended = move_to(standoff_p, target_q, args.move_steps, live, videos)
            target_ok = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
            t_pn, t_rn = pn.clone(), rn.clone()
            still = live & ~ended
            if still.any():
                target_ok, t_pn, t_rn, ended2 = move_to(
                    target_p, target_q, args.move_steps, still, videos, pos_tol=0.008)
                ended |= ended2
            pre_tilt = tilt_deg()
            pre_shift = xy_shift_cm()
            oR = matrix_from_quat(obj.data.root_quat_w)
            op = obj.data.root_pos_w
            tcp = X.tcp_pos_w(env)
            rel = torch.einsum('nji,nj->ni', oR, tcp - op)
            max_tilt = pre_tilt.clone()
            grasp_close = torch.zeros(env.num_envs, dtype=torch.long, device=dev)
            ended_close = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
            closing = live & ~ended
            for k in range(args.close_steps):
                if not closing.any():
                    break
                g = torch.full((env.num_envs,), 2.0 * (k + 1) / args.close_steps - 1.0, device=dev)
                now_ended = step(closing, grip=g, videos=videos)
                grasp_close += (closing & X.xsim_grasp_flags(env)['is_grasped']).long()
                ended_close |= now_ended
                ended |= now_ended
                closing &= ~ended
                max_tilt = torch.maximum(max_tilt, tilt_deg())
            post_shift = xy_shift_cm()
            close_max_tilt = max_tilt.clone()
            lifting = live & ~ended
            ended_lift = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
            for k in range(args.lift_steps + args.hold_steps):
                if not lifting.any():
                    break
                now_ended = step(lifting, grip=torch.ones(env.num_envs, device=dev),
                                 lift=0.5 if k < args.lift_steps else 0.0, videos=videos)
                ended_lift |= now_ended
                ended |= now_ended
                lifting &= ~ended
                max_tilt = torch.maximum(max_tilt, tilt_deg())
            grasped = X.xsim_grasp_flags(env)['is_grasped']
            metrics = getattr(env, '_xsim_state', {}).get('metrics') or {}
            if 'lift' in metrics:
                lift_cm = metrics['lift'] * 100.0
            else:
                lift_cm = (obj.data.root_pos_w[:, 2] - obj.data.default_root_state[:, 2]) * 100.0
            hold_ok = (lift_cm >= args.lift_target_cm) & grasped & live & ~ended
            final_tilt = tilt_deg()
            for i, idx in enumerate(chunk):
                cand = cands[idx]
                if ended_close[i]:
                    phase = 'close'
                elif ended_lift[i]:
                    phase = 'lift'
                else:
                    phase = None
                row = {
                    'candidate': idx, 'axis': cand['axis'], 'roll_rad': cand['roll_rad'],
                    'offset_fractions': cand.get('position_offset_principal_fractions'),
                    'standoff_reached': bool(standoff_ok[i]),
                    'standoff_err_cm': pn[i].item() * 100,
                    'standoff_rot_err_deg': rn[i].item(),
                    'target_reached': bool(target_ok[i]),
                    'target_err_cm': t_pn[i].item() * 100,
                    'target_rot_err_deg': t_rn[i].item(),
                    'pre_close_tilt_deg': pre_tilt[i].item(),
                    'pre_close_xy_shift_cm': pre_shift[i].item(),
                    'tcp_in_object_frame_cm': (rel[i] * 100).tolist(),
                    'close_grasp_steps': int(grasp_close[i]),
                    'close_max_tilt_deg': close_max_tilt[i].item(),
                    'post_close_xy_shift_cm': post_shift[i].item(),
                    'ended_early': phase,
                    'final_lift_cm': lift_cm[i].item(),
                    'final_grasp': int(grasped[i]),
                    'max_tilt_deg': max_tilt[i].item(),
                    'final_tilt_deg': final_tilt[i].item(),
                    'measured_lift_and_hold': bool(hold_ok[i]),
                }
                results.append(row)
                print(f"cand {idx:4d} axis{row['axis']} | standoff {int(row['standoff_reached'])} "
                      f"target {int(row['target_reached'])} "
                      f"| pre-close tilt {row['pre_close_tilt_deg']:5.1f} shift {row['pre_close_xy_shift_cm']:4.1f}cm "
                      f"| close grasp steps {row['close_grasp_steps']:3d} max tilt {row['max_tilt_deg']:5.1f} "
                      f"| lift {row['final_lift_cm']:5.1f}cm grasp {row['final_grasp']} "
                      f"| hold {int(row['measured_lift_and_hold'])}", flush=True)
            for rec in videos:
                rec.close()
    finally:
        if env is not None:
            env.close()
            out = args.out_dir / 'validation.json'
            out.write_text(json.dumps({
                'task': args.task,
                'object': object_name,
                'mesh': str(usd),
                'controller': args.controller,
                'num_envs': n_envs,
                'status': 'measured physics diagnostics; not named task success',
                'lift_target_cm': args.lift_target_cm, 'results': results}, indent=2))
            print('Saved:', out, flush=True)
    good = [r for r in results if r['measured_lift_and_hold']]
    print(f'{len(good)}/{len(results)} candidates lifted >= {args.lift_target_cm} cm with grasp held')
    for r in sorted(good, key=lambda r: r['max_tilt_deg'])[:10]:
        print('  best:', r['candidate'], 'max_tilt', round(r['max_tilt_deg'], 1), 'lift', round(r['final_lift_cm'], 1))


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
