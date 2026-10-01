#!/usr/bin/env python3
"""Capture the actual Isaac Sim scene with visual-only waypoint markers."""
import argparse
import json
import struct
import sys
import zlib
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--task', default='VLA-YCBSugarBox-G1-Waypoint-v0')
parser.add_argument('--output', type=Path)
parser.add_argument('--camera-eye', type=float, nargs=3, metavar=('X', 'Y', 'Z'))
parser.add_argument('--camera-target', type=float, nargs=3, metavar=('X', 'Y', 'Z'))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
args.experience = str(ROOT / 'configs' / ('ycb.python.headless.rendering.kit' if args.headless else 'ycb.python.rendering.kit'))
output = args.output or ROOT / 'outputs/waypoints' / datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
output.mkdir(parents=True, exist_ok=False)
args.kit_args = f'--portable-root {output.resolve()}/kit'
app = AppLauncher(args).app

import numpy as np
import torch
import gymnasium as gym
import vla_isaaclab
import omni.usd
from pxr import Gf, UsdGeom
from isaaclab_tasks.utils import parse_env_cfg
from vla_isaaclab.envs.common import camera_cfg

MARKER_COLORS = [
    (0.7, 0.7, 0.7),
    (0.05, 0.55, 1.),
    (0.75, 0.2, 0.9),
    (1., 0.55, 0.05),
    (0.1, 1., 0.25),
    (1., 0.2, 0.2),
]
MUSTARD_DEMO = ROOT / 'assets/xsim/kitchen_env/mustard/demos/mustard_rl.npy'
# X-Sim Mustard-Place metadata.npz: tracking sites after pop(start).
MUSTARD_NUM_WAYPOINTS = 5
MUSTARD_FIRST_WAYPOINT_IDX = 89
MUSTARD_LAST_WAYPOINT_IDX = 205


def mustard_demo_frames():
    interval = (MUSTARD_LAST_WAYPOINT_IDX - MUSTARD_FIRST_WAYPOINT_IDX) // MUSTARD_NUM_WAYPOINTS
    frames = list(range(MUSTARD_FIRST_WAYPOINT_IDX, MUSTARD_LAST_WAYPOINT_IDX - interval, interval))
    frames.append(MUSTARD_LAST_WAYPOINT_IDX)
    return frames


def png(path, rgb):
    """Write RGB PNG without adding an image-library dependency."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = rgb.shape
    def chunk(kind, payload):
        return struct.pack('!I', len(payload)) + kind + payload + struct.pack('!I', zlib.crc32(kind + payload) & 0xffffffff)
    raw = b''.join(b'\0' + row.tobytes() for row in rgb)
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', w, h, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


def _as_local(env, xyz):
    if not torch.is_tensor(xyz):
        xyz = torch.tensor(xyz, device=env.device, dtype=torch.float32)
    return xyz.to(env.device, dtype=torch.float32).reshape(3)


def waypoint_points(env, task):
    """Local object-center path: command waypoints, else a scene demo, else spawn."""
    if 'target_pose' in env.command_manager.active_terms:
        cmd = env.command_manager.get_term('target_pose')
        named = [(name, getattr(cmd, name)) for name in ('start', 'lift', 'mid', 'hover', 'destination') if hasattr(cmd, name)]
        if len(named) >= 2:
            return [(name, _as_local(env, pos)) for name, pos in named]
    if 'MustardPlace' in task and MUSTARD_DEMO.is_file():
        traj = np.load(MUSTARD_DEMO)[0, :, :3]
        return [(f'wp{i}', _as_local(env, traj[idx])) for i, idx in enumerate(mustard_demo_frames())]
    spawn = env.scene['object'].data.root_pos_w[0] - env.scene.env_origins[0]
    return [('start', _as_local(env, spawn))]


def main():
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    cfg.seed = 0
    for name in ('camera', 'cam_left_high', 'cam_left_wrist', 'cam_right_wrist'):
        if hasattr(cfg.scene, name):
            setattr(cfg.scene, name, None)
    if args.camera_eye and args.camera_target:
        cfg.scene.cam_side = camera_cfg(tuple(args.camera_eye), tuple(args.camera_target))
    elif args.task == 'VLA-YCBSugarBox-G1-Waypoint-v0':
        cfg.scene.cam_side = camera_cfg((0.15, 0.45, 1.20), (-0.14, -0.29, 0.74))
    elif 'MustardPlace' in args.task:
        # Offset of the X-Sim camera: open side of the kitchen, G1 not in front.
        cfg.scene.cam_side = camera_cfg((-0.55, -1.25, 1.35), (0.48, 0.02, 0.30))
    elif not hasattr(cfg.scene, 'cam_side') or cfg.scene.cam_side is None:
        cfg.scene.cam_side = camera_cfg(
            getattr(cfg, 'camera_eye', (0.35, 1.90, 1.85)),
            getattr(cfg, 'camera_target', (-0.15, -0.30, 0.72)),
        )
    cfg.scene.cam_side.width = 1280
    cfg.scene.cam_side.height = 960
    env = gym.make(args.task, cfg=cfg).unwrapped
    try:
        env.reset(seed=0)
        origin = env.scene.env_origins[0]
        named = waypoint_points(env, args.task)
        names = [item[0] for item in named]
        points = [item[1] for item in named]
        colors = MARKER_COLORS[:len(named)]
        world = [(p + origin).cpu().tolist() for p in points]
        stage = omni.usd.get_context().get_stage()
        for name, point, color in zip(names, world, colors):
            sphere = UsdGeom.Sphere.Define(stage, '/World/WaypointPicture/' + name)
            sphere.CreateRadiusAttr(0.012)
            sphere.AddTranslateOp().Set(Gf.Vec3d(*point))
            sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        # No collision or rigid-body APIs: markers never participate in physics.
        if len(world) >= 2:
            line = UsdGeom.BasisCurves.Define(stage, '/World/WaypointPicture/path')
            line.CreateTypeAttr('linear')
            line.CreateCurveVertexCountsAttr([len(world)])
            line.CreatePointsAttr([Gf.Vec3f(*p) for p in world])
            line.CreateWidthsAttr([0.004])
            line.SetWidthsInterpolation('constant')
            line.CreateDisplayColorAttr([Gf.Vec3f(0.9, 0.9, 0.9)])
        camera = env.scene['cam_side']
        def capture(filename):
            # Render only. No env.step: robot/object states stay at reset.
            for _ in range(12):
                env.sim.render()
            camera.update(env.step_dt, force_recompute=True)
            rgb = camera.data.output['rgb'][0, ..., :3].detach().cpu().numpy()
            png(output / filename, rgb)
            print('Saved:', (output / filename).resolve(), flush=True)
        capture('waypoints_scene.png')
        def hide(prim_path, filename):
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                print('Skip hide, missing prim:', prim_path, flush=True)
                return
            imageable = UsdGeom.Imageable(prim)
            previous = imageable.GetVisibilityAttr().Get()
            try:
                imageable.MakeInvisible()
                capture(filename)
            finally:
                imageable.GetVisibilityAttr().Set(previous or UsdGeom.Tokens.inherited)
        hide('/World/envs/env_0/Robot', 'waypoints_robot_hidden.png')
        hide('/World/envs/env_0/Object', 'waypoints_object_hidden.png')
        mustard_frames = mustard_demo_frames() if 'MustardPlace' in args.task and MUSTARD_DEMO.is_file() else None
        markers = {}
        for i, (name, p, color) in enumerate(zip(names, points, colors)):
            entry = {'local_position_m': p.cpu().tolist(), 'color_rgb': color}
            if mustard_frames is not None:
                entry['demo_frame'] = mustard_frames[i]
            markers[name] = entry
        (output / 'waypoints.json').write_text(json.dumps({
            'task': args.task,
            'description': 'Actual simulator renders. Hidden images are visibility-only; physics is unchanged. Lines show desired object-center path, not a measured trajectory.',
            'mustard_demo_frames': mustard_frames,
            'markers': markers,
        }, indent=2))
        print('Markers:', ', '.join(names), flush=True)
    finally:
        env.close()

if __name__ == '__main__':
    try:
        with torch.inference_mode():
            main()
    except BaseException:
        import traceback
        traceback.print_exc()
        raise
    finally:
        app.close()
