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
parser.add_argument('--output', type=Path)
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


def png(path, rgb):
    """Write RGB PNG without adding an image-library dependency."""
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    h, w, _ = rgb.shape
    def chunk(kind, payload):
        return struct.pack('!I', len(payload)) + kind + payload + struct.pack('!I', zlib.crc32(kind + payload) & 0xffffffff)
    raw = b''.join(b'\0' + row.tobytes() for row in rgb)
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', w, h, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


def main():
    cfg = parse_env_cfg('VLA-YCBSugarBox-G1-Waypoint-v0', device=args.device, num_envs=1)
    cfg.seed = 0
    for name in ('camera', 'cam_left_high', 'cam_left_wrist', 'cam_right_wrist'):
        if hasattr(cfg.scene, name):
            setattr(cfg.scene, name, None)
    cfg.scene.cam_side.width = 1280
    cfg.scene.cam_side.height = 960
    env = gym.make('VLA-YCBSugarBox-G1-Waypoint-v0', cfg=cfg).unwrapped
    try:
        env.reset(seed=0)
        cmd = env.command_manager.get_term('target_pose')
        origin = env.scene.env_origins[0]
        start = cmd.start.clone()
        lift = start + start.new_tensor((0., 0., 0.06))
        transfer = cmd.destination + start.new_tensor((0., 0., 0.06))
        destination = cmd.destination.clone()
        points = [start, lift, transfer, destination]
        world = [(p + origin).cpu().tolist() for p in points]
        stage = omni.usd.get_context().get_stage()
        colors = [(0.7, 0.7, 0.7), (0.05, 0.55, 1.), (1., 0.55, 0.05), (0.1, 1., 0.25)]
        names = ['start', 'lift_blue', 'transfer_orange', 'placement_release_green']
        for name, point, color in zip(names, world, colors):
            sphere = UsdGeom.Sphere.Define(stage, '/World/WaypointPicture/' + name)
            sphere.CreateRadiusAttr(0.004)
            sphere.AddTranslateOp().Set(Gf.Vec3d(*point))
            sphere.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        # No collision or rigid-body APIs: markers never participate in physics.
        line = UsdGeom.BasisCurves.Define(stage, '/World/WaypointPicture/path')
        line.CreateTypeAttr('linear')
        line.CreateCurveVertexCountsAttr([4])
        line.CreatePointsAttr([Gf.Vec3f(*p) for p in world])
        line.CreateWidthsAttr([0.0015])
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
        obj = stage.GetPrimAtPath('/World/envs/env_0/Object')
        if not obj.IsValid():
            raise RuntimeError('Cannot find object prim for visibility-only capture')
        imageable = UsdGeom.Imageable(obj)
        previous = imageable.GetVisibilityAttr().Get()
        try:
            imageable.MakeInvisible()
            capture('waypoints_box_hidden.png')
        finally:
            imageable.GetVisibilityAttr().Set(previous or UsdGeom.Tokens.inherited)
        (output / 'waypoints.json').write_text(json.dumps({
            'description': 'Actual simulator renders. Second image hides only box geometry; physics is unchanged. Lines show desired box-center path, not a measured trajectory.',
            'markers': {name: {'local_position_m': p.cpu().tolist(), 'color_rgb': color} for name, p, color in zip(names, points, colors)},
        }, indent=2))
        print('Blue = lift; orange = transfer; green = placement/release; gray = start.', flush=True)
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
