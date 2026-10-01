#!/usr/bin/env python3
"""Save chosen, physics-validated grasp candidates as a small JSON library.

Poses are regenerated from the object mesh (deterministic), not typed in, and
carry their measured validation results from every listed run. Needs the Isaac
app only for pxr (mesh loading); no simulation is run.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'rl'))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--object', required=True)
parser.add_argument('--indices', type=int, nargs='+', required=True)
parser.add_argument('--validations', type=Path, nargs='+', required=True,
                    help='validation.json files (original + repeats) holding these candidates')
parser.add_argument('--hand-folder', type=Path, default=ROOT / 'outputs/grasp_geometry')
parser.add_argument('--margin', type=float, default=0.01)
parser.add_argument('--offsets', type=float, nargs='+', default=[-0.25, 0.0, 0.25])
parser.add_argument('--out', type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False
args.experience = str(ROOT / 'configs/ycb.python.headless.kit')
app = AppLauncher(args).app


def main():
    import numpy as np
    import grasp_geometry as G

    rel_usd, orient = G.YCB_OBJECTS[args.object]
    points = G.load_mesh_points(ROOT / 'assets/YCB' / rel_usd)
    hand = np.load(args.hand_folder / 'hand_closure.npz')
    cands, _ = G.generate_candidates(points, hand, margin=args.margin, offsets=tuple(args.offsets))

    runs = {}
    for path in args.validations:
        for row in json.loads(path.read_text())['results']:
            runs.setdefault(row['candidate'], []).append({
                'source': str(path.relative_to(ROOT)) if path.is_absolute() else str(path),
                'measured_lift_and_hold': row['measured_lift_and_hold'],
                'final_lift_cm': row['final_lift_cm'],
                'max_tilt_deg': row['max_tilt_deg'],
                'close_grasp_steps': row['close_grasp_steps'],
            })
    grasps = []
    for idx in args.indices:
        c = cands[idx]
        grasps.append({
            'candidate': idx, 'axis': c['axis'], 'roll_rad': c['roll_rad'],
            'position_offset_principal_fractions': c['position_offset_principal_fractions'],
            'palm_position_asset_m': np.asarray(c['palm_position_asset_m']).tolist(),
            'palm_rotation_asset_from_palm': np.asarray(c['palm_rotation_asset_from_palm']).tolist(),
            'validation_runs': runs.get(idx, []),
        })
    out = args.out or (ROOT / 'assets/grasps' / f'{args.object}.json')
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        'object': args.object,
        'status': 'measured physics diagnostics; not named task success',
        'frame': 'palm_position_asset_m / palm_rotation_asset_from_palm: left_hand_palm_link pose in the object root frame',
        'generation': {'margin_m': args.margin, 'offsets': args.offsets, 'script': 'rl/grasp_geometry.py'},
        'grasps': grasps,
    }, indent=1))
    print('Saved', out, 'with', len(grasps), 'grasps', flush=True)


if __name__ == '__main__':
    try:
        main()
    finally:
        app.close()
