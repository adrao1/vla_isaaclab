"""Object-agnostic geometric grasp proposals for the left Dex3 tripod hand.

Ported from the earlier sugar-box-only scripts (generate / expand / filter).
Everything is derived from the object's mesh vertices and the measured hand
closure geometry (outputs/grasp_geometry/hand_closure.npz, object independent).
Proposals are UNVALIDATED until executed in physics.
"""
import itertools

import numpy as np

# name -> (USD path relative to assets/YCB, upright rotation of the mesh).
# 'x+90': Y-up mesh (sugar box etc.); 'x-90': mesh up is -Y (mustard bottle, as the
# scene preview spawns it); 'none': already Z-up (dinnerware).
YCB_OBJECTS = {
    'cracker_box': ('Axis_Aligned_Physics/003_cracker_box.usd', 'x+90'),
    'sugar_box': ('Axis_Aligned_Physics/004_sugar_box.usd', 'x+90'),
    'tomato_soup_can': ('Axis_Aligned_Physics/005_tomato_soup_can.usd', 'x+90'),
    'mustard_bottle': ('Axis_Aligned_Physics/006_mustard_bottle.usd', 'x-90'),
    'bowl': ('dinnerware/024_bowl/024_bowl_physics.usd', 'none'),
}
# Paths relative to assets/. Spawn orientation comes from the scene, not this table.
XSIM_OBJECTS = {
    'xsim_mustard': 'xsim/kitchen_env/mustard/usd/mustard.usd',
}
OBJECT_NAMES = tuple(YCB_OBJECTS) + tuple(XSIM_OBJECTS)


def object_usd(project_root, name):
    """Absolute USD path for a named object asset."""
    if name in YCB_OBJECTS:
        return project_root / 'assets/YCB' / YCB_OBJECTS[name][0]
    return project_root / 'assets' / XSIM_OBJECTS[name]


def upright_quat(kind, yaw_rad):
    """wxyz quaternion: mesh turned upright first, then yawed about world z."""
    s2 = np.sqrt(0.5)
    base = {'x+90': (s2, s2, 0.0, 0.0), 'x-90': (s2, -s2, 0.0, 0.0), 'none': (1.0, 0.0, 0.0, 0.0)}[kind]
    h = yaw_rad / 2
    yw, yz = np.cos(h), np.sin(h)
    bw, bx, by, bz = base
    # yaw (w,0,0,z) * base
    return (float(yw * bw - yz * bz), float(yw * bx - yz * by),
            float(yw * by + yz * bx), float(yw * bz + yz * bw))


def load_mesh_points(usd_path):
    """All mesh vertices of a USD file, in the asset frame, in metres."""
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(usd_path))
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    cache = UsdGeom.XformCache()
    chunks = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        pts = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not pts:
            continue
        pts = np.asarray(pts, dtype=np.float64)
        m = np.array(cache.GetLocalToWorldTransform(prim)).reshape(4, 4)
        chunks.append((np.c_[pts, np.ones(len(pts))] @ m)[:, :3] * mpu)
    if not chunks:
        raise RuntimeError(f'No meshes found in {usd_path}')
    return np.concatenate(chunks)


def quat_to_matrix(q):
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _unit(v):
    return v / np.linalg.norm(v)


def generate_candidates(points, hand, margin=0.01, rolls=8, offsets=(-0.25, 0.0, 0.25)):
    """Palm poses (asset frame) that put the tripod's pad midpoint at the bbox
    centre with the closing direction along a principal axis.

    Returns (candidates, principal_spans). Order matches the earlier scripts:
    base candidates (axis, sign, roll) then 27 position offsets each.
    """
    pads = hand['pad_centers_palm_m']
    fractions = hand['fractions']
    _, axes = np.linalg.eigh(np.cov(points.T))
    local = points @ axes
    lo, hi = local.min(0), local.max(0)
    center = axes @ ((lo + hi) / 2)
    spans = hi - lo
    opposite = pads[:, 1:].mean(axis=1)
    widths = np.linalg.norm(opposite - pads[:, 0], axis=1)
    base = []
    for axis in range(3):
        eligible = np.flatnonzero(widths >= spans[axis] + 2 * margin)
        if not len(eligible):
            continue
        k = eligible[np.argmin(widths[eligible])]
        u = _unit(opposite[k] - pads[k, 0])
        v = pads[k, 1] - pads[k, 2]
        v = _unit(v - u * np.dot(u, v))
        hand_frame = np.column_stack((u, v, np.cross(u, v)))
        midpoint = (pads[k, 0] + opposite[k]) / 2
        for sign in (-1, 1):
            a = sign * axes[:, axis]
            b = axes[:, (axis + 1) % 3]
            for roll in np.linspace(0, 2 * np.pi, rolls, endpoint=False):
                v_obj = np.cos(roll) * b + np.sin(roll) * np.cross(a, b)
                target = np.column_stack((a, v_obj, np.cross(a, v_obj)))
                rotation = target @ hand_frame.T
                base.append({
                    'axis': axis, 'roll_rad': float(roll),
                    'reference_closure_fraction': float(fractions[k]),
                    'object_span_m': float(spans[axis]),
                    'palm_position_asset_m': (center - rotation @ midpoint),
                    'palm_rotation_asset_from_palm': rotation,
                })
    out = []
    for i, cand in enumerate(base):
        for fr in itertools.product(offsets, repeat=3):
            row = dict(cand)
            row['base_candidate_index'] = i
            row['position_offset_principal_fractions'] = list(fr)
            row['palm_position_asset_m'] = cand['palm_position_asset_m'] + axes @ (spans * np.array(fr))
            out.append(row)
    return out, spans


def screen_candidates(candidates, points, hand, object_rotation_w, object_position_w,
                      min_clearance=0.002, mode='hull'):
    """Open-hand vs object overlap (convex hull LP) and clearance above the
    object's bottom. Returns one row per candidate."""
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull
    boxes = hand['collision_corners_palm_m'][0]  # fully open
    links = hand['collision_links']
    object_eq = ConvexHull(points).equations
    bottom = (points @ object_rotation_w.T + object_position_w)[:, 2].min()

    def overlaps(corners):
        eq = np.vstack((object_eq, ConvexHull(corners).equations))
        res = linprog(np.zeros(3), A_ub=eq[:, :3], b_ub=-eq[:, 3],
                      bounds=[(None, None)] * 3, method='highs')
        if res.status not in (0, 2):
            raise RuntimeError(f'Intersection test failed: {res.message}')
        return res.status == 0

    rows = []
    for index, cand in enumerate(candidates):
        r, t = cand['palm_rotation_asset_from_palm'], cand['palm_position_asset_m']
        asset_boxes = boxes @ r.T + t
        if mode == 'points':  # concave objects: any mesh vertex inside a hand box
            bad = [str(link) for link, box in zip(links, asset_boxes) if _points_in_box(points, box)]
        else:
            bad = [str(link) for link, box in zip(links, asset_boxes) if overlaps(box)]
        world = asset_boxes @ object_rotation_w.T + object_position_w
        clearance = float(world[..., 2].min() - bottom)
        rows.append({'candidate_index': index, 'open_hand_overlap_links': bad,
                     'clearance_m': clearance,
                     'passes': (not bad) and clearance >= min_clearance})
    return rows


def generate_local_candidates(points, hand, margin=0.01, anchors=400, rolls=8,
                              radius=0.015, knn=30, seed=0):
    """Local-width proposals for thin or concave features (rims, handles).

    Samples surface points, estimates the normal n by local PCA, measures the
    object's thickness along n within `radius` of the normal line, and fits the
    hand aperture to that thickness. Closing direction is +-n; roll is swept.
    """
    from scipy.spatial import cKDTree
    pads = hand['pad_centers_palm_m']
    fractions = hand['fractions']
    opposite = pads[:, 1:].mean(axis=1)
    widths = np.linalg.norm(opposite - pads[:, 0], axis=1)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(points), size=min(anchors, len(points)), replace=False)
    tree = cKDTree(points)
    out = []
    for ai in pick:
        p = points[ai]
        _, nn = tree.query(p, k=knn)
        nb = points[nn]
        n = np.linalg.eigh(np.cov(nb.T))[1][:, 0]  # smallest-variance direction
        q = points - p
        along = q @ n
        radial = np.linalg.norm(q - np.outer(along, n), axis=1)
        sel = radial < radius
        if sel.sum() < 10:
            continue
        lo, hi = along[sel].min(), along[sel].max()
        thickness = hi - lo
        eligible = np.flatnonzero(widths >= thickness + 2 * margin)
        if not len(eligible):
            continue
        k = eligible[np.argmin(widths[eligible])]
        mid = p + n * (lo + hi) / 2
        u = _unit(opposite[k] - pads[k, 0])
        v = pads[k, 1] - pads[k, 2]
        v = _unit(v - u * np.dot(u, v))
        hand_frame = np.column_stack((u, v, np.cross(u, v)))
        midpoint = (pads[k, 0] + opposite[k]) / 2
        b0 = _unit(np.cross(n, [1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.cross(n, [0.0, 1.0, 0.0]))
        for sign in (-1, 1):
            a = sign * n
            for roll in np.linspace(0, 2 * np.pi, rolls, endpoint=False):
                v_obj = np.cos(roll) * b0 + np.sin(roll) * np.cross(a, b0)
                target = np.column_stack((a, v_obj, np.cross(a, v_obj)))
                rotation = target @ hand_frame.T
                out.append({
                    'axis': -1, 'roll_rad': float(roll), 'anchor_index': int(ai),
                    'local_thickness_m': float(thickness),
                    'reference_closure_fraction': float(fractions[k]),
                    'palm_position_asset_m': mid - rotation @ midpoint,
                    'palm_rotation_asset_from_palm': rotation,
                })
    return out


def _points_in_box(points, corners, tol=0.001):
    from scipy.spatial import ConvexHull
    eq = ConvexHull(corners).equations
    return bool(np.any(np.all(points @ eq[:, :3].T + eq[:, 3] < -tol, axis=1)))
