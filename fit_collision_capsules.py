"""
Read DFQ hand STL meshes, fit capsules via PCA, and print corrected
fromto + size values to replace the hand collision geoms in
g1_mjx_with_dfq_hands.xml.

For palm (hand_base) meshes the visual geom has an explicit pos/quat offset
that is applied before fitting.

Run from repo root:
  pixi run python fit_collision_capsules.py
"""

import os
import numpy as np

ASSETS = "sbto/models/unitree_g1/assets"


# ---------------------------------------------------------------------------
# STL reader (binary + ASCII fallback)
# ---------------------------------------------------------------------------

def read_stl(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        header = f.read(80)
        raw = f.read()

    # ASCII STL starts with "solid"
    if header[:5].lower() == b"solid":
        verts = []
        for line in (header + raw).decode("utf-8", errors="ignore").splitlines():
            line = line.strip()
            if line.startswith("vertex"):
                verts.append([float(x) for x in line.split()[1:4]])
        if verts:
            return np.array(verts, dtype=np.float64)

    # Binary STL
    n_tri = np.frombuffer(raw[:4], dtype=np.uint32)[0]
    buf = np.frombuffer(raw[4:], dtype=np.uint8)
    verts = []
    for i in range(n_tri):
        off = i * 50          # 12 normal + 3*12 verts + 2 attr
        for j in range(3):
            v = np.frombuffer(buf[off + 12 + j * 12: off + 12 + j * 12 + 12],
                              dtype=np.float32).astype(np.float64)
            verts.append(v.copy())
    return np.array(verts)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def quat_to_rot(q_wxyz) -> np.ndarray:
    """MuJoCo wxyz quaternion → 3×3 rotation matrix."""
    w, x, y, z = q_wxyz
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)    ],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)    ],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def fit_capsule(verts: np.ndarray, radius_percentile: float = 90.0):
    """
    Fit a capsule to a point cloud via PCA.
    Returns (pt1, pt2, radius_p90, radius_max, bbox_short_r).
    pt1/pt2: capsule axis endpoints
    radius_p90: pct-ile of radial distances (filters connector/flange outliers)
    radius_max: full bounding radius
    bbox_short_r: half the shorter bbox cross-section (tightest estimate)
    """
    center = verts.mean(axis=0)
    c = verts - center
    _, _, Vt = np.linalg.svd(c, full_matrices=False)
    axis = Vt[0]                         # first principal component

    projs = c @ axis
    pt1 = center + axis * projs.min()
    pt2 = center + axis * projs.max()

    # Radial distance from every vertex to the axis line
    along = np.outer(projs, axis)
    radials = np.linalg.norm(c - along, axis=1)
    radius_max = radials.max()
    radius_pct = float(np.percentile(radials, radius_percentile))

    # Bbox-based: half the shorter cross-sectional dimension
    bb_min = verts.min(axis=0)
    bb_max = verts.max(axis=0)
    bb_size = bb_max - bb_min
    # Sort by size: longest = capsule axis, shorter two = cross-section
    sorted_dims = np.sort(bb_size)
    bbox_short_r = sorted_dims[1] / 2.0   # second-largest = wider cross-section half

    return pt1, pt2, radius_pct, radius_max, bbox_short_r


def f3(v):
    return f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f}"


# ---------------------------------------------------------------------------
# Link table
# Each entry: geom_col_name, stl_file, geom_pos_in_body, geom_quat_wxyz
# pos/quat = None means the visual geom sits at body origin with no rotation.
# ---------------------------------------------------------------------------

LINKS = [
    # ── Left palm ──────────────────────────────────────────────────────────
    # visual geom: pos="0.0415 0 0"  quat="0.707107 0 0 0.707107"
    ("left_hand_collision",  "L_hand_base_link.STL",
     [0.0415, 0.0, 0.0], [0.707107, 0.0, 0.0, 0.707107]),

    # ── Left thumb ─────────────────────────────────────────────────────────
    ("L_thumb_base_col",  "Link11_L.STL", None, None),
    ("L_thumb_prox_col",  "Link12_L.STL", None, None),
    ("L_thumb_inter_col", "Link13_L.STL", None, None),
    ("L_thumb_dist_col",  "Link14_L.STL", None, None),

    # ── Left index ─────────────────────────────────────────────────────────
    ("L_index_prox_col",  "Link15_L.STL", None, None),
    ("L_index_inter_col", "Link16_L.STL", None, None),

    # ── Left middle ────────────────────────────────────────────────────────
    ("L_middle_prox_col",  "Link17_L.STL", None, None),
    ("L_middle_inter_col", "Link18_L.STL", None, None),

    # ── Left ring ──────────────────────────────────────────────────────────
    ("L_ring_prox_col",  "Link19_L.STL", None, None),
    ("L_ring_inter_col", "Link20_L.STL", None, None),

    # ── Left pinky ─────────────────────────────────────────────────────────
    ("L_pinky_prox_col",  "Link21_L.STL", None, None),
    ("L_pinky_inter_col", "Link22_L.STL", None, None),

    # ── Right palm ─────────────────────────────────────────────────────────
    # visual geom: pos="0.0415 0 0"  quat="0 0.707107 -0.707107 0"
    ("right_hand_collision", "R_hand_base_link.STL",
     [0.0415, 0.0, 0.0], [0.0, 0.707107, -0.707107, 0.0]),

    # ── Right thumb ────────────────────────────────────────────────────────
    ("R_thumb_base_col",  "Link11_R.STL", None, None),
    ("R_thumb_prox_col",  "Link12_R.STL", None, None),
    ("R_thumb_inter_col", "Link13_R.STL", None, None),
    ("R_thumb_dist_col",  "Link14_R.STL", None, None),

    # ── Right index ────────────────────────────────────────────────────────
    ("R_index_prox_col",  "Link15_R.STL", None, None),
    ("R_index_inter_col", "Link16_R.STL", None, None),

    # ── Right middle ───────────────────────────────────────────────────────
    ("R_middle_prox_col",  "Link17_R.STL", None, None),
    ("R_middle_inter_col", "Link18_R.STL", None, None),

    # ── Right ring ─────────────────────────────────────────────────────────
    ("R_ring_prox_col",  "Link19_R.STL", None, None),
    ("R_ring_inter_col", "Link20_R.STL", None, None),

    # ── Right pinky ────────────────────────────────────────────────────────
    ("R_pinky_prox_col",  "Link21_R.STL", None, None),
    ("R_pinky_inter_col", "Link22_R.STL", None, None),
]


# ---------------------------------------------------------------------------
# Current values from XML (for comparison)
# ---------------------------------------------------------------------------

CURRENT = {
    "left_hand_collision":  ('type="cylinder"', 'fromto="0 0 0 0.0415 0 0"',   'size="0.02"'),
    "right_hand_collision": ('type="cylinder"', 'fromto="0 0 0 0.0415 0 0"',   'size="0.02"'),
    "L_thumb_base_col":     ('',                'fromto="0 0 0 0.010 0.010 -0.009"',  'size="0.010"'),
    "L_thumb_prox_col":     ('',                'fromto="0 0 0 0.044 -0.035 -0.001"', 'size="0.010"'),
    "L_thumb_inter_col":    ('',                'fromto="0 0 0 0.020 -0.010 -0.001"', 'size="0.009"'),
    "L_thumb_dist_col":     ('',                'fromto="0 0 0 0.022 0 0"',           'size="0.009"'),
    "L_index_prox_col":     ('',                'fromto="0 0 0 -0.002 -0.032 -0.001"','size="0.010"'),
    "L_index_inter_col":    ('',                'fromto="0 0 0 0 -0.022 0"',          'size="0.009"'),
    "L_middle_prox_col":    ('',                'fromto="0 0 0 -0.002 -0.032 -0.001"','size="0.010"'),
    "L_middle_inter_col":   ('',                'fromto="0 0 0 0 -0.024 0"',          'size="0.009"'),
    "L_ring_prox_col":      ('',                'fromto="0 0 0 -0.002 -0.032 -0.001"','size="0.010"'),
    "L_ring_inter_col":     ('',                'fromto="0 0 0 0 -0.022 0"',          'size="0.009"'),
    "L_pinky_prox_col":     ('',                'fromto="0 0 0 -0.002 -0.032 -0.001"','size="0.010"'),
    "L_pinky_inter_col":    ('',                'fromto="0 0 0 0 -0.019 0"',          'size="0.009"'),
    "R_thumb_base_col":     ('',                'fromto="0 0 0 -0.009 0.011 -0.009"', 'size="0.010"'),
    "R_thumb_prox_col":     ('',                'fromto="0 0 0 0.044 0.035 -0.001"',  'size="0.010"'),
    "R_thumb_inter_col":    ('',                'fromto="0 0 0 0.020 0.010 -0.001"',  'size="0.009"'),
    "R_thumb_dist_col":     ('',                'fromto="0 0 0 0.022 0 0"',           'size="0.009"'),
    "R_index_prox_col":     ('',                'fromto="0 0 0 -0.003 0.032 -0.001"', 'size="0.010"'),
    "R_index_inter_col":    ('',                'fromto="0 0 0 0 0.022 0"',           'size="0.009"'),
    "R_middle_prox_col":    ('',                'fromto="0 0 0 -0.002 0.032 -0.001"', 'size="0.010"'),
    "R_middle_inter_col":   ('',                'fromto="0 0 0 0 0.024 0"',           'size="0.009"'),
    "R_ring_prox_col":      ('',                'fromto="0 0 0 -0.002 0.032 -0.001"', 'size="0.010"'),
    "R_ring_inter_col":     ('',                'fromto="0 0 0 0 0.022 0"',           'size="0.009"'),
    "R_pinky_prox_col":     ('',                'fromto="0 0 0 -0.002 0.032 -0.001"', 'size="0.010"'),
    "R_pinky_inter_col":    ('',                'fromto="0 0 0 0 0.019 0"',           'size="0.009"'),
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("DFQ hand collision capsule fit from STL meshes")
    print("=" * 70)
    print()

    results = {}

    for (geom_name, stl_file, pos, quat) in LINKS:
        path = os.path.join(ASSETS, stl_file)
        if not os.path.exists(path):
            print(f"[MISSING] {geom_name}: {stl_file}")
            continue

        verts = read_stl(path)

        # Apply visual geom transform (pos/quat) to get body-frame vertices
        if pos is not None and quat is not None:
            R = quat_to_rot(quat)
            verts = (R @ verts.T).T + np.array(pos)

        pt1, pt2, r_p90, r_max, r_bbox = fit_capsule(verts)
        # Recommended: min of p90 and bbox, rounded up 0.5mm
        r_rec = round(min(r_p90, r_bbox) + 0.0005, 3)

        bb_min = verts.min(axis=0)
        bb_max = verts.max(axis=0)
        bb_size = bb_max - bb_min

        results[geom_name] = (pt1, pt2, r_rec)

        cur = CURRENT.get(geom_name, ("", "", ""))
        print(f"── {geom_name}  ({stl_file})")
        print(f"   bbox: [{f3(bb_min)}] → [{f3(bb_max)}]  size=[{f3(bb_size)}]")
        print(f"   CURRENT : {cur[1]}  {cur[2]}")
        print(f"   r_p90={r_p90:.4f}  r_max={r_max:.4f}  r_bbox={r_bbox:.4f}  → REC={r_rec:.4f}")
        print(f"   FITTED  : fromto=\"{f3(pt1)}   {f3(pt2)}\"  size=\"{r_rec:.4f}\"")
        print()

    # -----------------------------------------------------------------------
    # Print XML snippet for copy-paste (finger geoms only, not palm which
    # may need special handling)
    # -----------------------------------------------------------------------
    print()
    print("=" * 70)
    print("Suggested XML geom replacements (copy into g1_mjx_with_dfq_hands.xml)")
    print("=" * 70)
    print()

    palm_names = {"left_hand_collision", "right_hand_collision"}
    for (geom_name, _, _, _) in LINKS:
        if geom_name not in results:
            continue
        pt1, pt2, radius = results[geom_name]
        # Round radius up slightly for safety margin
        r = round(radius + 0.001, 3)

        if geom_name in palm_names:
            print(f'<!-- {geom_name}: fitted palm capsule -->')
            print(f'<geom name="{geom_name}" class="collision" type="capsule" size="{r:.3f}" rgba=".2 .6 .2 .2"')
            print(f'     fromto="{f3(pt1)}   {f3(pt2)}"/>')
        else:
            print(f'<geom name="{geom_name}" class="collision" fromto="{f3(pt1)}   {f3(pt2)}" size="{r:.3f}"/>')
        print()


if __name__ == "__main__":
    main()
