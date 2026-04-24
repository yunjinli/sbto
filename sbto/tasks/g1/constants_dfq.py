"""
Constants for G1 29-DOF body + Inspire Hand DFQ (12 DOF per hand = 24 total).

Joint ordering mirrors the actuator declaration order in g1_mjx_with_dfq_hands.xml:
  [0-5]   left leg
  [6-11]  right leg
  [12-14] waist (yaw, roll, pitch)
  [15-21] left arm  (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw)
  [22-28] right arm (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw)
  [29-40] left DFQ hand  (thumb yaw/pitch/inter/distal, index/middle/ring/pinky prox+inter)
  [41-52] right DFQ hand (same order)
"""

from sbto.tasks.g1.constants import (
    N_FEET, N_HANDS,
    _cnt_sens_per_foot, _cnt_sens_dim_per_foot,
    _cnt_sens_per_hand, _cnt_sens_dim_per_hand,
    Sensors as _SensorsBase,
)

# ── Finger joint name lists (ordered as in the XML actuator section) ─────────

_LEFT_HAND_DFQ_JOINTS = [
    "L_thumb_proximal_yaw_joint",
    "L_thumb_proximal_pitch_joint",
    "L_thumb_intermediate_joint",
    "L_thumb_distal_joint",
    "L_index_proximal_joint",
    "L_index_intermediate_joint",
    "L_middle_proximal_joint",
    "L_middle_intermediate_joint",
    "L_ring_proximal_joint",
    "L_ring_intermediate_joint",
    "L_pinky_proximal_joint",
    "L_pinky_intermediate_joint",
]

_RIGHT_HAND_DFQ_JOINTS = [
    "R_thumb_proximal_yaw_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_intermediate_joint",
    "R_thumb_distal_joint",
    "R_index_proximal_joint",
    "R_index_intermediate_joint",
    "R_middle_proximal_joint",
    "R_middle_intermediate_joint",
    "R_ring_proximal_joint",
    "R_ring_intermediate_joint",
    "R_pinky_proximal_joint",
    "R_pinky_intermediate_joint",
]

NDOF_HAND = len(_LEFT_HAND_DFQ_JOINTS)   # 12 per hand


# ── Sensors ──────────────────────────────────────────────────────────────────

class Sensors(_SensorsBase):
    """Extends the base Sensors with DFQ fingertip tracking names."""

    # Fingertip position sensor names (declared in sensors/hands_dfq.xml)
    LEFT_FINGERTIP_POS = [
        "L_thumb_tip_pos",
        "L_index_tip_pos",
        "L_middle_tip_pos",
        "L_ring_tip_pos",
        "L_pinky_tip_pos",
    ]
    RIGHT_FINGERTIP_POS = [
        "R_thumb_tip_pos",
        "R_index_tip_pos",
        "R_middle_tip_pos",
        "R_ring_tip_pos",
        "R_pinky_tip_pos",
    ]

    TORQUES = (
        _SensorsBase.TORQUES          # first 23 body joints
        + [
            # Waist roll + pitch (completing the full 29-DOF body actuator set)
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
        + _LEFT_HAND_DFQ_JOINTS
        + _RIGHT_HAND_DFQ_JOINTS
    )


# ── DOF layout ────────────────────────────────────────────────────────────────

class _29DoF_DFQ:
    """Full G1 29-DOF body + 24-DOF DFQ hands = 53 actuated DOF."""

    NDOF_BODY = 29          # G1 body joints
    NDOF_HAND_PER_SIDE = NDOF_HAND   # 12
    NDOF_HANDS = 2 * NDOF_HAND       # 24
    NDOF_G1 = NDOF_BODY + NDOF_HANDS  # 53

    NQ_G1 = 7 + NDOF_G1    # 60  (7 free-joint coords + 53 joint angles)
    NV_G1 = 6 + NDOF_G1    # 59  (6 free-joint vel + 53 joint velocities)
    iNV_G1 = NQ_G1          # index where qvel for joints starts (in full state)

    # qpos/qvel index slices (full 53-joint block starts at qpos[7])
    IDX_JOINT_POS = list(range(7, 7 + NDOF_G1))
    IDX_JOINT_VEL = list(range(iNV_G1, iNV_G1 + NDOF_G1))

    # ── Sub-group indices: 0-based offsets WITHIN qpos[7:60] / qvel[6:59] ──
    # qpos layout (MuJoCo DFS tree order):
    #   [0:6]   left leg        (6)
    #   [6:12]  right leg       (6)
    #   [12:15] waist           (3)
    #   [15:22] left arm        (7)
    #   [22:34] left hand DFQ  (12)
    #   [34:41] right arm       (7)
    #   [41:53] right hand DFQ (12)
    IDX_LEFT_LEG    = list(range(0, 6))
    IDX_RIGHT_LEG   = list(range(6, 12))
    IDX_WAIST       = list(range(12, 15))
    IDX_LEFT_ARM    = list(range(15, 22))
    IDX_LEFT_HAND   = list(range(22, 34))
    IDX_RIGHT_ARM   = list(range(34, 41))
    IDX_RIGHT_HAND  = list(range(41, 53))

    # Body-only joints (non-contiguous — excludes hand indices)
    IDX_BODY_JOINTS = IDX_LEFT_LEG + IDX_RIGHT_LEG + IDX_WAIST + IDX_LEFT_ARM + IDX_RIGHT_ARM

    # Convenience sub-groups matching _25DoF interface
    IDX_HIP_KNEE        = [0, 3, 6, 9]
    IDX_SHOULDER_PITCH  = [15, 22]
    IDX_WAIST_YAW       = 12

    # Joint ranges for the body DOFs (same as _25DoF but extended to 29)
    RESTRICTED_JOINT_RANGE_BODY = (
        # Left leg
        (-1.57, 1.57), (-0.5, 0.5), (-0.5, 0.5), (0, 1.57), (-0.5, 0.7), (-0.2, 0.2),
        # Right leg
        (-1.57, 1.57), (-0.5, 0.5), (-0.5, 0.5), (0, 1.57), (-0.5, 0.7), (-0.2, 0.2),
        # Waist
        (-0.5, 0.5), (-0.52, 0.52), (-0.52, 0.52),
        # Left arm
        (-1.57, 1.57), (-0.2, 1.57), (-1.0, 1.0), (-1.0, 1.57), (-1.0, 1.0),
        (-1.61, 1.61), (-1.61, 1.61),
        # Right arm
        (-1.57, 1.57), (-1.57, 0.2), (-1.0, 1.0), (-1.0, 1.57), (-1.0, 1.0),
        (-1.61, 1.61), (-1.61, 1.61),
    )

    # Finger ranges (per hand, same for left and right)
    RESTRICTED_JOINT_RANGE_HAND = (
        (-0.1, 1.3),   # thumb yaw
        (-0.1, 0.6),   # thumb proximal pitch
        (0.0, 0.8),    # thumb intermediate
        (0.0, 1.2),    # thumb distal
        (0.0, 1.7),    # index proximal
        (0.0, 1.7),    # index intermediate
        (0.0, 1.7),    # middle proximal
        (0.0, 1.7),    # middle intermediate
        (0.0, 1.7),    # ring proximal
        (0.0, 1.7),    # ring intermediate
        (0.0, 1.7),    # pinky proximal
        (0.0, 1.7),    # pinky intermediate
    )

    RESTRICTED_JOINT_RANGE = (
        RESTRICTED_JOINT_RANGE_BODY
        + RESTRICTED_JOINT_RANGE_HAND   # left hand
        + RESTRICTED_JOINT_RANGE_HAND   # right hand
    )


class _29DoF_DFQ_Obj(_29DoF_DFQ):
    """Same as _29DoF_DFQ but with an additional free-floating object in the scene."""

    iNV_G1 = _29DoF_DFQ.NQ_G1 + 7    # NQ + 7 for object free joint

    IDX_JOINT_POS = list(range(7, 7 + _29DoF_DFQ.NDOF_G1))
    IDX_JOINT_VEL = list(range(iNV_G1, iNV_G1 + _29DoF_DFQ.NDOF_G1))

    IDX_BOX_POS   = list(range(_29DoF_DFQ.NQ_G1, _29DoF_DFQ.NQ_G1 + 3))
    IDX_BOX_QUAT  = list(range(_29DoF_DFQ.NQ_G1 + 3, _29DoF_DFQ.NQ_G1 + 7))

    IDX_BOX_LINVEL = list(range(iNV_G1 + _29DoF_DFQ.NV_G1, iNV_G1 + _29DoF_DFQ.NV_G1 + 3))
    IDX_BOX_ANGVEL = list(range(iNV_G1 + _29DoF_DFQ.NV_G1 + 3, iNV_G1 + _29DoF_DFQ.NV_G1 + 6))
