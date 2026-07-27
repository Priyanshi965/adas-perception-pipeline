"""
body_language.py — Derive interpretable body-posture features from pose keypoints.

The crossing-intent model needs signals the bounding box cannot provide:
whether a pedestrian is *leaning* toward the road, has their *head turned* to
check traffic, has a *foot advanced* off the curb, or is mid-*stride*. These are
the cues a human driver reads. This module turns a 17-keypoint COCO skeleton
into a fixed-length vector of such cues.

Design rules
────────────
1. Scale-invariant: every pixel distance is divided by the person's bbox height,
   so a pedestrian near the camera and one far away produce comparable features.
2. Ego-lateral: features that have a left/right sign (lean, head turn, leading
   foot) are optionally mirrored by `mirror=True` so a pedestrian approaching
   from the left and one from the right map to the same representation. The
   caller (dataset builder) decides the sign from the track's horizontal
   velocity; positive x is defined as "toward the road / crossing direction".
3. Robust to missing joints: low-confidence keypoints are masked out and their
   dependent features fall back to a neutral 0.0, with a companion visibility
   value so the model can tell "leaning zero" from "leg not seen".

COCO-17 keypoint order (ultralytics YOLO-pose):
  0 nose        1 left_eye     2 right_eye    3 left_ear     4 right_ear
  5 left_shldr  6 right_shldr  7 left_elbow   8 right_elbow  9 left_wrist
 10 right_wrist 11 left_hip    12 right_hip   13 left_knee   14 right_knee
 15 left_ankle  16 right_ankle
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

# ─── Keypoint indices ─────────────────────────────────────────────────────────
NOSE = 0
L_EYE, R_EYE = 1, 2
L_EAR, R_EAR = 3, 4
L_SHO, R_SHO = 5, 6
L_ELB, R_ELB = 7, 8
L_WRI, R_WRI = 9, 10
L_HIP, R_HIP = 11, 12
L_KNE, R_KNE = 13, 14
L_ANK, R_ANK = 15, 16

# Fixed feature order — MUST stay stable (model input depends on it).
POSE_FEATURE_NAMES: List[str] = [
    "torso_lean",      # signed lean of torso from vertical (+ = toward road)
    "head_turn",       # signed nose offset from shoulder midline (+ = toward road)
    "head_pitch",      # nose height above shoulder line (down-tilt when small/neg)
    "body_frontal",    # shoulder width / height: wide=facing camera, narrow=profile
    "stance_width",    # ankle horizontal separation / height (planted vs stepping)
    "step_extension",  # leading-ankle horizontal offset from hips (+ = foot toward road)
    "leg_lift",        # vertical ankle asymmetry / height (mid-stride gait)
    "knee_bend",       # mean knee flexion (0 straight → 1 deeply bent)
    "arm_extension",   # wrist reach beyond torso / height (hailing / bracing)
    "lower_body_vis",  # fraction of hip/knee/ankle joints confidently seen
]
POSE_FEATURE_DIM = len(POSE_FEATURE_NAMES)

_EPS = 1e-6


def _valid(conf: np.ndarray, idx: int, thr: float) -> bool:
    return bool(conf[idx] >= thr)


def _mid(pts: np.ndarray, a: int, b: int) -> np.ndarray:
    return (pts[a] + pts[b]) / 2.0


def _angle_from_vertical(dx: float, dy: float) -> float:
    """
    Signed angle (radians) of vector (dx, dy) away from straight-up.
    Image y grows downward, so 'up' is -y. Returns 0 for a perfectly upright
    torso; sign follows dx (positive dx → positive angle).
    Result is squashed to roughly [-1, 1] by dividing by pi/2.
    """
    ang = np.arctan2(dx, -dy)          # 0 when pointing up, +ve to the right
    return float(np.clip(ang / (np.pi / 2.0), -2.0, 2.0))


def _knee_angle(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> Optional[float]:
    v1 = hip - knee
    v2 = ankle - knee
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < _EPS or n2 < _EPS:
        return None
    cos = float(np.dot(v1, v2) / (n1 * n2))
    cos = max(-1.0, min(1.0, cos))
    ang = np.arccos(cos)               # pi = straight leg, small = bent
    return float(np.clip((np.pi - ang) / np.pi, 0.0, 1.0))   # 0 straight → 1 bent


def compute_pose_features(
    kpts_xy: Sequence[Sequence[float]],
    kpt_conf: Optional[Sequence[float]],
    bbox: Sequence[float],
    mirror: bool = False,
    conf_thr: float = 0.3,
) -> Dict[str, float]:
    """
    Convert one person's keypoints into named body-language features.

    Args:
        kpts_xy:   (17, 2) pixel coordinates in COCO order.
        kpt_conf:  (17,) per-keypoint confidence, or None (treat all as visible).
        bbox:      [x, y, w, h] of the person (for scale normalisation).
        mirror:    if True, flip the sign of all lateral features (ego-lateral).
        conf_thr:  keypoints below this confidence are treated as missing.

    Returns:
        dict of POSE_FEATURE_NAMES → float. Missing-joint features default to 0.0.
    """
    pts = np.asarray(kpts_xy, dtype=np.float32).reshape(-1, 2)
    n = pts.shape[0]
    if kpt_conf is None:
        conf = np.ones(n, dtype=np.float32)
    else:
        conf = np.asarray(kpt_conf, dtype=np.float32).reshape(-1)

    h = max(float(bbox[3]), 1.0)       # bbox height = scale unit
    s = -1.0 if mirror else 1.0        # lateral sign flip

    feats = {name: 0.0 for name in POSE_FEATURE_NAMES}

    have_sho = _valid(conf, L_SHO, conf_thr) and _valid(conf, R_SHO, conf_thr)
    have_hip = _valid(conf, L_HIP, conf_thr) and _valid(conf, R_HIP, conf_thr)

    sho_mid = _mid(pts, L_SHO, R_SHO) if have_sho else None
    hip_mid = _mid(pts, L_HIP, R_HIP) if have_hip else None

    # ── torso lean (shoulder-mid relative to hip-mid) ──
    if sho_mid is not None and hip_mid is not None:
        dx = sho_mid[0] - hip_mid[0]
        dy = sho_mid[1] - hip_mid[1]
        feats["torso_lean"] = s * _angle_from_vertical(dx, dy)

    # ── head turn / pitch (nose vs shoulder line) ──
    if _valid(conf, NOSE, conf_thr) and sho_mid is not None:
        sho_w = abs(pts[L_SHO][0] - pts[R_SHO][0]) + _EPS
        feats["head_turn"] = s * float(np.clip((pts[NOSE][0] - sho_mid[0]) / sho_w, -2.0, 2.0))
        feats["head_pitch"] = float(np.clip((sho_mid[1] - pts[NOSE][1]) / h, -1.0, 1.0))

    # ── body frontal-ness (shoulder width / height) ──
    if have_sho:
        feats["body_frontal"] = float(np.clip(abs(pts[L_SHO][0] - pts[R_SHO][0]) / h, 0.0, 1.0))

    # ── ankle-based stance / step / gait ──
    have_lank = _valid(conf, L_ANK, conf_thr)
    have_rank = _valid(conf, R_ANK, conf_thr)
    if have_lank and have_rank:
        feats["stance_width"] = float(np.clip(abs(pts[L_ANK][0] - pts[R_ANK][0]) / h, 0.0, 1.0))
        feats["leg_lift"] = float(np.clip(abs(pts[L_ANK][1] - pts[R_ANK][1]) / h, 0.0, 1.0))
    if hip_mid is not None:
        ank_x = [pts[i][0] for i, ok in ((L_ANK, have_lank), (R_ANK, have_rank)) if ok]
        if ank_x:
            lead = max(ank_x, key=lambda x: s * (x - hip_mid[0]))
            feats["step_extension"] = s * float(np.clip((lead - hip_mid[0]) / h, -1.5, 1.5))

    # ── knee bend (mean of visible legs) ──
    knee_vals = []
    for hip_i, kne_i, ank_i in ((L_HIP, L_KNE, L_ANK), (R_HIP, R_KNE, R_ANK)):
        if all(_valid(conf, j, conf_thr) for j in (hip_i, kne_i, ank_i)):
            kb = _knee_angle(pts[hip_i], pts[kne_i], pts[ank_i])
            if kb is not None:
                knee_vals.append(kb)
    if knee_vals:
        feats["knee_bend"] = float(np.mean(knee_vals))

    # ── arm extension (max wrist reach beyond torso centre) ──
    centre_x = sho_mid[0] if sho_mid is not None else (hip_mid[0] if hip_mid is not None else None)
    if centre_x is not None:
        reach = [abs(pts[w][0] - centre_x) for w in (L_WRI, R_WRI) if _valid(conf, w, conf_thr)]
        if reach:
            feats["arm_extension"] = float(np.clip(max(reach) / h, 0.0, 1.5))

    # ── lower-body visibility (helps model discount unreliable gait feats) ──
    lower = [L_HIP, R_HIP, L_KNE, R_KNE, L_ANK, R_ANK]
    feats["lower_body_vis"] = float(np.mean([1.0 if _valid(conf, j, conf_thr) else 0.0 for j in lower]))

    return feats


def features_to_vector(feats: Dict[str, float]) -> np.ndarray:
    """Pack a feature dict into the canonical fixed-order vector."""
    return np.array([feats.get(name, 0.0) for name in POSE_FEATURE_NAMES], dtype=np.float32)


def zero_vector() -> np.ndarray:
    """Neutral vector for frames where no pose was detected."""
    return np.zeros(POSE_FEATURE_DIM, dtype=np.float32)
