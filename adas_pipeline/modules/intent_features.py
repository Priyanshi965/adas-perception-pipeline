"""
intent_features.py — Canonical feature layout for crossing-intent prediction.

Single source of truth shared by:
  • the offline JAAD dataset builder (datasets/build_jaad_features.py)
  • the trainer (train_intent.py)
  • the live inference predictor (modules/intent_predictor.py)

so all three agree on channel order and windowing. Getting this wrong is how
future-leak and feature-misalignment bugs creep in, so it lives in exactly one
place.

Per-frame feature vector (FULL_FEATURE_NAMES, 28 dims):
  kinematic (8): cx, cy, w, h, vx, vy, ax, ay      (trajectory, normalised)
  pose     (10): body-language features             (see body_language.py)
  aux       (2): look, action                        (JAAD GT — optional/leaky)
  lidar     (4): range, ptcount, height, present    (nuScenes only — see below)
  radar     (4): vr, vx, vy, present                (nuScenes only — see below)

The last two groups are the multimodal-fusion channels used by the nuScenes
camera+LiDAR+radar extension (datasets/build_nuscenes_features.py). They are
**appended after aux** so the kin/pose/aux column slices are byte-identical to
the original 20-dim layout — existing JAAD/PIE models (which store their own
`active_cols` in the .npz) keep loading and running unchanged. For camera-only
datasets these 8 columns are simply left as zeros and never selected.

For nuScenes the kinematic block carries **BEV/ego-frame** motion (not 2D camera
projection, which is discontinuous across the 6 cameras): cx,cy = longitudinal/
lateral ego-frame position, w,h = box footprint, vx,vy,ax,ay = BEV velocity/accel.
Pose is still estimated from the projected camera crop (the camera modality).

Windowing (observe→predict, JAAD/PIE standard):
  A sample at timeline index i observes points [i-OBS+1 .. i] and is labelled 1
  iff a crossing frame occurs in the *future* window [i+1 .. i+TTE]. The observed
  window never overlaps the label horizon, so the model predicts rather than
  describes the present.
"""

from typing import List, Tuple

import numpy as np

from modules.body_language import POSE_FEATURE_NAMES, POSE_FEATURE_DIM

KINEMATIC_NAMES: List[str] = ["cx", "cy", "w", "h", "vx", "vy", "ax", "ay"]
AUX_NAMES: List[str] = ["look", "action"]
# Multimodal fusion channels (nuScenes camera+LiDAR+radar extension).
LIDAR_NAMES: List[str] = ["lid_range", "lid_ptcount", "lid_height", "lid_present"]
RADAR_NAMES: List[str] = ["rad_vr", "rad_vx", "rad_vy", "rad_present"]

FULL_FEATURE_NAMES: List[str] = (
    KINEMATIC_NAMES + list(POSE_FEATURE_NAMES) + AUX_NAMES + LIDAR_NAMES + RADAR_NAMES
)
FULL_DIM = len(FULL_FEATURE_NAMES)

KIN_DIM = len(KINEMATIC_NAMES)
POSE_DIM = POSE_FEATURE_DIM
AUX_DIM = len(AUX_NAMES)
LIDAR_DIM = len(LIDAR_NAMES)
RADAR_DIM = len(RADAR_NAMES)

# Column slices into the full per-frame vector. kin/pose/aux keep their original
# offsets so pre-existing 20-dim models remain valid; lidar/radar are appended.
KIN_SLICE = slice(0, KIN_DIM)
POSE_SLICE = slice(KIN_DIM, KIN_DIM + POSE_DIM)
AUX_SLICE = slice(KIN_DIM + POSE_DIM, KIN_DIM + POSE_DIM + AUX_DIM)
LIDAR_SLICE = slice(AUX_SLICE.stop, AUX_SLICE.stop + LIDAR_DIM)
RADAR_SLICE = slice(LIDAR_SLICE.stop, LIDAR_SLICE.stop + RADAR_DIM)


def active_columns(
    use_kinematics: bool,
    use_pose: bool,
    use_aux: bool = False,
    use_lidar: bool = False,
    use_radar: bool = False,
) -> np.ndarray:
    """Return the column indices selected by the given feature toggles."""
    cols: List[int] = []
    if use_kinematics:
        cols += list(range(KIN_SLICE.start, KIN_SLICE.stop))
    if use_pose:
        cols += list(range(POSE_SLICE.start, POSE_SLICE.stop))
    if use_aux:
        cols += list(range(AUX_SLICE.start, AUX_SLICE.stop))
    if use_lidar:
        cols += list(range(LIDAR_SLICE.start, LIDAR_SLICE.stop))
    if use_radar:
        cols += list(range(RADAR_SLICE.start, RADAR_SLICE.stop))
    return np.array(cols, dtype=np.int64)


def active_dim(use_kinematics: bool, use_pose: bool, use_aux: bool = False,
               use_lidar: bool = False, use_radar: bool = False) -> int:
    return int(len(active_columns(use_kinematics, use_pose, use_aux, use_lidar, use_radar)))


def compute_kinematics(
    bbox_seq: List[List[float]], frame_width: int, frame_height: int
) -> np.ndarray:
    """
    Causal per-frame kinematics for a full track timeline: (T, 8).

    Each row depends only on the current and previous points, so a row at index i
    is safe to use in an observation window ending at i (no future leakage).
    """
    T = len(bbox_seq)
    out = np.zeros((T, KIN_DIM), dtype=np.float32)
    prev_c = None
    prev_v = (0.0, 0.0)
    for i, (x, y, w, h) in enumerate(bbox_seq):
        cx = (x + w / 2) / max(frame_width, 1)
        cy = (y + h / 2) / max(frame_height, 1)
        wn = w / max(frame_width, 1)
        hn = h / max(frame_height, 1)
        if prev_c is None:
            vx = vy = ax = ay = 0.0
        else:
            vx = cx - prev_c[0]
            vy = cy - prev_c[1]
            ax = vx - prev_v[0]
            ay = vy - prev_v[1]
        out[i] = [cx, cy, wn, hn, vx, vy, ax, ay]
        prev_c = (cx, cy)
        prev_v = (vx, vy)
    return out


def make_windows(
    feats: np.ndarray,
    cross: np.ndarray,
    obs_len: int,
    tte: int,
    stride: int = 1,
) -> Tuple[List[np.ndarray], List[int], List[int]]:
    """
    Slide observe→predict windows over one track's feature timeline.

    Args:
        feats: (T, FULL_DIM) per-frame features for the track.
        cross: (T,) per-frame crossing label (1/0; -1 for unknown).
        obs_len, tte, stride: window geometry (all in timeline steps).

    Returns:
        (windows, labels, end_indices) where each window is (obs_len, FULL_DIM).

    Label = crossing *onset*: a sample is only taken when the pedestrian is NOT
    yet crossing at i (cross[i]==0), and it is positive iff a crossing *begins*
    within (i, i+tte]. Windows where the pedestrian is already crossing are not
    prediction points and are skipped. This makes the task "will they step into
    the road?" — the ADAS-relevant question — rather than "are they currently
    mid-crossing?", which bbox trajectory alone can already answer and which
    would trivially inflate accuracy.
    """
    T = feats.shape[0]
    windows: List[np.ndarray] = []
    labels: List[int] = []
    ends: List[int] = []
    # i is the last observed index; need obs history behind and tte future ahead
    for i in range(obs_len - 1, T - 1, stride):
        if cross[i] != 0:
            continue  # unknown (-1) or already crossing (1): not a prediction point
        fut = cross[i + 1:min(T, i + 1 + tte)]
        fut = fut[fut >= 0]
        if fut.size == 0:
            continue  # no known future label → cannot supervise
        label = int((fut == 1).any())   # does a crossing begin within the horizon?
        windows.append(feats[i - obs_len + 1:i + 1].astype(np.float32))
        labels.append(label)
        ends.append(i)
    return windows, labels, ends
