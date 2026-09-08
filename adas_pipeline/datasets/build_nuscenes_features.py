"""
build_nuscenes_features.py — Camera+LiDAR+radar feature timelines for nuScenes.

Turns nuScenes pedestrian tracks (datasets/nuscenes_loader.py) into the canonical
(T, FULL_DIM=28) intent feature timeline consumed by train_intent.py and the
inference predictor, filling the multimodal channels:

  KIN  (0:8)   BEV/ego-frame trajectory (x,y footprint + velocity/accel)
  POSE (8:18)  body language from RTMPose on the projected camera crop
  AUX  (18:20) unused for nuScenes (zeros)
  LID  (20:24) lid_range, lid_ptcount, lid_height, lid_present
  RAD  (24:28) rad_vr, rad_vx, rad_vy, rad_present

Label (nuScenes has none): per-step `cross` = 1 when the pedestrian is inside the
ego forward-path corridor, else 0, else -1 (behind the ego / not a meaningful
prediction state). intent_features.make_windows then turns that into a leak-free
"will they *enter* the corridor within TTE" onset label — the same observe→predict
formulation used for JAAD/PIE, so the training/eval code is unchanged.

3D boxes are 2 Hz; the ego-frame track is linearly interpolated to
config.NUSC_TIMELINE_HZ. LiDAR point counts come straight from the annotation
(num_lidar_pts — no blob load). Radar velocity is read from the (tiny) radar
sweeps. Features are cached per scene so re-runs and the pose step are paid once.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Dict, List, Optional

import numpy as np

import config
from modules import body_language as bl
from modules import intent_features as ifeat
from datasets import nuscenes_loader as nl

logger = logging.getLogger("build_nuscenes_features")

RADAR_CHANNELS = [
    "RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT",
    "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT",
]


# ─── interpolation of the ego-frame track to a fixed rate ─────────────────────

def _interpolate_track(track: nl.NuscPedTrack, hz: float):
    """Resample the keyframe ego-frame track to `hz`.

    Returns arrays over T uniform timesteps:
        pos   (T,2)  ego-frame (x forward, y left)
        z     (T,)   box centre height
        size  (T,3)  (w, l, h)
        lidpt (T,)   num_lidar_pts (piecewise-constant, from nearest keyframe)
        kf_idx(T,)   index of the nearest keyframe (for radar/pose lookup)
    Continuous quantities are linearly interpolated; discrete ones (point counts,
    pose, radar) are attached from the nearest keyframe by kf_idx.
    """
    fr = track.frames
    ts = np.array([f.timestamp for f in fr], dtype=np.float64)
    t0, t1 = ts[0], ts[-1]
    if t1 <= t0:
        return None
    n = int(round((t1 - t0) * hz)) + 1
    grid = t0 + np.arange(n) / hz

    kx = np.array([f.x for f in fr]); ky = np.array([f.y for f in fr])
    kz = np.array([f.z for f in fr])
    ksz = np.array([f.size for f in fr], dtype=np.float64)   # (K,3)
    klp = np.array([f.num_lidar_pts for f in fr], dtype=np.float64)

    x = np.interp(grid, ts, kx); y = np.interp(grid, ts, ky)
    z = np.interp(grid, ts, kz)
    size = np.stack([np.interp(grid, ts, ksz[:, j]) for j in range(3)], axis=1)
    lidpt = np.interp(grid, ts, klp)
    # nearest keyframe index + distance for each grid point (for radar/pose
    # attachment and for gap detection).
    dist = np.abs(grid[:, None] - ts[None, :])
    kf_idx = dist.argmin(axis=1)
    # A grid point is "valid" only if a real keyframe sits within NUSC_MAX_GAP_S;
    # points inside a long annotation gap (occlusion / out of FOV) are fabricated
    # by interpolation and must not become prediction points or label horizons.
    valid = dist.min(axis=1) <= config.NUSC_MAX_GAP_S

    return {
        "pos": np.stack([x, y], axis=1), "z": z, "size": size,
        "lidpt": lidpt, "kf_idx": kf_idx, "grid": grid, "valid": valid,
    }


# ─── derived corridor-crossing label ──────────────────────────────────────────

def _corridor_label(pos: np.ndarray) -> np.ndarray:
    """Per-step label: 1 inside forward-path corridor, 0 outside, -1 behind ego.

    make_windows then converts this to a leak-free crossing-*onset* target.
    """
    x, y = pos[:, 0], pos[:, 1]
    inside = (
        (x >= config.NUSC_CORRIDOR_MIN_X_M)
        & (x <= config.NUSC_CORRIDOR_LOOKAHEAD_M)
        & (np.abs(y) <= config.NUSC_CORRIDOR_HALF_WIDTH_M)
    )
    cross = np.where(inside, 1, 0).astype(np.int64)
    cross[x < config.NUSC_CORRIDOR_MIN_X_M] = -1   # behind ego → not a prediction pt
    return cross


# ─── BEV kinematics into the KIN slice ────────────────────────────────────────

def _bev_kinematics(pos: np.ndarray, size: np.ndarray) -> np.ndarray:
    """(T,8) trajectory features in the ego frame, normalised to ~unit scale.

    Columns mirror the canonical KIN names but carry BEV semantics:
      cx = x/range, cy = y/range, w = width/2, h = length/2,
      vx,vy = per-step Δposition (normalised), ax,ay = per-step Δvelocity.
    """
    R = config.NUSC_RANGE_NORM_M
    T = pos.shape[0]
    out = np.zeros((T, ifeat.KIN_DIM), dtype=np.float32)
    cx = pos[:, 0] / R
    cy = pos[:, 1] / R
    out[:, 0] = cx
    out[:, 1] = cy
    out[:, 2] = size[:, 0] / 2.0          # width (m)/2
    out[:, 3] = size[:, 1] / 2.0          # length (m)/2
    out[1:, 4] = np.diff(cx)              # vx
    out[1:, 5] = np.diff(cy)              # vy
    out[2:, 6] = np.diff(out[1:, 4])      # ax
    out[2:, 7] = np.diff(out[1:, 5])      # ay
    return out


# ─── LiDAR channels ───────────────────────────────────────────────────────────

def _lidar_channels(pos: np.ndarray, z: np.ndarray, lidpt: np.ndarray) -> np.ndarray:
    """(T,4): range, log-normalised point count, height, presence flag."""
    T = pos.shape[0]
    out = np.zeros((T, ifeat.LIDAR_DIM), dtype=np.float32)
    rng = np.linalg.norm(pos, axis=1)
    out[:, 0] = rng / config.NUSC_RANGE_NORM_M
    out[:, 1] = np.log1p(lidpt) / np.log1p(config.NUSC_LIDAR_PTS_NORM)
    out[:, 2] = z / 3.0                        # ~[0,1] for pedestrian heights
    out[:, 3] = (lidpt > 0).astype(np.float32)
    return out


# ─── radar velocity per keyframe (loads the tiny radar sweeps) ────────────────

def _radar_feature_per_keyframe(nusc, track: nl.NuscPedTrack) -> np.ndarray:
    """(K,4) radar features per keyframe: vr, vx_comp, vy_comp, present.

    All 5 radars are transformed into the ego frame and returns within
    NUSC_RADAR_ASSOC_RADIUS_M of the pedestrian are aggregated (median). vr is the
    radial (line-of-sight) component of the ego-motion-compensated velocity.
    """
    from nuscenes.utils.data_classes import RadarPointCloud
    from pyquaternion import Quaternion

    K = len(track.frames)
    out = np.zeros((K, ifeat.RADAR_DIM), dtype=np.float32)
    for i, f in enumerate(track.frames):
        sample = nusc.get("sample", f.sample_token)
        pts_xy, pts_v = [], []
        for ch in RADAR_CHANNELS:
            tok = sample["data"].get(ch)
            if tok is None:
                continue
            sd = nusc.get("sample_data", tok)
            cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            path = os.path.join(nusc.dataroot, sd["filename"])
            if not os.path.exists(path):
                continue
            try:
                pc = RadarPointCloud.from_file(path)
            except Exception:
                continue
            arr = pc.points  # (18, N)
            if arr.shape[1] == 0:
                continue
            rot = Quaternion(cs["rotation"]).rotation_matrix
            trans = np.array(cs["translation"]).reshape(3, 1)
            xyz = rot @ arr[0:3, :] + trans                    # sensor→ego
            vcomp = rot @ np.vstack([arr[8:10, :], np.zeros((1, arr.shape[1]))])
            pts_xy.append(xyz[0:2, :])
            pts_v.append(vcomp[0:2, :])
        if not pts_xy:
            continue
        XY = np.concatenate(pts_xy, axis=1)      # (2, M)
        V = np.concatenate(pts_v, axis=1)        # (2, M)
        d = np.linalg.norm(XY - np.array([[f.x], [f.y]]), axis=0)
        sel = d <= config.NUSC_RADAR_ASSOC_RADIUS_M
        if not sel.any():
            continue
        vx = float(np.median(V[0, sel])); vy = float(np.median(V[1, sel]))
        rnorm = max(np.hypot(f.x, f.y), 1e-6)
        vr = (vx * f.x + vy * f.y) / rnorm       # radial (line-of-sight)
        out[i] = [np.tanh(vr / 5.0), np.tanh(vx / 5.0), np.tanh(vy / 5.0), 1.0]
    return out


# ─── pose per keyframe (RTMPose on the projected crop) ────────────────────────

def _pose_feature_per_keyframe(track: nl.NuscPedTrack, pose_est) -> np.ndarray:
    """(K,POSE_DIM) body-language features per keyframe, 0 where unavailable."""
    import cv2
    K = len(track.frames)
    out = np.zeros((K, ifeat.POSE_DIM), dtype=np.float32)
    cache: Dict[str, "np.ndarray"] = {}
    for i, f in enumerate(track.frames):
        pr = f.pose_ref
        if pr is None or not os.path.exists(pr.image_path):
            continue
        img = cache.get(pr.image_path)
        if img is None:
            img = cv2.imread(pr.image_path)
            if img is None:
                continue
            cache[pr.image_path] = img
        res = pose_est.estimate_crop(img, pr.bbox_xywh)
        if res is None:
            continue
        # Ego-lateral mirror sign: pedestrian moving to image-right vs left.
        feats = bl.compute_pose_features(
            res["keypoints"], res["kpt_conf"], pr.bbox_xywh,
            mirror=(f.y < 0), conf_thr=config.POSE_KPT_CONF_THR,
        )
        out[i] = bl.features_to_vector(feats)
    return out


# ─── assemble one track into a (T, FULL_DIM) timeline ─────────────────────────

def build_track(nusc, track: nl.NuscPedTrack, pose_est, with_pose: bool,
                with_radar: bool = True) -> Optional[Dict]:
    interp = _interpolate_track(track, config.NUSC_TIMELINE_HZ)
    if interp is None:
        return None
    pos, z, size, lidpt, kf_idx, valid = (
        interp["pos"], interp["z"], interp["size"], interp["lidpt"],
        interp["kf_idx"], interp["valid"],
    )
    T = pos.shape[0]
    if T < config.INTENT_OBS_LEN + 1:
        return None

    full = np.zeros((T, ifeat.FULL_DIM), dtype=np.float32)
    full[:, ifeat.KIN_SLICE] = _bev_kinematics(pos, size)
    # LiDAR channels come from the ANNOTATION (num_lidar_pts) + geometry — no point
    # blob is loaded — so they work from the metadata pack alone.
    full[:, ifeat.LIDAR_SLICE] = _lidar_channels(pos, z, lidpt)

    if with_radar:  # radar needs the sensor blobs; skipped in meta-only mode
        rad_kf = _radar_feature_per_keyframe(nusc, track)      # (K,4)
        full[:, ifeat.RADAR_SLICE] = rad_kf[kf_idx]            # attach nearest kf

    if with_pose and pose_est is not None:
        pose_kf = _pose_feature_per_keyframe(track, pose_est)  # (K,POSE_DIM)
        full[:, ifeat.POSE_SLICE] = pose_kf[kf_idx]

    cross = _corridor_label(pos)
    cross[~valid] = -1     # steps inside an annotation gap: excluded, never labelled
    return {
        "ped_id": track.ped_id,
        "video_id": track.video_id,     # scene name (per-scene cache key)
        "scene": track.scene_name,
        "feats": full,
        "cross": cross,
    }


# ─── dataset build with per-scene cache ───────────────────────────────────────

def build_dataset(
    scene_tokens: Optional[List[str]] = None,
    with_pose: bool = True,
    force: bool = False,
    nusc=None,
    with_radar: bool = True,
    require_blobs: bool = True,
) -> List[Dict]:
    """Featurise nuScenes pedestrian tracks, caching per scene.

    Returns a flat list of track dicts (each carries its `scene` for scene-level,
    leak-free train/val/test splitting in the trainer).

    meta-only mode (`require_blobs=False`, `with_pose=False`, `with_radar=False`):
    processes ALL scenes using only the annotation pack — BEV kinematics + LiDAR
    density/geometry + the derived label — no camera or radar blobs touched. This
    is what lets the full 850-scene trainval run from `v1.0-trainval_meta.tgz`
    alone, without the ~300 GB of sensor blobs.
    """
    from nuscenes.nuscenes import NuScenes
    if nusc is None:
        nusc = NuScenes(version=config.NUSC_VERSION, dataroot=config.NUSC_DATAROOT,
                        verbose=False)
    if scene_tokens is None:
        # With blobs: only scenes whose sensor files exist (partial-download safe).
        # Meta-only: every scene, since kin+lidar+label need no blob.
        scene_tokens = (nl.available_scene_tokens(nusc) if require_blobs
                        else [s["token"] for s in nusc.scene])

    os.makedirs(config.NUSC_CACHE_DIR, exist_ok=True)
    # Cache tag encodes which modalities were extracted so meta-only and full-blob
    # builds of the same scene never collide.
    tag = f"{'pose' if with_pose else 'nopose'}_{'rad' if with_radar else 'norad'}"
    pose_est = None
    if with_pose:
        from modules.pose_estimator import PoseEstimator
        pose_est = PoseEstimator()

    all_built: List[Dict] = []
    for n, scene_tok in enumerate(scene_tokens, 1):
        scene = nusc.get("scene", scene_tok)
        cache = os.path.join(config.NUSC_CACHE_DIR, f"{scene['name']}_{tag}.pkl")
        if os.path.exists(cache) and not force:
            with open(cache, "rb") as fh:
                all_built.extend(pickle.load(fh))
            continue
        tracks = nl.load_nuscenes_tracks(nusc, [scene_tok], with_pose=with_pose)
        built = []
        for t in tracks:
            d = build_track(nusc, t, pose_est, with_pose, with_radar=with_radar)
            if d is not None:
                built.append(d)
        with open(cache, "wb") as fh:
            pickle.dump(built, fh)
        logger.info("[%d/%d] scene %s: %d tracks → %s",
                    n, len(scene_tokens), scene["name"], len(built),
                    os.path.basename(cache))
        all_built.extend(built)

    n_pos = sum(int((d["cross"] == 1).any()) for d in all_built)
    logger.info("nuScenes: %d tracks (%d ever enter corridor) from %d scenes",
                len(all_built), n_pos, len(scene_tokens))
    return all_built
