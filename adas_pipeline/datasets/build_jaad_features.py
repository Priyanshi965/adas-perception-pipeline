"""
build_jaad_features.py — Offline feature extraction for intent training.

For each JAAD pedestrian track this builds a per-frame feature *timeline*:
  1. downsample the track to a fixed frame rate (config.JAAD_TIMELINE_STRIDE)
  2. read those frames from the clip
  3. run pose on the GT box crop → body-language features
  4. compute causal kinematics from the box trajectory
  5. mirror lateral features to ego-lateral coords using the track's net motion
  6. store [kinematics(8) | pose(10) | aux(2)] + per-frame cross label

Results are cached per video as a pickle so re-runs are incremental and pose
(the expensive part) is computed once. The trainer consumes these caches and
slices observe→predict windows from them.

Usage:
  python -m datasets.build_jaad_features --videos 30
  python -m datasets.build_jaad_features --videos 30 --no-pose   # kinematics only
"""

import argparse
import logging
import os
import pickle
import sys
from typing import Dict, List, Optional

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config
from datasets import jaad_loader as jl
from modules import body_language as bl
from modules import intent_features as ifeat

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_jaad_features")

CACHE_DIR = os.path.join(config.CHECKPOINT_DIR, "jaad_features")
CLIPS_DIR = os.path.join(config.JAAD_ROOT, "JAAD_clips")


def _clip_path(video_id: str) -> str:
    return os.path.join(CLIPS_DIR, f"{video_id}.mp4")


def _read_frames(video_id: str, frame_idxs: List[int]) -> Dict[int, "np.ndarray"]:
    """Read a specific set of frame indices from a clip. Returns {idx: BGR}."""
    import cv2

    out: Dict[int, np.ndarray] = {}
    path = _clip_path(video_id)
    if not os.path.exists(path):
        return out
    cap = cv2.VideoCapture(path)
    wanted = sorted(set(frame_idxs))
    for idx in wanted:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            out[idx] = frame
    cap.release()
    return out


def build_track(track: jl.PedTrack, pose_est, with_pose: bool,
                fw: int, fh: int, stride: int) -> Optional[Dict]:
    """Featurise one pedestrian track into a timeline dict, or None if unusable."""
    pts = track.frames[::stride]
    if len(pts) < config.INTENT_OBS_LEN + 1:
        return None

    # Ego-lateral mirror sign: net horizontal motion of the box centre.
    x0 = pts[0].bbox[0] + pts[0].bbox[2] / 2
    x1 = pts[-1].bbox[0] + pts[-1].bbox[2] / 2
    mirror = (x1 - x0) < 0     # normalise rightward-moving as the canonical direction

    bbox_seq = [p.bbox for p in pts]
    kin = ifeat.compute_kinematics(bbox_seq, fw, fh)     # (T, 8)

    frames_bgr = _read_frames(track.video_id, [p.frame for p in pts]) if with_pose else {}

    T = len(pts)
    full = np.zeros((T, ifeat.FULL_DIM), dtype=np.float32)
    cross = np.full(T, -1, dtype=np.int64)

    for i, p in enumerate(pts):
        full[i, ifeat.KIN_SLICE] = kin[i]
        if with_pose and p.frame in frames_bgr:
            res = pose_est.estimate_crop(frames_bgr[p.frame], p.bbox)
            if res is not None:
                feats = bl.compute_pose_features(
                    res["keypoints"], res["kpt_conf"], p.bbox,
                    mirror=mirror, conf_thr=config.POSE_KPT_CONF_THR,
                )
                full[i, ifeat.POSE_SLICE] = bl.features_to_vector(feats)
        full[i, ifeat.AUX_SLICE] = [p.look, p.action]
        if p.cross is not None:
            cross[i] = p.cross

    return {
        "ped_id": track.ped_id,
        "video_id": track.video_id,
        "feats": full,
        "cross": cross,
        "mirror": mirror,
    }


def build_video(video_id: str, pose_est, with_pose: bool, force: bool = False) -> List[Dict]:
    """Build (or load cached) featurised tracks for one video."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    tag = "pose" if with_pose else "kin"
    cache = os.path.join(CACHE_DIR, f"{video_id}_{tag}.pkl")
    if os.path.exists(cache) and not force:
        with open(cache, "rb") as f:
            return pickle.load(f)

    xml = os.path.join(config.JAAD_ANNOTATIONS_DIR, f"{video_id}.xml")
    if not os.path.exists(xml):
        return []
    tracks = [t for t in jl.parse_video(xml)
              if any(f.cross is not None for f in t.frames)]

    fw, fh = config.JAAD_FRAME_WIDTH, config.JAAD_FRAME_HEIGHT
    stride = config.JAAD_TIMELINE_STRIDE
    built = []
    for t in tracks:
        d = build_track(t, pose_est, with_pose, fw, fh, stride)
        if d is not None:
            built.append(d)

    with open(cache, "wb") as f:
        pickle.dump(built, f)
    logger.info(f"  {video_id}: {len(built)} tracks featurised → {os.path.basename(cache)}")
    return built


def build_dataset(video_ids: List[str], with_pose: bool, force: bool = False) -> List[Dict]:
    pose_est = None
    if with_pose:
        from modules.pose_estimator import PoseEstimator
        pose_est = PoseEstimator()

    all_tracks: List[Dict] = []
    for n, vid in enumerate(video_ids, 1):
        all_tracks.extend(build_video(vid, pose_est, with_pose, force))
        if n % 10 == 0:
            logger.info(f"[{n}/{len(video_ids)}] videos done, {len(all_tracks)} tracks so far")
    logger.info(f"Dataset: {len(all_tracks)} tracks from {len(video_ids)} videos "
                f"(pose={'on' if with_pose else 'off'})")
    return all_tracks


def parse_args():
    p = argparse.ArgumentParser(description="Build JAAD intent features")
    p.add_argument("--videos", type=int, default=30, help="number of videos (from the start)")
    p.add_argument("--no-pose", action="store_true", help="kinematics only (fast, no clips)")
    p.add_argument("--force", action="store_true", help="ignore cache, rebuild")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    vids = jl.available_video_ids(config.JAAD_ANNOTATIONS_DIR)[: args.videos]
    build_dataset(vids, with_pose=not args.no_pose, force=args.force)
