"""
feature_builder.py — Dataset-agnostic core for intent feature timelines.

Shared by the JAAD and PIE offline builders so the featurisation + caching logic
lives in exactly one place (windowing itself already lives in intent_features).
A caller supplies the three things that actually differ between datasets:

  * tracks       — List[PedTrack] from jaad_loader or pie_loader
  * frame_reader — callable(video_id, frame_idxs) -> {idx: BGR ndarray}
                   (JAAD seeks the mp4; PIE reads pre-extracted PNGs)
  * cache_dir / dims / stride

Everything downstream — kinematics, pose crop, body-language features, the aux
channels, and the per-frame cross label — is identical across datasets and is
defined here so the two builders can never drift apart. Output layout matches the
canonical 20-dim vector in intent_features (KIN | POSE | AUX); OBD ego-motion
fusion is a separate, later change and is deliberately NOT wired in here.
"""

import logging
import os
import pickle
from typing import Callable, Dict, List, Optional

import numpy as np

import config
from modules import body_language as bl
from modules import intent_features as ifeat

logger = logging.getLogger("feature_builder")

FrameReader = Callable[[str, List[int]], Dict[int, "np.ndarray"]]


def sanitize_video_id(video_id: str) -> str:
    """PIE video ids contain a set separator ('set01/video_0001'); make them
    safe to embed in a cache filename."""
    return video_id.replace("/", "_").replace("\\", "_")


def build_track(track, pose_est, with_pose: bool, fw: int, fh: int,
                stride: int, frame_reader: FrameReader) -> Optional[Dict]:
    """Featurise one pedestrian track into a timeline dict, or None if unusable."""
    pts = track.frames[::stride]
    if len(pts) < config.INTENT_OBS_LEN + 1:
        return None

    # Ego-lateral mirror sign: net horizontal motion of the box centre.
    x0 = pts[0].bbox[0] + pts[0].bbox[2] / 2
    x1 = pts[-1].bbox[0] + pts[-1].bbox[2] / 2
    mirror = (x1 - x0) < 0

    bbox_seq = [p.bbox for p in pts]
    kin = ifeat.compute_kinematics(bbox_seq, fw, fh)     # (T, 8)

    frames_bgr = frame_reader(track.video_id, [p.frame for p in pts]) if with_pose else {}

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


def build_from_tracks(
    tracks: List,
    frame_reader: FrameReader,
    cache_dir: str,
    pose_est,
    with_pose: bool,
    fw: int,
    fh: int,
    stride: int,
    force: bool = False,
) -> List[Dict]:
    """
    Featurise a list of PedTracks, caching per video id.

    Tracks are grouped by their video_id so caches stay per-video (incremental
    re-runs), exactly like the original JAAD builder. Returns all featurised
    track dicts across the given tracks.
    """
    os.makedirs(cache_dir, exist_ok=True)
    tag = "pose" if with_pose else "kin"

    by_vid: Dict[str, List] = {}
    for t in tracks:
        by_vid.setdefault(t.video_id, []).append(t)

    all_built: List[Dict] = []
    for n, (vid, vts) in enumerate(sorted(by_vid.items()), 1):
        cache = os.path.join(cache_dir, f"{sanitize_video_id(vid)}_{tag}.pkl")
        if os.path.exists(cache) and not force:
            with open(cache, "rb") as f:
                all_built.extend(pickle.load(f))
            continue
        built = []
        for t in vts:
            d = build_track(t, pose_est, with_pose, fw, fh, stride, frame_reader)
            if d is not None:
                built.append(d)
        with open(cache, "wb") as f:
            pickle.dump(built, f)
        logger.info("  %s: %d tracks featurised → %s", vid, len(built), os.path.basename(cache))
        all_built.extend(built)

    logger.info("Dataset: %d tracks from %d videos (pose=%s)",
                len(all_built), len(by_vid), "on" if with_pose else "off")
    return all_built
