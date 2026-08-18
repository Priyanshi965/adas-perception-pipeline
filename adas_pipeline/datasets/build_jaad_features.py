"""
build_jaad_features.py — Offline feature extraction for JAAD intent training.

Thin dataset adapter over the shared datasets/feature_builder core: it supplies
JAAD tracks (jaad_loader) and an mp4-seeking frame reader, and delegates the
actual featurisation/caching to feature_builder so JAAD and PIE stay in lockstep.

For each JAAD pedestrian track this builds a per-frame feature timeline:
  1. downsample the track to a fixed frame rate (config.JAAD_TIMELINE_STRIDE)
  2. read those frames from the clip
  3. run pose on the GT box crop → body-language features
  4. compute causal kinematics from the box trajectory
  5. store [kinematics(8) | pose(10) | aux(2)] + per-frame cross label

Results are cached per video as a pickle so re-runs are incremental and pose
(the expensive part) is computed once.

Usage:
  python -m datasets.build_jaad_features --videos 30
  python -m datasets.build_jaad_features --videos 30 --no-pose   # kinematics only
"""

import argparse
import logging
import os
import sys
from typing import Dict, List

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config
from datasets import feature_builder as fb
from datasets import jaad_loader as jl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_jaad_features")

CACHE_DIR = os.path.join(config.CHECKPOINT_DIR, "jaad_features")
CLIPS_DIR = os.path.join(config.JAAD_ROOT, "JAAD_clips")

FRAME_W, FRAME_H = config.JAAD_FRAME_WIDTH, config.JAAD_FRAME_HEIGHT
STRIDE = config.JAAD_TIMELINE_STRIDE


def make_frame_reader(clips_dir: str):
    """Return a frame_reader(video_id, frame_idxs) -> {idx: BGR} that seeks the
    clip mp4. JAAD video ids are flat ('video_0001')."""
    import cv2

    def reader(video_id: str, frame_idxs: List[int]) -> Dict[int, "np.ndarray"]:
        out: Dict[int, np.ndarray] = {}
        path = os.path.join(clips_dir, f"{video_id}.mp4")
        if not os.path.exists(path):
            return out
        cap = cv2.VideoCapture(path)
        for idx in sorted(set(frame_idxs)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if ok:
                out[idx] = frame
        cap.release()
        return out

    return reader


def build_dataset(video_ids: List[str], with_pose: bool, force: bool = False) -> List[Dict]:
    tracks = jl.load_tracks(config.JAAD_ANNOTATIONS_DIR, video_ids,
                            require_crossing_label=True)
    pose_est = None
    if with_pose:
        from modules.pose_estimator import PoseEstimator
        pose_est = PoseEstimator()
    reader = make_frame_reader(CLIPS_DIR)
    return fb.build_from_tracks(
        tracks, reader, CACHE_DIR, pose_est, with_pose,
        FRAME_W, FRAME_H, STRIDE, force,
    )


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
