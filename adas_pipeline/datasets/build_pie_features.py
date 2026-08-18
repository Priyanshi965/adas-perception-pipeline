"""
build_pie_features.py — Offline feature extraction for PIE intent training.

Identical featurisation to JAAD (shared datasets/feature_builder core), but:
  * tracks come from pie_loader (PedTrack objects, video_id = "setNN/video_XXXX")
  * frames are read from the pre-extracted PNGs (images/setNN/video_XXXX/NNNNN.png)
    rather than by seeking an mp4 — faster and exact.

Every requested frame PNG is ASSERTED to exist. If extraction is incomplete the
build fails loudly here instead of silently writing zero pose features (which
would train a meaningless "pose" model that looks fine). Cache lives in a
separate pie_features/ dir so it never collides with the JAAD cache.

Usage (run from adas_pipeline/):
  python -m datasets.build_pie_features                 # all present sets, pose
  python -m datasets.build_pie_features --no-pose       # kinematics only (fast)
  python -m datasets.build_pie_features --sets set05    # one set
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
from datasets import pie_loader as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_pie_features")

IMAGES_DIR = os.path.join(pl.PIE_DATA_ROOT, "images")
CACHE_DIR = os.path.join(config.CHECKPOINT_DIR, "pie_features")

# PIE clips are 1920x1080 @ ~30fps, same as JAAD, so the JAAD frame dims and
# timeline stride apply unchanged.
FRAME_W, FRAME_H = config.JAAD_FRAME_WIDTH, config.JAAD_FRAME_HEIGHT
STRIDE = config.JAAD_TIMELINE_STRIDE


def make_frame_reader(images_dir: str):
    """Return a frame_reader(video_id, frame_idxs) -> {idx: BGR} over PNGs.

    Filenames use the exact format the extractor wrote ("%05.f.png"). A missing
    frame is a hard error — see module docstring.
    """
    import cv2

    def reader(video_id: str, frame_idxs: List[int]) -> Dict[int, "np.ndarray"]:
        vdir = os.path.join(images_dir, *video_id.split("/"))
        out: Dict[int, np.ndarray] = {}
        for idx in sorted(set(frame_idxs)):
            path = os.path.join(vdir, "%05.f.png" % idx)
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"PIE frame PNG missing: {path}\n"
                    f"Extraction is incomplete for {video_id}. Re-run "
                    f"extract_and_save_images(extract_frame_type='annotated') "
                    f"before building features."
                )
            img = cv2.imread(path)
            if img is None:
                raise IOError(f"failed to read image {path}")
            out[idx] = img
        return out

    return reader


def build_dataset(with_pose: bool, set_ids=None, force: bool = False) -> List[Dict]:
    tracks = pl.load_pie_tracks(set_ids=set_ids)
    pose_est = None
    if with_pose:
        from modules.pose_estimator import PoseEstimator
        pose_est = PoseEstimator()
    reader = make_frame_reader(IMAGES_DIR)
    return fb.build_from_tracks(
        tracks, reader, CACHE_DIR, pose_est, with_pose,
        FRAME_W, FRAME_H, STRIDE, force,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Build PIE intent features")
    p.add_argument("--no-pose", action="store_true", help="kinematics only (fast, no images)")
    p.add_argument("--sets", nargs="*", default=None, help="subset of sets, e.g. set01 set05")
    p.add_argument("--force", action="store_true", help="ignore cache, rebuild")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(with_pose=not args.no_pose, set_ids=args.sets, force=args.force)
