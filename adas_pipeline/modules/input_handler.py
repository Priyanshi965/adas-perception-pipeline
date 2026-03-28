"""
input_handler.py — Validates input sources.

Supports two modes:
  - "video" : single video file (mp4, avi, mov)
  - "jaad"  : JAAD dataset — reads XML annotations + optional video clips
"""

import logging
import os
from typing import Dict, List

import cv2

import config

logger = logging.getLogger("input_handler")


# ─── Video mode ───────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}


def validate_video(video_path: str) -> Dict:
    """
    Validate a single video file and return its metadata.

    Returns:
        {
          "mode": "video",
          "video_path": str,
          "width": int, "height": int,
          "fps": float,
          "total_frames": int,
          "duration_sec": float,
        }
    Raises:
        FileNotFoundError / ValueError on bad input.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    ext = os.path.splitext(video_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported video format '{ext}'. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"OpenCV cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if width == 0 or height == 0:
        raise ValueError(f"Video has zero dimensions: {video_path}")
    if total_frames <= 0:
        raise ValueError(f"Video reports 0 frames (may be corrupted): {video_path}")

    duration_sec = total_frames / fps

    logger.info(
        f"Video validated: {os.path.basename(video_path)} | "
        f"{width}x{height} @ {fps:.1f}fps | "
        f"{total_frames} frames | {duration_sec:.1f}s"
    )

    return {
        "mode": "video",
        "video_path": os.path.abspath(video_path),
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
        "duration_sec": duration_sec,
    }


# ─── JAAD mode ────────────────────────────────────────────────────────────────

def validate_jaad(split: str = None, subset: str = None) -> Dict:
    """
    Validate JAAD dataset structure and return list of video IDs to process.

    Returns:
        {
          "mode": "jaad",
          "jaad_root": str,
          "annotations_dir": str,
          "video_ids": List[str],   # e.g. ["video_0001", ...]
          "total_videos": int,
        }
    """
    split = split or config.JAAD_SPLIT
    subset = subset or config.JAAD_SUBSET

    annotations_dir = config.JAAD_ANNOTATIONS_DIR
    if not os.path.isdir(annotations_dir):
        raise FileNotFoundError(
            f"JAAD annotations directory not found: {annotations_dir}\n"
            f"Expected JAAD at: {config.JAAD_ROOT}"
        )

    # Collect all annotation XML files
    xml_files = sorted(
        f for f in os.listdir(annotations_dir) if f.endswith(".xml")
    )
    if not xml_files:
        raise ValueError(f"No XML annotation files found in {annotations_dir}")

    # Optionally filter to a train/val/test split
    video_ids = _filter_by_split(xml_files, split, subset)

    # Respect max-videos cap
    if config.JAAD_MAX_VIDEOS:
        video_ids = video_ids[: config.JAAD_MAX_VIDEOS]

    logger.info(
        f"JAAD validated: {len(video_ids)} videos "
        f"(split={split}, subset={subset})"
    )

    return {
        "mode": "jaad",
        "jaad_root": os.path.abspath(config.JAAD_ROOT),
        "annotations_dir": os.path.abspath(annotations_dir),
        "video_ids": video_ids,
        "total_videos": len(video_ids),
        "output_path": annotations_dir,
    }


def _filter_by_split(xml_files: List[str], split: str, subset: str) -> List[str]:
    """
    Return video IDs filtered by the requested split/subset.
    Falls back to all videos if split files are missing.
    """
    split_dir = os.path.join(config.JAAD_SPLIT_IDS_DIR, split)

    if subset == "all" or not os.path.isdir(split_dir):
        return [os.path.splitext(f)[0] for f in xml_files]

    split_file = os.path.join(split_dir, f"{subset}.txt")
    if not os.path.exists(split_file):
        logger.warning(f"Split file not found: {split_file} — using all videos")
        return [os.path.splitext(f)[0] for f in xml_files]

    with open(split_file) as f:
        wanted = {line.strip() for line in f if line.strip()}

    all_ids = {os.path.splitext(f)[0] for f in xml_files}
    return sorted(all_ids & wanted)


# ─── Stage entry point ────────────────────────────────────────────────────────

def run(context: dict) -> dict:
    """Pipeline stage: validate input and inject metadata into context."""
    mode = context.get("mode", "video")

    if mode == "jaad":
        result = validate_jaad(
            split=context.get("jaad_split"),
            subset=context.get("jaad_subset"),
        )
    else:
        video_path = context.get("video_path")
        if not video_path:
            raise ValueError("context['video_path'] is required for video mode")
        result = validate_video(video_path)

    result["output_path"] = result.get("video_path", result.get("annotations_dir", ""))
    return result
