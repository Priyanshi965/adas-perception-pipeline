"""
frame_cleaner.py — Filters out blurry, dark, and corrupted frames.

Frames that pass are copied to output/clean_frames/.
Returns a filtered list of frame records.
"""

import logging
import os
import shutil
from typing import Dict, List, Tuple

import cv2
import numpy as np

import config

logger = logging.getLogger("frame_cleaner")


def is_blurry(gray: np.ndarray) -> bool:
    """Laplacian variance below threshold → blurry."""
    return cv2.Laplacian(gray, cv2.CV_64F).var() < config.BLUR_THRESHOLD


def is_dark(gray: np.ndarray) -> bool:
    """Mean pixel brightness below threshold → too dark."""
    return float(gray.mean()) < config.BRIGHTNESS_THRESHOLD


def is_corrupted(frame: np.ndarray) -> bool:
    """
    All-black or all-white frame → likely corrupted / blank.
    Uses the saturation ratio threshold from config.
    """
    total_pixels = frame.shape[0] * frame.shape[1]
    black_pixels = int(np.sum(np.all(frame == 0, axis=2)))
    white_pixels = int(np.sum(np.all(frame == 255, axis=2)))
    ratio = (black_pixels + white_pixels) / total_pixels
    return ratio > config.CORRUPTION_SATURATION_RATIO


def clean_frame(record: Dict) -> Tuple[bool, str]:
    """
    Check a single frame record. Return (passes, reject_reason).

    For annotation-only records (file_path is None) we skip image checks
    and pass them through — JAAD annotations are still useful.
    """
    if record.get("file_path") is None:
        return True, ""

    path = record["file_path"]
    if not os.path.exists(path):
        return False, "file_missing"

    frame = cv2.imread(path)
    if frame is None:
        return False, "opencv_decode_fail"

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if is_corrupted(frame):
        return False, "corrupted"
    if is_blurry(gray):
        return False, f"blurry (var={cv2.Laplacian(gray, cv2.CV_64F).var():.1f})"
    if is_dark(gray):
        return False, f"dark (mean={gray.mean():.1f})"

    return True, ""


def _adaptive_blur_threshold(records: List[Dict]) -> float:
    """
    Compute an adaptive blur threshold from the actual frame data.
    Uses the 20th percentile of Laplacian variances so we never remove
    more than ~20% of frames purely due to blur, even on low-quality video.
    Falls back to config.BLUR_THRESHOLD if frames can't be read.
    """
    variances = []
    for record in records[:200]:  # sample first 200 frames
        path = record.get("file_path")
        if not path or not os.path.exists(path):
            continue
        frame = cv2.imread(path)
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variances.append(cv2.Laplacian(gray, cv2.CV_64F).var())

    if not variances:
        return config.BLUR_THRESHOLD

    variances.sort()
    p20 = variances[int(len(variances) * 0.20)]
    # Use the lower of: config threshold OR 20th-percentile (adaptive)
    threshold = min(config.BLUR_THRESHOLD, p20 * 0.5)
    logger.info(
        f"Adaptive blur threshold: {threshold:.1f} "
        f"(config={config.BLUR_THRESHOLD}, p20={p20:.1f})"
    )
    return threshold


def clean_frames(records: List[Dict]) -> List[Dict]:
    """
    Filter the list of frame records. Copy passing frames to clean_frames/.
    Returns only records that passed cleaning (with updated file_path).
    """
    os.makedirs(config.CLEAN_FRAMES_DIR, exist_ok=True)

    # Use adaptive threshold to avoid wiping out entire videos
    blur_threshold = _adaptive_blur_threshold(records)

    passed = []
    stats = {"blurry": 0, "dark": 0, "corrupted": 0, "file_missing": 0, "opencv_decode_fail": 0}
    total = len(records)

    for i, record in enumerate(records):
        if i % 500 == 0 and i > 0:
            logger.info(f"  Cleaning progress: {i}/{total} frames ({len(passed)} passed so far)")

        path = record.get("file_path")
        if path is None:
            passed.append(record)
            continue

        if not os.path.exists(path):
            stats["file_missing"] += 1
            continue

        frame = cv2.imread(path)
        if frame is None:
            stats["opencv_decode_fail"] += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if is_corrupted(frame):
            stats["corrupted"] += 1
            continue
        if cv2.Laplacian(gray, cv2.CV_64F).var() < blur_threshold:
            stats["blurry"] += 1
            continue
        if is_dark(gray):
            stats["dark"] += 1
            continue

        src = path
        dst = os.path.join(config.CLEAN_FRAMES_DIR, os.path.basename(src))
        shutil.copy2(src, dst)
        passed.append({**record, "file_path": dst})

    total = len(records)
    removed = total - len(passed)
    logger.info(
        f"Frame cleaning: {len(passed)}/{total} passed "
        f"({removed} removed — {stats})"
    )
    return passed


# ─── Stage entry point ────────────────────────────────────────────────────────

def run(context: dict) -> dict:
    records = context.get("frame_records", [])
    clean = clean_frames(records)
    return {
        "frame_records": clean,
        "total_frames_clean": len(clean),
        "output_path": config.CLEAN_FRAMES_DIR,
    }
