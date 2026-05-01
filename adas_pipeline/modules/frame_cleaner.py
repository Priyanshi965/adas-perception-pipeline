"""
frame_cleaner.py — Filters and preprocesses video frames for ADAS analysis.

Pipeline per frame:
  1. Reject corrupted / unreadable frames
  2. Reject blurry frames (adaptive Laplacian threshold)
  3. Reject too-dark frames
  4. Perceptual-hash deduplication (remove near-identical frames)
  5. Gaussian denoising (reduce sensor/compression noise)
  6. CLAHE lighting normalisation (equalise contrast across day/night/tunnel)
  7. Copy cleaned frame to output/clean_frames/

WHY each step matters for behaviour analysis
─────────────────────────────────────────────
• Blur rejection        → blurry frames produce noisy bounding-box edges that
                          confuse trackers and inflate displacement vectors.
• Dark rejection        → detectors trained on bright imagery fail in very dark
                          frames, generating false-negatives that break tracks.
• Deduplication         → consecutive frames < 1% different carry no new motion
                          signal but add O(n) cost to every downstream stage.
• Gaussian denoising    → impulse/Gaussian sensor noise shifts bbox centroids
                          frame-to-frame, creating phantom velocity signals.
• CLAHE normalisation   → uneven illumination (headlights, shadows, tunnels)
                          causes confidence drops; normalising contrast lets the
                          detector operate at a stable operating point.
"""

import logging
import os
import shutil
from typing import Dict, List, Tuple

import cv2
import numpy as np

import config

logger = logging.getLogger("frame_cleaner")


# ─── Quality checks ───────────────────────────────────────────────────────────

def is_blurry(gray: np.ndarray, threshold: float) -> bool:
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold


def is_dark(gray: np.ndarray) -> bool:
    return float(gray.mean()) < config.BRIGHTNESS_THRESHOLD


def is_corrupted(frame: np.ndarray) -> bool:
    total = frame.shape[0] * frame.shape[1]
    black = int(np.sum(np.all(frame == 0, axis=2)))
    white = int(np.sum(np.all(frame == 255, axis=2)))
    return (black + white) / total > config.CORRUPTION_SATURATION_RATIO


# ─── Adaptive blur threshold ──────────────────────────────────────────────────

def _adaptive_blur_threshold(records: List[Dict]) -> float:
    """
    Sample up to 200 frames to compute the 20th-percentile Laplacian variance.
    Using a fixed threshold on degraded footage would wipe out entire videos;
    the adaptive approach never removes more than ~20 % of frames purely for blur.
    """
    variances = []
    for record in records[:200]:
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
    threshold = min(config.BLUR_THRESHOLD, p20 * 0.5)
    logger.info(
        f"Adaptive blur threshold: {threshold:.1f} "
        f"(config={config.BLUR_THRESHOLD}, p20={p20:.1f})"
    )
    return threshold


# ─── Perceptual-hash deduplication ────────────────────────────────────────────

def _phash(gray: np.ndarray, size: int = 16) -> np.ndarray:
    """
    Compute a difference-hash (dHash) of the frame.
    Resize to (size+1) x size, then compare adjacent pixels row-by-row
    to produce a size*size bit vector.  Hamming distance < threshold → duplicate.
    """
    resized = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]   # shape (size, size) bool
    return diff.flatten()


def _hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


# ─── Enhancement ──────────────────────────────────────────────────────────────

def denoise(frame: np.ndarray) -> np.ndarray:
    """
    Gaussian denoising with a 3×3 kernel (σ=1).
    Keeps edges sharp enough for detection while suppressing high-frequency
    sensor noise that perturbs bbox centroid estimates.
    """
    return cv2.GaussianBlur(frame, (3, 3), sigmaX=1.0)


def clahe_normalise(frame: np.ndarray) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalisation) on the L channel
    of LAB colour space.  Clip limit=2.0 prevents over-amplification of flat
    regions while significantly boosting contrast in under/over-exposed areas.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=config.CLAHE_CLIP_LIMIT,
                             tileGridSize=config.CLAHE_TILE_SIZE)
    l_eq = clahe.apply(l_ch)
    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# ─── Main cleaning function ───────────────────────────────────────────────────

def clean_frames(records: List[Dict]) -> List[Dict]:
    """
    Filter and preprocess frame records.
    Returns only records that passed all quality checks (with updated file_path
    pointing to the processed image in output/clean_frames/).
    """
    os.makedirs(config.CLEAN_FRAMES_DIR, exist_ok=True)

    blur_threshold = _adaptive_blur_threshold(records)

    passed: List[Dict] = []
    stats = {
        "blurry": 0, "dark": 0, "corrupted": 0,
        "file_missing": 0, "opencv_decode_fail": 0, "duplicate": 0,
    }
    total = len(records)

    # Rolling hash buffer for deduplication (last N hashes)
    last_hash: np.ndarray | None = None

    for i, record in enumerate(records):
        if i % 500 == 0 and i > 0:
            logger.info(
                f"  Cleaning progress: {i}/{total} "
                f"({len(passed)} passed, {stats})"
            )

        # Annotation-only records (JAAD without video frames) pass through
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

        # ── Quality gates ──────────────────────────────────────────────
        if is_corrupted(frame):
            stats["corrupted"] += 1
            continue
        if cv2.Laplacian(gray, cv2.CV_64F).var() < blur_threshold:
            stats["blurry"] += 1
            continue
        if is_dark(gray):
            stats["dark"] += 1
            continue

        # ── Perceptual-hash deduplication ──────────────────────────────
        h = _phash(gray)
        if last_hash is not None:
            hamming = _hamming(h, last_hash)
            if hamming < config.DUPLICATE_HASH_THRESHOLD:
                stats["duplicate"] += 1
                continue
        last_hash = h

        # ── Preprocessing ──────────────────────────────────────────────
        processed = denoise(frame)
        processed = clahe_normalise(processed)

        # ── Save ───────────────────────────────────────────────────────
        dst = os.path.join(config.CLEAN_FRAMES_DIR, os.path.basename(path))
        cv2.imwrite(dst, processed, [cv2.IMWRITE_JPEG_QUALITY, config.FRAME_QUALITY])
        passed.append({**record, "file_path": dst})

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
