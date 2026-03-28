"""
behavior_analyzer.py — Classifies what each tracked object is doing.

Uses displacement vectors from track_history to determine:
  pedestrian: stopping / walking / running / crossing
  vehicle:    stopping / driving

For JAAD frames, optionally cross-validates with JAAD ground-truth crossing labels.
"""

import logging
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

import config

logger = logging.getLogger("behavior_analyzer")


# ─── Displacement helpers ─────────────────────────────────────────────────────

def _center(bbox: List[int]) -> Tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2, y + h / 2


def _displacement_vector(bbox_a: List[int], bbox_b: List[int]) -> Tuple[float, float]:
    cx_a, cy_a = _center(bbox_a)
    cx_b, cy_b = _center(bbox_b)
    return cx_b - cx_a, cy_b - cy_a


def _smooth_displacements(
    bboxes: List[List[int]], window: int
) -> List[Tuple[float, float]]:
    """Compute frame-to-frame displacement vectors, smoothed over a rolling window."""
    raw = []
    for i in range(1, len(bboxes)):
        raw.append(_displacement_vector(bboxes[i - 1], bboxes[i]))

    if not raw:
        return [(0.0, 0.0)]

    # Rolling average
    smoothed = []
    buf: deque = deque(maxlen=window)
    for dx, dy in raw:
        buf.append((dx, dy))
        avg_dx = sum(v[0] for v in buf) / len(buf)
        avg_dy = sum(v[1] for v in buf) / len(buf)
        smoothed.append((avg_dx, avg_dy))
    return smoothed


# ─── Behavior classification ──────────────────────────────────────────────────

def classify_behavior(
    label: str,
    displacements: List[Tuple[float, float]],
    jaad_crossing: Optional[int] = None,
) -> str:
    """
    Classify behavior for a single object based on its displacement history.

    If jaad_crossing is not None (0 or 1), it overrides the crossing decision
    with the dataset ground truth.
    """
    if not displacements:
        return "stopping"

    # Average speed (magnitude)
    speeds = [np.hypot(dx, dy) for dx, dy in displacements]
    avg_speed = float(np.mean(speeds))

    # Recent frames: check for consecutive stopping
    recent = speeds[-config.STOP_MIN_FRAMES :]
    all_stopped = all(s < config.STOP_THRESHOLD for s in recent)

    if all_stopped:
        return "stopping"

    # Horizontal vs vertical component ratio
    avg_dx = float(np.mean([d[0] for d in displacements]))
    avg_dy = float(np.mean([d[1] for d in displacements]))
    total_motion = abs(avg_dx) + abs(avg_dy) + 1e-6
    horizontal_ratio = abs(avg_dx) / total_motion

    if label == "pedestrian":
        # JAAD ground truth override for crossing
        if jaad_crossing == 1:
            return "crossing"
        if jaad_crossing == 0 and horizontal_ratio < config.CROSSING_HORIZONTAL_RATIO:
            # Confirmed not-crossing by JAAD
            pass
        elif horizontal_ratio >= config.CROSSING_HORIZONTAL_RATIO:
            return "crossing"

        if avg_speed < config.WALK_THRESHOLD:
            return "walking"
        return "running"

    else:  # vehicle
        if avg_speed < config.WALK_THRESHOLD:
            return "driving_slow"
        return "driving"


# ─── Main analysis function ───────────────────────────────────────────────────

def analyze_behaviors(
    records: List[Dict], track_history: Dict[str, List]
) -> List[Dict]:
    """
    For every object in every frame, compute behavior using its track history.

    O(n) approach: pre-compute a behavior label for every (track_id, frame_id)
    pair by walking each track's history once in order, then do a O(1) lookup
    per detection. Avoids the O(n²) filter-per-frame that caused hangs.
    """
    # track_history: { track_id: [(frame_id, bbox, ts), ...] }

    # Step 1: for each track, walk its sorted history once and compute behavior
    # at each position using a rolling window of displacements.
    # Result: behavior_lookup[(track_id, frame_id)] = behavior_str
    behavior_lookup: Dict[Tuple[str, int], str] = {}

    for tid, entries in track_history.items():
        sorted_entries = sorted(entries, key=lambda e: e[0])
        bboxes = [e[1] for e in sorted_entries]
        frame_ids = [e[0] for e in sorted_entries]

        # Determine label from track_id prefix
        label = "pedestrian" if tid.startswith("Person") else "vehicle"

        # Compute rolling displacements incrementally — O(len(track)) total
        buf: deque = deque(maxlen=config.DISPLACEMENT_WINDOW)

        for i, (fid, bbox) in enumerate(zip(frame_ids, bboxes)):
            if i > 0:
                dx, dy = _displacement_vector(bboxes[i - 1], bbox)
                buf.append((dx, dy))

            displacements = list(buf) if buf else [(0.0, 0.0)]

            # jaad_crossing is per-detection, not per-track; use None here —
            # it will be applied as an override in the lookup step below.
            behavior = classify_behavior(label, displacements, jaad_crossing=None)
            behavior_lookup[(tid, fid)] = behavior

    logger.info(f"Pre-computed behaviors for {len(track_history)} tracks")

    # Step 2: annotate each detection using the lookup (O(1) per detection)
    enriched_records = []
    all_behaviors: Dict[str, int] = {}

    for record in records:
        frame_id = record["frame_id"]
        new_detections = []

        for det in record.get("detections", []):
            tid = det.get("track_id")
            jaad_crossing = det.get("jaad_crossing")

            behavior = behavior_lookup.get((tid, frame_id), "unknown") if tid else "unknown"

            # Apply JAAD ground-truth crossing override if available
            if jaad_crossing == 1:
                behavior = "crossing"
            elif jaad_crossing == 0 and behavior == "crossing":
                # JAAD says not crossing — downgrade to walking
                behavior = "walking"

            new_detections.append({**det, "behavior": behavior})
            all_behaviors[behavior] = all_behaviors.get(behavior, 0) + 1

        enriched_records.append({**record, "detections": new_detections})

    logger.info(f"Behavior distribution: {all_behaviors}")
    return enriched_records


# ─── Stage entry point ────────────────────────────────────────────────────────

def run(context: dict) -> dict:
    records = analyze_behaviors(
        context["frame_records"],
        context.get("track_history", {}),
    )
    return {
        "frame_records": records,
        "output_path": config.ANNOTATIONS_DIR,
    }
