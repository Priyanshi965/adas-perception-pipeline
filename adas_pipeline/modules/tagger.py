"""
tagger.py — Assigns object-level tags and scene-level safety tags.

Produces DANGER / SAFE for each frame with a plain-text reason.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

import config

logger = logging.getLogger("tagger")

# Object-level behavior → tag string
BEHAVIOR_TAGS = {
    "walking": "#walking",
    "running": "#running",
    "crossing": "#crossing",
    "driving": "#driving",
    "driving_slow": "#driving_slow",
    "stopping": "#stopping",
    "unknown": "#unknown",
}

# Base danger score by behavior (0.0 = totally safe, 1.0 = certain danger)
BEHAVIOR_BASE_SCORE = {
    "crossing":     0.75,
    "running":      0.50,
    "driving":      0.20,
    "driving_slow": 0.15,
    "walking":      0.15,
    "stopping":     0.10,
    "unknown":      0.10,
}


def _danger_score(
    det: Dict,
    vehicles: List[Dict],
    sudden_stop_frames: Dict[str, set],
    frame_id: int,
    track_lengths: Dict[str, int],
    frame_width: int,
    frame_height: int,
) -> float:
    """
    Compute a 0.0–1.0 danger probability for a single detection.

    Rule-based scoring:
      - Base score from behavior
      - +0.20 if pedestrian crossing within proximity of a vehicle
      - +0.25 if vehicle in a sudden-stop frame
      - +0.10 if occluded (hidden objects are more dangerous)
      - +0.15 if new object appearing in periphery (blind spot)
    """
    behavior = det.get("behavior", "unknown")
    score = BEHAVIOR_BASE_SCORE.get(behavior, 0.10)
    label = det.get("label", "")
    tid = det.get("track_id")
    bbox = det.get("bbox", [0, 0, 1, 1])

    # Proximity: crossing pedestrian near a vehicle
    if label == "pedestrian" and behavior == "crossing":
        for veh in vehicles:
            if _distance(bbox, veh.get("bbox", [0, 0, 1, 1])) < config.DANGER_PROXIMITY_PX:
                score += 0.20
                break

    # Sudden stop vehicle
    if label == "vehicle" and tid:
        if frame_id in sudden_stop_frames.get(tid, set()):
            score += 0.25

    # Occlusion
    occlusion = det.get("occlusion", 0)
    if occlusion == 2:   # full occlusion
        score += 0.10
    elif occlusion == 1: # partial
        score += 0.05

    # Blind spot: short history + periphery
    if tid and track_lengths.get(tid, 999) <= 2:
        if _is_in_periphery(bbox, frame_width, frame_height):
            score += 0.15

    return round(min(score, 1.0), 3)


def _center(bbox: List[int]) -> Tuple[float, float]:
    x, y, w, h = bbox
    return x + w / 2, y + h / 2


def _distance(bbox_a: List[int], bbox_b: List[int]) -> float:
    cx_a, cy_a = _center(bbox_a)
    cx_b, cy_b = _center(bbox_b)
    return float(np.hypot(cx_a - cx_b, cy_a - cy_b))


def _is_in_periphery(bbox: List[int], frame_width: int, frame_height: int) -> bool:
    """True if the object center is within the periphery band."""
    cx, cy = _center(bbox)
    margin_x = frame_width * config.PERIPHERY_FRACTION
    margin_y = frame_height * config.PERIPHERY_FRACTION
    return (
        cx < margin_x
        or cx > frame_width - margin_x
        or cy < margin_y
        or cy > frame_height - margin_y
    )


def _precompute_sudden_stops(track_history: Dict[str, List]) -> Dict[str, set]:
    """
    Walk each track's history once and return a set of frame_ids where a
    sudden stop occurred for that track. O(total detections) overall.

    Returns: { track_id: {frame_id, ...} }
    """
    sudden_stop_frames: Dict[str, set] = {}
    window = config.SUDDEN_STOP_FRAMES + 1

    for tid, entries in track_history.items():
        if not tid.startswith("Car"):
            continue  # only check vehicles

        sorted_entries = sorted(entries, key=lambda e: e[0])
        bboxes = [e[1] for e in sorted_entries]
        frame_ids = [e[0] for e in sorted_entries]

        # Compute speeds along the track
        speeds = [0.0]
        for i in range(1, len(bboxes)):
            cx_p, cy_p = _center(bboxes[i - 1])
            cx_c, cy_c = _center(bboxes[i])
            speeds.append(np.hypot(cx_c - cx_p, cy_c - cy_p))

        stops: set = set()
        for i in range(window, len(speeds)):
            start_speed = speeds[i - window]
            end_speed = speeds[i]
            if start_speed > config.WALK_THRESHOLD and end_speed < config.STOP_THRESHOLD:
                stops.add(frame_ids[i])

        if stops:
            sudden_stop_frames[tid] = stops

    return sudden_stop_frames


def _precompute_track_lengths(track_history: Dict[str, List]) -> Dict[str, int]:
    return {tid: len(entries) for tid, entries in track_history.items()}


def tag_all_frames(
    records: List[Dict],
    track_history: Dict[str, List],
    frame_width: int = 1280,
    frame_height: int = 720,
) -> List[Dict]:
    # Pre-compute expensive per-track data once
    sudden_stop_frames = _precompute_sudden_stops(track_history)
    track_lengths = _precompute_track_lengths(track_history)

    tagged = []
    danger_count = 0

    for record in records:
        frame_id = record["frame_id"]
        detections = record.get("detections", [])

        # Assign object-level tags + danger scores
        vehicles_in_frame = [d for d in detections if d["label"] == "vehicle"]
        tagged_dets = []
        for det in detections:
            behavior = det.get("behavior", "unknown")
            score = _danger_score(
                det, vehicles_in_frame, sudden_stop_frames,
                frame_id, track_lengths, frame_width, frame_height,
            )
            tagged_dets.append({
                **det,
                "object_tag": BEHAVIOR_TAGS.get(behavior, "#unknown"),
                "danger_score": score,
            })

        pedestrians = [d for d in tagged_dets if d["label"] == "pedestrian"]
        vehicles = [d for d in tagged_dets if d["label"] == "vehicle"]

        danger_reasons = []

        # Condition 1: crossing pedestrian near a vehicle
        for ped in pedestrians:
            if ped.get("behavior") != "crossing":
                continue
            for veh in vehicles:
                dist = _distance(ped["bbox"], veh["bbox"])
                if dist < config.DANGER_PROXIMITY_PX:
                    danger_reasons.append(
                        f"Pedestrian {ped.get('track_id', '?')} crossing within "
                        f"{dist:.0f}px of vehicle {veh.get('track_id', '?')}"
                    )

        # Condition 2: sudden vehicle stop (O(1) lookup)
        for veh in vehicles:
            tid = veh.get("track_id")
            if tid and frame_id in sudden_stop_frames.get(tid, set()):
                danger_reasons.append(
                    f"Vehicle {tid} stopped suddenly (rapid deceleration)"
                )

        # Condition 3: blind-spot — new object in periphery (O(1) lookup)
        for det in tagged_dets:
            tid = det.get("track_id")
            if tid and track_lengths.get(tid, 0) <= 2:
                if _is_in_periphery(det["bbox"], frame_width, frame_height):
                    danger_reasons.append(
                        f"{det['label'].capitalize()} {tid} appeared in blind spot"
                    )

        if danger_reasons:
            scene_tag = "DANGER"
            safety_reason = "; ".join(danger_reasons)
            danger_count += 1
        else:
            scene_tag = "SAFE"
            safety_reason = "No conflicts detected"

        tagged.append({
            **record,
            "detections": tagged_dets,
            "scene_tag": scene_tag,
            "safety_reason": safety_reason,
        })

    logger.info(f"Tagging complete: {danger_count}/{len(tagged)} frames tagged DANGER")
    return tagged


# ─── Stage entry point ────────────────────────────────────────────────────────

def run(context: dict) -> dict:
    frame_width = context.get("width", config.JAAD_FRAME_WIDTH)
    frame_height = context.get("height", config.JAAD_FRAME_HEIGHT)

    records = tag_all_frames(
        context["frame_records"],
        context.get("track_history", {}),
        frame_width,
        frame_height,
    )
    return {
        "frame_records": records,
        "output_path": config.ANNOTATIONS_DIR,
    }
