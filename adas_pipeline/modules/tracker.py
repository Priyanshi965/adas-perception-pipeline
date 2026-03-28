"""
tracker.py — Assigns consistent track IDs to objects across frames.

Two modes controlled by config.USE_BYTETRACK:
  True  → Use ByteTrack via ultralytics (re-runs detection with tracking)
  False → Simple IoU-based tracker (no extra dependencies)

For JAAD annotation-only mode the pedestrian ped_id from the XML is used
directly as the track_id, which is the most reliable option.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

import config

logger = logging.getLogger("tracker")


# ─── IoU helper ───────────────────────────────────────────────────────────────

def _iou(boxA: List[int], boxB: List[int]) -> float:
    """Compute IoU between two [x, y, w, h] boxes."""
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = aw * ah
    area_b = bw * bh
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


# ─── Simple IoU tracker ───────────────────────────────────────────────────────

class SimpleIoUTracker:
    """
    Greedy IoU tracker — assigns IDs to objects across consecutive frames.
    """

    def __init__(self):
        self.next_id: Dict[str, int] = {}   # label → next int ID
        self.active: List[Dict] = []         # last frame's tracked objects
        self.age: Dict[str, int] = {}        # track_id → frames since seen

    def _new_id(self, label: str) -> str:
        prefix = "Person" if label == "pedestrian" else "Car"
        idx = self.next_id.get(prefix, 0)
        self.next_id[prefix] = idx + 1
        return f"{prefix}_{idx}"

    def update(self, detections: List[Dict]) -> List[Dict]:
        """Match detections to existing tracks and return enriched list."""
        matched_ids: Dict[int, str] = {}  # det index → track_id
        used_tracks = set()

        for det_idx, det in enumerate(detections):
            best_iou = config.IOU_MATCH_THRESHOLD
            best_track_id = None

            for prev in self.active:
                if prev["label"] != det["label"]:
                    continue
                if prev["track_id"] in used_tracks:
                    continue
                iou = _iou(det["bbox"], prev["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = prev["track_id"]

            if best_track_id:
                matched_ids[det_idx] = best_track_id
                used_tracks.add(best_track_id)
            else:
                matched_ids[det_idx] = self._new_id(det["label"])

        # Age out old tracks
        new_active = []
        for prev in self.active:
            if prev["track_id"] not in used_tracks:
                self.age[prev["track_id"]] = self.age.get(prev["track_id"], 0) + 1
            else:
                self.age[prev["track_id"]] = 0

        # Build enriched detections and new active list
        result = []
        for det_idx, det in enumerate(detections):
            track_id = matched_ids[det_idx]
            enriched = {**det, "track_id": track_id}
            result.append(enriched)
            new_active.append(enriched)
            self.age[track_id] = 0

        # Keep tracks that are still "alive" (not aged out)
        surviving = [
            p for p in self.active
            if self.age.get(p["track_id"], 0) < config.MAX_TRACK_AGE
            and p["track_id"] not in {r["track_id"] for r in result}
        ]
        self.active = result + surviving
        return result


# ─── ByteTrack mode ───────────────────────────────────────────────────────────

def _track_with_bytetrack(records: List[Dict]) -> List[Dict]:
    """
    Re-run detection with ByteTrack enabled on each frame image.
    Falls back to SimpleIoUTracker if no image is available.
    """
    from ultralytics import YOLO

    model_path = config.YOLO_MODEL_PATH
    if not os.path.exists(model_path):
        model_path = config.YOLO_MODEL_FALLBACK
    model = YOLO(model_path)

    iou_tracker = SimpleIoUTracker()  # fallback for annotation-only frames
    enriched = []

    for record in records:
        frame_id = record["frame_id"]
        file_path = record.get("file_path")

        if file_path and os.path.exists(file_path):
            results = model.track(
                source=file_path,
                conf=config.DETECTION_CONFIDENCE,
                classes=list(config.TARGET_CLASS_IDS),
                persist=True,
                verbose=False,
            )
            detections = _parse_bytetrack_results(results)
        else:
            # No image — use annotation IDs from JAAD or IoU tracker
            detections = _assign_jaad_track_ids(record.get("detections", []))
            if not any("track_id" in d for d in detections):
                detections = iou_tracker.update(detections)

        enriched.append({**record, "detections": detections})

    return enriched


def _parse_bytetrack_results(results) -> List[Dict]:
    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id not in config.TARGET_CLASS_IDS:
                continue
            track_id = int(box.id[0]) if box.id is not None else -1
            prefix = "Person" if cls_id in config.PEDESTRIAN_CLASS_IDS else "Car"
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label = "pedestrian" if cls_id in config.PEDESTRIAN_CLASS_IDS else "vehicle"
            detections.append(
                {
                    "label": label,
                    "class_id": cls_id,
                    "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    "confidence": round(float(box.conf[0]), 4),
                    "track_id": f"{prefix}_{track_id}",
                }
            )
    return detections


def _assign_jaad_track_ids(detections: List[Dict]) -> List[Dict]:
    """Use JAAD ped_id as track_id for annotation-only detections."""
    result = []
    for det in detections:
        ped_id = det.get("ped_id")
        if ped_id:
            prefix = "Person" if det["label"] == "pedestrian" else "Car"
            track_id = f"{prefix}_{ped_id}"
        else:
            track_id = None  # will be filled by IoU tracker
        result.append({**det, "track_id": track_id})
    return result


# ─── Track history builder ────────────────────────────────────────────────────

def build_track_history(records: List[Dict]) -> Dict[str, List]:
    """
    Returns { track_id: [(frame_id, bbox, timestamp), ...] }
    across all frame records.
    """
    history: Dict[str, List] = {}
    for record in records:
        fid = record["frame_id"]
        ts = record.get("timestamp", "")
        for det in record.get("detections", []):
            tid = det.get("track_id")
            if tid:
                history.setdefault(tid, []).append((fid, det["bbox"], ts))
    return history


# ─── Stage entry point ────────────────────────────────────────────────────────

def run(context: dict) -> dict:
    records = context["frame_records"]

    if config.USE_BYTETRACK:
        try:
            logger.info("Using ByteTrack tracker")
            enriched = _track_with_bytetrack(records)
        except Exception as e:
            logger.warning(f"ByteTrack failed ({e}), falling back to IoU tracker")
            enriched = _run_iou_tracker(records)
    else:
        logger.info("Using simple IoU tracker")
        enriched = _run_iou_tracker(records)

    track_history = build_track_history(enriched)
    logger.info(f"Tracking complete: {len(track_history)} unique tracks")

    return {
        "frame_records": enriched,
        "track_history": track_history,
        "output_path": config.ANNOTATIONS_DIR,
    }


def _run_iou_tracker(records: List[Dict]) -> List[Dict]:
    tracker = SimpleIoUTracker()
    enriched = []
    for record in records:
        dets = record.get("detections", [])
        # Use JAAD IDs if available
        dets = _assign_jaad_track_ids(dets)
        # Fill in any missing track_ids with IoU tracker
        needs_tracking = [d for d in dets if not d.get("track_id")]
        already_tracked = [d for d in dets if d.get("track_id")]
        tracked_new = tracker.update(needs_tracking)
        enriched.append({**record, "detections": already_tracked + tracked_new})
    return enriched
