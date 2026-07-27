"""
detector.py — YOLOv8 object detection per frame.

Detects pedestrians and vehicles, saves per-frame JSON annotations.
"""

import json
import logging
import os
from typing import Dict, List, Optional

import config

logger = logging.getLogger("detector")

# Lazy import — ultralytics only loaded when needed
_model = None


def _get_model():
    """
    Load the configured detection backbone (config.DETECTOR_BACKEND).

    Default is RT-DETR — a transformer detector (NMS-free, stronger under
    occlusion/crowding) — the "more adaptable algorithm" upgrade over YOLOv8.
    "yolo11"/"yolov8" remain selectable. All share ultralytics' predict API and
    the COCO class ids used elsewhere, so downstream stages are unchanged.
    """
    global _model
    if _model is None:
        backend = getattr(config, "DETECTOR_BACKEND", "yolov8")
        weights = getattr(config, "DETECTOR_WEIGHTS", {}).get(backend, config.YOLO_MODEL_PATH)
        fallback = getattr(config, "DETECTOR_FALLBACK", {}).get(backend, config.YOLO_MODEL_FALLBACK)

        model_path = weights
        if not os.path.exists(model_path):
            logger.warning(
                f"Detector weights not found at {model_path}, "
                f"falling back to '{fallback}' (will auto-download)"
            )
            model_path = fallback

        if backend == "rtdetr":
            from ultralytics import RTDETR
            _model = RTDETR(model_path)
        else:
            from ultralytics import YOLO
            _model = YOLO(model_path)
        logger.info(f"Detector loaded: backend={backend} weights={model_path}")
    return _model


def _label_for_class(class_id: int) -> str:
    if class_id in config.PEDESTRIAN_CLASS_IDS:
        return "pedestrian"
    if class_id in config.VEHICLE_CLASS_IDS:
        return "vehicle"
    return "unknown"


def detect_frame(frame_path: str, frame_id: int) -> List[Dict]:
    """
    Run YOLOv8 on a single frame image.

    Returns list of detection dicts:
        {
          "label": "pedestrian" | "vehicle",
          "class_id": int,
          "bbox": [x, y, w, h],
          "confidence": float,
        }
    """
    model = _get_model()
    results = model.predict(
        source=frame_path,
        conf=config.DETECTION_CONFIDENCE,
        classes=list(config.TARGET_CLASS_IDS),
        verbose=False,
    )

    detections = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id not in config.TARGET_CLASS_IDS:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append(
                {
                    "label": _label_for_class(cls_id),
                    "class_id": cls_id,
                    "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                    "confidence": round(float(box.conf[0]), 4),
                }
            )

    return detections


def detect_from_jaad_annotations(record: Dict) -> List[Dict]:
    """
    When in annotation-only mode (no image), convert JAAD ground-truth
    bounding boxes into detection-format dicts.
    """
    detections = []
    for ann in record.get("jaad_annotations", []):
        label = "pedestrian" if "pedestrian" in ann.get("label", "").lower() else "vehicle"
        detections.append(
            {
                "label": label,
                "class_id": 0 if label == "pedestrian" else 2,
                "bbox": ann["bbox"],
                "confidence": 1.0,  # ground truth → perfect confidence
                "ped_id": ann.get("ped_id"),
                "jaad_action": ann.get("action"),
                "jaad_crossing": ann.get("crossing"),
            }
        )
    return detections


def run_detection(records: List[Dict]) -> List[Dict]:
    """
    Run detection on all clean frame records.
    Saves per-frame JSON to output/annotations/.
    Returns frame records enriched with 'detections' key.
    """
    os.makedirs(config.ANNOTATIONS_DIR, exist_ok=True)
    enriched = []
    total = len(records)

    for record in records:
        frame_id = record["frame_id"]
        file_path = record.get("file_path")

        if record.get("jaad_annotations"):
            # JAAD mode: always use ground-truth bboxes + crossing labels.
            # Even when a frame image exists on disk, YOLO would discard the
            # crossing intent labels that live in the XML annotations.
            detections = detect_from_jaad_annotations(record)
        elif file_path and os.path.exists(file_path):
            # Video mode: run YOLO on the extracted frame image
            detections = detect_frame(file_path, frame_id)
        else:
            detections = []

        # Save per-frame annotation JSON
        ann_filename = f"frame_{frame_id:06d}_detections.json"
        ann_path = os.path.join(config.ANNOTATIONS_DIR, ann_filename)
        with open(ann_path, "w") as f:
            json.dump(
                {
                    "frame_id": frame_id,
                    "timestamp": record.get("timestamp"),
                    "source_frame": record.get("source_frame"),
                    "video_id": record.get("video_id"),
                    "detections": detections,
                },
                f,
                indent=2,
            )

        enriched.append({**record, "detections": detections, "annotation_path": ann_path})

        if len(enriched) % 100 == 0 and enriched:
            logger.info(f"  Detection progress: {len(enriched)}/{total} frames")

    total_detections = sum(len(r["detections"]) for r in enriched)
    logger.info(
        f"Detection complete: {len(enriched)} frames, "
        f"{total_detections} total objects"
    )
    return enriched


# ─── Stage entry point ────────────────────────────────────────────────────────

def run(context: dict) -> dict:
    records = run_detection(context["frame_records"])
    return {
        "frame_records": records,
        "output_path": config.ANNOTATIONS_DIR,
    }
