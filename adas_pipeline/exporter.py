"""
exporter.py — Writes the final ML-ready dataset as JSON + CSV.

JSON: one record per frame.
CSV: one row per object per frame (flat, pandas-friendly).
"""

import csv
import json
import logging
import os
from typing import Dict, List

import config

logger = logging.getLogger("exporter")


def export(records: List[Dict]) -> Dict[str, str]:
    """
    Write output/final/dataset.json and dataset.csv.
    Returns paths to both output files.
    """
    os.makedirs(config.FINAL_DIR, exist_ok=True)

    json_path = os.path.join(config.FINAL_DIR, config.OUTPUT_JSON_NAME)
    csv_path = os.path.join(config.FINAL_DIR, config.OUTPUT_CSV_NAME)

    _write_json(records, json_path)
    _write_csv(records, csv_path)

    logger.info(f"Exported {len(records)} frames → {json_path}, {csv_path}")
    return {"json": json_path, "csv": csv_path}


def _build_json_record(record: Dict) -> Dict:
    """Shape a frame record into the final JSON schema."""
    objects = []
    for det in record.get("detections", []):
        obj = {
            "id": det.get("track_id", "unknown"),
            "label": det.get("label"),
            "bbox": det.get("bbox"),
            "behavior": det.get("behavior", "unknown"),
            "confidence": det.get("confidence"),
            "object_tag": det.get("object_tag"),
            "danger_score": det.get("danger_score"),
        }
        # Include JAAD ground-truth fields if present
        if "jaad_action" in det:
            obj["jaad_action"] = det["jaad_action"]
        if "jaad_crossing" in det:
            obj["jaad_crossing"] = det["jaad_crossing"]
        objects.append(obj)

    return {
        "frame_id": record["frame_id"],
        "file_path": record.get("file_path"),
        "timestamp": record.get("timestamp"),
        "source_frame": record.get("source_frame"),
        "video_id": record.get("video_id"),
        "objects": objects,
        "scene_tag": record.get("scene_tag", "SAFE"),
        "safety_reason": record.get("safety_reason", ""),
    }


def _write_json(records: List[Dict], path: str):
    json_records = [_build_json_record(r) for r in records]
    with open(path, "w") as f:
        json.dump(json_records, f, indent=2)
    logger.info(f"JSON written: {path} ({len(json_records)} records)")


CSV_COLUMNS = [
    "frame_id",
    "timestamp",
    "source_frame",
    "video_id",
    "object_id",
    "label",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "behavior",
    "confidence",
    "danger_score",
    "object_tag",
    "jaad_action",
    "jaad_crossing",
    "scene_tag",
    "safety_reason",
]


def _write_csv(records: List[Dict], path: str):
    rows = []
    for record in records:
        frame_id = record["frame_id"]
        ts = record.get("timestamp", "")
        sf = record.get("source_frame", "")
        vid = record.get("video_id", "")
        scene_tag = record.get("scene_tag", "SAFE")
        safety_reason = record.get("safety_reason", "")

        detections = record.get("detections", [])
        if not detections:
            # Keep frames with no detections as a single empty row
            rows.append(
                {
                    "frame_id": frame_id,
                    "timestamp": ts,
                    "source_frame": sf,
                    "video_id": vid,
                    "object_id": None,
                    "label": None,
                    "bbox_x": None,
                    "bbox_y": None,
                    "bbox_w": None,
                    "bbox_h": None,
                    "behavior": None,
                    "confidence": None,
                    "object_tag": None,
                    "jaad_action": None,
                    "jaad_crossing": None,
                    "scene_tag": scene_tag,
                    "safety_reason": safety_reason,
                }
            )
        else:
            for det in detections:
                bbox = det.get("bbox", [None, None, None, None])
                rows.append(
                    {
                        "frame_id": frame_id,
                        "timestamp": ts,
                        "source_frame": sf,
                        "video_id": vid,
                        "object_id": det.get("track_id"),
                        "label": det.get("label"),
                        "bbox_x": bbox[0] if bbox else None,
                        "bbox_y": bbox[1] if bbox else None,
                        "bbox_w": bbox[2] if bbox else None,
                        "bbox_h": bbox[3] if bbox else None,
                        "behavior": det.get("behavior"),
                        "confidence": det.get("confidence"),
                        "object_tag": det.get("object_tag"),
                        "danger_score": det.get("danger_score"),
                        "jaad_action": det.get("jaad_action"),
                        "jaad_crossing": det.get("jaad_crossing"),
                        "scene_tag": scene_tag,
                        "safety_reason": safety_reason,
                    }
                )

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"CSV written: {path} ({len(rows)} rows)")


# ─── Stage entry point ────────────────────────────────────────────────────────

def run(context: dict) -> dict:
    paths = export(context["frame_records"])
    return {
        "output_path": config.FINAL_DIR,
        "json_path": paths["json"],
        "csv_path": paths["csv"],
    }
