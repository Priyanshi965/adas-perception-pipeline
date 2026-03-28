"""
xml_exporter.py — Generates CVAT-compatible XML annotations from pipeline output.

Format matches JAAD's own annotation schema so it can be fed back into
any tool that understands CVAT 1.1 XML.
"""

import json
import os
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Dict, List
from xml.dom import minidom


def _prettify(elem: ET.Element) -> str:
    raw = ET.tostring(elem, encoding="unicode")
    return minidom.parseString(raw).toprettyxml(indent="  ")


def generate_xml(dataset_json_path: str, output_xml_path: str, video_id: str = "pipeline_output"):
    """
    Read dataset.json and write a CVAT 1.1 XML annotation file.
    One <track> per unique object ID, one <box> per frame appearance.
    """
    with open(dataset_json_path, encoding="utf-8") as f:
        records = json.load(f)

    # Group detections by track_id
    tracks: Dict[str, List[Dict]] = {}
    track_labels: Dict[str, str] = {}

    for record in records:
        frame_id = record["frame_id"]
        for obj in record.get("objects", []):
            tid = obj.get("id") or "unknown"
            tracks.setdefault(tid, []).append({
                "frame": frame_id,
                "bbox": obj.get("bbox") or [0, 0, 0, 0],
                "behavior": obj.get("behavior", "unknown"),
                "danger_score": obj.get("danger_score", 0.0),
                "confidence": obj.get("confidence", 1.0),
                "jaad_crossing": obj.get("jaad_crossing"),
                "jaad_action": obj.get("jaad_action"),
                "scene_tag": record.get("scene_tag", "SAFE"),
                "safety_reason": record.get("safety_reason", ""),
            })
            track_labels[tid] = obj.get("label", "pedestrian")

    # Build XML tree
    root = ET.Element("annotations")
    ET.SubElement(root, "version").text = "1.1"

    # Meta
    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = video_id
    ET.SubElement(task, "size").text = str(len(records))
    ET.SubElement(task, "created").text = datetime.now().isoformat()
    ET.SubElement(task, "source").text = "ADAS Pipeline"

    labels_el = ET.SubElement(task, "labels")
    for lname in ["pedestrian", "vehicle"]:
        lbl = ET.SubElement(labels_el, "label")
        ET.SubElement(lbl, "name").text = lname

    ET.SubElement(meta, "dumped").text = datetime.now().isoformat()

    # Tracks
    for tid, boxes in tracks.items():
        label = track_labels.get(tid, "pedestrian")
        track_el = ET.SubElement(root, "track", label=label, id=tid)

        for box_data in sorted(boxes, key=lambda b: b["frame"]):
            x, y, w, h = [int(v) for v in box_data["bbox"]]
            box_el = ET.SubElement(
                track_el, "box",
                frame=str(box_data["frame"]),
                xtl=str(x),
                ytl=str(y),
                xbr=str(x + w),
                ybr=str(y + h),
                outside="0",
                occluded="0",
                keyframe="1",
            )
            # Attributes
            attrs = {
                "behavior": box_data["behavior"],
                "danger_score": str(box_data["danger_score"]),
                "scene_tag": box_data["scene_tag"],
                "safety_reason": box_data["safety_reason"],
            }
            if box_data.get("jaad_crossing") is not None:
                attrs["cross"] = "crossing" if box_data["jaad_crossing"] == 1 else "not-crossing"
            if box_data.get("jaad_action"):
                attrs["action"] = box_data["jaad_action"]

            for name, value in attrs.items():
                attr_el = ET.SubElement(box_el, "attribute", name=name)
                attr_el.text = str(value)

    os.makedirs(os.path.dirname(output_xml_path), exist_ok=True)
    with open(output_xml_path, "w", encoding="utf-8") as f:
        f.write(_prettify(root))

    return output_xml_path
