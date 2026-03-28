"""
frame_extractor.py — Extracts frames from a video file OR from JAAD video clips.

Saves numbered JPGs to output/frames/ and returns their paths + timestamps.
"""

import logging
import os
import xml.etree.ElementTree as ET
from typing import Dict, List

import cv2

import config

logger = logging.getLogger("frame_extractor")


# ─── Video mode ───────────────────────────────────────────────────────────────

def extract_from_video(video_path: str, fps: float) -> List[Dict]:
    """
    Extract every Nth frame from a video file.

    Returns list of:
        {"frame_id": int, "file_path": str, "timestamp": str, "source_frame": int}
    """
    os.makedirs(config.FRAMES_DIR, exist_ok=True)
    interval = config.FRAME_SAMPLE_INTERVAL

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    records = []
    source_frame_idx = 0
    saved_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if source_frame_idx % interval == 0:
            timestamp = _frame_to_timestamp(source_frame_idx, fps)
            filename = f"frame_{saved_idx:06d}.{config.FRAME_FORMAT}"
            out_path = os.path.join(config.FRAMES_DIR, filename)
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, config.FRAME_QUALITY])
            records.append(
                {
                    "frame_id": saved_idx,
                    "file_path": out_path,
                    "timestamp": timestamp,
                    "source_frame": source_frame_idx,
                }
            )
            saved_idx += 1

        source_frame_idx += 1

    cap.release()
    logger.info(
        f"Extracted {saved_idx} frames from {os.path.basename(video_path)} "
        f"(every {interval} frames, {source_frame_idx} total source frames)"
    )
    return records


# ─── JAAD mode ────────────────────────────────────────────────────────────────

def extract_from_jaad(video_ids: List[str], annotations_dir: str) -> List[Dict]:
    """
    For each JAAD video, parse its XML to get annotated frame numbers,
    then extract those exact frames from the corresponding video clip.

    If no video clips are present, fall back to returning annotation-only records
    (no image saved — frame_file will be None) so the rest of the pipeline can
    still run using the annotation data.

    Returns list of:
        {
          "frame_id": int,
          "file_path": str | None,
          "timestamp": str,
          "source_frame": int,
          "video_id": str,
          "jaad_annotations": List[Dict],   ← ground-truth pedestrian data
        }
    """
    os.makedirs(config.FRAMES_DIR, exist_ok=True)
    records = []
    global_frame_id = 0

    for vid_idx, video_id in enumerate(video_ids):
        xml_path = os.path.join(annotations_dir, f"{video_id}.xml")
        if not os.path.exists(xml_path):
            logger.warning(f"Missing annotation XML for {video_id}, skipping")
            continue

        annotated_frames = _parse_jaad_xml(xml_path)

        # Try to find the video clip
        video_path = _find_jaad_video(video_id)

        if video_path:
            frame_records = _extract_jaad_frames_from_video(
                video_path, video_id, annotated_frames, global_frame_id
            )
        else:
            logger.warning(
                f"No video clip for {video_id} — annotation-only mode"
            )
            frame_records = _annotation_only_records(
                video_id, annotated_frames, global_frame_id
            )

        records.extend(frame_records)
        global_frame_id += len(frame_records)
        logger.info(f"  [{vid_idx+1}/{len(video_ids)}] {video_id}: {len(frame_records)} frames extracted")

    logger.info(f"JAAD extraction complete: {len(records)} total frame records")
    return records


def _parse_jaad_xml(xml_path: str) -> Dict[int, List[Dict]]:
    """
    Parse a JAAD annotation XML and return a dict of:
        { frame_number: [ pedestrian_annotation, ... ] }

    JAAD XML specifics:
      - <track label="pedestrian|ped|people"> — no id attr on the track element
      - <box xtl ytl xbr ybr frame outside occluded>
      - outside="1" means the object is not visible — skip those boxes
      - child <attribute name="id"> holds the pedestrian ID (e.g. "0_1_3b")
      - child <attribute name="cross"> holds "crossing" or "not-crossing"
      - child <attribute name="action"> holds "standing" or "walking"
      - child <attribute name="occlusion"> holds "none", "part", or "full"
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Labels that represent pedestrians
    PED_LABELS = {"pedestrian", "ped", "people"}

    frames: Dict[int, List[Dict]] = {}

    for track in root.findall(".//track"):
        raw_label = track.get("label", "pedestrian").lower()
        label = "pedestrian" if raw_label in PED_LABELS else "vehicle"

        for box in track.findall("box"):
            # Skip boxes where the object is outside the frame
            if box.get("outside", "0") == "1":
                continue

            frame_num = int(box.get("frame", 0))

            # Bounding box — JAAD uses xtl/ytl/xbr/ybr
            xtl = float(box.get("xtl", 0))
            ytl = float(box.get("ytl", 0))
            xbr = float(box.get("xbr", 0))
            ybr = float(box.get("ybr", 0))
            bbox = [int(xtl), int(ytl), int(xbr - xtl), int(ybr - ytl)]

            # Attributes are child text elements: <attribute name="x">value</attribute>
            attrs = {
                a.get("name"): (a.text or "").strip()
                for a in box.findall("attribute")
            }

            # Pedestrian ID lives inside attrs, not on the track element
            ped_id = attrs.get("id") or attrs.get("old_id") or f"unk_{frame_num}"

            # "cross" field: "crossing" → 1, "not-crossing" → 0
            cross_val = attrs.get("cross", "not-crossing")
            crossing = 1 if cross_val == "crossing" else 0

            # Occlusion: "none" → 0, "part" → 1, "full" → 2
            occ_map = {"none": 0, "part": 1, "full": 2}
            occlusion = occ_map.get(attrs.get("occlusion", "none"), 0)

            annotation = {
                "ped_id": ped_id,
                "label": label,
                "bbox": bbox,
                "action": attrs.get("action", "unknown"),
                "crossing": crossing,
                "occlusion": occlusion,
            }

            frames.setdefault(frame_num, []).append(annotation)

    return frames


def _find_jaad_video(video_id: str) -> str | None:
    """Search common locations for a JAAD video clip."""
    candidates = [
        os.path.join(config.JAAD_ROOT, "JAAD_clips", f"{video_id}.mp4"),
        os.path.join(config.JAAD_ROOT, "videos", f"{video_id}.mp4"),
        os.path.join(config.JAAD_ROOT, "clips", f"{video_id}.mp4"),
        os.path.join(config.JAAD_ROOT, f"{video_id}.mp4"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def _extract_jaad_frames_from_video(
    video_path: str,
    video_id: str,
    annotated_frames: Dict[int, List[Dict]],
    global_offset: int,
) -> List[Dict]:
    """
    Extract annotated frames from a JAAD video clip by reading sequentially.
    Sequential read is far faster than per-frame seeking in MP4.
    Only saves frames whose number is in the annotated set.
    Also applies FRAME_SAMPLE_INTERVAL so we don't save every annotated frame
    when the annotation density is very high.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    wanted = set(annotated_frames.keys())
    interval = config.FRAME_SAMPLE_INTERVAL

    # Apply sampling: keep every Nth annotated frame to control output size
    wanted_sorted = sorted(wanted)
    wanted_sampled = set(wanted_sorted[i] for i in range(0, len(wanted_sorted), interval))

    records = []
    local_id = 0
    current_frame = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame in wanted_sampled:
            filename = f"{video_id}_frame_{current_frame:06d}.{config.FRAME_FORMAT}"
            out_path = os.path.join(config.FRAMES_DIR, filename)
            cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, config.FRAME_QUALITY])
            records.append({
                "frame_id": global_offset + local_id,
                "file_path": out_path,
                "timestamp": _frame_to_timestamp(current_frame, fps),
                "source_frame": current_frame,
                "video_id": video_id,
                "jaad_annotations": annotated_frames[current_frame],
            })
            local_id += 1

        current_frame += 1

    cap.release()
    return records


def _annotation_only_records(
    video_id: str,
    annotated_frames: Dict[int, List[Dict]],
    global_offset: int,
) -> List[Dict]:
    """Create frame records with no image file (annotation-only mode)."""
    records = []
    for local_id, frame_num in enumerate(sorted(annotated_frames.keys())):
        records.append(
            {
                "frame_id": global_offset + local_id,
                "file_path": None,
                "timestamp": _frame_to_timestamp(frame_num, 30.0),
                "source_frame": frame_num,
                "video_id": video_id,
                "jaad_annotations": annotated_frames[frame_num],
            }
        )
    return records


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _frame_to_timestamp(frame_num: int, fps: float) -> str:
    total_sec = frame_num / fps
    h = int(total_sec // 3600)
    m = int((total_sec % 3600) // 60)
    s = total_sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# ─── Stage entry point ────────────────────────────────────────────────────────

def run(context: dict) -> dict:
    mode = context.get("mode", "video")

    if mode == "jaad":
        records = extract_from_jaad(
            video_ids=context["video_ids"],
            annotations_dir=context["annotations_dir"],
        )
    else:
        records = extract_from_video(
            video_path=context["video_path"],
            fps=context.get("fps", 30.0),
        )

    return {
        "frame_records": records,
        "total_frames_extracted": len(records),
        "output_path": config.FRAMES_DIR,
    }
