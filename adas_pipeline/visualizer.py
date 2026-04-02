"""
visualizer.py — Renders annotated video output from the pipeline dataset.

For each frame record it draws:
  - Bounding boxes colored by danger_score (green → yellow → red)
  - Track ID, behavior, and danger score label
  - Scene tag banner (DANGER / SAFE) at the top
  - Mini danger bar per object

Works in two modes:
  - Image mode: uses extracted frame images from output/clean_frames/
  - Synthetic mode: draws on a plain background (when no images exist — JAAD
    annotation-only runs)

Usage:
    python visualizer.py                        # reads output/final/dataset.json
    python visualizer.py --input path/to.json
    python visualizer.py --max-frames 500       # limit for quick preview
    python visualizer.py --fps 10
"""

import argparse
import json
import logging
import os
import sys

import cv2
import numpy as np

import config

logger = logging.getLogger("visualizer")

# ─── Color helpers ────────────────────────────────────────────────────────────

def _score_to_bgr(score: float):
    """Map 0.0→green, 0.5→yellow, 1.0→red in BGR."""
    score = max(0.0, min(1.0, score))
    if score < 0.5:
        t = score / 0.5
        r, g = int(255 * t), 255
    else:
        t = (score - 0.5) / 0.5
        r, g = 255, int(255 * (1 - t))
    return (0, g, r)   # BGR


LABEL_COLOR = {
    "pedestrian": (255, 200, 50),   # blue-ish
    "vehicle":    (50, 200, 255),   # orange-ish
}

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.45
THICKNESS  = 1

# ─── Frame canvas ─────────────────────────────────────────────────────────────

def _load_or_synthetic(file_path, width: int, height: int) -> np.ndarray:
    """Return a BGR frame image, either from disk or synthesized."""
    if file_path and os.path.exists(file_path):
        img = cv2.imread(file_path)
        if img is not None:
            return img

    # Synthetic: dark gray road-like background
    canvas = np.full((height, width, 3), 40, dtype=np.uint8)
    # Draw a simple lane marker
    mid_x = width // 2
    for y in range(0, height, 40):
        cv2.line(canvas, (mid_x, y), (mid_x, y + 20), (80, 80, 80), 2)
    return canvas


# ─── Drawing ──────────────────────────────────────────────────────────────────

def _draw_danger_bar(img, x: int, y: int, score: float, bar_w: int = 40, bar_h: int = 6):
    """Draw a small filled danger bar above a bounding box."""
    filled = int(bar_w * score)
    cv2.rectangle(img, (x, y - bar_h - 2), (x + bar_w, y - 2), (60, 60, 60), -1)
    if filled > 0:
        color = _score_to_bgr(score)
        cv2.rectangle(img, (x, y - bar_h - 2), (x + filled, y - 2), color, -1)


def _draw_detection(img, det: dict):
    bbox  = det.get("bbox") or [0, 0, 10, 10]
    x, y, w, h = [int(v) for v in bbox]
    score = float(det.get("danger_score") or 0.0)
    behavior = det.get("behavior", "?")
    tid   = det.get("track_id") or det.get("id") or "?"

    color = _score_to_bgr(score)

    # Bounding box
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

    # Danger bar above box
    _draw_danger_bar(img, x, y, score, bar_w=w)

    # Label text: "Person_3b | crossing | 0.95"
    text = f"{tid} | {behavior} | {score:.2f}"
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
    tx = max(x, 2)
    ty = max(y - bar_h_offset(y, th), th + 4)

    cv2.rectangle(img, (tx - 2, ty - th - 3), (tx + tw + 2, ty + 2), (20, 20, 20), -1)
    cv2.putText(img, text, (tx, ty), FONT, FONT_SCALE, color, THICKNESS, cv2.LINE_AA)


def bar_h_offset(y: int, text_h: int) -> int:
    return 18 + text_h if y > 30 else -(18 + text_h)


def _draw_scene_banner(img, scene_tag: str, safety_reason: str, frame_id: int, timestamp: str):
    h, w = img.shape[:2]
    banner_h = 32

    if scene_tag == "DANGER":
        bg = (0, 0, 180)
        fg = (255, 255, 255)
    else:
        bg = (0, 120, 0)
        fg = (255, 255, 255)

    cv2.rectangle(img, (0, 0), (w, banner_h), bg, -1)

    tag_text = f"[{scene_tag}]  Frame {frame_id}  {timestamp}"
    cv2.putText(img, tag_text, (8, 22), FONT, 0.55, fg, 1, cv2.LINE_AA)

    # Truncate reason to fit
    reason_x = 350
    max_reason_w = w - reason_x - 10
    reason = safety_reason or ""
    while reason:
        (rw, _), _ = cv2.getTextSize(reason, FONT, 0.42, 1)
        if rw <= max_reason_w:
            break
        reason = reason[: len(reason) - 4] + "..."
    cv2.putText(img, reason, (reason_x, 22), FONT, 0.42, (220, 220, 220), 1, cv2.LINE_AA)


# ─── Main render loop ─────────────────────────────────────────────────────────

def render_video(
    dataset_path: str,
    output_path: str,
    fps: float = 10.0,
    frame_width: int = 1920,
    frame_height: int = 1080,
    max_frames: int = None,
):
    logger.info(f"Loading dataset: {dataset_path}")
    with open(dataset_path, encoding="utf-8") as f:
        records = json.load(f)

    if max_frames:
        records = records[:max_frames]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Try H.264 (browser-compatible) first, fall back to mp4v
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {output_path}")

    logger.info(f"Rendering {len(records)} frames -> {output_path}")

    for i, record in enumerate(records):
        frame_id  = record.get("frame_id", i)
        timestamp = record.get("timestamp", "")
        scene_tag = record.get("scene_tag", "SAFE")
        reason    = record.get("safety_reason", "")
        file_path = record.get("file_path")
        objects   = record.get("objects") or record.get("detections") or []

        img = _load_or_synthetic(file_path, frame_width, frame_height)

        # Resize to target if needed
        if img.shape[1] != frame_width or img.shape[0] != frame_height:
            img = cv2.resize(img, (frame_width, frame_height))

        # Draw detections
        for obj in objects:
            _draw_detection(img, obj)

        # Draw scene banner
        _draw_scene_banner(img, scene_tag, reason, frame_id, timestamp)

        writer.write(img)

        if i % 500 == 0:
            logger.info(f"  Rendered {i}/{len(records)} frames...")

    writer.release()
    logger.info(f"Video saved: {output_path}")
    return output_path


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Render annotated video from pipeline output")
    parser.add_argument("--input",  default=os.path.join(config.FINAL_DIR, config.OUTPUT_JSON_NAME))
    parser.add_argument("--output", default=os.path.join(config.FINAL_DIR, "annotated_video.mp4"))
    parser.add_argument("--fps",    type=float, default=config.JAAD_FPS)
    parser.add_argument("--width",  type=int,   default=config.JAAD_FRAME_WIDTH)
    parser.add_argument("--height", type=int,   default=config.JAAD_FRAME_HEIGHT)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Only render first N frames (quick preview)")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    args = parse_args()
    render_video(
        dataset_path=args.input,
        output_path=args.output,
        fps=args.fps,
        frame_width=args.width,
        frame_height=args.height,
        max_frames=args.max_frames,
    )
