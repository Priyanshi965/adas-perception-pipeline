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


INTENT_COLORS = {
    "crossing":     (0,   80, 255),   # orange-red  — high attention
    "not_crossing": (50, 220,  50),   # green        — safe
}


def _draw_intent_badge(img, x: int, y: int, w: int, intent: str, prob: float, conf: float):
    """Draw a compact intent badge below the bounding box bottom edge."""
    label   = "CROSS" if intent == "crossing" else "SAFE"
    pct     = int(prob * 100)
    text    = f"{label} {pct}%"
    color   = INTENT_COLORS.get(intent, (200, 200, 200))

    (tw, th), _ = cv2.getTextSize(text, FONT, 0.42, 1)
    bx = max(x, 2)
    by = y + w + 14   # below the bottom of the bbox
    # Background pill
    cv2.rectangle(img, (bx - 3, by - th - 3), (bx + tw + 3, by + 3), (15, 15, 15), -1)
    cv2.rectangle(img, (bx - 3, by - th - 3), (bx + tw + 3, by + 3), color, 1)
    cv2.putText(img, text, (bx, by), FONT, 0.42, color, 1, cv2.LINE_AA)

    # Thin confidence bar inside badge
    bar_total = tw + 6
    bar_filled = int(bar_total * conf)
    if bar_filled > 0:
        cv2.line(img, (bx - 3, by + 3), (bx - 3 + bar_filled, by + 3), color, 2)


def _draw_detection(img, det: dict):
    bbox  = det.get("bbox") or [0, 0, 10, 10]
    x, y, w, h = [int(v) for v in bbox]
    score = float(det.get("danger_score") or 0.0)
    behavior = det.get("behavior", "?")
    tid   = det.get("track_id") or det.get("id") or "?"
    intent  = det.get("intent")
    prob    = float(det.get("crossing_prob") or 0.0)
    conf    = float(det.get("intent_conf")   or 0.0)

    color = _score_to_bgr(score)

    # Bounding box — thicker for pedestrians with crossing intent
    thickness = 3 if intent == "crossing" else 2
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

    # Danger bar above box
    _draw_danger_bar(img, x, y, score, bar_w=w)

    # Label text: "Person_3b | crossing | 0.95"
    text = f"{tid} | {behavior} | {score:.2f}"
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
    tx = max(x, 2)
    ty = max(y - bar_h_offset(y, th), th + 4)

    cv2.rectangle(img, (tx - 2, ty - th - 3), (tx + tw + 2, ty + 2), (20, 20, 20), -1)
    cv2.putText(img, text, (tx, ty), FONT, FONT_SCALE, color, THICKNESS, cv2.LINE_AA)

    # Intent badge (pedestrians only)
    if intent is not None:
        _draw_intent_badge(img, x, y, h, intent, prob, conf)


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

    # Auto-detect actual frame dimensions from the first readable image
    for rec in records[:20]:
        fp = rec.get("file_path")
        if fp and os.path.exists(fp):
            tmp = cv2.imread(fp)
            if tmp is not None:
                frame_height, frame_width = tmp.shape[:2]
                break

    logger.info(f"Rendering {len(records)} frames -> {output_path}  [{frame_width}x{frame_height}]")

    # Build rendered frames list
    frames = []
    for i, record in enumerate(records):
        frame_id  = record.get("frame_id", i)
        timestamp = record.get("timestamp", "")
        scene_tag = record.get("scene_tag", "SAFE")
        reason    = record.get("safety_reason", "")
        file_path = record.get("file_path")
        objects   = record.get("objects") or record.get("detections") or []

        img = _load_or_synthetic(file_path, frame_width, frame_height)
        if img.shape[1] != frame_width or img.shape[0] != frame_height:
            img = cv2.resize(img, (frame_width, frame_height))
        for obj in objects:
            _draw_detection(img, obj)
        _draw_scene_banner(img, scene_tag, reason, frame_id, timestamp)
        frames.append(img)

        if i % 500 == 0:
            logger.info(f"  Rendered {i}/{len(records)} frames...")

    # ── Encoder waterfall ──────────────────────────────────────────────
    # Priority (browser-playable H.264 first):
    # 1) PyAV         → true H.264 MP4, pure Python, best choice
    # 2) ffmpeg CLI   → re-encode via subprocess if ffmpeg is on PATH
    # 3) imageio      → H.264 via bundled ffmpeg binary
    # 4) OpenCV avc1  → H.264 on Windows if codec is installed
    # 5) OpenCV mp4v  → MPEG-4 fallback (download-only, not inline)

    def _encode_pyav(frames, output_path, fps, w, h):
        import av  # noqa: PLC0415
        container = av.open(output_path, mode="w")
        stream = container.add_stream("h264", rate=int(fps))
        stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
        stream.options = {"crf": "23", "preset": "fast"}
        for img in frames:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frm = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            for pkt in stream.encode(frm):
                container.mux(pkt)
        for pkt in stream.encode(None):
            container.mux(pkt)
        container.close()

    def _encode_ffmpeg_cli(frames, output_path, fps, w, h):
        """Write frames as raw pipe to ffmpeg CLI → H.264 MP4."""
        import subprocess, shutil  # noqa: PLC0415
        if not shutil.which("ffmpeg"):
            raise RuntimeError("ffmpeg not on PATH")
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "bgr24",
            "-r", str(fps),
            "-i", "pipe:0",
            "-vcodec", "libx264", "-crf", "23",
            "-preset", "fast", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for img in frames:
            proc.stdin.write(img.tobytes())
        proc.stdin.close()
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg CLI exited non-zero")

    def _encode_imageio(frames, output_path, fps, w, h):
        import imageio  # noqa: PLC0415
        writer = imageio.get_writer(
            output_path, fps=fps, codec="libx264", quality=None,
            output_params=["-crf", "23", "-preset", "fast",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )
        for img in frames:
            writer.append_data(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        writer.close()

    def _encode_opencv_avc1(frames, output_path, fps, w, h):
        """OpenCV with H.264 fourcc — works on Windows with codec installed."""
        wr = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
        if not wr.isOpened():
            raise RuntimeError("OpenCV avc1 writer failed to open")
        for img in frames:
            wr.write(img)
        wr.release()
        # Verify the file was actually written (avc1 silently falls back to nothing on some platforms)
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
            raise RuntimeError("OpenCV avc1 produced empty/missing file")

    def _encode_opencv_mp4v(frames, output_path, fps, w, h):
        """MPEG-4 Part 2 — not browser-streamable but works as a download."""
        wr = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        if not wr.isOpened():
            raise RuntimeError("OpenCV mp4v writer failed to open")
        for img in frames:
            wr.write(img)
        wr.release()

    encoders = [
        ("PyAV H.264",        _encode_pyav),
        ("ffmpeg CLI H.264",  _encode_ffmpeg_cli),
        ("imageio H.264",     _encode_imageio),
        ("OpenCV avc1 H.264", _encode_opencv_avc1),
        ("OpenCV mp4v",       _encode_opencv_mp4v),
    ]

    used_encoder = None
    for name, fn in encoders:
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
            fn(frames, output_path, fps, frame_width, frame_height)
            if os.path.exists(output_path) and os.path.getsize(output_path) > 500:
                logger.info(f"Video saved ({name}): {output_path}")
                used_encoder = name
                break
        except Exception as e:
            logger.warning(f"Encoder '{name}' failed: {e} — trying next...")

    if not used_encoder:
        raise RuntimeError(
            "All video encoders failed. Install 'av' (pip install av) for H.264 "
            "browser support, or ensure ffmpeg is on PATH."
        )

    # If we fell back to mp4v (not browser-playable), warn loudly so the user
    # knows the inline player may show a grey screen.
    if "mp4v" in used_encoder:
        logger.warning(
            "VIDEO CODEC WARNING: mp4v (MPEG-4 Part 2) was used as a last resort. "
            "This file is NOT playable inline in browsers (Chrome/Firefox/Safari require H.264). "
            "Install PyAV: pip install av"
        )

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
