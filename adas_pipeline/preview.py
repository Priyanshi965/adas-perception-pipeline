"""
Quick annotated video preview.
Tries to play the rendered annotated_video.mp4/.avi first.
Falls back to live rendering from dataset.json if the video file is missing or empty.
Controls: SPACE = pause/resume  |  Q = quit  |  LEFT/RIGHT = step frames (JSON mode only)
"""
import json, sys, cv2, os
sys.path.insert(0, os.path.dirname(__file__))
import config
from visualizer import _load_or_synthetic, _draw_detection, _draw_scene_banner

FINAL_DIR = config.FINAL_DIR
DATASET   = os.path.join(FINAL_DIR, config.OUTPUT_JSON_NAME)


def play_video_file(path):
    """Play a rendered video file directly with cv2."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return False
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return False

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = max(1, int(1000 / fps))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Playing {path}  [{w}x{h} @ {fps:.1f}fps, {total} frames]")
    print("SPACE=pause  Q=quit")

    cv2.namedWindow("ADAS Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ADAS Preview", min(w, 1280), min(h, 720))

    paused = False
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop
                continue
        key = cv2.waitKey(0 if paused else delay) & 0xFF
        if not paused:
            cv2.imshow("ADAS Preview", frame)
        if   key == ord('q'):  break
        elif key == ord(' '):  paused = not paused

    cap.release()
    cv2.destroyAllWindows()
    return True


def play_from_json():
    """Live-render frames from dataset.json (fallback when video file is missing)."""
    if not os.path.exists(DATASET):
        print(f"ERROR: No dataset found at {DATASET}")
        return

    print(f"Loading {DATASET} ...")
    with open(DATASET, encoding="utf-8") as f:
        records = json.load(f)
    print(f"{len(records)} frames — SPACE=pause  Q=quit  ←/→=step")

    # detect actual frame size
    w, h = config.FRAME_WIDTH, config.FRAME_HEIGHT
    for r in records[:20]:
        fp = r.get("file_path")
        if fp and os.path.exists(fp):
            tmp = cv2.imread(fp)
            if tmp is not None:
                h, w = tmp.shape[:2]
                break

    delay  = max(1, int(1000 / 30))
    paused = False
    idx    = 0

    cv2.namedWindow("ADAS Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ADAS Preview", min(w, 1280), min(h, 720))

    while 0 <= idx < len(records):
        record    = records[idx]
        frame_id  = record.get("frame_id", idx)
        scene_tag = record.get("scene_tag", "SAFE")
        reason    = record.get("safety_reason", "")
        timestamp = record.get("timestamp", "")
        objects   = record.get("objects") or record.get("detections") or []
        file_path = record.get("file_path")

        img = _load_or_synthetic(file_path, w, h)
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h))

        for obj in objects:
            _draw_detection(img, obj)
        _draw_scene_banner(img, scene_tag, reason, frame_id, timestamp)

        cv2.putText(img, f"Frame {idx+1}/{len(records)}", (w - 185, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 140, 180), 1, cv2.LINE_AA)

        cv2.imshow("ADAS Preview", img)
        key = cv2.waitKey(0 if paused else delay) & 0xFF

        if   key == ord('q'):          break
        elif key == ord(' '):          paused = not paused
        elif key in (81, 2):           idx = max(0, idx - 1)
        elif key in (83, 3):           idx = min(len(records) - 1, idx + 1)
        elif not paused:               idx += 1

    cv2.destroyAllWindows()
    print("Done.")


def main():
    # Try rendered video files first (prefer mp4, fall back to avi)
    for ext in ("annotated_video.mp4", "annotated_video.avi"):
        path = os.path.join(FINAL_DIR, ext)
        if os.path.exists(path) and os.path.getsize(path) > 10_000:
            if play_video_file(path):
                return

    print("Rendered video not found or empty — falling back to live JSON render...")
    play_from_json()


if __name__ == "__main__":
    main()
