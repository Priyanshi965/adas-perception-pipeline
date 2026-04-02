"""
Quick annotated video preview — reads dataset.json and renders live to a window.
Controls: SPACE = pause/resume  |  Q = quit  |  LEFT/RIGHT = step frames
"""
import json, sys, cv2, numpy as np, os
sys.path.insert(0, os.path.dirname(__file__))
import config
from visualizer import (
    _load_or_synthetic, _draw_detection, _draw_scene_banner,
)

DATASET = os.path.join(config.FINAL_DIR, config.OUTPUT_JSON_NAME)
FPS     = 30

def main():
    print(f"Loading {DATASET} ...")
    with open(DATASET, encoding="utf-8") as f:
        records = json.load(f)
    print(f"{len(records)} frames — SPACE=pause  Q=quit  ←/→=step")

    # detect frame size from first real frame
    sample_path = next((r.get("file_path") for r in records if r.get("file_path")), None)
    if sample_path and os.path.exists(sample_path):
        tmp = cv2.imread(sample_path)
        h, w = tmp.shape[:2]
    else:
        w, h = config.FRAME_WIDTH, config.FRAME_HEIGHT

    delay = max(1, int(1000 / FPS))
    paused = False
    idx    = 0

    cv2.namedWindow("ADAS Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("ADAS Preview", min(w, 1280), min(h, 720))

    while 0 <= idx < len(records):
        record     = records[idx]
        frame_id   = record.get("frame_id", idx)
        scene_tag  = record.get("scene_tag", "SAFE")
        reason     = record.get("safety_reason", "")
        timestamp  = record.get("timestamp", "")
        objects    = record.get("objects") or record.get("detections") or []
        file_path  = record.get("file_path")

        img = _load_or_synthetic(file_path, w, h)
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h))

        for obj in objects:
            _draw_detection(img, obj)
        _draw_scene_banner(img, scene_tag, reason, frame_id, timestamp)

        # frame counter overlay
        cv2.putText(img, f"Frame {idx+1}/{len(records)}", (w-180, h-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100,140,180), 1, cv2.LINE_AA)

        cv2.imshow("ADAS Preview", img)
        key = cv2.waitKey(0 if paused else delay) & 0xFF

        if   key == ord('q'):           break
        elif key == ord(' '):           paused = not paused
        elif key == 81 or key == 2:     idx = max(0, idx - 1)   # left arrow
        elif key == 83 or key == 3:     idx = min(len(records)-1, idx + 1)  # right
        elif not paused:                idx += 1

    cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
