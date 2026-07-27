"""
pose_estimator.py — 17-keypoint body-pose extraction (YOLO-pose).

Two entry points:
  • PoseEstimator.estimate_crop(frame, bbox)  — run pose on a padded crop around a
    known box (used by the offline JAAD dataset builder, which has GT boxes).
  • PoseEstimator.estimate_frame(frame)        — run pose on a whole frame and return
    every person; used by the live pipeline stage, which then matches skeletons to
    tracked pedestrian boxes by IoU.

The pipeline stage run() attaches, to every pedestrian detection that has an
image available:
    det["keypoints"]     — (17, 2) pixel coords
    det["kpt_conf"]      — (17,) confidences
    det["pose_features"] — dict of body_language features (see body_language.py)
"""

import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import config
from modules import body_language as bl

logger = logging.getLogger("pose_estimator")

_model = None


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO

        path = config.POSE_MODEL_PATH
        if not os.path.exists(path):
            logger.warning(f"Pose weights not at {path}; downloading {config.POSE_MODEL_FALLBACK}")
            path = config.POSE_MODEL_FALLBACK
        _model = YOLO(path)
        logger.info(f"Pose model loaded: {path}")
    return _model


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    """IoU of two [x, y, w, h] boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2, bx2, by2 = ax + aw, ay + ah, bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class PoseEstimator:
    def __init__(self):
        self._m = _get_model()
        self.conf = getattr(config, "POSE_CONF_THRESHOLD", 0.35)
        self.pad = getattr(config, "POSE_CROP_PAD", 0.15)

    # ── whole-frame pose (returns all persons) ──
    def estimate_frame(self, frame_bgr) -> List[Dict]:
        r = self._m.predict(frame_bgr, conf=self.conf, verbose=False)[0]
        out: List[Dict] = []
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            return out
        xy = r.keypoints.xy.cpu().numpy()          # (N, 17, 2)
        cf = (r.keypoints.conf.cpu().numpy()
              if r.keypoints.conf is not None else np.ones(xy.shape[:2], np.float32))
        boxes = r.boxes.xywh.cpu().numpy()         # centre-form
        for i in range(len(boxes)):
            cx, cy, w, h = boxes[i]
            out.append({
                "bbox": [float(cx - w / 2), float(cy - h / 2), float(w), float(h)],
                "keypoints": xy[i],
                "kpt_conf": cf[i],
            })
        return out

    # ── crop-based pose around a known box (GT-box path) ──
    def estimate_crop(self, frame_bgr, bbox: Sequence[float]) -> Optional[Dict]:
        H, W = frame_bgr.shape[:2]
        x, y, w, h = bbox
        px, py = w * self.pad, h * self.pad
        x1 = max(0, int(x - px)); y1 = max(0, int(y - py))
        x2 = min(W, int(x + w + px)); y2 = min(H, int(y + h + py))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        crop = frame_bgr[y1:y2, x1:x2]
        r = self._m.predict(crop, conf=self.conf, verbose=False)[0]
        if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
            return None
        # Pick the person whose box best overlaps the (crop-local) target box
        target_local = [x - x1, y - y1, w, h]
        boxes = r.boxes.xywh.cpu().numpy()
        best_i, best_iou = 0, -1.0
        for i in range(len(boxes)):
            cx, cy, bw, bh = boxes[i]
            iou = _iou(target_local, [cx - bw / 2, cy - bh / 2, bw, bh])
            if iou > best_iou:
                best_i, best_iou = i, iou
        xy = r.keypoints.xy.cpu().numpy()[best_i].copy()          # crop-local coords
        cf = (r.keypoints.conf.cpu().numpy()[best_i]
              if r.keypoints.conf is not None else np.ones(17, np.float32))
        xy[:, 0] += x1                                            # map back to frame
        xy[:, 1] += y1
        return {"keypoints": xy, "kpt_conf": cf, "match_iou": float(best_iou)}


_estimator: Optional[PoseEstimator] = None


def _get_estimator() -> PoseEstimator:
    global _estimator
    if _estimator is None:
        _estimator = PoseEstimator()
    return _estimator


# ─── Pipeline stage ───────────────────────────────────────────────────────────

def run(context: dict) -> dict:
    """
    Attach keypoints + body-language features to pedestrian detections.

    Reads:  context["frame_records"] — each with "detections"/"objects" and a
            readable image at record["file_path"].
    Writes: det["keypoints"], det["kpt_conf"], det["pose_features"].
    Skips gracefully (leaves detections untouched) when pose is disabled or the
    frame image is unavailable.
    """
    if not getattr(config, "POSE_ENABLED", True):
        logger.info("Pose disabled (config.POSE_ENABLED=False) — skipping")
        return {"frame_records": context.get("frame_records", [])}

    import cv2

    records = context.get("frame_records", [])
    est = _get_estimator()
    kpt_thr = getattr(config, "POSE_KPT_CONF_THR", 0.30)
    match_iou = getattr(config, "POSE_MATCH_IOU", 0.30)

    enriched, n_posed = [], 0
    for record in records:
        path = record.get("file_path")
        key = "detections" if "detections" in record else "objects"
        dets = record.get(key, [])
        peds = [d for d in dets if d.get("label") == "pedestrian"]

        if not peds or not path or not os.path.exists(path):
            enriched.append(record)
            continue

        frame = cv2.imread(path)
        if frame is None:
            enriched.append(record)
            continue

        persons = est.estimate_frame(frame)
        new_dets = []
        for d in dets:
            if d.get("label") != "pedestrian" or not persons:
                new_dets.append(d)
                continue
            best = max(persons, key=lambda p: _iou(d["bbox"], p["bbox"]))
            if _iou(d["bbox"], best["bbox"]) < match_iou:
                new_dets.append(d)
                continue
            feats = bl.compute_pose_features(best["keypoints"], best["kpt_conf"],
                                             d["bbox"], mirror=False, conf_thr=kpt_thr)
            new_dets.append({
                **d,
                "keypoints": best["keypoints"].tolist(),
                "kpt_conf": best["kpt_conf"].tolist(),
                "pose_features": feats,
            })
            n_posed += 1
        enriched.append({**record, key: new_dets})

    logger.info(f"Pose stage: attached skeletons to {n_posed} pedestrian detections")
    return {"frame_records": enriched}
