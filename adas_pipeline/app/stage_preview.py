"""
stage_preview.py — Render per-stage preview images for the web dashboard.

Given the pipeline's in-memory ``frame_records`` (which still carry keypoints and
intent, unlike the reduced dataset.json), this picks ONE representative frame and
renders a series of images that each visualise what a single pipeline stage
produces on that frame:

    clean     → the cleaned input frame (no overlays)
    detect    → RT-DETR bounding boxes, coloured by class
    track     → boxes coloured/labelled by ByteTrack track ID
    pose      → RTMPose 17-keypoint skeletons on pedestrians
    behavior  → boxes labelled with the motion classification
    intent    → final danger colouring + crossing-intent badge + scene banner

Returns a manifest describing the frame and the stage images written, so the
frontend can build an interactive step-by-step explorer. Never raises for a
missing/degenerate frame — returns an empty manifest instead.
"""

import logging
import os
from typing import Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger("stage_preview")

# COCO-17 skeleton edges (index pairs) for pose rendering.
_SKELETON = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6),
]

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_MAX_W = 960  # downscale wide frames so the browser payload stays light

# BGR palette
_PED = (255, 190, 40)     # cyan-blue
_VEH = (40, 190, 255)     # orange
_SKEL = (90, 235, 120)    # green
_JOINT = (60, 220, 255)   # yellow


# ─── helpers ──────────────────────────────────────────────────────────────────

def _dets(record: Dict) -> List[Dict]:
    return record.get("detections") or record.get("objects") or []


def _score_to_bgr(score: float):
    score = max(0.0, min(1.0, float(score)))
    if score < 0.5:
        r, g = int(255 * (score / 0.5)), 255
    else:
        r, g = 255, int(255 * (1 - (score - 0.5) / 0.5))
    return (0, g, r)


def _track_color(tid) -> tuple:
    """Deterministic vivid colour from a track id."""
    h = abs(hash(str(tid)))
    # spread across hue via three offset primes, keep it bright
    return (60 + h % 180, 60 + (h // 7) % 180, 60 + (h // 13) % 180)


def _ibox(bbox):
    x, y, w, h = [int(round(float(v))) for v in bbox]
    return x, y, w, h


def _label(img, x, y, text, color, scale=0.5):
    (tw, th), _ = cv2.getTextSize(text, _FONT, scale, 1)
    ty = max(y - 6, th + 4)
    cv2.rectangle(img, (x - 2, ty - th - 4), (x + tw + 3, ty + 3), (18, 18, 18), -1)
    cv2.putText(img, text, (x, ty), _FONT, scale, color, 1, cv2.LINE_AA)


def _has_keypoints(det: Dict) -> bool:
    k = det.get("keypoints")
    if k is None:
        return False
    try:
        arr = np.asarray(k, dtype=float)
        return arr.shape[0] >= 17 and arr.shape[-1] >= 2
    except Exception:
        return False


# ─── per-stage drawing ──────────────────────────────────────────────────────────

def _draw_detect(img, dets):
    for d in dets:
        if not d.get("bbox"):
            continue
        x, y, w, h = _ibox(d["bbox"])
        color = _PED if d.get("label") == "pedestrian" else _VEH
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        conf = d.get("confidence")
        tag = d.get("label", "obj")
        if conf is not None:
            tag = f"{tag} {float(conf):.2f}"
        _label(img, x, y, tag, color, 0.45)


def _draw_track(img, dets):
    for d in dets:
        if not d.get("bbox"):
            continue
        x, y, w, h = _ibox(d["bbox"])
        tid = d.get("track_id") or d.get("id") or "?"
        color = _track_color(tid)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        _label(img, x, y, str(tid), color, 0.45)


def _draw_pose(img, dets, kpt_thr=0.3):
    for d in dets:
        if d.get("label") != "pedestrian" or not d.get("bbox"):
            continue
        x, y, w, h = _ibox(d["bbox"])
        cv2.rectangle(img, (x, y), (x + w, y + h), (70, 70, 70), 1)
        if not _has_keypoints(d):
            continue
        kp = np.asarray(d["keypoints"], dtype=float)
        sc = d.get("kpt_conf")
        sc = np.asarray(sc, dtype=float) if sc is not None else np.ones(len(kp))
        for a, b in _SKELETON:
            if a < len(kp) and b < len(kp) and sc[a] >= kpt_thr and sc[b] >= kpt_thr:
                pa = (int(kp[a][0]), int(kp[a][1]))
                pb = (int(kp[b][0]), int(kp[b][1]))
                cv2.line(img, pa, pb, _SKEL, 2, cv2.LINE_AA)
        for i in range(min(17, len(kp))):
            if sc[i] >= kpt_thr:
                cv2.circle(img, (int(kp[i][0]), int(kp[i][1])), 3, _JOINT, -1, cv2.LINE_AA)


def _draw_behavior(img, dets):
    for d in dets:
        if not d.get("bbox"):
            continue
        x, y, w, h = _ibox(d["bbox"])
        color = _PED if d.get("label") == "pedestrian" else _VEH
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        _label(img, x, y, str(d.get("behavior", "unknown")), color, 0.45)


def _draw_intent(img, dets, scene_tag, reason, frame_id):
    for d in dets:
        if not d.get("bbox"):
            continue
        x, y, w, h = _ibox(d["bbox"])
        score = float(d.get("danger_score") or 0.0)
        color = _score_to_bgr(score)
        intent = d.get("intent")
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3 if intent == "crossing" else 2)
        tid = d.get("track_id") or d.get("id") or "?"
        _label(img, x, y, f"{tid} | {score:.2f}", color, 0.45)
        if intent is not None:
            prob = int(float(d.get("crossing_prob") or 0.0) * 100)
            txt = f"{'CROSS' if intent == 'crossing' else 'SAFE'} {prob}%"
            ic = (0, 80, 255) if intent == "crossing" else (50, 220, 50)
            by = y + h + 16
            (tw, th), _ = cv2.getTextSize(txt, _FONT, 0.42, 1)
            cv2.rectangle(img, (x - 2, by - th - 4), (x + tw + 4, by + 3), (12, 12, 12), -1)
            cv2.rectangle(img, (x - 2, by - th - 4), (x + tw + 4, by + 3), ic, 1)
            cv2.putText(img, txt, (x, by), _FONT, 0.42, ic, 1, cv2.LINE_AA)
    # scene banner
    W = img.shape[1]
    bg = (0, 0, 180) if scene_tag == "DANGER" else (0, 120, 0)
    cv2.rectangle(img, (0, 0), (W, 30), bg, -1)
    head = f"[{scene_tag}]  Frame {frame_id}"
    cv2.putText(img, head, (8, 21), _FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ─── frame selection + orchestration ────────────────────────────────────────────

def _pick_frame(records: List[Dict]) -> Optional[Dict]:
    """Deterministic pick: readable frame with >=1 pedestrian and highest total
    danger; fall back to most objects; then any readable frame."""
    best = None
    best_key = None
    for rec in records:
        fp = rec.get("file_path")
        if not fp or not os.path.exists(fp):
            continue
        dets = _dets(rec)
        if not dets:
            continue
        n_ped = sum(1 for d in dets if d.get("label") == "pedestrian")
        danger = sum(float(d.get("danger_score") or 0.0) for d in dets)
        # sort key: prefer having pedestrians, then total danger, then object count
        key = (1 if n_ped else 0, round(danger, 4), len(dets), -int(rec.get("frame_id", 0)))
        if best_key is None or key > best_key:
            best_key, best = key, rec
    return best


_STAGE_META = {
    "clean":    ("Frame Cleaning", "Denoised, CLAHE-normalised input frame that feeds perception."),
    "detect":   ("Detection (RT-DETR)", "Pedestrians and vehicles localised as bounding boxes."),
    "track":    ("Tracking (ByteTrack)", "Each object gets a stable ID across frames (colour = ID)."),
    "pose":     ("Pose (RTMPose)", "17-keypoint skeletons drive the body-language features."),
    "behavior": ("Behavior", "Motion classified: walking / crossing / running / stopping / driving."),
    "intent":   ("Intent + Danger", "Crossing prediction, danger colouring, and the scene tag."),
}


def render_stage_previews(frame_records: List[Dict], out_dir: str,
                          prefix: str = "step", kpt_thr: float = 0.3) -> Dict:
    """Render per-stage previews for one representative frame.

    Returns a manifest dict (empty ``{"stages": []}`` if nothing renderable).
    Never raises — preview generation must never fail the job.
    """
    try:
        if not frame_records:
            return {"stages": []}
        rec = _pick_frame(frame_records)
        if rec is None:
            return {"stages": []}

        base = cv2.imread(rec["file_path"])
        if base is None:
            return {"stages": []}

        # Downscale wide frames (keeps skeleton coords consistent via scale factor)
        H, W = base.shape[:2]
        if W > _MAX_W:
            scale = _MAX_W / W
            base = cv2.resize(base, (int(W * scale), int(H * scale)))
        else:
            scale = 1.0

        dets = _dets(rec)
        # Scale bbox + keypoints to the (possibly) resized canvas.
        sdets = []
        for d in dets:
            nd = dict(d)
            if d.get("bbox"):
                nd["bbox"] = [float(v) * scale for v in d["bbox"]]
            if _has_keypoints(d):
                nd["keypoints"] = (np.asarray(d["keypoints"], dtype=float) * scale)
            sdets.append(nd)

        n_ped = sum(1 for d in sdets if d.get("label") == "pedestrian")
        n_veh = sum(1 for d in sdets if d.get("label") == "vehicle")
        scene_tag = rec.get("scene_tag", "SAFE")
        frame_id = rec.get("frame_id", 0)
        has_pose = any(_has_keypoints(d) for d in sdets)

        os.makedirs(out_dir, exist_ok=True)

        renderers = {
            "clean":    lambda im: None,
            "detect":   lambda im: _draw_detect(im, sdets),
            "track":    lambda im: _draw_track(im, sdets),
            "pose":     lambda im: _draw_pose(im, sdets, kpt_thr),
            "behavior": lambda im: _draw_behavior(im, sdets),
            "intent":   lambda im: _draw_intent(im, sdets, scene_tag,
                                                rec.get("safety_reason", ""), frame_id),
        }
        order = ["clean", "detect", "track", "pose", "behavior", "intent"]
        if not has_pose:
            order.remove("pose")  # pose disabled or no keypoints on this frame

        stat = {
            "clean": f"{W}x{H}",
            "detect": f"{n_ped + n_veh} objects",
            "track": f"{n_ped + n_veh} tracks",
            "pose": f"{n_ped} skeleton(s)",
            "behavior": f"{n_ped} pedestrian(s)",
            "intent": scene_tag,
        }

        stages = []
        for key in order:
            img = base.copy()
            renderers[key](img)
            fname = f"{prefix}_{key}.jpg"
            cv2.imwrite(os.path.join(out_dir, fname), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 86])
            label, desc = _STAGE_META[key]
            stages.append({"key": key, "label": label, "desc": desc,
                           "file": fname, "stat": stat.get(key, "")})

        return {
            "frame_id": frame_id,
            "scene_tag": scene_tag,
            "n_ped": n_ped,
            "n_veh": n_veh,
            "stages": stages,
        }
    except Exception as e:  # never fail the job over a cosmetic preview
        logger.warning(f"stage preview generation failed: {e}")
        return {"stages": []}
