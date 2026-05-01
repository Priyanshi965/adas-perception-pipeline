"""
tracker.py — Assigns consistent track IDs across frames using pre-computed detections.

IMPORTANT: This tracker NEVER re-runs YOLO. It works entirely on the bounding
boxes produced by Stage 4 (detector.py). Re-running detection at this stage is
the #1 cause of hour-long pipeline hangs on large datasets.

Two modes (config.USE_BYTETRACK):
  True  → ByteSort: IoU + confidence-weighted cost matrix (ByteTrack-style logic,
           no subprocess, no extra model load, O(D²) per frame)
  False → SimpleIoU: greedy O(D²) IoU matching (fastest, good enough for JAAD)

JAAD annotation-only frames bypass both trackers — the XML ped_id is used
directly as track_id (most reliable, zero cost).
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

import config

logger = logging.getLogger("tracker")


# ─── IoU helper ───────────────────────────────────────────────────────────────

def _iou(a: List[int], b: List[int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


# ─── Hungarian assignment (pure NumPy, no scipy needed) ──────────────────────

def _greedy_match(cost: np.ndarray, threshold: float) -> List[Tuple[int, int]]:
    """
    Greedy matching: repeatedly pick the lowest-cost (row, col) pair.
    O(n²) — fast for typical detection counts (<200 per frame).
    Returns list of (det_idx, track_idx) pairs with cost < threshold.
    """
    used_rows, used_cols = set(), set()
    pairs = []
    indices = np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]
    for r, c in indices:
        if cost[r, c] >= threshold:
            break
        if r in used_rows or c in used_cols:
            continue
        pairs.append((int(r), int(c)))
        used_rows.add(r)
        used_cols.add(c)
    return pairs


# ─── Simple IoU tracker ───────────────────────────────────────────────────────

class SimpleIoUTracker:
    """Greedy IoU tracker — fast, zero dependencies."""

    def __init__(self):
        self._next: Dict[str, int] = {}
        self._active: List[Dict]   = []
        self._age: Dict[str, int]  = {}

    def _new_id(self, label: str) -> str:
        prefix = "Person" if label == "pedestrian" else "Car"
        idx = self._next.get(prefix, 0)
        self._next[prefix] = idx + 1
        return f"{prefix}_{idx}"

    def update(self, detections: List[Dict]) -> List[Dict]:
        if not detections:
            for p in self._active:
                self._age[p["track_id"]] = self._age.get(p["track_id"], 0) + 1
            self._active = [
                p for p in self._active
                if self._age.get(p["track_id"], 0) < config.MAX_TRACK_AGE
            ]
            return []

        if not self._active:
            result = [{**d, "track_id": self._new_id(d["label"])} for d in detections]
            self._active = result[:]
            return result

        n_d, n_t = len(detections), len(self._active)
        cost = np.ones((n_d, n_t), dtype=np.float32)
        for i, det in enumerate(detections):
            for j, trk in enumerate(self._active):
                if det["label"] == trk["label"]:
                    cost[i, j] = 1.0 - _iou(det["bbox"], trk["bbox"])

        threshold = 1.0 - config.IOU_MATCH_THRESHOLD
        pairs = _greedy_match(cost, threshold)

        matched_det  = {r for r, _ in pairs}
        matched_trk  = {c for _, c in pairs}
        det_to_track = {r: self._active[c]["track_id"] for r, c in pairs}

        result = []
        for i, det in enumerate(detections):
            tid = det_to_track.get(i, self._new_id(det["label"]))
            result.append({**det, "track_id": tid})
            self._age[tid] = 0

        for j, trk in enumerate(self._active):
            if j not in matched_trk:
                tid = trk["track_id"]
                self._age[tid] = self._age.get(tid, 0) + 1

        self._active = result + [
            p for p in self._active
            if self._active.index(p) not in matched_trk
            and self._age.get(p["track_id"], 0) < config.MAX_TRACK_AGE
        ]
        seen = {r["track_id"] for r in result}
        self._active = result + [p for p in self._active if p["track_id"] not in seen]
        return result


# ─── ByteSort (ByteTrack-style, using pre-computed confidences) ───────────────

class ByteSortTracker:
    """
    ByteTrack-inspired tracker that uses pre-computed detection boxes + scores.
    High-confidence (>= conf_high) detections are matched first; low-confidence
    detections are used for a second-pass recovery of lost tracks.
    No YOLO re-run — works entirely on the bboxes from Stage 4.
    """

    def __init__(
        self,
        conf_high: float = 0.6,
        conf_low:  float = 0.1,
        iou_thresh: float = None,
        max_age:    int   = None,
    ):
        self.conf_high  = conf_high
        self.conf_low   = conf_low
        self.iou_thresh = iou_thresh or config.IOU_MATCH_THRESHOLD
        self.max_age    = max_age    or config.MAX_TRACK_AGE
        self._next: Dict[str, int] = {}
        self._tracks: List[Dict]   = []

    def _new_id(self, label: str) -> str:
        prefix = "Person" if label == "pedestrian" else "Car"
        idx = self._next.get(prefix, 0)
        self._next[prefix] = idx + 1
        return f"{prefix}_{idx}"

    def _match(self, dets: List[Dict]) -> Tuple[Dict[int, str], List[int]]:
        if not self._tracks or not dets:
            return {}, list(range(len(dets)))

        cost = np.ones((len(dets), len(self._tracks)), dtype=np.float32)
        for i, d in enumerate(dets):
            for j, t in enumerate(self._tracks):
                if d["label"] == t["label"]:
                    cost[i, j] = 1.0 - _iou(d["bbox"], t["bbox"])

        threshold = 1.0 - self.iou_thresh
        pairs = _greedy_match(cost, threshold)
        matched_det = {r: self._tracks[c]["track_id"] for r, c in pairs}
        unmatched   = [i for i in range(len(dets)) if i not in matched_det]
        matched_trk = {c for _, c in pairs}

        for j, trk in enumerate(self._tracks):
            if j not in matched_trk:
                trk["age"] = trk.get("age", 0) + 1

        return matched_det, unmatched

    def update(self, detections: List[Dict]) -> List[Dict]:
        high = [d for d in detections if d.get("confidence", 1.0) >= self.conf_high]
        low  = [d for d in detections if d.get("confidence", 1.0) <  self.conf_high
                                      and d.get("confidence", 1.0) >= self.conf_low]

        matched, unmatched_high = self._match(high)
        result: List[Dict] = []
        matched_tids = set()

        for i, det in enumerate(high):
            tid = matched.get(i, self._new_id(det["label"]))
            result.append({**det, "track_id": tid})
            matched_tids.add(tid)

        lost_tracks = [t for t in self._tracks if t["track_id"] not in matched_tids]
        if low and lost_tracks:
            cost2 = np.ones((len(low), len(lost_tracks)), dtype=np.float32)
            for i, d in enumerate(low):
                for j, t in enumerate(lost_tracks):
                    if d["label"] == t["label"]:
                        cost2[i, j] = 1.0 - _iou(d["bbox"], t["bbox"])
            pairs2 = _greedy_match(cost2, 1.0 - self.iou_thresh)
            low_matched = {r: lost_tracks[c]["track_id"] for r, c in pairs2}
            for i, det in enumerate(low):
                if i in low_matched:
                    tid = low_matched[i]
                    result.append({**det, "track_id": tid})
                    matched_tids.add(tid)

        new_tracks = []
        for r in result:
            new_tracks.append({
                "track_id": r["track_id"],
                "bbox":     r["bbox"],
                "label":    r["label"],
                "conf":     r.get("confidence", 1.0),
                "age":      0,
            })
        for t in self._tracks:
            if t["track_id"] not in matched_tids and t.get("age", 0) < self.max_age:
                new_tracks.append(t)

        self._tracks = new_tracks
        return result


# ─── JAAD helper ──────────────────────────────────────────────────────────────

def _assign_jaad_ids(detections: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Split detections into those with a JAAD ped_id (track assigned) and those without."""
    with_id, without_id = [], []
    for det in detections:
        ped_id = det.get("ped_id")
        if ped_id:
            prefix = "Person" if det.get("label") == "pedestrian" else "Car"
            with_id.append({**det, "track_id": f"{prefix}_{ped_id}"})
        elif det.get("track_id"):
            with_id.append(det)
        else:
            without_id.append(det)
    return with_id, without_id


# ─── Main tracking function ───────────────────────────────────────────────────

def _run_tracker(records: List[Dict], use_bytetrack: bool) -> List[Dict]:
    tracker = ByteSortTracker() if use_bytetrack else SimpleIoUTracker()
    enriched = []
    total = len(records)

    for i, record in enumerate(records):
        if i % 1000 == 0 and i > 0:
            logger.info(f"  Tracking progress: {i}/{total} frames")

        raw_dets = record.get("detections", [])

        already_tracked = []
        needs_tracking  = []
        for det in raw_dets:
            ped_id = det.get("ped_id")
            if ped_id:
                prefix = "Person" if det.get("label") == "pedestrian" else "Car"
                already_tracked.append({**det, "track_id": f"{prefix}_{ped_id}"})
            elif det.get("track_id"):
                already_tracked.append(det)
            else:
                needs_tracking.append(det)

        if needs_tracking:
            tracked = tracker.update(needs_tracking)
        else:
            tracked = []

        enriched.append({
            **record,
            "detections": already_tracked + tracked,
        })

    return enriched


# ─── Track history ────────────────────────────────────────────────────────────

def build_track_history(records: List[Dict]) -> Dict[str, List]:
    history: Dict[str, List] = {}
    for record in records:
        fid = record["frame_id"]
        ts  = record.get("timestamp", "")
        for det in record.get("detections", []):
            tid = det.get("track_id")
            if tid:
                history.setdefault(tid, []).append((fid, det["bbox"], ts))
    return history


# ─── Stage entry point ────────────────────────────────────────────────────────

def run(context: dict) -> dict:
    records = context["frame_records"]

    mode = "ByteSort" if config.USE_BYTETRACK else "Simple IoU"
    logger.info(f"Using {mode} tracker on {len(records)} frames "
                f"(pre-computed detections - no YOLO re-run)")

    enriched      = _run_tracker(records, config.USE_BYTETRACK)
    track_history = build_track_history(enriched)

    n_pedestrian = sum(1 for tid in track_history if tid.startswith("Person"))
    n_vehicle    = sum(1 for tid in track_history if not tid.startswith("Person"))
    logger.info(
        f"Tracking complete: {len(track_history)} unique tracks "
        f"({n_pedestrian} pedestrian, {n_vehicle} vehicle)"
    )

    return {
        "frame_records": enriched,
        "track_history": track_history,
        "output_path":   config.ANNOTATIONS_DIR,
    }
