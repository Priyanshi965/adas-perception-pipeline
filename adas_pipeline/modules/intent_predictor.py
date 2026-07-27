"""
intent_predictor.py — Pedestrian crossing-intent prediction (inference).

Predicts, causally and per frame, whether each tracked pedestrian is about to
cross, from a fused feature timeline:
    kinematics (bbox trajectory) + body-language (pose keypoints).

This is a *prediction*, not a description of the present: at frame t the model
sees only frames [t-OBS_LEN+1 .. t] and estimates whether a crossing begins in
the near future. Contrast with the previous version, which labelled a whole
track from its full (including future) history and stamped that single label on
every frame — that was non-causal and is gone.

Model file (config.INTENT_MODEL_PATH, produced by train_intent.py):
    NumPy .npz with 2-layer-LSTM weights plus metadata: feat_mean/std,
    active_cols (which of the 20 canonical features the model was trained on),
    and obs_len. Inference selects/normalises features identically to training.

If no trained model is found, the predictor logs a warning and returns a neutral
0.5 for every pedestrian (it does NOT invent confident predictions).
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np

import config
from modules import intent_features as ifeat
from modules import body_language as bl

logger = logging.getLogger("intent_predictor")


# ─── NumPy LSTM (torch-free inference) ────────────────────────────────────────

class _LSTMModel:
    def __init__(self, W: Dict[str, np.ndarray]):
        self.W = W

    @staticmethod
    def _sig(x): return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
    @staticmethod
    def _tanh(x): return np.tanh(np.clip(x, -30, 30))

    def _layer(self, seq, Wih, Whh, b):
        H = Wih.shape[0] // 4
        h = np.zeros(H, np.float32); c = np.zeros(H, np.float32); out = []
        for t in range(len(seq)):
            g = Wih @ seq[t] + Whh @ h + b
            i = self._sig(g[:H]); f = self._sig(g[H:2*H])
            gg = self._tanh(g[2*H:3*H]); o = self._sig(g[3*H:4*H])
            c = f * c + i * gg
            h = o * self._tanh(c)
            out.append(h)
        return np.stack(out), h

    def forward(self, seq: np.ndarray) -> float:
        out0, _ = self._layer(seq, self.W["lstm_Wih"], self.W["lstm_Whh"], self.W["lstm_b"])
        if "lstm_Wih_1" in self.W:
            _, h = self._layer(out0, self.W["lstm_Wih_1"], self.W["lstm_Whh_1"], self.W["lstm_b_1"])
        else:
            h = out0[-1]
        logit = float(np.dot(self.W["fc_W"], h)) + float(np.ravel(self.W["fc_b"])[0])
        return float(self._sig(np.array([logit]))[0])


# ─── Predictor ────────────────────────────────────────────────────────────────

class IntentPredictor:
    def __init__(self):
        self.model: Optional[_LSTMModel] = None
        self.mean = None
        self.std = None
        self.cols = None
        self.obs_len = config.INTENT_OBS_LEN
        self.threshold = getattr(config, "INTENT_THRESHOLD", 0.5)
        self._warned = False
        self._load()

    def _load(self):
        path = getattr(config, "INTENT_MODEL_PATH", None)
        if not (path and os.path.exists(path)):
            logger.warning("No trained intent model — returning neutral 0.5 predictions. "
                           "Train one: python train_intent.py --pose")
            return
        try:
            npz = np.load(path, allow_pickle=False)
            keys = set(npz.files)
            if "active_cols" not in keys:
                logger.warning(f"Model at {path} is a legacy format without metadata — ignoring. "
                               "Retrain with the current train_intent.py.")
                return
            self.model = _LSTMModel({k: npz[k] for k in keys
                                     if k.startswith(("lstm_", "fc_"))})
            self.mean = npz["feat_mean"]; self.std = npz["feat_std"]
            self.cols = npz["active_cols"].astype(int)
            self.obs_len = int(npz["obs_len"])
            if "threshold" in keys:
                self.threshold = float(npz["threshold"])
            logger.info(f"Intent model loaded from {path} "
                        f"({len(self.cols)} features, obs_len={self.obs_len}, thr={self.threshold:.3f})")
        except Exception as e:
            logger.warning(f"Failed to load intent model ({e}); neutral predictions")

    def _prep_window(self, timeline: np.ndarray, end: int) -> np.ndarray:
        """Select + normalise the obs_len window of full-feature rows ending at `end`."""
        lo = end - self.obs_len + 1
        if lo < 0:
            pad = np.repeat(timeline[:1], -lo, axis=0)
            win = np.vstack([pad, timeline[:end + 1]])
        else:
            win = timeline[lo:end + 1]
        win = win[:, self.cols]
        return (win - self.mean) / self.std

    def predict_timeline(self, timeline: np.ndarray) -> List[float]:
        """Per-frame crossing probabilities for one track's (T, FULL_DIM) timeline."""
        if self.model is None:
            return [0.5] * len(timeline)
        return [self.model.forward(self._prep_window(timeline, i).astype(np.float32))
                for i in range(len(timeline))]


_predictor: Optional[IntentPredictor] = None


def _get() -> IntentPredictor:
    global _predictor
    if _predictor is None:
        _predictor = IntentPredictor()
    return _predictor


# ─── Pipeline stage ───────────────────────────────────────────────────────────

def _pose_vec(det: Dict) -> np.ndarray:
    pf = det.get("pose_features")
    if isinstance(pf, dict):
        return bl.features_to_vector(pf)
    return bl.zero_vector()


def run(context: dict) -> dict:
    """
    Annotate each pedestrian detection with a causal crossing-intent prediction.

    Reads:  context["frame_records"], context["track_history"]
    Writes: det["intent"] ("crossing"|"not_crossing"), det["crossing_prob"],
            det["intent_conf"].
    """
    records = context.get("frame_records", [])
    predictor = _get()
    fw = context.get("width", context.get("frame_width", config.JAAD_FRAME_WIDTH))
    fh = context.get("height", context.get("frame_height", config.JAAD_FRAME_HEIGHT))
    thr = predictor.threshold

    # Gather each pedestrian track's ordered frames from the records themselves,
    # so kinematics and the pose features attached upstream stay aligned.
    # per-track: list of (frame_id, bbox, pose_vec, aux)
    from collections import defaultdict
    seq = defaultdict(list)
    for rec in records:
        key = "detections" if "detections" in rec else "objects"
        fid = rec.get("frame_id")
        for det in rec.get(key, []):
            if det.get("label") != "pedestrian":
                continue
            tid = det.get("track_id") or det.get("id")
            if tid is None:
                continue
            seq[tid].append((fid, det.get("bbox"), _pose_vec(det),
                             np.array([det.get("look", 0), det.get("action", 0)], np.float32)))

    # Build per-track full-feature timelines and predict causally.
    pred: Dict[str, Dict[int, float]] = {}
    for tid, items in seq.items():
        items.sort(key=lambda z: (z[0] is None, z[0]))
        bboxes = [it[1] for it in items if it[1] is not None]
        if len(bboxes) != len(items):
            continue
        kin = ifeat.compute_kinematics(bboxes, fw, fh)          # (T, 8)
        T = len(items)
        timeline = np.zeros((T, ifeat.FULL_DIM), np.float32)
        timeline[:, ifeat.KIN_SLICE] = kin
        for i, it in enumerate(items):
            timeline[i, ifeat.POSE_SLICE] = it[2]
            timeline[i, ifeat.AUX_SLICE] = it[3]
        probs = predictor.predict_timeline(timeline)
        pred[tid] = {items[i][0]: probs[i] for i in range(T)}

    # Stamp per-frame predictions back onto detections.
    enriched, n_cross = [], 0
    for rec in records:
        key = "detections" if "detections" in rec else "objects"
        fid = rec.get("frame_id")
        new = []
        for det in rec.get(key, []):
            tid = det.get("track_id") or det.get("id")
            if det.get("label") == "pedestrian" and tid in pred and fid in pred[tid]:
                p = pred[tid][fid]
                intent = "crossing" if p >= thr else "not_crossing"
                if intent == "crossing":
                    n_cross += 1
                det = {**det, "intent": intent, "crossing_prob": round(float(p), 4),
                       "intent_conf": round(abs(p - 0.5) * 2.0, 4)}
            new.append(det)
        enriched.append({**rec, key: new})

    logger.info(f"Intent prediction complete — crossing frame-detections: {n_cross}")
    return {"frame_records": enriched}
