"""
intent_predictor.py — Pedestrian crossing-intent prediction.

Architecture
────────────
Input features per time-step (8 values):
  [cx_norm, cy_norm, w_norm, h_norm,   # normalised bbox centre + size
   vx, vy,                              # velocity (px/frame, smoothed)
   ax, ay]                              # acceleration (2nd derivative)

Model options (controlled by config.INTENT_MODEL):
  "lstm"   → 2-layer LSTM → FC → sigmoid   (default, best accuracy)
  "gru"    → 2-layer GRU  → FC → sigmoid   (faster, similar accuracy)
  "mlp"    → flatten sequence → 3-layer MLP (baseline, no temporal modelling)

Output:
  crossing_prob  float  0–1   (≥0.5 → predicted crossing)
  intent_label   str    "crossing" | "not_crossing"

Sequence handling:
  • Uses the last INTENT_SEQ_LEN frames of each track's history.
  • Pads with zeros if the track is shorter than SEQ_LEN.
  • Inference is online: called once per frame for each active pedestrian.

Usage (inference only — no training needed):
  predictor = IntentPredictor()
  result = predictor.predict_track(bbox_history, frame_width, frame_height)
  # result = {"crossing_prob": 0.87, "intent": "crossing"}

Training:
  See train_intent.py for the full supervised training loop.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

import config

logger = logging.getLogger("intent_predictor")

# ─── Feature extraction ───────────────────────────────────────────────────────

FEATURE_DIM = 8   # must match model input_size


def extract_features(
    bbox_history: List[List[int]],
    frame_width: int,
    frame_height: int,
    seq_len: int,
) -> np.ndarray:
    """
    Convert a list of [x, y, w, h] bboxes into a (seq_len, FEATURE_DIM) array.

    Features:
      cx_norm  = (x + w/2) / frame_width        — horizontal position
      cy_norm  = (y + h/2) / frame_height        — vertical position
      w_norm   = w / frame_width                 — bbox width (proxy for depth)
      h_norm   = h / frame_height                — bbox height
      vx       = Δcx over last frame (smoothed)
      vy       = Δcy over last frame (smoothed)
      ax       = Δvx (acceleration)
      ay       = Δvy (acceleration)
    """
    # Take the last seq_len bboxes
    bboxes = bbox_history[-seq_len:]

    def to_centre(bbox):
        x, y, w, h = bbox
        cx = (x + w / 2) / max(frame_width, 1)
        cy = (y + h / 2) / max(frame_height, 1)
        wn = w / max(frame_width, 1)
        hn = h / max(frame_height, 1)
        return cx, cy, wn, hn

    centres = [to_centre(b) for b in bboxes]

    rows = []
    prev_v = (0.0, 0.0)
    for i, (cx, cy, wn, hn) in enumerate(centres):
        if i == 0:
            vx, vy = 0.0, 0.0
            ax, ay = 0.0, 0.0
        else:
            pcx, pcy, _, _ = centres[i - 1]
            vx = cx - pcx
            vy = cy - pcy
            ax = vx - prev_v[0]
            ay = vy - prev_v[1]
        prev_v = (vx, vy)
        rows.append([cx, cy, wn, hn, vx, vy, ax, ay])

    seq = np.array(rows, dtype=np.float32)   # (T, 8)

    # Pad front with zeros if shorter than seq_len
    if len(seq) < seq_len:
        pad = np.zeros((seq_len - len(seq), FEATURE_DIM), dtype=np.float32)
        seq = np.vstack([pad, seq])

    return seq   # (seq_len, 8)


# ─── Model ────────────────────────────────────────────────────────────────────

class _LSTMModel:
    """Pure-NumPy single-layer LSTM for inference (no PyTorch/TF required)."""

    def __init__(self, weights: Dict[str, np.ndarray]):
        self.W  = weights  # keys: lstm_Wih, lstm_Whh, lstm_b, fc_W, fc_b

    @staticmethod
    def _sigmoid(x):   return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
    @staticmethod
    def _tanh(x):      return np.tanh(np.clip(x, -30, 30))

    def forward(self, seq: np.ndarray) -> float:
        """seq: (T, input_size)  →  scalar crossing probability."""
        T, _ = seq.shape
        W = self.W

        def _lstm_layer(inp_seq, Wih, Whh, b):
            H = Wih.shape[0] // 4
            h = np.zeros(H, dtype=np.float32)
            c = np.zeros(H, dtype=np.float32)
            out = []
            for t in range(len(inp_seq)):
                gates = Wih @ inp_seq[t] + Whh @ h + b
                i_g = self._sigmoid(gates[0:H])
                f_g = self._sigmoid(gates[H:2*H])
                g_g = self._tanh(   gates[2*H:3*H])
                o_g = self._sigmoid(gates[3*H:4*H])
                c   = f_g * c + i_g * g_g
                h   = o_g * self._tanh(c)
                out.append(h)
            return np.stack(out), h

        # Layer 0
        out0, _ = _lstm_layer(seq, W["lstm_Wih"], W["lstm_Whh"], W["lstm_b"])

        # Layer 1 (if weights present, else use layer-0 output directly)
        if "lstm_Wih_1" in W:
            _, h = _lstm_layer(out0, W["lstm_Wih_1"], W["lstm_Whh_1"], W["lstm_b_1"])
        else:
            h = out0[-1]

        # FC layer — fc_b may be shape (1,) so flatten before scalar cast
        logit = float(np.dot(W["fc_W"], h)) + float(np.ravel(W["fc_b"])[0])
        return float(self._sigmoid(np.array([logit]))[0])


def _make_random_weights(hidden: int = 64, input_size: int = FEATURE_DIM) -> Dict:
    """
    Initialise random weights for offline/demo use when no trained model exists.
    The predictor will still run but output ~50 % confidence until trained.
    """
    rng = np.random.default_rng(42)
    scale = 0.01
    return {
        "lstm_Wih": rng.normal(0, scale, (4 * hidden, input_size)).astype(np.float32),
        "lstm_Whh": rng.normal(0, scale, (4 * hidden, hidden)).astype(np.float32),
        "lstm_b":   np.zeros(4 * hidden, dtype=np.float32),
        "fc_W":     rng.normal(0, scale, (hidden,)).astype(np.float32),
        "fc_b":     np.array([0.0], dtype=np.float32),
    }


# ─── Public predictor class ───────────────────────────────────────────────────

class IntentPredictor:
    """
    Wraps the LSTM model and handles weight loading / fallback.

    Weight file (config.INTENT_MODEL_PATH):
      NumPy .npz with keys: lstm_Wih, lstm_Whh, lstm_b, fc_W, fc_b

    If the file does not exist the predictor falls back to random weights
    (outputs ~50 % confidence) and logs a warning once.
    """

    def __init__(self):
        self._model: Optional[_LSTMModel] = None
        self._warned_untrained = False
        self._load_model()

    def _load_model(self):
        path = getattr(config, "INTENT_MODEL_PATH", None)
        if path and os.path.exists(path):
            try:
                npz = np.load(path, allow_pickle=False)
                weights = {k: npz[k] for k in npz.files}
                self._model = _LSTMModel(weights)
                logger.info(f"Intent model loaded from {path}")
                return
            except Exception as e:
                logger.warning(f"Failed to load intent model ({e}); using random weights")
        else:
            logger.warning(
                "No trained intent model found at config.INTENT_MODEL_PATH. "
                "Run train_intent.py to train one. Using untrained weights for now."
            )
        self._model = _LSTMModel(_make_random_weights(
            hidden=getattr(config, "INTENT_HIDDEN_SIZE", 64)
        ))

    def predict_track(
        self,
        bbox_history: List[List[int]],
        frame_width: int = 1920,
        frame_height: int = 1080,
    ) -> Dict:
        """
        Run intent prediction for a single pedestrian track.

        Args:
            bbox_history: list of [x, y, w, h] bboxes ordered oldest→newest
            frame_width:  frame resolution (for normalisation)
            frame_height: frame resolution

        Returns:
            {
              "crossing_prob": float,          # 0–1
              "intent":        str,            # "crossing" | "not_crossing"
              "confidence":    float,          # distance from 0.5, mapped to 0–1
            }
        """
        if not bbox_history:
            return {"crossing_prob": 0.5, "intent": "not_crossing", "confidence": 0.0}

        seq_len = getattr(config, "INTENT_SEQ_LEN", 16)
        seq = extract_features(bbox_history, frame_width, frame_height, seq_len)
        prob = self._model.forward(seq)
        threshold = getattr(config, "INTENT_THRESHOLD", 0.5)
        intent = "crossing" if prob >= threshold else "not_crossing"
        confidence = abs(prob - 0.5) * 2.0   # 0 = uncertain, 1 = very confident

        return {
            "crossing_prob": round(float(prob), 4),
            "intent":        intent,
            "confidence":    round(float(confidence), 4),
        }

    def predict_batch(
        self,
        tracks: Dict[str, List[List[int]]],
        frame_width: int = 1920,
        frame_height: int = 1080,
    ) -> Dict[str, Dict]:
        """Predict intent for every track in the dict.  Returns {track_id: result}."""
        return {
            tid: self.predict_track(bboxes, frame_width, frame_height)
            for tid, bboxes in tracks.items()
        }


# ─── Stage entry point ────────────────────────────────────────────────────────

# Module-level singleton so the model is loaded once per process
_predictor: Optional[IntentPredictor] = None


def _get_predictor() -> IntentPredictor:
    global _predictor
    if _predictor is None:
        _predictor = IntentPredictor()
    return _predictor


def run(context: dict) -> dict:
    """
    Pipeline stage: annotate every pedestrian detection with intent prediction.

    Reads:   context["frame_records"]   — list of frame dicts with detections
             context["track_history"]   — {track_id: [(frame_id, bbox, ts), ...]}
    Writes:  detection["intent"]        — "crossing" | "not_crossing"
             detection["crossing_prob"] — float
             detection["intent_conf"]   — float
    """
    records      = context.get("frame_records", [])
    track_history = context.get("track_history", {})
    predictor    = _get_predictor()

    # Build bbox-only history per track (ordered by frame_id)
    track_bboxes: Dict[str, List[List[int]]] = {}
    for tid, entries in track_history.items():
        sorted_entries = sorted(entries, key=lambda e: e[0])
        track_bboxes[tid] = [e[1] for e in sorted_entries]

    # Detect frame dimensions from context (set by input_handler)
    # input_handler sets "width"/"height" for video mode; fall back to JAAD defaults
    fw = context.get("width", context.get("frame_width",  1920))
    fh = context.get("height", context.get("frame_height", 1080))

    # Pre-compute predictions (one call per track)
    intent_cache = predictor.predict_batch(track_bboxes, fw, fh)

    # Annotate detections
    enriched = []
    for record in records:
        new_dets = []
        for det in record.get("detections", record.get("objects", [])):
            tid = det.get("track_id") or det.get("id")
            label = det.get("label", "")
            if label == "pedestrian" and tid and tid in intent_cache:
                result = intent_cache[tid]
                det = {
                    **det,
                    "intent":        result["intent"],
                    "crossing_prob": result["crossing_prob"],
                    "intent_conf":   result["confidence"],
                }
            new_dets.append(det)
        key = "detections" if "detections" in record else "objects"
        enriched.append({**record, key: new_dets})

    n_crossing = sum(
        1 for r in enriched
        for d in (r.get("detections") or r.get("objects") or [])
        if d.get("intent") == "crossing"
 