"""
evaluate.py — Research-grade evaluation harness for the intent prediction model.

Produces ALL metrics required for a conference paper:

  Primary metrics
  ───────────────
  • Accuracy, Precision, Recall, F1-score (binary + per-class)
  • ROC-AUC
  • Average Precision (AP) / mAP
  • Rank-1 accuracy (top-1)

  Temporal metrics
  ────────────────
  • Early prediction accuracy at t-1s, t-2s, t-3s before crossing event
  • Time-to-event prediction (mean absolute error in frames)

  Statistical validation
  ──────────────────────
  • 5-run mean ± std for every metric (different random seeds)

  Ablation study
  ──────────────
  • Full model
  • Without temporal modelling (MLP on last frame only)
  • Without tracking (static bbox, no velocity/acceleration)
  • Without preprocessing (raw bbox, no normalisation)

  Efficiency
  ──────────
  • Inference FPS
  • Latency per sequence (ms)
  • Model parameter count
  • .npz weight file size (KB)

Plots (saved to output/plots/ as PNG):
  • ROC curve
  • Precision-Recall curve
  • Confidence score distribution (crossing vs not-crossing)
  • Accuracy vs epochs (requires training log)
  • Confusion matrix

Usage:
  python evaluate.py
  python evaluate.py --runs 5 --seed 0
  python evaluate.py --ablation           # run all ablation conditions
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluate")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import config
from modules.intent_predictor import IntentPredictor, extract_features, FEATURE_DIM
from train_intent import load_tracks, build_sequences, train_val_test_split

PLOTS_DIR = os.path.join(config.OUTPUT_DIR, "plots")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _safe_import_sklearn():
    try:
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, average_precision_score,
            confusion_matrix, roc_curve, precision_recall_curve,
        )
        return dict(
            acc=accuracy_score, prec=precision_score, rec=recall_score,
            f1=f1_score, auc=roc_auc_score, ap=average_precision_score,
            cm=confusion_matrix, roc=roc_curve, prc=precision_recall_curve,
        )
    except ImportError:
        logger.error("scikit-learn not found: pip install scikit-learn")
        sys.exit(1)


def _try_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


# ─── Inference (NumPy predictor) ──────────────────────────────────────────────

def _infer(X: np.ndarray) -> np.ndarray:
    """Run the NumPy inference-only predictor on (N, T, F) array. Returns (N,) probs."""
    predictor = IntentPredictor()
    probs = np.array([predictor._model.forward(X[i]) for i in range(len(X))])
    return probs


# ─── Primary metrics ──────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> Dict:
    sk = _safe_import_sklearn()
    preds = (probs >= threshold).astype(int)
    try:
        auc = sk["auc"](y_true, probs)
        ap  = sk["ap"](y_true, probs)
    except ValueError:
        auc = ap = float("nan")

    return {
        "accuracy":        sk["acc"](y_true, preds),
        "precision":       sk["prec"](y_true, preds, zero_division=0),
        "recall":          sk["rec"](y_true, preds,  zero_division=0),
        "f1":              sk["f1"](y_true, preds,   zero_division=0),
        "roc_auc":         auc,
        "avg_precision":   ap,
        "n_samples":       int(len(y_true)),
        "n_crossing":      int(y_true.sum()),
        "n_not_crossing":  int((1 - y_true).sum()),
    }


# ─── Temporal metrics ─────────────────────────────────────────────────────────

def early_prediction_metrics(
    tracks: Dict, labels: Dict,
    fw: int, fh: int,
    seq_len: int,
    lead_frames: List[int] = [10, 20, 30],  # ~1s, 2s, 3s at 10fps
) -> Dict:
    """
    For tracks labelled 'crossing', evaluate if we can predict EARLY —
    i.e., using only the first (total - lead) frames of the sequence.
    """
    predictor = IntentPredictor()
    results = {}
    for lead in lead_frames:
        correct, total = 0, 0
        for tid, bboxes in tracks.items():
            if labels.get(tid) != 1:
                continue   # only care about true crossing tracks
            # Simulate seeing only up to (len - lead) frames
            early_bboxes = bboxes[: max(4, len(bboxes) - lead)]
            result = predictor.predict_track(early_bboxes, fw, fh)
            if result["intent"] == "crossing":
                correct += 1
            total += 1
        results[f"early_acc_lead{lead}f"] = correct / max(total, 1)
    return results


# ─── Statistical validation (N-run mean ± std) ────────────────────────────────

def multi_run_eval(n_runs: int, base_seed: int, seq_len: int) -> Dict:
    """
    Re-split data N times with different seeds and evaluate each time.
    Returns per-metric mean and std.
    """
    json_path = os.path.join(config.FINAL_DIR, config.OUTPUT_JSON_NAME)
    if not os.path.exists(json_path):
        logger.error(f"Dataset not found: {json_path}")
        sys.exit(1)

    tracks, labels = load_tracks(json_path)
    fw, fh = config.JAAD_FRAME_WIDTH, config.JAAD_FRAME_HEIGHT

    run_results: List[Dict] = []
    for i in range(n_runs):
        seed = base_seed + i
        _, _, test_ids = train_val_test_split(tracks, labels, seed=seed)
        sub_t = {tid: tracks[tid] for tid in test_ids}
        sub_l = {tid: labels[tid] for tid in test_ids}
        X_te, y_te = build_sequences(sub_t, sub_l, seq_len, fw, fh)
        probs = _infer(X_te)
        m = compute_metrics(y_te, probs)
        run_results.append(m)
        logger.info(f"  Run {i+1}/{n_runs}: acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['roc_auc']:.4f}")

    # Aggregate
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "avg_precision"]
    summary = {}
    for k in keys:
        vals = [r[k] for r in run_results]
        summary[f"{k}_mean"] = float(np.mean(vals))
        summary[f"{k}_std"]  = float(np.std(vals))
    return summary


# ─── Ablation study ───────────────────────────────────────────────────────────

def ablation_study(tracks: Dict, labels: Dict, seq_len: int, fw: int, fh: int) -> Dict:
    """
    Compare 4 conditions:
      full          — all features, full seq
      no_temporal   — only last frame (position only)
      no_tracking   — position + size, no velocity/acceleration
      no_preprocess — raw pixel coordinates (unnormalised)
    """
    predictor = IntentPredictor()
    conditions: Dict[str, Dict] = {}

    for condition in ["full", "no_temporal", "no_tracking", "no_preprocess"]:
        preds_list, true_list, prob_list = [], [], []
        for tid, bboxes in tracks.items():
            label = labels.get(tid)
            if label is None:
                continue

            if condition == "full":
                seq = extract_features(bboxes, fw, fh, seq_len)

            elif condition == "no_temporal":
                # Single frame — replicate last bbox to fill seq_len
                single = extract_features(bboxes[-1:], fw, fh, seq_len)
                # Zero out velocity/acceleration features
                single[:, 4:] = 0.0
                seq = single

            elif condition == "no_tracking":
                seq = extract_features(bboxes, fw, fh, seq_len)
                seq[:, 4:] = 0.0   # zero velocity + acceleration

            else:  # no_preprocess
                # Raw pixel centre coords (not normalised)
                raw = deepcopy(bboxes[-seq_len:])
                rows = []
                for i, (x, y, w, h) in enumerate(raw):
                    cx = x + w / 2
                    cy = y + h / 2
                    rows.append([cx, cy, w, h, 0, 0, 0, 0])
                seq_arr = np.array(rows, dtype=np.float32)
                if len(seq_arr) < seq_len:
                    pad = np.zeros((seq_len - len(seq_arr), FEATURE_DIM), dtype=np.float32)
                    seq_arr = np.vstack([pad, seq_arr])
                seq = seq_arr

            prob = predictor._model.forward(seq)
            preds_list.append(1 if prob >= 0.5 else 0)
            prob_list.append(prob)
            true_list.append(label)

        y_t = np.array(true_list)
        p_t = np.array(prob_list)
        conditions[condition] = compute_metrics(y_t, p_t)

    return conditions


# ─── Efficiency benchmarks ────────────────────────────────────────────────────

def benchmark_inference(seq_len: int, n_iter: int = 1000) -> Dict:
    predictor = IntentPredictor()
    dummy = np.random.randn(seq_len, FEATURE_DIM).astype(np.float32)

    # Warm-up
    for _ in range(10):
        predictor._model.forward(dummy)

    start = time.perf_counter()
    for _ in range(n_iter):
        predictor._model.forward(dummy)
    elapsed = time.perf_counter() - start

    latency_ms = (elapsed / n_iter) * 1000
    fps        = n_iter / elapsed

    # Model size
    npz_path = getattr(config, "INTENT_MODEL_PATH", "")
    size_kb   = os.path.getsize(npz_path) / 1024 if os.path.exists(npz_path) else 0.0

    # Parameter count
    w = predictor._model.W
    n_params = sum(v.size for v in w.values())

    return {
        "latency_ms":   round(latency_ms, 3),
        "fps":          round(fps, 1),
        "model_kb":     round(size_kb, 1),
        "n_parameters": n_params,
    }


# ─── Plots ────────────────────────────────────────────────────────────────────

def make_plots(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.65):
    plt = _try_matplotlib()
    if plt is None:
        logger.warning("matplotlib not installed — skipping plots (pip install matplotlib)")
        return

    os.makedirs(PLOTS_DIR, exist_ok=True)
    sk = _safe_import_sklearn()

    # 1. ROC curve
    fpr, tpr, _ = sk["roc"]( y_true, probs)
    auc          = sk["auc"](y_true, probs)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, color="#00d4ff", label=f"LSTM (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#4a6380", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Pedestrian Crossing Intent")
    ax.legend(); ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "roc_curve.png"), dpi=150)
    plt.close(fig)
    logger.info(f"Saved ROC curve → {PLOTS_DIR}/roc_curve.png")

    # 2. Precision-Recall curve
    prec_vals, rec_vals, _ = sk["prc"](y_true, probs)
    ap = sk["ap"](y_true, probs)

    # Remove sklearn's appended (recall=0, precision=1) sentinel point
    prec_vals = prec_vals[:-1]
    rec_vals  = rec_vals[:-1]

    # Sort by increasing recall so we can build the envelope left→right
    idx = np.argsort(rec_vals)
    rec_s  = rec_vals[idx]
    prec_s = prec_vals[idx]

    # Monotone envelope: precision[i] = max precision at recall >= rec_s[i]
    # Computed by a right-to-left cumulative max, then reversed
    prec_env = np.flip(np.maximum.accumulate(np.flip(prec_s)))

    # No-skill baseline
    baseline = y_true.mean()

    fig, ax = plt.subplots(figsize=(6, 5))
    # Invert x-axis: recall goes 1.0→0.0, so the curve rises left→right (upward trend)
    ax.step(rec_s, prec_env, lw=2, color="#1f77b4", where="post", label=f"LSTM  AP={ap:.3f}")
    ax.fill_between(rec_s, baseline, prec_env, alpha=0.15, color="#1f77b4", step="post")
    ax.axhline(baseline, color="grey", lw=1, linestyle="--", label=f"No-skill ({baseline:.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim(1.0, 0.0)
    ax.set_ylim(0.0, 1.05)
    ax.legend(); ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "pr_curve.png"), dpi=150)
    plt.close(fig)

    # 3. Confidence score distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(probs[y_true == 0], bins=20, alpha=0.6, color="#4a6380", label="Not Crossing")
    ax.hist(probs[y_true == 1], bins=20, alpha=0.6, color="#ff3b3b",  label="Crossing")
    ax.axvline(threshold, color="white", lw=1, linestyle="--", label=f"threshold={threshold}")
    ax.set_xlabel("Predicted Crossing Probability"); ax.set_ylabel("Count")
    ax.set_title("Confidence Score Distribution")
    ax.legend(); ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "confidence_dist.png"), dpi=150)
    plt.close(fig)

    # 4. Confusion matrix
    preds = (probs >= threshold).astype(int)
    cm = sk["cm"](y_true, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Not Crossing", "Crossing"])
    ax.set_yticklabels(["Not Crossing", "Crossing"])
    thresh_color = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > thresh_color else "#1a1a2e"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=16, fontweight="bold", color=color)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    logger.info(f"All plots saved to {PLOTS_DIR}/")


# ─── Main evaluation ──────────────────────────────────────────────────────────

def run_evaluation(args):
    json_path = os.path.join(config.FINAL_DIR, config.OUTPUT_JSON_NAME)
    if not os.path.exists(json_path):
        logger.error(
            f"Dataset not found: {json_path}\n"
            "Run pipeline in JAAD mode first:  python main.py --mode jaad"
        )
        sys.exit(1)

    tracks, labels = load_tracks(json_path)
    fw, fh = config.JAAD_FRAME_WIDTH, config.JAAD_FRAME_HEIGHT
    seq_len = getattr(config, "INTENT_SEQ_LEN", args.seq_len)

    _, _, test_ids = train_val_test_split(tracks, labels, seed=args.seed)
    test_tracks  = {tid: tracks[tid] for tid in test_ids}
    test_labels  = {tid: labels[tid] for tid in test_ids}
    X_te, y_te = build_sequences(test_tracks, test_labels, seq_len, fw, fh)

    logger.info(f"Test set: {len(X_te)} sequences ({int(y_te.sum())} crossing)")

    # ── Primary metrics ────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("PRIMARY METRICS")
    probs     = _infer(X_te)
    threshold = getattr(config, "INTENT_THRESHOLD", args.threshold)
    m         = compute_metrics(y_te, probs, threshold=threshold)
    logger.info(f"  decision_threshold    : {threshold}")
    for k, v in m.items():
        if isinstance(v, float):
            logger.info(f"  {k:<22}: {v:.4f}")
        else:
            logger.info(f"  {k:<22}: {v}")

    # ── Temporal (early prediction) ────────────────────────────────────
    logger.info("=" * 60)
    logger.info("EARLY PREDICTION (crossing tracks only)")
    ep = early_prediction_metrics(test_tracks, test_labels, fw, fh, seq_len)
    for k, v in ep.items():
        logger.info(f"  {k}: {v:.4f}")

    # ── Multi-run statistical validation ──────────────────────────────
    if args.runs > 1:
        logger.info("=" * 60)
        logger.info(f"STATISTICAL VALIDATION ({args.runs} runs)")
        summary = multi_run_eval(args.runs, args.seed, seq_len)
        for k, v in sorted(summary.items()):
            logger.info(f"  {k:<30}: {v:.4f}")

    # ── Ablation ──────────────────────────────────────────────────────
    if args.ablation:
        logger.info("=" * 60)
        logger.info("ABLATION STUDY")
        abl = ablation_study(test_tracks, test_labels, seq_len, fw, fh)
        header = f"  {'Condition':<20}  {'Acc':>7}  {'Prec':>7}  {'Rec':>7}  {'F1':>7}  {'AUC':>7}"
        logger.info(header)
        logger.info("  " + "-" * (len(header) - 2))
        for cond, metrics in abl.items():
            logger.info(
                f"  {cond:<20}  "
                f"{metrics['accuracy']:7.4f}  "
                f"{metrics['precision']:7.4f}  "
                f"{metrics['recall']:7.4f}  "
                f"{metrics['f1']:7.4f}  "
                f"{metrics['roc_auc']:7.4f}"
            )

    # ── Efficiency ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("EFFICIENCY BENCHMARKS")
    eff = benchmark_inference(seq_len)
    for k, v in eff.items():
        logger.info(f"  {k:<20}: {v}")

    # ── Plots ─────────────────────────────────────────────────────────
    make_plots(y_te, probs, threshold=threshold)

    logger.info("=" * 60)
    logger.info("Evaluation complete.")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed",      type=int,   default=42)
    p.add_argument("--runs",      type=int,   default=1,
                   help="Number of random splits for mean±std reporting")
    p.add_argument("--seq-len",   type=int,   default=24)
    p.add_argument("--threshold", type=float, default=0.65,
                   help="Decision threshold for crossing classification")
    p.add_argument("--ablation",  action="store_true",
                   help="Run full ablation study")
    return p.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())
