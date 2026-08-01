"""
evaluate.py — Evaluation harness for the crossing-intent model.

Evaluates a trained model (config.INTENT_MODEL_PATH) on held-out JAAD tracks using
the same leak-free observe→predict windowing as training, and reports:

  • Accuracy, Precision, Recall, F1, ROC-AUC, Average Precision
  • Confusion matrix
  • Early-prediction curve: accuracy as a function of time-to-event, i.e. how far
    ahead of the crossing the model is already right (the metric that matters for
    an ADAS warning system)
  • Efficiency: inference latency, throughput, and model file size

Plots (output/plots/): roc_curve.png, pr_curve.png, confusion_matrix.png,
early_prediction.png.

For the pose-vs-baseline ablation, train two models and evaluate each:
  python train_intent.py --videos 40 --no-pose   &&  python evaluate.py
  python train_intent.py --videos 40 --pose      &&  python evaluate.py

Usage:
  python evaluate.py --videos 40
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("evaluate")

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)

import config
from datasets import jaad_loader as jl
from datasets import build_jaad_features as bjf
from modules import intent_features as ifeat
from modules.intent_predictor import IntentPredictor
from train_intent import split_tracks

PLOTS_DIR = os.path.join(config.OUTPUT_DIR, "plots")


def collect_samples(tracks, ids, predictor) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the model over each track and gather (prob, label, tte) per window.

    tte = timeline steps from the window end to the *first* future crossing frame
    (np.inf if the positive label is due to a crossing farther than that first
    one is tracked; used only for the early-prediction curve, positives only).
    """
    probs, labels, ttes = [], [], []
    for i in ids:
        tr = tracks[i]
        per_frame = predictor.predict_timeline(tr["feats"])   # causal, per timeline step
        W, y, ends = ifeat.make_windows(tr["feats"], tr["cross"],
                                        config.INTENT_OBS_LEN, config.INTENT_TTE,
                                        config.INTENT_SAMPLE_STRIDE)
        cross = tr["cross"]
        for yi, end in zip(y, ends):
            probs.append(per_frame[end])
            labels.append(yi)
            # distance to first future crossing frame within horizon
            tte = np.inf
            if yi == 1:
                fut = cross[end + 1:end + 1 + config.INTENT_TTE]
                hit = np.where(fut == 1)[0]
                if len(hit):
                    tte = int(hit[0] + 1)
            ttes.append(tte)
    return np.array(probs), np.array(labels), np.array(ttes)


def early_prediction_curve(probs, labels, ttes, thr) -> List[Tuple[int, float, int]]:
    """Accuracy on positives bucketed by time-to-event (how far ahead we're right)."""
    out = []
    pos = labels == 1
    for lo in range(0, config.INTENT_TTE, 3):
        hi = lo + 3
        m = pos & (ttes > lo) & (ttes <= hi)
        if m.sum() == 0:
            continue
        acc = float(((probs[m] >= thr).astype(int) == 1).mean())
        out.append((hi, acc, int(m.sum())))
    return out


def save_plots(probs, labels, thr, early):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
    except Exception as e:
        logger.warning(f"Plotting skipped ({e})")
        return
    os.makedirs(PLOTS_DIR, exist_ok=True)
    if len(np.unique(labels)) > 1:
        fpr, tpr, _ = roc_curve(labels, probs)
        plt.figure(); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], "--")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC")
        plt.savefig(os.path.join(PLOTS_DIR, "roc_curve.png"), dpi=120); plt.close()

        prec, rec, _ = precision_recall_curve(labels, probs)
        plt.figure(); plt.plot(rec, prec)
        plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall")
        plt.savefig(os.path.join(PLOTS_DIR, "pr_curve.png"), dpi=120); plt.close()

    cm = confusion_matrix(labels, (probs >= thr).astype(int), labels=[0, 1])
    plt.figure(); plt.imshow(cm, cmap="Blues")
    for (r, c), v in np.ndenumerate(cm):
        plt.text(c, r, str(v), ha="center", va="center")
    plt.xticks([0, 1], ["not_cross", "cross"]); plt.yticks([0, 1], ["not_cross", "cross"])
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=120); plt.close()

    if early:
        hs = [h for h, _, _ in early]; accs = [a for _, a, _ in early]
        plt.figure(); plt.plot(hs, accs, "-o")
        plt.xlabel("Time-to-event (timeline steps ahead)"); plt.ylabel("Accuracy on crossings")
        plt.title("Early-prediction accuracy"); plt.ylim(0, 1)
        plt.savefig(os.path.join(PLOTS_DIR, "early_prediction.png"), dpi=120); plt.close()
    logger.info(f"Plots saved → {PLOTS_DIR}")


def write_report(metrics, args, use_pose, thr, thr_src):
    """Persist metrics to <out>/eval_report.json + eval_report.md and copy figures.

    Without this, evaluate.py's numbers exist only in the console and the paper's
    results are not reproducible from anything committed. This writes a permanent,
    committable record of exactly what produced the reported numbers.
    """
    import json
    import shutil
    import datetime
    out_dir = args.out or os.path.join(REPO_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    cmd = (f"python adas_pipeline/evaluate.py --videos {args.videos} --seed {args.seed}"
           + (f" --threshold {args.threshold}" if args.threshold is not None else ""))
    meta = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "command": cmd,
        "videos": args.videos,
        "seed": args.seed,
        "threshold": float(thr),
        "threshold_source": thr_src,
        "features": "POSE+KIN" if use_pose else "KIN only",
        "obs_len": config.INTENT_OBS_LEN,
        "tte": config.INTENT_TTE,
    }
    with open(os.path.join(out_dir, "eval_report.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "metrics": metrics}, f, indent=2)

    cm = metrics["confusion"]
    lines = [
        "# Crossing-Intent Model — Single-Split Evaluation",
        "",
        f"- Generated: {meta['generated']}",
        f"- Command: `{meta['command']}`",
        f"- Videos: {meta['videos']} · Seed: {meta['seed']} · Features: {meta['features']}",
        f"- Observation window: {meta['obs_len']} steps · Prediction horizon (TTE): {meta['tte']} steps",
        f"- Operating threshold: {meta['threshold']:.3f} ({meta['threshold_source']})",
        f"- Samples: {metrics['samples']} (positives {metrics['positives']})",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy | {metrics['accuracy']:.4f} |",
        f"| Precision | {metrics['precision']:.4f} |",
        f"| Recall | {metrics['recall']:.4f} |",
        f"| F1 | {metrics['f1']:.4f} |",
    ]
    if "roc_auc" in metrics:
        lines += [f"| ROC-AUC | {metrics['roc_auc']:.4f} |",
                  f"| Avg-Precision | {metrics['avg_precision']:.4f} |"]
    lines += [
        "",
        "### Confusion matrix",
        "",
        "| | Pred: not-cross | Pred: cross |",
        "|---|---|---|",
        f"| **True: not-cross** | {cm['tn']} | {cm['fp']} |",
        f"| **True: cross** | {cm['fn']} | {cm['tp']} |",
        "",
    ]
    with open(os.path.join(out_dir, "eval_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    for name in ("roc_curve.png", "pr_curve.png", "confusion_matrix.png",
                 "early_prediction.png"):
        src_png = os.path.join(PLOTS_DIR, name)
        if os.path.exists(src_png):
            shutil.copy2(src_png, os.path.join(out_dir, name))
    logger.info(f"Report + figures saved → {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate crossing-intent model")
    ap.add_argument("--videos", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold", type=float, default=None,
                    help="Operating threshold override; default uses the model's calibrated threshold.")
    ap.add_argument("--out", type=str, default=None,
                    help="Dir for eval_report.json/.md + copied figures (default: <repo>/results).")
    args = ap.parse_args()

    predictor = IntentPredictor()
    if predictor.model is None:
        logger.error("No trained model at config.INTENT_MODEL_PATH — run train_intent.py first.")
        sys.exit(1)
    use_pose = bool(int(np.load(config.INTENT_MODEL_PATH)["use_pose"]))
    logger.info(f"Evaluating model (features: {'POSE+KIN' if use_pose else 'KIN only'})")

    vids = jl.available_video_ids(config.JAAD_ANNOTATIONS_DIR)[: args.videos]
    tracks = bjf.build_dataset(vids, with_pose=use_pose, force=False)
    tracks = [t for t in tracks if (t["cross"] >= 0).any()]
    _, _, te_ids = split_tracks(tracks, args.seed)     # identical split to training
    logger.info(f"Held-out test tracks: {len(te_ids)}")

    probs, labels, ttes = collect_samples(tracks, te_ids, predictor)
    if len(probs) == 0:
        logger.error("No test windows — increase --videos."); sys.exit(1)

    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, average_precision_score,
                                 confusion_matrix)
    thr = predictor.threshold if args.threshold is None else args.threshold
    thr_src = "model" if args.threshold is None else "override"
    logger.info(f"Operating threshold ({thr_src}): {thr:.3f}")
    preds = (probs >= thr).astype(int)

    metrics = {
        "samples": int(len(probs)),
        "positives": int(labels.sum()),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
    }
    if len(np.unique(labels)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(labels, probs))
        metrics["avg_precision"] = float(average_precision_score(labels, probs))
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    metrics["confusion"] = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

    logger.info("─" * 56)
    logger.info(f"  Samples  : {metrics['samples']}  (+ve {metrics['positives']})")
    logger.info(f"  Accuracy : {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall   : {metrics['recall']:.4f}")
    logger.info(f"  F1       : {metrics['f1']:.4f}")
    if "roc_auc" in metrics:
        logger.info(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
        logger.info(f"  Avg-Prec : {metrics['avg_precision']:.4f}")
    logger.info(f"  Confusion: TN={tn} FP={fp} FN={fn} TP={tp}")

    if len(np.unique(labels)) > 1:
        from sklearn.metrics import precision_score as _p, recall_score as _r, f1_score as _f
        logger.info("  Threshold sweep (thr → P / R / F1):")
        for t in (0.30, 0.40, 0.50, 0.60, 0.70):
            pr = (probs >= t).astype(int)
            logger.info(f"    thr={t:.2f}  P={_p(labels, pr, zero_division=0):.3f} "
                        f"R={_r(labels, pr, zero_division=0):.3f} "
                        f"F1={_f(labels, pr, zero_division=0):.3f}")

    early = early_prediction_curve(probs, labels, ttes, thr)
    logger.info("  Early-prediction accuracy (crossings, by steps-ahead):")
    for hi, acc, n in early:
        logger.info(f"    ≤{hi:2d} steps ahead: acc={acc:.3f} (n={n})")
    metrics["early_prediction"] = [{"steps_ahead": hi, "accuracy": acc, "n": n}
                                   for hi, acc, n in early]

    # Efficiency
    t0 = time.time()
    _ = [predictor.predict_timeline(tracks[te_ids[0]]["feats"]) for _ in range(3)]
    dt = (time.time() - t0) / max(3 * len(tracks[te_ids[0]]["feats"]), 1)
    size_kb = os.path.getsize(config.INTENT_MODEL_PATH) / 1024
    metrics["latency_ms_per_frame"] = float(dt * 1000)
    metrics["throughput_fps"] = float(1 / dt) if dt > 0 else None
    metrics["model_size_kb"] = float(size_kb)
    logger.info(f"  Latency  : {dt*1000:.2f} ms/frame  |  ~{1/dt:.0f} fps  |  model {size_kb:.1f} KB")

    save_plots(probs, labels, thr, early)
    write_report(metrics, args, use_pose, thr, thr_src)


if __name__ == "__main__":
    main()
