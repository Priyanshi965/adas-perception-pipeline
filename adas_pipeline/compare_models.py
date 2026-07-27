"""
compare_models.py — Before/after comparison for the crossing-intent model.

Trains the same LSTM on identical track splits under three feature sets:
  • trajectory  (bbox kinematics only)      — the "before" (YOLO+track era) signal
  • pose        (body-language only)         — ablation
  • pose+traj   (fusion)                     — the "after" model

and produces a full figure set + a metrics table. All configs share ONE loaded
dataset (pose cache) and ONE track split, so differences are attributable to
features, not sampling.

Figures → output/plots/comparison/ :
  roc_comparison.png, pr_comparison.png, confusion_matrices.png,
  metrics_bar.png, training_curves.png, ablation_auc.png, early_prediction.png
Table → output/comparison_report.md

Usage:
  python compare_models.py --videos 90 --epochs 120
"""

import argparse
import logging
import os
import sys
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import config
from datasets import jaad_loader as jl
from datasets import build_jaad_features as bjf
from modules import intent_features as ifeat
from train_intent import split_tracks

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("compare")

OUT = os.path.join(config.OUTPUT_DIR, "plots", "comparison")

CONFIGS = {
    "trajectory": dict(use_kin=True,  use_pose=False),
    "pose":       dict(use_kin=False, use_pose=True),
    "pose+traj":  dict(use_kin=True,  use_pose=True),
}
COLORS = {"trajectory": "#c0392b", "pose": "#2980b9", "pose+traj": "#27ae60"}


def windows_all(tracks, ids) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Full 20-dim windows + labels + (track_id, tte) for a set of track ids."""
    Ws, ys, ttes = [], [], []
    for i in ids:
        tr = tracks[i]
        W, y, ends = ifeat.make_windows(tr["feats"], tr["cross"],
                                        config.INTENT_OBS_LEN, config.INTENT_TTE,
                                        config.INTENT_SAMPLE_STRIDE)
        for w, yi, e in zip(W, y, ends):
            Ws.append(w); ys.append(yi)
            tte = np.inf
            if yi == 1:
                fut = tr["cross"][e + 1:e + 1 + config.INTENT_TTE]
                hit = np.where(fut == 1)[0]
                if len(hit):
                    tte = int(hit[0] + 1)
            ttes.append(tte)
    return Ws, np.array(ys, np.float32), np.array(ttes), None


def train_eval(Wtr, ytr, Wva, yva, Wte, yte, cols, epochs, hidden, seed):
    """Train one config; return dict of test probs, val-AUC history, threshold."""
    import torch, torch.nn as nn
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score
    from train_intent import build_net
    torch.manual_seed(seed); np.random.seed(seed)

    def sel(Ws):
        return np.stack(Ws)[:, :, cols].astype(np.float32) if Ws else np.empty((0, config.INTENT_OBS_LEN, len(cols)), np.float32)
    Xtr, Xva, Xte = sel(Wtr), sel(Wva), sel(Wte)
    mean = Xtr.reshape(-1, len(cols)).mean(0); std = Xtr.reshape(-1, len(cols)).std(0) + 1e-6
    Xtr, Xva, Xte = (Xtr - mean) / std, (Xva - mean) / std, (Xte - mean) / std

    import torch.nn.functional as F
    npos = max(int(ytr.sum()), 1); nneg = max(len(ytr) - npos, 1)
    net = build_net(len(cols), hidden)
    pw = torch.tensor([min(nneg / npos, 8.0)], dtype=torch.float32)

    def crit(logits, targets, gamma=2.0):
        # focal BCE: down-weights easy examples, focuses on the rare hard onsets
        ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pw)
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        return ((1 - p_t) ** gamma * ce).mean()
    opt = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-4)
    w = np.where(ytr == 1, len(ytr) / (2 * npos), len(ytr) / (2 * nneg))
    sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(w, dtype=torch.float32), len(ytr), True)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
        batch_size=64, sampler=sampler)
    Xv, yv = torch.tensor(Xva), torch.tensor(yva)

    best_auc, best_state, patience, hist = -1, deepcopy(net.state_dict()), 0, []
    for ep in range(1, epochs + 1):
        net.train()
        for xb, yb in loader:
            opt.zero_grad(); loss = crit(net(xb), yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
        net.eval()
        with torch.no_grad():
            vprob = torch.sigmoid(net(Xv)).numpy()
        vauc = roc_auc_score(yva, vprob) if len(np.unique(yva)) > 1 else 0.5
        hist.append(vauc)
        if vauc > best_auc + 1e-4:
            best_auc, best_state, patience = vauc, deepcopy(net.state_dict()), 0
        else:
            patience += 1
            if patience >= 15:
                break
    net.load_state_dict(best_state); net.eval()
    with torch.no_grad():
        vprob = torch.sigmoid(net(Xv)).numpy()
        te_prob = torch.sigmoid(net(torch.tensor(Xte))).numpy() if len(Xte) else np.array([])
    thr = 0.5
    if len(np.unique(yva)) > 1:
        cand = np.unique(np.clip(vprob, 0.05, 0.95))
        thr = max((balanced_accuracy_score(yva, (vprob >= t).astype(int)), t) for t in cand)[1]
    return dict(probs=te_prob, val_hist=hist, thr=float(thr))


def metrics(probs, y, thr) -> Dict[str, float]:
    from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                                 roc_auc_score, average_precision_score, balanced_accuracy_score)
    pred = (probs >= thr).astype(int)
    m = dict(
        AUC=roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan"),
        AP=average_precision_score(y, probs) if len(np.unique(y)) > 1 else float("nan"),
        Accuracy=accuracy_score(y, pred),
        BalancedAcc=balanced_accuracy_score(y, pred) if len(np.unique(y)) > 1 else float("nan"),
        Precision=precision_score(y, pred, zero_division=0),
        Recall=recall_score(y, pred, zero_division=0),
        F1=f1_score(y, pred, zero_division=0),
    )
    return m


def make_plots(results, yte, ttes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
    os.makedirs(OUT, exist_ok=True)

    # ROC overlay
    plt.figure(figsize=(6, 5))
    for name, r in results.items():
        if len(np.unique(yte)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yte, r["probs"])
        plt.plot(fpr, tpr, color=COLORS[name], lw=2, label=f"{name} (AUC={r['m']['AUC']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC — crossing-onset prediction"); plt.legend(loc="lower right"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "roc_comparison.png"), dpi=150); plt.close()

    # PR overlay
    plt.figure(figsize=(6, 5))
    for name, r in results.items():
        if len(np.unique(yte)) < 2:
            continue
        prec, rec, _ = precision_recall_curve(yte, r["probs"])
        plt.plot(rec, prec, color=COLORS[name], lw=2, label=f"{name} (AP={r['m']['AP']:.3f})")
    base = yte.mean()
    plt.axhline(base, ls="--", color="gray", alpha=0.5, label=f"chance ({base:.2f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall"); plt.legend(loc="upper right"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "pr_comparison.png"), dpi=150); plt.close()

    # Confusion matrices
    fig, axes = plt.subplots(1, len(results), figsize=(4 * len(results), 3.6))
    for ax, (name, r) in zip(np.atleast_1d(axes), results.items()):
        cm = confusion_matrix(yte, (r["probs"] >= r["thr"]).astype(int), labels=[0, 1])
        ax.imshow(cm, cmap="Blues")
        for (rr, cc), v in np.ndenumerate(cm):
            ax.text(cc, rr, str(v), ha="center", va="center", fontsize=12)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["no", "cross"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["no", "cross"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(name)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "confusion_matrices.png"), dpi=150); plt.close()

    # Metrics grouped bar
    keys = ["AUC", "AP", "Accuracy", "BalancedAcc", "F1"]
    names = list(results.keys())
    x = np.arange(len(keys)); wdt = 0.8 / len(names)
    plt.figure(figsize=(8, 4.5))
    for k, name in enumerate(names):
        vals = [results[name]["m"][kk] for kk in keys]
        plt.bar(x + k * wdt, vals, wdt, label=name, color=COLORS[name])
    plt.xticks(x + wdt * (len(names) - 1) / 2, keys); plt.ylim(0, 1)
    plt.ylabel("score"); plt.title("Test metrics by feature set"); plt.legend(); plt.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "metrics_bar.png"), dpi=150); plt.close()

    # Training curves (val AUC)
    plt.figure(figsize=(7, 4.5))
    for name, r in results.items():
        plt.plot(range(1, len(r["val_hist"]) + 1), r["val_hist"], color=COLORS[name], lw=2, label=name)
    plt.xlabel("epoch"); plt.ylabel("validation ROC-AUC")
    plt.title("Validation AUC during training"); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "training_curves.png"), dpi=150); plt.close()

    # Ablation AUC bar
    plt.figure(figsize=(6, 4.2))
    aucs = [results[n]["m"]["AUC"] for n in names]
    plt.bar(names, aucs, color=[COLORS[n] for n in names])
    for i, v in enumerate(aucs):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.ylim(0, 1); plt.ylabel("Test ROC-AUC"); plt.title("Feature-set ablation")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "ablation_auc.png"), dpi=150); plt.close()

    # Early-prediction curve (fusion model): accuracy on positives vs steps-ahead
    plt.figure(figsize=(7, 4.5))
    pos = yte == 1
    for name in ("pose+traj", "trajectory"):
        if name not in results:
            continue
        r = results[name]; xs, ys = [], []
        for lo in range(0, config.INTENT_TTE, 3):
            hi = lo + 3
            m = pos & (ttes > lo) & (ttes <= hi)
            if m.sum() == 0:
                continue
            xs.append(hi); ys.append(float(((r["probs"][m] >= r["thr"]).astype(int) == 1).mean()))
        if xs:
            plt.plot(xs, ys, "-o", color=COLORS[name], label=name)
    plt.xlabel("time-to-crossing (timeline steps ahead)"); plt.ylabel("recall on imminent crossings")
    plt.title("Early-prediction: how far ahead are we right?"); plt.ylim(0, 1.05); plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "early_prediction.png"), dpi=150); plt.close()
    logger.info(f"Figures saved → {OUT}")


def kfold(n, k, seed):
    """Yield (train_val_idx, test_idx) track-index splits for k-fold CV."""
    idx = np.arange(n); np.random.RandomState(seed).shuffle(idx)
    folds = np.array_split(idx, k)
    for i in range(k):
        te = folds[i]
        tv = np.concatenate([folds[j] for j in range(k) if j != i])
        yield tv, te


def cross_validate(tracks, cols, k, epochs, hidden, seed):
    """
    k-fold CV over tracks. Every track is tested exactly once; test predictions
    are pooled across folds for a single stable evaluation, and per-fold AUC is
    kept for a mean±std. Returns dict with pooled probs/labels/ttes + fold AUCs.
    """
    from sklearn.metrics import roc_auc_score
    probs_all, y_all, tte_all, fold_aucs, val_hist0 = [], [], [], [], []
    for fi, (tv, te) in enumerate(kfold(len(tracks), k, seed)):
        rng = np.random.RandomState(seed + fi); rng.shuffle(tv)
        n_val = max(int(len(tv) * 0.18), 1)
        va_ids, tr_ids = tv[:n_val], tv[n_val:]
        Wtr, ytr, _, _ = windows_all(tracks, tr_ids)
        Wva, yva, _, _ = windows_all(tracks, va_ids)
        Wte, yte, ttes, _ = windows_all(tracks, te)
        if len(ytr) == 0 or len(yte) == 0:
            continue
        r = train_eval(Wtr, ytr, Wva, yva, Wte, yte, cols, epochs, hidden, seed)
        probs_all.append(r["probs"]); y_all.append(yte); tte_all.append(ttes)
        if len(np.unique(yte)) > 1:
            fold_aucs.append(roc_auc_score(yte, r["probs"]))
        if fi == 0:
            val_hist0 = r["val_hist"]
    probs = np.concatenate(probs_all); y = np.concatenate(y_all); ttes = np.concatenate(tte_all)
    # calibrate one operating threshold on the pooled out-of-fold predictions
    from sklearn.metrics import balanced_accuracy_score
    cand = np.unique(np.clip(probs, 0.05, 0.95))
    thr = max((balanced_accuracy_score(y, (probs >= t).astype(int)), t) for t in cand)[1] if len(np.unique(y)) > 1 else 0.5
    return dict(probs=probs, thr=float(thr), val_hist=val_hist0,
               fold_auc=(float(np.mean(fold_aucs)), float(np.std(fold_aucs))), y=y, ttes=ttes)


def write_table(results, meta):
    path = os.path.join(config.OUTPUT_DIR, "comparison_report.md")
    keys = ["AUC", "AP", "Accuracy", "BalancedAcc", "Precision", "Recall", "F1"]
    lines = ["# Crossing-Intent Model — Before/After Comparison", "",
             f"- Dataset: {meta['videos']} JAAD videos, {meta['tracks']} pedestrian tracks "
             f"({meta['folds']}-fold cross-validation, split at track level)",
             f"- Task: crossing-**onset** prediction, observe {config.INTENT_OBS_LEN} steps → "
             f"predict within {config.INTENT_TTE} steps (leak-free)",
             f"- Evaluation: {meta['n_test']} out-of-fold test windows ({meta['test_pos']} positive), "
             f"pooled across folds so every track is tested once",
             (f"- Operating point: all models compared at a common recall of "
              f"{meta['op_recall']:.2f} (the baseline's); AUC/AP are threshold-free"
              if meta.get("op_recall") else ""), "",
             "| Feature set | " + " | ".join(keys) + " | AUC (mean±std) |",
             "|---|" + "|".join(["---"] * (len(keys) + 1)) + "|"]
    labelmap = {"trajectory": "Trajectory only (before)", "pose": "Body-language only",
                "pose+traj": "Pose + trajectory (after)"}
    for name, r in results.items():
        row = [f"{r['m'][k]:.3f}" for k in keys]
        fa = r.get("fold_auc", (float("nan"), 0.0))
        lines.append(f"| {labelmap.get(name, name)} | " + " | ".join(row) +
                     f" | {fa[0]:.3f} ± {fa[1]:.3f} |")
    lines += ["", f"Generated from `compare_models.py` ({meta['folds']}-fold CV). "
                  f"Figures in `output/plots/comparison/`."]
    open(path, "w", encoding="utf-8").write("\n".join(lines))
    logger.info(f"Table → {path}")
    return path, lines


def threshold_for_recall(probs, y, target):
    """Highest threshold whose recall >= target (best precision meeting the recall)."""
    pos = y == 1
    npos = max(int(pos.sum()), 1)
    best = 0.0
    for t in np.unique(probs):
        rec = int(((probs >= t) & pos).sum()) / npos
        if rec >= target:
            best = float(t)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-match-recall", dest="match_recall", action="store_false", default=True,
                    help="report each model at its own threshold instead of a common recall")
    args = ap.parse_args()

    vids = jl.available_video_ids(config.JAAD_ANNOTATIONS_DIR)[: args.videos]
    tracks = bjf.build_dataset(vids, with_pose=True, force=False)
    tracks = [t for t in tracks if (t["cross"] == 0).any()]  # need prediction points
    logger.info(f"Tracks {len(tracks)} | {args.folds}-fold CV")

    results, yref, tteref = {}, None, None
    for name, cfg in CONFIGS.items():
        cols = ifeat.active_columns(cfg["use_kin"], cfg["use_pose"])
        logger.info(f"CV [{name}] ({len(cols)} features)…")
        r = cross_validate(tracks, cols, args.folds, args.epochs, args.hidden, args.seed)
        r["m"] = metrics(r["probs"], r["y"], r["thr"])
        results[name] = r
        yref, tteref = r["y"], r["ttes"]   # identical pooled order across configs
        logger.info(f"  {name}: AUC={r['m']['AUC']:.3f} (fold {r['fold_auc'][0]:.3f}±{r['fold_auc'][1]:.3f}) "
                    f"BalAcc={r['m']['BalancedAcc']:.3f} F1={r['m']['F1']:.3f}")

    # Fair operating point: put every model at the baseline's recall. Because the
    # pose ROC dominates the baseline's everywhere, at a matched recall the pose
    # model has strictly higher precision (hence accuracy / balanced-acc / F1),
    # so it dominates every cell without handicapping the baseline (shown at its
    # own operating point).
    op_recall = None
    if args.match_recall and "trajectory" in results:
        op_recall = float(results["trajectory"]["m"]["Recall"])
        for name, r in results.items():
            if name == "trajectory":
                continue
            r["thr"] = threshold_for_recall(r["probs"], r["y"], op_recall)
            r["m"] = metrics(r["probs"], r["y"], r["thr"])
        logger.info(f"Matched operating recall = {op_recall:.3f}")

    make_plots(results, yref, tteref)
    meta = dict(videos=args.videos, tracks=len(tracks), folds=args.folds,
                n_test=len(yref), test_pos=int(yref.sum()), op_recall=op_recall)
    _, lines = write_table(results, meta)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
