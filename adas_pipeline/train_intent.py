"""
train_intent.py — Supervised training for pedestrian crossing-intent prediction.

Uses the JAAD observe→predict formulation (no future leakage): a sample observes
INTENT_OBS_LEN timeline steps and is labelled by whether a crossing occurs within
the next INTENT_TTE steps. Features come from datasets/build_jaad_features.py and
may be kinematics-only (bbox trajectory) or kinematics + pose (body language),
controlled by config.INTENT_USE_KINEMATICS / INTENT_USE_POSE or the CLI flags.

The trained model is a 2-layer LSTM exported to NumPy .npz so the pipeline's
inference predictor needs no PyTorch. The .npz also stores feature normalisation
stats and the active-column set so inference stays perfectly aligned with training.

Usage:
  pip install torch scikit-learn
  python train_intent.py --videos 40 --pose            # fusion model
  python train_intent.py --videos 40 --no-pose         # bbox-only baseline
"""

import argparse
import logging
import os
import random
import sys
from typing import Dict, List, Tuple

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import config
from datasets import jaad_loader as jl
from datasets import build_jaad_features as bjf
from modules import intent_features as ifeat

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("train_intent")


# ─── Sample assembly ──────────────────────────────────────────────────────────

def windows_for_track(track: Dict, cols: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (N, obs_len, D) windows and (N,) labels for one track."""
    W, y, _ = ifeat.make_windows(
        track["feats"], track["cross"],
        config.INTENT_OBS_LEN, config.INTENT_TTE, config.INTENT_SAMPLE_STRIDE,
    )
    if not W:
        return np.empty((0, config.INTENT_OBS_LEN, len(cols)), np.float32), np.empty((0,), np.float32)
    X = np.stack(W)[:, :, cols].astype(np.float32)
    return X, np.array(y, dtype=np.float32)


def split_tracks(tracks: List[Dict], seed: int) -> Tuple[List, List, List]:
    """70/15/15 split at TRACK level (prevents window leakage across splits)."""
    idx = list(range(len(tracks)))
    random.Random(seed).shuffle(idx)
    n = len(idx)
    a, b = int(n * 0.70), int(n * 0.85)
    return idx[:a], idx[a:b], idx[b:]


def assemble(tracks: List[Dict], ids: List[int], cols: np.ndarray):
    Xs, ys = [], []
    for i in ids:
        X, y = windows_for_track(tracks[i], cols)
        if len(X):
            Xs.append(X); ys.append(y)
    if not Xs:
        return np.empty((0, config.INTENT_OBS_LEN, len(cols)), np.float32), np.empty((0,), np.float32)
    return np.concatenate(Xs), np.concatenate(ys)


# ─── Model ────────────────────────────────────────────────────────────────────

def _import_torch():
    try:
        import torch, torch.nn as nn
        return torch, nn
    except ImportError:
        logger.error("PyTorch required for training: pip install torch")
        sys.exit(1)


def build_net(input_dim: int, hidden: int):
    torch, nn = _import_torch()

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden, num_layers=2, batch_first=True, dropout=0.3)
            self.fc = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :]).squeeze(-1)

    return Net()


def export_npz(net, path, mean, std, cols, args, threshold=0.5):
    torch, _ = _import_torch()
    sd = net.state_dict()
    np.savez(
        path,
        threshold=np.float32(threshold),
        lstm_Wih=sd["lstm.weight_ih_l0"].cpu().numpy(),
        lstm_Whh=sd["lstm.weight_hh_l0"].cpu().numpy(),
        lstm_b=(sd["lstm.bias_ih_l0"] + sd["lstm.bias_hh_l0"]).cpu().numpy(),
        lstm_Wih_1=sd["lstm.weight_ih_l1"].cpu().numpy(),
        lstm_Whh_1=sd["lstm.weight_hh_l1"].cpu().numpy(),
        lstm_b_1=(sd["lstm.bias_ih_l1"] + sd["lstm.bias_hh_l1"]).cpu().numpy(),
        fc_W=sd["fc.weight"].cpu().numpy().squeeze(),
        fc_b=sd["fc.bias"].cpu().numpy(),
        feat_mean=mean.astype(np.float32),
        feat_std=std.astype(np.float32),
        active_cols=cols.astype(np.int64),
        obs_len=np.int64(config.INTENT_OBS_LEN),
        use_kinematics=np.int64(int(args.use_kin)),
        use_pose=np.int64(int(args.use_pose)),
    )
    logger.info(f"Saved model → {path}")


# ─── Train ────────────────────────────────────────────────────────────────────

def train(args):
    torch, nn = _import_torch()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    vids = jl.available_video_ids(config.JAAD_ANNOTATIONS_DIR)[: args.videos]
    tracks = bjf.build_dataset(vids, with_pose=args.use_pose, force=False)
    tracks = [t for t in tracks if (t["cross"] >= 0).any()]
    if len(tracks) < 8:
        logger.error(f"Only {len(tracks)} usable tracks — increase --videos."); sys.exit(1)

    cols = ifeat.active_columns(args.use_kin, args.use_pose, use_aux=args.use_aux)
    logger.info(f"Feature set: kin={args.use_kin} pose={args.use_pose} aux={args.use_aux} "
                f"→ {len(cols)} dims: {[ifeat.FULL_FEATURE_NAMES[c] for c in cols]}")

    tr_ids, va_ids, te_ids = split_tracks(tracks, args.seed)
    X_tr, y_tr = assemble(tracks, tr_ids, cols)
    X_va, y_va = assemble(tracks, va_ids, cols)
    X_te, y_te = assemble(tracks, te_ids, cols)
    logger.info(f"Samples: train={len(X_tr)} val={len(X_va)} test={len(X_te)} | "
                f"train +ve={int(y_tr.sum())}/{len(y_tr)}")
    if len(X_tr) == 0 or len(X_va) == 0:
        logger.error("Empty train/val split — increase --videos."); sys.exit(1)

    # Standardise using train stats only
    mean = X_tr.reshape(-1, X_tr.shape[-1]).mean(0)
    std = X_tr.reshape(-1, X_tr.shape[-1]).std(0) + 1e-6
    def norm(X): return (X - mean) / std
    X_tr, X_va, X_te = norm(X_tr), norm(X_va), norm(X_te)

    n_pos = max(int(y_tr.sum()), 1); n_neg = max(len(y_tr) - n_pos, 1)
    pos_weight = torch.tensor([min(n_neg / n_pos, 5.0)], dtype=torch.float32)

    net = build_net(len(cols), args.hidden)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    Xt = torch.tensor(X_tr); yt = torch.tensor(y_tr)
    Xv = torch.tensor(X_va); yv = torch.tensor(y_va)
    ds = torch.utils.data.TensorDataset(Xt, yt)
    # class-balanced sampler
    w = np.where(y_tr == 1, len(y_tr) / (2 * n_pos), len(y_tr) / (2 * n_neg))
    sampler = torch.utils.data.WeightedRandomSampler(torch.tensor(w, dtype=torch.float32), len(y_tr), True)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, sampler=sampler)

    from copy import deepcopy
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, classification_report)

    # Model selection by validation ROC-AUC (threshold-free → robust to the class
    # imbalance and the tiny val sets that JAAD track-level splits produce).
    best_auc, best_ep, patience, best_state = -1.0, 0, 0, deepcopy(net.state_dict())
    os.makedirs(os.path.dirname(config.INTENT_MODEL_PATH), exist_ok=True)
    for ep in range(1, args.epochs + 1):
        net.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = crit(net(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            vlogits = net(Xv)
            vl = crit(vlogits, yv).item()
            vprob = torch.sigmoid(vlogits).numpy()
        vauc = roc_auc_score(y_va, vprob) if len(np.unique(y_va)) > 1 else 0.5
        sched.step(vl)
        if ep % 10 == 0 or ep == 1:
            logger.info(f"epoch {ep:3d}  val_loss={vl:.4f}  val_auc={vauc:.3f}")
        if vauc > best_auc + 1e-4:
            best_auc, best_ep, patience = vauc, ep, 0
            best_state = deepcopy(net.state_dict())
        else:
            patience += 1
            if patience >= args.patience:
                logger.info(f"early stop @ {ep} (best {best_ep}, val_auc={best_auc:.3f})"); break

    # Reload best weights and calibrate the operating threshold on validation
    # (maximise F1) so the deployed model isn't stuck predicting one class at 0.5.
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        vprob = torch.sigmoid(net(Xv)).numpy()
    # Calibrate by max balanced-accuracy, not F1: with a positive-heavy val set,
    # max-F1 rewards predicting everything positive and collapses the threshold.
    thr = 0.5
    if len(np.unique(y_va)) > 1:
        from sklearn.metrics import balanced_accuracy_score
        cand = np.unique(np.clip(vprob, 0.05, 0.95))
        scores = [(balanced_accuracy_score(y_va, (vprob >= t).astype(int)), t) for t in cand]
        thr = max(scores)[1] if scores else 0.5
    logger.info(f"Calibrated threshold (val, max balanced-acc): {thr:.3f}")
    export_npz(net, config.INTENT_MODEL_PATH, mean, std, cols, args, threshold=float(thr))
    torch.save(net.state_dict(), config.INTENT_MODEL_PATH + ".pt")

    # ── Test evaluation (best weights, calibrated threshold) ──
    with torch.no_grad():
        probs = torch.sigmoid(net(torch.tensor(X_te))).numpy() if len(X_te) else np.array([])
    logger.info("=" * 56)
    logger.info(f"TEST SET  (feature set: {'POSE+KIN' if args.use_pose else 'KIN only'})")
    if len(X_te) and len(np.unique(y_te)) > 1:
        preds = (probs >= thr).astype(int)
        logger.info(f"  Accuracy : {accuracy_score(y_te, preds):.4f}")
        logger.info(f"  Precision: {precision_score(y_te, preds, zero_division=0):.4f}")
        logger.info(f"  Recall   : {recall_score(y_te, preds, zero_division=0):.4f}")
        logger.info(f"  F1       : {f1_score(y_te, preds, zero_division=0):.4f}")
        logger.info(f"  ROC-AUC  : {roc_auc_score(y_te, probs):.4f}")
        logger.info("\n" + classification_report(y_te, preds, target_names=["not_cross", "cross"]))
    else:
        logger.warning("Test set too small / single-class — increase --videos for a real number.")
    logger.info(f"Model → {config.INTENT_MODEL_PATH}")


def parse_args():
    p = argparse.ArgumentParser(description="Train pedestrian intent model")
    p.add_argument("--videos", type=int, default=config.__dict__.get("TRAIN_VIDEOS", 40))
    p.add_argument("--pose", dest="use_pose", action="store_true", default=config.INTENT_USE_POSE)
    p.add_argument("--no-pose", dest="use_pose", action="store_false")
    p.add_argument("--no-kin", dest="use_kin", action="store_false", default=config.INTENT_USE_KINEMATICS)
    p.add_argument("--aux", dest="use_aux", action="store_true", default=False,
                   help="include JAAD look/action GT as features (leaky for deployment)")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=config.INTENT_HIDDEN_SIZE)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
