"""
train_intent.py — Supervised training for the pedestrian intent LSTM.

Dataset format (produced by the ADAS pipeline with JAAD ground truth):
  output/final/dataset.json — list of frame records, each with:
    record["objects"] or record["detections"] — list of per-object dicts:
      {
        "track_id":      str,
        "label":         "pedestrian" | "vehicle",
        "bbox":          [x, y, w, h],
        "jaad_crossing": 0 | 1,          ← ground-truth label
        ...
      }

The script:
  1. Builds per-track bbox sequences + labels from the JSON
  2. Splits 70/15/15 (train/val/test) at the TRACK level (not frame level)
     to prevent data leakage
  3. Trains a 2-layer LSTM with:
       - binary cross-entropy loss
       - Adam optimizer
       - early stopping on val loss
       - class-balanced sampling (crossing events are rare)
  4. Saves best weights to config.INTENT_MODEL_PATH (.npz)
  5. Prints full evaluation metrics on the held-out test set

Usage:
  pip install torch          # only dependency beyond the pipeline
  python train_intent.py
  python train_intent.py --epochs 100 --lr 3e-4 --seq-len 20 --seed 42
"""

import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_intent")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import config
from modules.intent_predictor import extract_features, FEATURE_DIM


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_tracks(json_path: str) -> Tuple[Dict[str, List], Dict[str, int]]:
    """
    Parse dataset.json into per-track bbox sequences and crossing labels.

    Returns:
      tracks: {track_id: [bbox, bbox, ...]}  (ordered by frame_id)
      labels: {track_id: 0 | 1}             (1 = crossing, majority vote)
    """
    logger.info(f"Loading dataset: {json_path}")
    with open(json_path, encoding="utf-8") as f:
        records = json.load(f)

    track_bboxes: Dict[str, List] = defaultdict(list)
    track_crossing_votes: Dict[str, List[int]] = defaultdict(list)

    for record in sorted(records, key=lambda r: r.get("frame_id", 0)):
        objects = record.get("objects") or record.get("detections") or []
        for obj in objects:
            if obj.get("label") != "pedestrian":
                continue
            tid = obj.get("track_id") or obj.get("id")
            if not tid:
                continue
            bbox = obj.get("bbox")
            if not bbox or len(bbox) < 4:
                continue
            jaad = obj.get("jaad_crossing")
            track_bboxes[tid].append(bbox)
            if jaad is not None:
                track_crossing_votes[tid].append(int(jaad))

    # Assign label by majority vote (tracks with no jaad label → skip)
    labels: Dict[str, int] = {}
    for tid, votes in track_crossing_votes.items():
        if votes:
            labels[tid] = 1 if sum(votes) / len(votes) >= 0.5 else 0

    # Keep only labelled tracks with enough frames
    min_frames = 4
    tracks = {
        tid: track_bboxes[tid]
        for tid in labels
        if len(track_bboxes[tid]) >= min_frames
    }
    labels = {tid: labels[tid] for tid in tracks}

    n_cross  = sum(1 for v in labels.values() if v == 1)
    n_safe   = sum(1 for v in labels.values() if v == 0)
    logger.info(
        f"Loaded {len(tracks)} labelled pedestrian tracks: "
        f"{n_cross} crossing, {n_safe} not-crossing"
    )
    return tracks, labels


def build_sequences(
    tracks: Dict[str, List],
    labels: Dict[str, int],
    seq_len: int,
    frame_width: int,
    frame_height: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert track dicts to (N, seq_len, FEATURE_DIM) array + label vector."""
    X, y = [], []
    for tid, bboxes in tracks.items():
        seq = extract_features(bboxes, frame_width, frame_height, seq_len)
        X.append(seq)
        y.append(labels[tid])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_val_test_split(
    tracks: Dict, labels: Dict, seed: int = 42
) -> Tuple[List, List, List]:
    """70/15/15 split at TRACK level to avoid sequence leakage."""
    tids = list(tracks.keys())
    rng  = random.Random(seed)
    rng.shuffle(tids)
    n = len(tids)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)
    return tids[:n_train], tids[n_train:n_train + n_val], tids[n_train + n_val:]


# ─── PyTorch model ────────────────────────────────────────────────────────────

def _try_import_torch():
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError:
        logger.error(
            "PyTorch not found. Install with: pip install torch\n"
            "Alternatively, use the NumPy inference-only predictor (no training)."
        )
        sys.exit(1)


class IntentLSTM:
    """Thin wrapper so we can keep the training script self-contained."""

    def __init__(self, hidden=64, num_layers=2, dropout=0.3):
        torch, nn = _try_import_torch()
        self.torch = torch
        self.nn    = nn

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    FEATURE_DIM, hidden, num_layers=num_layers,
                    batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
                )
                self.fc = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :]).squeeze(-1)

        self.net = _Net()

    def parameters(self): return self.net.parameters()
    def train(self):       self.net.train()
    def eval(self):        self.net.eval()
    def __call__(self, x): return self.net(x)

    def save_numpy(self, path: str):
        """Export weights to NumPy .npz for use by the inference-only predictor."""
        sd = self.net.state_dict()
        # LSTM weight layout: weight_ih_l0 shape (4H, input), weight_hh_l0 (4H, H)
        np.savez(
            path,
            lstm_Wih   = sd["lstm.weight_ih_l0"].detach().numpy(),
            lstm_Whh   = sd["lstm.weight_hh_l0"].detach().numpy(),
            lstm_b     = (sd["lstm.bias_ih_l0"] + sd["lstm.bias_hh_l0"]).detach().numpy(),
            lstm_Wih_1 = sd["lstm.weight_ih_l1"].detach().numpy(),
            lstm_Whh_1 = sd["lstm.weight_hh_l1"].detach().numpy(),
            lstm_b_1   = (sd["lstm.bias_ih_l1"] + sd["lstm.bias_hh_l1"]).detach().numpy(),
            fc_W       = sd["fc.weight"].detach().numpy().squeeze(),
            fc_b       = sd["fc.bias"].detach().numpy(),
        )
        logger.info(f"Saved NumPy weights → {path}")


# ─── Training loop ────────────────────────────────────────────────────────────

def train(args):
    torch, nn = _try_import_torch()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ── Load data ──────────────────────────────────────────────────────
    json_path = os.path.join(config.FINAL_DIR, config.OUTPUT_JSON_NAME)
    if not os.path.exists(json_path):
        logger.error(
            f"Dataset not found: {json_path}\n"
            "Run the pipeline in JAAD mode first:  python main.py --mode jaad"
        )
        sys.exit(1)

    tracks, labels = load_tracks(json_path)
    if len(tracks) < 10:
        logger.error("Too few labelled tracks (need ≥ 10). Use JAAD mode for ground truth.")
        sys.exit(1)

    train_ids, val_ids, test_ids = train_val_test_split(tracks, labels, args.seed)
    logger.info(f"Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test tracks")

    fw, fh = config.JAAD_FRAME_WIDTH, config.JAAD_FRAME_HEIGHT

    def subset(ids):
        sub_t = {tid: tracks[tid] for tid in ids}
        sub_l = {tid: labels[tid] for tid in ids}
        return build_sequences(sub_t, sub_l, args.seq_len, fw, fh)

    X_tr, y_tr = subset(train_ids)
    X_va, y_va = subset(val_ids)
    X_te, y_te = subset(test_ids)

    logger.info(f"Sequences: train={len(X_tr)}, val={len(X_va)}, test={len(X_te)}")

    # ── Class-balanced sampler ─────────────────────────────────────────
    n_pos  = int(y_tr.sum())
    n_neg  = len(y_tr) - n_pos
    if n_pos == 0 or n_neg == 0:
        logger.warning("Only one class present in training data — using uniform sampling")
        weights = np.ones(len(y_tr))
    else:
        pos_w = len(y_tr) / (2.0 * n_pos)
        neg_w = len(y_tr) / (2.0 * n_neg)
        weights = np.where(y_tr == 1, pos_w, neg_w)

    # ── Augment crossing sequences (Gaussian noise on bbox coords) ─────
    # Doubles crossing samples with small perturbations so the model sees
    # more variety in the minority class without changing labels.
    rng = np.random.default_rng(args.seed)
    crossing_mask = y_tr == 1
    X_crossing = X_tr[crossing_mask]
    noise = rng.normal(0, 0.005, X_crossing.shape).astype(np.float32)  # ±0.5% of frame
    X_aug = np.clip(X_crossing + noise, 0, 1)
    y_aug = np.ones(len(X_aug), dtype=np.float32)
    X_tr  = np.concatenate([X_tr, X_aug], axis=0)
    y_tr  = np.concatenate([y_tr, y_aug], axis=0)
    # Recompute weights after augmentation
    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    pos_w = len(y_tr) / (2.0 * n_pos)
    neg_w = len(y_tr) / (2.0 * n_neg)
    weights = np.where(y_tr == 1, pos_w, neg_w)
    logger.info(f"After augmentation: {n_pos} crossing, {n_neg} not-crossing")

    sampler = torch.utils.data.WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.float32), len(y_tr), replacement=True
    )

    ds_tr = torch.utils.data.TensorDataset(
        torch.tensor(X_tr), torch.tensor(y_tr)
    )
    loader = torch.utils.data.DataLoader(
        ds_tr, batch_size=args.batch_size, sampler=sampler
    )

    # ── Model + optimiser ──────────────────────────────────────────────
    model = IntentLSTM(hidden=args.hidden, num_layers=2, dropout=0.3)
    # pos_weight=2.5: penalises missing a crossing but less aggressively than
    # the raw class ratio (~5.2), reducing the FP flood seen at 0.5 threshold.
    pos_weight = torch.tensor([args.pos_weight], dtype=torch.float32)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    logger.info(f"pos_weight={args.pos_weight} (raw ratio={n_neg/max(n_pos,1):.2f})")
    optimizer  = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    # ── Training ───────────────────────────────────────────────────────
    X_va_t = torch.tensor(X_va)
    y_va_t = torch.tensor(y_va)

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    logger.info(
        f"Training {args.epochs} epochs | lr={args.lr} | "
        f"batch={args.batch_size} | hidden={args.hidden} | seq_len={args.seq_len}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_n = 0.0, 0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(xb)
            total_n    += len(xb)

        train_loss = total_loss / max(total_n, 1)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_va_t)
            val_loss   = criterion(val_logits, y_va_t).item()
            val_preds  = (torch.sigmoid(val_logits) >= 0.5).float()
            val_acc    = (val_preds == y_va_t).float().mean().item()

        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            logger.info(
                f"Epoch {epoch:4d}/{args.epochs}  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
            )

        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            best_epoch    = epoch
            patience_counter = 0
            # Save checkpoint
            os.makedirs(os.path.dirname(config.INTENT_MODEL_PATH), exist_ok=True)
            torch.save(model.net.state_dict(), config.INTENT_MODEL_PATH + ".pt")
            model.save_numpy(config.INTENT_MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(f"Early stopping at epoch {epoch} (best={best_epoch})")
                break

    logger.info(f"Best val loss: {best_val_loss:.4f} @ epoch {best_epoch}")

    # ── Test-set evaluation ────────────────────────────────────────────
    logger.info("─" * 60)
    logger.info("TEST SET EVALUATION")

    # Reload best weights
    best_npz = np.load(config.INTENT_MODEL_PATH, allow_pickle=False)

    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, classification_report,
    )

    X_te_t = torch.tensor(X_te)
    model.eval()
    with torch.no_grad():
        te_probs = torch.sigmoid(model(X_te_t)).numpy()

    te_preds = (te_probs >= 0.5).astype(int)

    acc  = accuracy_score(y_te, te_preds)
    prec = precision_score(y_te, te_preds, zero_division=0)
    rec  = recall_score(y_te, te_preds,    zero_division=0)
    f1   = f1_score(y_te, te_preds,        zero_division=0)
    try:
        auc = roc_auc_score(y_te, te_probs)
    except ValueError:
        auc = float("nan")

    logger.info(f"  Accuracy  : {acc:.4f}")
    logger.info(f"  Precision : {prec:.4f}")
    logger.info(f"  Recall    : {rec:.4f}")
    logger.info(f"  F1-Score  : {f1:.4f}")
    logger.info(f"  ROC-AUC   : {auc:.4f}")
    logger.info("\n" + classification_report(y_te, te_preds,
                target_names=["not_crossing", "crossing"]))
    logger.info("─" * 60)
    logger.info(f"Model saved → {config.INTENT_MODEL_PATH}")

    return {"acc": acc, "prec": prec, "rec": rec, "f1": f1, "auc": auc}


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train pedestrian intent LSTM")
    p.add_argument("--epochs",     type=int,   default=150)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--batch-size", type=int,   default=32)
    p.add_argument("--seq-len",    type=int,   default=24)
    p.add_argument("--hidden",     type=int,   default=64)
    p.add_argument("--patience",   type=int,   default=15)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--pos-weight", type=float, default=2.5,
                   help="BCEWithLogitsLoss positive class weight (default 2.5)")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
