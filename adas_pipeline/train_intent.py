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
from datasets import build_pie_features as bpf
from modules import intent_features as ifeat

try:
    from datasets import build_nuscenes_features as bnf
    from datasets import nuscenes_loader as nl
except Exception:  # nuscenes-devkit optional until the dataset is used
    bnf = nl = None

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


def split_tracks_by_scene(tracks: List[Dict], seed: int) -> Tuple[List, List, List]:
    """70/15/15 split at SCENE level so no scene's pedestrians appear in two
    splits (the honest split for nuScenes; track-level would leak context).

    Positives (tracks that ever enter the corridor) cluster in a few scenes, so a
    naive scene shuffle can hand val/test a positive-free scene — fatal on the
    tiny mini split (single-class val → meaningless model selection). We therefore
    **stratify by whether a scene contains any positive track**: positive-bearing
    and negative-only scenes are each split 70/15/15, then combined. This keeps
    the split leak-free at the scene level while guaranteeing positives land in
    every partition whenever there are enough positive scenes.
    """
    def scene_of(t): return t.get("scene", t["video_id"])
    pos_scenes = {scene_of(t) for t in tracks if (t["cross"] == 1).any()}
    all_scenes = {scene_of(t) for t in tracks}
    grp_pos = sorted(pos_scenes)
    grp_neg = sorted(all_scenes - pos_scenes)
    rng = random.Random(seed)
    tr_s, va_s, te_s = set(), set(), set()
    for grp in (grp_pos, grp_neg):
        rng.shuffle(grp)
        n = len(grp)
        a, b = int(round(n * 0.70)), int(round(n * 0.85))
        # Ensure val/test are non-empty when the group has >=3 scenes.
        if n >= 3:
            a = min(a, n - 2); b = min(max(b, a + 1), n - 1)
        tr_s |= set(grp[:a]); va_s |= set(grp[a:b]); te_s |= set(grp[b:])
    def pick(S):
        return [i for i, t in enumerate(tracks) if scene_of(t) in S]
    return pick(tr_s), pick(va_s), pick(te_s)


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
        use_lidar=np.int64(int(getattr(args, "use_lidar", False))),
        use_radar=np.int64(int(getattr(args, "use_radar", False))),
    )
    logger.info(f"Saved model → {path}")


# ─── Train ────────────────────────────────────────────────────────────────────

def calibrate_threshold(y, p, min_recall=0.7):
    """Pick an operating threshold that maximises F1 subject to recall >= min_recall.

    Returns None if y is single-class (caller should calibrate elsewhere). Falls
    back to max balanced-accuracy if the recall floor is unreachable. This prevents
    the model from being deployed at a threshold where it predicts only one class.
    """
    if len(np.unique(y)) < 2:
        return None
    from sklearn.metrics import f1_score, recall_score, balanced_accuracy_score
    cand = np.unique(np.round(p, 3))
    cand = cand[(cand > p.min()) & (cand < p.max())]
    if len(cand) == 0:
        return 0.5
    best_t, best_f1 = None, -1.0
    for t in cand:
        pred = (p >= t).astype(int)
        if recall_score(y, pred, zero_division=0) < min_recall:
            continue
        f = f1_score(y, pred, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    if best_t is None:
        best_t = max((balanced_accuracy_score(y, (p >= t).astype(int)), t) for t in cand)[1]
    return float(best_t)


def train(args):
    torch, nn = _import_torch()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    use_amp = (device.type == "cuda" and not args.no_amp)
    logger.info(f"Training device: {device}  (AMP={'on' if use_amp else 'off'})")

    # Output path is dataset-specific so runs never clobber each other.
    model_path = config.INTENT_MODEL_PATH
    if args.dataset == "pie":
        base, ext = os.path.splitext(config.INTENT_MODEL_PATH)
        model_path = base + "_pie" + ext
    elif args.dataset == "nuscenes":
        model_path = config.NUSC_MODEL_PATH

    if args.dataset == "pie":
        # PIE reuses the shared feature builder; all present sets, all tracks.
        # The track-level split below IS the random split (leak-free by track).
        tracks = bpf.build_dataset(with_pose=args.use_pose, set_ids=None, force=False)
    elif args.dataset == "nuscenes":
        if bnf is None:
            logger.error("nuscenes-devkit not importable — pip install nuscenes-devkit"); sys.exit(1)
        if args.meta_only:
            # No sensor blobs: camera(pose) + radar are unavailable, so disable them
            # and train on trajectory + LiDAR (both come from the annotation pack).
            if args.use_pose or args.use_radar:
                logger.warning("--meta-only: disabling pose+radar (need blobs); "
                               "training on KIN+LiDAR from the metadata pack.")
            args.use_pose = False; args.use_radar = False
            tracks = bnf.build_dataset(scene_tokens=None, with_pose=False,
                                       with_radar=False, require_blobs=False, force=False)
        else:
            tracks = bnf.build_dataset(scene_tokens=None, with_pose=args.use_pose,
                                       force=False)
    else:
        vids = jl.available_video_ids(config.JAAD_ANNOTATIONS_DIR)[: args.videos]
        tracks = bjf.build_dataset(vids, with_pose=args.use_pose, force=False)
    tracks = [t for t in tracks if (t["cross"] >= 0).any()]
    if len(tracks) < 8:
        logger.error(f"Only {len(tracks)} usable tracks — increase --videos / dataset."); sys.exit(1)

    cols = ifeat.active_columns(args.use_kin, args.use_pose, use_aux=args.use_aux,
                                use_lidar=args.use_lidar, use_radar=args.use_radar)
    logger.info(f"Feature set: kin={args.use_kin} pose={args.use_pose} aux={args.use_aux} "
                f"lidar={args.use_lidar} radar={args.use_radar} "
                f"→ {len(cols)} dims: {[ifeat.FULL_FEATURE_NAMES[c] for c in cols]}")

    # nuScenes splits at scene level (no scene leaks across train/val/test);
    # JAAD/PIE split at track level (their 'scene' is one clip).
    if args.dataset == "nuscenes":
        tr_ids, va_ids, te_ids = split_tracks_by_scene(tracks, args.seed)
        def _sc(ids):
            return sorted({tracks[i].get("scene", tracks[i]["video_id"]) for i in ids})
        def _pos_sc(ids):
            return sorted({tracks[i].get("scene", tracks[i]["video_id"]) for i in ids
                           if (tracks[i]["cross"] == 1).any()})
        logger.info(f"Scene split — train {_sc(tr_ids)}  (+ve scenes: {_pos_sc(tr_ids)})")
        logger.info(f"Scene split — val   {_sc(va_ids)}  (+ve scenes: {_pos_sc(va_ids)})")
        logger.info(f"Scene split — test  {_sc(te_ids)}  (+ve scenes: {_pos_sc(te_ids)})")
    else:
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
    # Single imbalance mechanism: focal loss with a mild pos_weight cap. Previously
    # a class-balanced sampler AND pos_weight<=5 both compensated at once, biasing
    # the model toward positive and collapsing its probabilities into a narrow band.
    pos_weight = torch.tensor([min(n_neg / n_pos, 2.0)], dtype=torch.float32, device=device)

    class FocalLoss(nn.Module):
        def __init__(self, gamma=2.0, pos_weight=None):
            super().__init__(); self.gamma = gamma; self.pos_weight = pos_weight
        def forward(self, logits, targets):
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=self.pos_weight, reduction="none")
            p = torch.sigmoid(logits)
            p_t = p * targets + (1 - p) * (1 - targets)
            return ((1 - p_t) ** self.gamma * bce).mean()

    net = build_net(len(cols), args.hidden).to(device)
    crit = FocalLoss(gamma=2.0, pos_weight=pos_weight)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    Xt = torch.tensor(X_tr); yt = torch.tensor(y_tr)
    Xv = torch.tensor(X_va, device=device); yv = torch.tensor(y_va, device=device)
    ds = torch.utils.data.TensorDataset(Xt, yt)
    # Plain shuffled loader — focal loss handles imbalance; no balanced sampler.
    # Whole training set is small enough to keep pinned in host RAM; batches move
    # to the GPU in the loop. (This trainer is I/O-free once features are cached.)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                         pin_memory=(device.type == "cuda"))

    from copy import deepcopy
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, classification_report)

    # Model selection by validation ROC-AUC (threshold-free → robust to the class
    # imbalance and the tiny val sets that JAAD track-level splits produce).
    best_auc, best_ep, patience, best_state = -1.0, 0, 0, deepcopy(net.state_dict())
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    for ep in range(1, args.epochs + 1):
        net.train()
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = crit(net(xb), yb)
            scaler.scale(loss).backward()
            if use_amp:
                scaler.unscale_(opt)   # unscale_ raises when the scaler is disabled
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        net.eval()
        with torch.no_grad():
            vlogits = net(Xv)
            vl = crit(vlogits, yv).item()
            vprob = torch.sigmoid(vlogits).cpu().numpy()
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
        vprob = torch.sigmoid(net(Xv)).cpu().numpy()
    # Calibrate the operating threshold: maximise F1 subject to a minimum-recall
    # floor (prevents collapse to all-negative), falling back to max balanced-acc
    # if the floor is unreachable. If val is single-class, calibrate on TRAIN
    # instead of silently defaulting to 0.5.
    thr = calibrate_threshold(y_va, vprob)
    if thr is None:
        with torch.no_grad():
            tprob = torch.sigmoid(net(torch.tensor(X_tr, device=device))).cpu().numpy()
        thr = calibrate_threshold(y_tr, tprob) or 0.5
        logger.warning("val single-class — threshold calibrated on TRAIN split")
    logger.info(f"Calibrated threshold: {thr:.3f}")
    logger.info(f"Val prob spread: min={vprob.min():.3f} med={np.median(vprob):.3f} "
                f"max={vprob.max():.3f}  |  frac>=thr={(vprob >= thr).mean():.3f}")
    export_npz(net, model_path, mean, std, cols, args, threshold=float(thr))
    torch.save(net.state_dict(), model_path + ".pt")

    # ── Test evaluation (best weights, calibrated threshold) ──
    with torch.no_grad():
        probs = (torch.sigmoid(net(torch.tensor(X_te, device=device))).cpu().numpy()
                 if len(X_te) else np.array([]))
    groups = [g for g, on in (("KIN", args.use_kin), ("POSE", args.use_pose),
              ("AUX", args.use_aux), ("LIDAR", args.use_lidar),
              ("RADAR", args.use_radar)) if on]
    logger.info("=" * 56)
    logger.info(f"TEST SET  (feature set: {'+'.join(groups)})")
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
    logger.info(f"Model → {model_path}")


def parse_args():
    p = argparse.ArgumentParser(description="Train pedestrian intent model")
    p.add_argument("--dataset", choices=["jaad", "pie", "nuscenes"], default="jaad",
                   help="jaad (flat clips, --videos), pie (3-set subset), or "
                        "nuscenes (camera+LiDAR+radar fusion)")
    p.add_argument("--videos", type=int, default=config.__dict__.get("TRAIN_VIDEOS", 40))
    p.add_argument("--pose", dest="use_pose", action="store_true", default=config.INTENT_USE_POSE)
    p.add_argument("--no-pose", dest="use_pose", action="store_false")
    p.add_argument("--no-kin", dest="use_kin", action="store_false", default=config.INTENT_USE_KINEMATICS)
    p.add_argument("--aux", dest="use_aux", action="store_true", default=False,
                   help="include JAAD look/action GT as features (leaky for deployment)")
    p.add_argument("--lidar", dest="use_lidar", action="store_true", default=False,
                   help="include nuScenes LiDAR channels (range/density/height)")
    p.add_argument("--radar", dest="use_radar", action="store_true", default=False,
                   help="include nuScenes radar channels (radial/xy velocity)")
    p.add_argument("--meta-only", dest="meta_only", action="store_true", default=False,
                   help="nuScenes: train on trajectory+LiDAR from the metadata pack "
                        "alone (no camera/radar blobs needed — full trainval works)")
    p.add_argument("--cpu", action="store_true", default=False, help="force CPU training")
    p.add_argument("--no-amp", action="store_true", default=False,
                   help="disable mixed precision on GPU")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=config.INTENT_HIDDEN_SIZE)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
