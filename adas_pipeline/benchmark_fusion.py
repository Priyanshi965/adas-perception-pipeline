"""
benchmark_fusion.py — Real-time latency profiler for the nuScenes fusion model.

"Optimize the time / feasible for real-time" means knowing where the milliseconds
go. This decomposes the per-frame budget into the stages that actually cost time:

  1. model inference   — the crossing-intent network itself, three ways:
        (a) NumPy LSTM  (the deployed torch-free path, one window at a time)
        (b) Torch CPU   (batched)
        (c) Torch CUDA  (batched, the real-time path)
  2. feature extraction — only with nuScenes data present:
        radar sweep load + association, and RTMPose on the camera crop.

The model benchmark runs with NO dataset (synthetic windows of the right shape),
so you get the inference budget immediately; feature timing needs the blobs.

Usage:
  python benchmark_fusion.py                 # model-inference budget only
  python benchmark_fusion.py --features 40   # + profile feature extraction
  python benchmark_fusion.py --export-onnx   # dump an ONNX graph of the model
"""

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import config
from modules import intent_features as ifeat


def _synthetic_npz(n_cols: int, hidden: int, path: str):
    """Write a random-weight model in the trainer's .npz format for timing."""
    rng = np.random.default_rng(0)
    def r(*s): return rng.standard_normal(s).astype(np.float32) * 0.1
    np.savez(
        path, threshold=np.float32(0.5),
        lstm_Wih=r(4 * hidden, n_cols), lstm_Whh=r(4 * hidden, hidden), lstm_b=r(4 * hidden),
        lstm_Wih_1=r(4 * hidden, hidden), lstm_Whh_1=r(4 * hidden, hidden), lstm_b_1=r(4 * hidden),
        fc_W=r(hidden), fc_b=r(1),
        feat_mean=np.zeros(n_cols, np.float32), feat_std=np.ones(n_cols, np.float32),
        active_cols=ifeat.active_columns(True, True, use_lidar=True, use_radar=True),
        obs_len=np.int64(config.INTENT_OBS_LEN),
        use_kinematics=np.int64(1), use_pose=np.int64(1),
        use_lidar=np.int64(1), use_radar=np.int64(1),
    )


def _time(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters * 1e3   # ms/iter


def bench_model(args):
    from modules.intent_predictor import IntentPredictor
    cols = ifeat.active_columns(True, True, use_lidar=True, use_radar=True)
    D, obs = len(cols), config.INTENT_OBS_LEN
    model_path = config.NUSC_MODEL_PATH
    tmp = None
    if not os.path.exists(model_path):
        tmp = os.path.join(config.CHECKPOINT_DIR, "_bench_synth.npz")
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        _synthetic_npz(D, config.INTENT_HIDDEN_SIZE, tmp)
        model_path = tmp
        print(f"(no trained nuScenes model — timing a synthetic {D}-dim model)")

    print("\n" + "=" * 62)
    print(f"MODEL INFERENCE BUDGET  (obs_len={obs}, fusion_dims={D})")
    print("=" * 62)

    # (a) deployed NumPy path — one window at a time (per-pedestrian, per-frame)
    pred = IntentPredictor(model_path=model_path)
    win = np.random.randn(obs, ifeat.FULL_DIM).astype(np.float32)
    ms_np = _time(lambda: pred.model.forward(pred._prep_window(win, obs - 1)), args.iters)
    print(f"(a) NumPy LSTM, 1 window        : {ms_np:6.3f} ms  "
          f"→ {1000/ms_np:7.0f} windows/s  ({int((1000/ms_np))} peds @1 win/frame)")

    # (b)/(c) Torch batched — the many-pedestrians-per-frame real-time path
    try:
        import torch
        import train_intent as ti
        net = ti.build_net(D, config.INTENT_HIDDEN_SIZE).eval()
        for B in (args.batch, ):
            xb = torch.randn(B, obs, D)
            ms_cpu = _time(lambda: net(xb).detach(), max(args.iters // 5, 5))
            print(f"(b) Torch CPU, batch={B:<4d}        : {ms_cpu:6.3f} ms  "
                  f"→ {B*1000/ms_cpu:7.0f} windows/s")
            if torch.cuda.is_available():
                netc = net.cuda(); xbc = xb.cuda()
                torch.cuda.synchronize()
                def _c():
                    with torch.no_grad(): netc(xbc); torch.cuda.synchronize()
                ms_gpu = _time(_c, args.iters)
                print(f"(c) Torch CUDA, batch={B:<4d}       : {ms_gpu:6.3f} ms  "
                      f"→ {B*1000/ms_gpu:7.0f} windows/s  "
                      f"({B} peds/frame in {ms_gpu:.2f} ms)")
    except Exception as e:
        print(f"(torch batched path skipped: {e})")

    if tmp and os.path.exists(tmp):
        os.remove(tmp)
    return ms_np


def bench_features(n_keyframes: int):
    """Profile the fusion feature-extraction stages on real nuScenes keyframes."""
    if not os.path.isdir(config.NUSC_DATAROOT):
        print(f"\n(feature profiling skipped — no nuScenes at {config.NUSC_DATAROOT})")
        return
    from nuscenes.nuscenes import NuScenes
    from datasets import nuscenes_loader as nl
    from datasets import build_nuscenes_features as bnf
    from modules.pose_estimator import PoseEstimator

    print("\n" + "=" * 62)
    print("FEATURE EXTRACTION BUDGET  (real nuScenes keyframes)")
    print("=" * 62)
    nusc = NuScenes(version=config.NUSC_VERSION, dataroot=config.NUSC_DATAROOT, verbose=False)
    scene_tok = nusc.scene[0]["token"]
    tracks = nl.load_nuscenes_tracks(nusc, [scene_tok], with_pose=True)
    if not tracks:
        print("no pedestrian tracks in first scene"); return
    pose_est = PoseEstimator()

    t0 = time.perf_counter(); nrad = 0
    for t in tracks[: max(1, n_keyframes // max(len(tracks[0].frames), 1))]:
        bnf._radar_feature_per_keyframe(nusc, t); nrad += len(t.frames)
    ms_rad = (time.perf_counter() - t0) / max(nrad, 1) * 1e3
    print(f"radar load+associate / keyframe : {ms_rad:6.2f} ms")

    t0 = time.perf_counter(); npose = 0
    for t in tracks:
        p = bnf._pose_feature_per_keyframe(t, pose_est); npose += len(t.frames)
        if npose >= n_keyframes: break
    ms_pose = (time.perf_counter() - t0) / max(npose, 1) * 1e3
    print(f"RTMPose / pedestrian-keyframe   : {ms_pose:6.2f} ms")
    print("\nNote: LiDAR density + geometry are read from the annotation "
          "(≈0 ms); radar sweeps are tiny; pose dominates and is the first "
          "target for batching / caching.")


def export_onnx():
    import torch
    import train_intent as ti
    cols = ifeat.active_columns(True, True, use_lidar=True, use_radar=True)
    net = ti.build_net(len(cols), config.INTENT_HIDDEN_SIZE).eval()
    out = os.path.join(config.CHECKPOINT_DIR, "intent_model_nuscenes.onnx")
    dummy = torch.randn(1, config.INTENT_OBS_LEN, len(cols))
    torch.onnx.export(net, dummy, out, input_names=["window"], output_names=["logit"],
                      dynamic_axes={"window": {0: "batch"}}, opset_version=17)
    print(f"ONNX graph → {out}")


def main():
    p = argparse.ArgumentParser(description="nuScenes fusion latency benchmark")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--batch", type=int, default=32, help="pedestrians/frame for batched timing")
    p.add_argument("--features", type=int, default=0, metavar="N",
                   help="also profile feature extraction over ~N keyframes (needs data)")
    p.add_argument("--export-onnx", action="store_true")
    args = p.parse_args()

    bench_model(args)
    if args.features:
        bench_features(args.features)
    if args.export_onnx:
        export_onnx()


if __name__ == "__main__":
    main()
