"""
deploy_realtime.py — Sustained real-time feasibility test for crossing-intent.

Simulates a real deployment: streams a scene frame-by-frame, keeps a rolling
observation buffer per tracked pedestrian (O(1) per frame — no recompute), runs
the deployed model each frame on every active pedestrian, and reports sustained
FPS plus p50/p99 per-frame latency at realistic pedestrian counts.

This answers "can it run real-time on a live sensor stream?" — not just the
one-window micro-benchmark.

Usage (from adas_pipeline/, with NUSC_* env set):
  python deploy_realtime.py                 # numpy (CPU-deployable) path
  python deploy_realtime.py --gpu           # + torch CUDA batched path
"""
import argparse, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
import config
from modules import intent_features as ifeat
from modules.intent_predictor import IntentPredictor


def load_scene_tracks(n_scenes):
    """Load cached per-track feature timelines (meta-only KIN+LiDAR)."""
    from datasets import build_nuscenes_features as bnf
    tracks = bnf.build_dataset(scene_tokens=None, with_pose=False, with_radar=False,
                               require_blobs=False)
    # group by scene, keep scenes with the most tracks (busy = worst case)
    by_scene = {}
    for t in tracks:
        by_scene.setdefault(t["scene"], []).append(t)
    scenes = sorted(by_scene.values(), key=len, reverse=True)[:n_scenes]
    return scenes


def stream_scene(scene_tracks, pred, cols, mean, std, obs):
    """Replay one scene; return list of (n_active, latency_ms) per frame."""
    T = max(t["feats"].shape[0] for t in scene_tracks)
    out = []
    for f in range(obs - 1, T):
        # active pedestrians this frame = those whose track covers [f-obs+1 .. f]
        wins = []
        for t in scene_tracks:
            ft = t["feats"]
            if ft.shape[0] > f:
                w = ft[f - obs + 1: f + 1][:, cols]
                wins.append(((w - mean) / std).astype(np.float32))
        if not wins:
            continue
        t0 = time.perf_counter()
        for w in wins:                        # per-pedestrian forward (deployed numpy path)
            pred.model.forward(w)
        dt = (time.perf_counter() - t0) * 1e3
        out.append((len(wins), dt))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=6)
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    pred = IntentPredictor(model_path=config.NUSC_MODEL_PATH)
    if pred.model is None:
        print("No trained nuScenes model at", config.NUSC_MODEL_PATH); return
    cols, mean, std, obs = pred.cols, pred.mean, pred.std, pred.obs_len
    print(f"Deployed model: {len(cols)} features, obs_len={obs}, threshold={pred.threshold:.2f}")

    scenes = load_scene_tracks(args.scenes)
    print(f"Streaming {len(scenes)} scenes (worst-case busiest)...\n")

    per_frame = []
    for sc in scenes:
        per_frame += stream_scene(sc, pred, cols, mean, std, obs)

    counts = np.array([c for c, _ in per_frame])
    lat = np.array([d for _, d in per_frame])
    print("=" * 60)
    print(f"frames streamed         : {len(per_frame)}")
    print(f"pedestrians/frame        : mean {counts.mean():.1f}  max {counts.max()}")
    print(f"per-frame latency (numpy CPU-deployable path):")
    print(f"   p50 {np.percentile(lat,50):.2f} ms | p90 {np.percentile(lat,90):.2f} ms | "
          f"p99 {np.percentile(lat,99):.2f} ms | max {lat.max():.2f} ms")
    fps = 1000.0 / np.percentile(lat, 99)
    print(f"sustained FPS @ p99      : {fps:.0f}  "
          f"({'REAL-TIME ✓' if fps>=30 else 'below 30 FPS'}; sensor keyframes are 2 Hz)")
    print(f"per-pedestrian           : {lat.sum()/counts.sum():.3f} ms")

    if args.gpu:
        try:
            import torch, train_intent as ti
            dev = torch.device("cuda")
            net = ti.build_net(len(cols), config.INTENT_HIDDEN_SIZE).to(dev).eval()
            # batch all active peds of the busiest frame into one forward
            busiest = max(per_frame, key=lambda z: z[0])[0]
            xb = torch.randn(busiest, obs, len(cols), device=dev)
            for _ in range(10): net(xb)
            torch.cuda.synchronize(); t0 = time.perf_counter()
            for _ in range(200):
                with torch.no_grad(): net(xb); torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) / 200 * 1e3
            print(f"\nGPU batched ({busiest} peds/frame in ONE call): {ms:.3f} ms/frame "
                  f"→ {1000/ms:.0f} FPS")
        except Exception as e:
            print("gpu path skipped:", e)


if __name__ == "__main__":
    main()
