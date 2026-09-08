"""
check_nuscenes.py — Verify a nuScenes install and preview the derived label.

Run this once after downloading nuScenes (before training) to confirm the blobs
are laid out where config expects and that the crossing-corridor label produces a
sane positive rate. It loads the geometry only (no pose/radar), so it is fast.

Usage:
  python -m datasets.check_nuscenes                # uses config.NUSC_VERSION/ROOT
  python -m datasets.check_nuscenes --version v1.0-mini --dataroot ../nuscenes
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config
from datasets import nuscenes_loader as nl
from datasets import build_nuscenes_features as bnf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--version", default=config.NUSC_VERSION)
    p.add_argument("--dataroot", default=config.NUSC_DATAROOT)
    p.add_argument("--meta-only", dest="meta_only", action="store_true",
                   help="use every scene from the metadata pack (no sensor blobs needed; "
                        "kin+LiDAR+label only)")
    args = p.parse_args()

    print(f"nuScenes version : {args.version}")
    print(f"dataroot         : {os.path.abspath(args.dataroot)}")
    if not os.path.isdir(args.dataroot):
        print("\nERROR: dataroot does not exist. Download nuScenes and either put it "
              "there or set config.NUSC_DATAROOT / pass --dataroot.")
        sys.exit(1)

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    print(f"scenes (metadata): {len(nusc.scene)}")

    if args.meta_only:
        avail = [s["token"] for s in nusc.scene]
        print(f"scenes usable    : {len(avail)}  (meta-only: kin+LiDAR+label, no blobs)")
    else:
        # Partial-download safe: how many scenes actually have blobs on disk?
        avail = nl.available_scene_tokens(nusc)
        print(f"scenes on disk   : {len(avail)}"
              + ("  (partial download)" if len(avail) < len(nusc.scene) else "  (complete)"))
        if not avail:
            print("\nERROR: no scene blobs found on disk. Either extract the sample/blob "
                  "archives, or pass --meta-only to train on kin+LiDAR from the metadata "
                  "pack alone (see docs/nuscenes_fusion.md)."); sys.exit(1)

    # Official splits present?
    for split in (("mini_train", "mini_val") if "mini" in args.version else ("train", "val")):
        toks = set(nl.scene_tokens_for_split(nusc, split)) & set(avail)
        print(f"  split {split:11s}: {len(toks)} scenes on disk")

    # Geometry-only track load over a sample of AVAILABLE scenes + label preview.
    n_preview = min(40 if args.meta_only else 5, len(avail))
    scene_toks = avail[:n_preview]
    tracks = nl.load_nuscenes_tracks(nusc, scene_toks, with_pose=False)
    print(f"\nped tracks (first {len(scene_toks)} scenes): {len(tracks)}")
    if not tracks:
        print("No pedestrian tracks found — check the annotation blobs."); return

    from modules import intent_features as ifeat
    LAT_THR = 0.3   # m: below this the pedestrian barely moved sideways over the horizon
    tot_win = tot_pos = ego_only = ever = gapped = 0
    for t in tracks:
        interp = bnf._interpolate_track(t, config.NUSC_TIMELINE_HZ)
        if interp is None:
            continue
        pos, valid = interp["pos"], interp["valid"]
        gapped += int((~valid).sum())
        cross = bnf._corridor_label(pos)
        cross[~valid] = -1
        ever += int((cross == 1).any())
        _, y, ends = ifeat.make_windows(
            np.zeros((len(cross), ifeat.FULL_DIM), np.float32), cross,
            config.INTENT_OBS_LEN, config.INTENT_TTE, config.INTENT_SAMPLE_STRIDE)
        tot_win += len(y); tot_pos += int(np.sum(y))
        # Ego-motion-only diagnostic: for each positive, how far did the PEDESTRIAN
        # move laterally between the window end and the corridor entry it predicts?
        for yi, e in zip(y, ends):
            if yi != 1:
                continue
            fut = cross[e + 1: e + 1 + config.INTENT_TTE]
            rel = np.argmax(fut == 1)
            entry = e + 1 + int(rel)
            if abs(pos[entry, 1] - pos[e, 1]) < LAT_THR:
                ego_only += 1

    print(f"tracks entering corridor      : {ever}/{len(tracks)}")
    print(f"timeline steps in annot. gaps : {gapped} (excluded as cross=-1)")
    print(f"crossing-onset windows        : {tot_win}")
    if tot_win:
        print(f"positive (will-cross) rate    : {100*tot_pos/tot_win:.1f}%  "
              f"({tot_pos} positives)")
    if tot_pos:
        frac = 100 * ego_only / tot_pos
        print(f"ego-motion-only positives     : {frac:.1f}%  ({ego_only}/{tot_pos})  "
              f"[pedestrian |Δy| < {LAT_THR} m to entry]")
        if frac > 40:
            print("  ⚠ High: many positives are the ego driving up to a laterally-static\n"
                  "    pedestrian, not a real sideways step-in. Consider lowering\n"
                  "    config.NUSC_CORRIDOR_LOOKAHEAD_M before trusting the AUC.")
    print("\nIf the rates look sane, build features and train:")
    print("  python train_intent.py --dataset nuscenes --pose --lidar --radar")


if __name__ == "__main__":
    main()
