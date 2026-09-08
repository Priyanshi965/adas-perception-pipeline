# nuScenes Camera + LiDAR + Radar Crossing-Intent Fusion

This extends the existing pedestrian crossing-intent pipeline to **nuScenes** as a
feature-level **camera + LiDAR + radar** fusion model. It reuses the project's
architecture unchanged — `PedTrack → feature timeline → leak-free observe→predict
windows → LSTM → NumPy export for torch-free inference` — and adds two new sensor
channel groups plus a nuScenes-specific data path.

---

## ⚠️ Read this first: the label is *derived*, not annotated

JAAD and PIE ship a per-frame "will this pedestrian cross the ego path" label.
**nuScenes does not.** It only provides 3D boxes + attributes
(`pedestrian.moving/standing/sitting`).

So the crossing label here is **derived from each pedestrian's future 3D
trajectory**: at time *t* a pedestrian is a *positive* if — while currently
**outside** the ego forward-path corridor — it **enters** that corridor within the
next `INTENT_TTE` steps (~1.5 s). The corridor is `|y| ≤ 1.75 m` and
`0 ≤ x ≤ 30 m` in the ego frame (x forward, y left), configurable in `config.py`.

Consequences, stated plainly:

- This is **leak-free** (verified): the observation window ends strictly before
  the corridor entry it predicts.
- It is a **different task definition** from JAAD/PIE. The resulting ROC-AUC is
  **not** an apples-to-apples fourth benchmark row next to JAAD (0.781) / PIE
  (0.797). Report it as *"nuScenes multimodal crossing (derived label)"*, a new
  task — not as "the same model, now on nuScenes."

---

## What was added / changed

| File | Change |
|------|--------|
| `modules/intent_features.py` | Appended `LIDAR` (4) + `RADAR` (4) channel groups **after** `aux`, so `kin/pose/aux` column offsets stay byte-identical → existing JAAD/PIE `.npz` models still load. `FULL_DIM` 20 → 28. New `use_lidar`/`use_radar` toggles in `active_columns`. |
| `datasets/nuscenes_loader.py` | Loads pedestrian tracks in the **ego frame** (x fwd, y left), per keyframe: 3D position, size, `num_lidar/radar_pts`, `moving` attribute, and the best camera + 2D box for pose. |
| `datasets/build_nuscenes_features.py` | Interpolates the 2 Hz track to `NUSC_TIMELINE_HZ` (10 Hz), derives the corridor label, and fills the `(T, 28)` timeline: **BEV kinematics**, **RTMPose** from the camera crop, **LiDAR** (range/density/height), **radar** (radial + xy velocity). Cached per scene. |
| `datasets/check_nuscenes.py` | One-shot install verification + label-distribution preview. |
| `train_intent.py` | `--dataset nuscenes`, `--lidar`, `--radar`; **scene-level** train/val/test split (no scene leaks across splits); GPU + AMP training. |
| `benchmark_fusion.py` | Real-time latency profiler (model + feature stages). |
| `config.py` | `NUSC_*` settings (dataroot, version, corridor, timeline Hz, radar radius, norms). |

### Feature layout (28 dims)

```
KIN  [0:8]   BEV/ego trajectory: cx=x/60, cy=y/60, w, h, vx, vy, ax, ay   (camera-free motion)
POSE [8:18]  body language from RTMPose on the projected camera crop       ← camera
AUX  [18:20] unused for nuScenes (zeros)
LID  [20:24] lid_range, lid_ptcount(log), lid_height, lid_present          ← LiDAR
RAD  [24:28] rad_vr, rad_vx, rad_vy, rad_present                           ← radar
```

Kinematics are computed in **BEV/ego frame**, not by 2D camera projection —
a pedestrian visible across nuScenes' 6 cameras would otherwise produce garbage
velocity/acceleration at camera boundaries. The camera contributes **pose**.

---

## 1. Get the data (you must do this — it needs your nuScenes account)

nuScenes requires registration + license acceptance at
<https://www.nuscenes.org/nuscenes>. Claude cannot download it for you.

**Start with `v1.0-mini`** (~4 GB, 10 scenes) to get the pipeline working
end-to-end, then move to `v1.0-trainval` **keyframes** for real numbers. Mini is
too small for a statistically meaningful scene-level train/val/test split (its
val will often be single-class — the trainer's threshold calibration handles that
but the number isn't trustworthy).

Extract so the layout is:

```
E:\datacleaning\nuscenes\          ←  config.NUSC_DATAROOT
├── v1.0-mini/                     (the JSON tables)
├── samples/                       (keyframe sensor blobs)
├── sweeps/                        (non-keyframe sweeps — optional; only for --pose dense)
└── maps/
```

Then verify:

```bash
python -m datasets.check_nuscenes --version v1.0-mini --dataroot ../nuscenes
```

You should see scene counts, the official split sizes, and a **positive
(will-cross) rate** — sanity-check that it's neither ~0 % nor ~100 %.

### Fastest path to real numbers: metadata-only (NO blob download)

The ~300 GB of sensor blobs are only needed for the **camera (pose)** and
**radar** channels. The **crossing label, BEV trajectory (KIN), and the LiDAR
channels** (range, point-density from `num_lidar_pts`, height) all come from the
**annotation pack**, `v1.0-trainval_meta.tgz` (~0.4 GB). So you can train a real
**trajectory + LiDAR** fusion model on the **full 850-scene trainval** with only
the meta download:

1. Download **`v1.0-trainval_meta.tgz`** and extract into
   `E:\datacleaning\nuscenes-trainval\` (gives `v1.0-trainval/  maps/`).
2. Run:

```powershell
$env:NUSC_VERSION = "v1.0-trainval"; $env:NUSC_DATAROOT = "E:\datacleaning\nuscenes-trainval"
E:\datacleaning\.venv\Scripts\python.exe -m datasets.check_nuscenes --meta-only
E:\datacleaning\.venv\Scripts\python.exe train_intent.py --dataset nuscenes --meta-only --lidar
```

`--meta-only` disables pose+radar (no blobs), auto-uses **all 850 scenes** (850
scenes → hundreds of positive scenes, vs mini's 5), and feature extraction is
fast (no RTMPose). Add the camera+radar channels later by downloading blobs and
dropping `--meta-only` (see below).

### Adding camera+radar later: trainval blobs (partial download is fine)

mini can't score the task (see results below). For trustworthy metrics use
`v1.0-trainval` — and you do **not** need all ~300 GB. On the nuScenes download
page under **Full dataset (v1.0) → Trainval**:

1. **`v1.0-trainval_meta.tgz`** (~0.4 GB) — **required**, the JSON tables for all
   850 scenes.
2. **1–3 of `v1.0-trainval01_blobs.tgz` … `10_blobs.tgz`** (~30 GB each) — each
   holds a subset of scenes' sensor data; two or three gives a few hundred scenes
   (hundreds of positive scenes).

Extract them all into a **separate** folder (keep mini intact), e.g.
`E:\datacleaning\nuscenes-trainval\`, giving `v1.0-trainval/ samples/ sweeps/
maps/`. The pipeline auto-detects which scenes are actually on disk
(`available_scene_tokens`), so a partial download simply trains on the scenes you
have. Point the code at it with env vars — no file edits:

```powershell
$env:NUSC_VERSION = "v1.0-trainval"
$env:NUSC_DATAROOT = "E:\datacleaning\nuscenes-trainval"
E:\datacleaning\.venv\Scripts\python.exe -m datasets.check_nuscenes
E:\datacleaning\.venv\Scripts\python.exe train_intent.py --dataset nuscenes --pose --lidar --radar
```

## 2. Install the devkit

nuScenes-devkit pins `numpy<2`, which fails to build on this project's
`numpy 2.x` / Python 3.13. Install it **without deps** and add the small ones:

```bash
pip install --no-deps nuscenes-devkit
pip install pyquaternion cachetools shapely
```

The core API + geometry work fine under numpy 2 (verified).

## 3. Train

```bash
# Full fusion: camera(pose) + BEV kinematics + LiDAR + radar
python train_intent.py --dataset nuscenes --pose --lidar --radar

# Ablations — measure each modality's contribution:
python train_intent.py --dataset nuscenes --pose                 # camera only
python train_intent.py --dataset nuscenes --pose --lidar         # + LiDAR
python train_intent.py --dataset nuscenes --no-pose --lidar --radar   # no camera
```

First run featurises + caches per scene (the slow part is RTMPose); later runs
reuse `checkpoints/nusc_features/`. The model is written to
`checkpoints/intent_model_nuscenes.npz` (+ `.pt`). The trainer prints test-set
Accuracy / Precision / Recall / F1 / ROC-AUC on the held-out **scenes**.

---

## Results on nuScenes-mini — the pipeline runs, but mini can't score it

Full run on real v1.0-mini: 210 pedestrian tracks / 10 scenes, features cached in
~7 min (RTMPose-dominated), GPU training in seconds. **Everything works
end-to-end.** But mini has only **5 scenes containing any crossing positive**
(~35 positive tracks total), so no honest split gives a stable metric.

Leave-one-positive-scene-out CV (train on the other scenes, test on the held-out
one), test ROC-AUC per fold:

| held-out scene | KIN | KIN+POSE | FULL (cam+LiDAR+radar) |
|---|---|---|---|
| scene-0061 | 0.190 | 0.284 | 0.552 |
| scene-0553 | 0.448 | 0.353 | 0.364 |
| scene-0916 | 0.980 | 0.641 | 0.461 |
| scene-1094 | 0.368 | 0.552 | 0.610 |
| scene-1100 | 0.819 | 0.264 | 0.621 |
| **mean ± std** | **0.561 ± 0.293** | **0.419 ± 0.151** | **0.521 ± 0.097** |

**Read this honestly:** all three are ~chance, and per-fold AUC swings from 0.19
to 0.98 — pure small-data noise. A single scene-0061 split earlier showed
KIN 0.26 → KIN+POSE 0.72 (pose "carrying" the signal); CV shows that was a
one-fold artifact that does not survive. FULL fusion has the **lowest variance**
(±0.097) but no mean lift. The per-scene diagnostic (`check_nuscenes` extended)
shows the label is clean (ego-motion-only positives ≤21%, median lateral step
~1–3 m), so this is a **data-volume limit, not a label/code bug**.

**Conclusion:** use mini to prove the pipeline (done); get real numbers from
`v1.0-trainval`, where ~700 scenes yield hundreds of positive scenes instead of 5.
Downloading even 2–3 of the 10 trainval blob files (~30 GB each) already gives a
few hundred scenes — enough for a trustworthy CV. Re-run the exact same commands
with `NUSC_VERSION="v1.0-trainval"`.

## Results on nuScenes-trainval (metadata-only: trajectory + LiDAR)

Full 850-scene trainval, meta pack only (no blobs), scene-level 70/15/15 split
(disjoint scenes, leak-free windows): **10,437 pedestrian tracks / 575
crossing-positive**; train 159,364 / val 34,641 / **test 35,837** windows.

Model = KIN (BEV trajectory) + LiDAR (range, `num_lidar_pts` density, height).
Held-out **test** set (556 positives across 114 scenes):

| Metric | Value |
|---|---|
| ROC-AUC | **0.993** |
| Accuracy | 0.989 |
| Precision | 0.61 |
| Recall | 0.75 |
| F1 | 0.68 |
| val ROC-AUC | 0.995 |

This is a **stable, trustworthy** number (34 positive test scenes), unlike mini
(0.5 ± 0.3 over one test scene). **But read it in context:** the *derived*
corridor label is far more determined by current position + velocity than
JAAD/PIE's human-annotated crossing *decision*, so **0.99 here is not comparable
to JAAD 0.781 / PIE 0.797** — those predict a human's intent; this predicts a
geometric event (entering the ego corridor) that trajectory already largely
implies. It is a different, easier task by construction. The value is that the
**pipeline, split, and fusion plumbing are validated at scale on real data**, and
the LiDAR/(camera/radar) channels can now be measured for genuine lift.

**Modality ablation (identical split):**

| Feature set | dims | test ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| KIN only (trajectory) | 8 | **0.9948** | 0.73 | 0.72 | **0.72** |
| KIN + LiDAR | 12 | 0.9927 | 0.61 | 0.75 | 0.68 |

**Honest finding: LiDAR adds no lift on this label — trajectory alone is
marginally better.** The derived corridor event is (by construction) a geometric
function of ego-frame position + velocity, which the KIN channels already carry;
the LiDAR range/density channels are largely redundant with position, so they add
noise, not signal. Two implications:

1. On the *current* derived label, **KIN-only is the model to deploy** — the
   fusion channels don't earn their place. (The fusion *infrastructure* is
   validated and ready; the label just doesn't need it.)
2. To make multimodal fusion genuinely pay off you need a label that trajectory
   *cannot* trivially predict — i.e. a human-intent-style label (does the person
   *decide* to cross, from body language) rather than a geometric one. That is
   what the **camera/pose** channel targets, and it needs the sensor blobs. LiDAR
   and radar would then contribute orthogonal depth/velocity cues the camera
   lacks. This is the clear next experiment once blobs are available.

## The decisive test: real camera+LiDAR+radar (85-scene subset)

To rule out "LiDAR is just a weak channel," we downloaded one trainval keyframe
partition (`v1.0-trainval01_keyframes.tgz`, ~4.5 GB via AWS S3 — 1/7th the size of
the full blobs) and built the **real** camera(RTMPose)+LiDAR+radar features on all
85 of its scenes, then ablated on one leak-free scene split (test = 140 positives
/ 4,882 windows), using the production focal-loss trainer:

| feature set | test ROC-AUC | F1 |
|---|---|---|
| **KIN (trajectory)** | **0.956** | **0.523** |
| KIN + POSE (camera body-language) | 0.855 | 0.292 |
| KIN + LiDAR + radar | 0.885 | 0.217 |
| FULL (all modalities) | 0.869 | 0.195 |

**Trajectory alone wins; every added modality — including the real pose channel —
hurts.** This agrees with the 850-scene result (KIN 0.995 ≥ KIN+LiDAR 0.993). The
finding is robust across scales and across the actual camera signal.

### Why — and what it means for "multifusion"

The reason is the **label**, not the sensors. nuScenes has no crossing-intent
annotation, so the label is *derived geometrically* — "does the pedestrian enter
the ego corridor" — which is (almost by definition) a function of position +
velocity. Trajectory saturates it, so extra channels only add capacity to overfit.

Contrast the project's JAAD/PIE result, where pose fusion **did** help
(ROC-AUC 0.730 → 0.781): those datasets ship a **human-annotated** crossing
*intent* label — a decision that body posture genuinely predicts *before* the
trajectory reveals it. That is the regime where multimodal fusion pays off.

**Conclusion.** The multimodal pipeline is fully built, optimized, and validated
on real nuScenes camera+LiDAR+radar data. But **on any label derivable from
nuScenes' 3D boxes, fusion does not beat trajectory** — a genuine, reproducible
finding, not a plumbing bug. Making fusion pay off needs a human-intent-style
label (which nuScenes lacks) — so for *intent* fusion, JAAD/PIE remain the right
data, and nuScenes' value is 3D range/velocity for tasks like **3D detection or
trajectory forecasting**, not this derived crossing label.

## Visual test-batch viewer (in the dashboard)

Since nuScenes has no separate "sample" set, the web UI runs the trained model
over a batch of **held-out test scenes** and animates it. Launch:

```powershell
cd E:\datacleaning\adas_pipeline
$env:NUSC_VERSION="v1.0-trainval"; $env:NUSC_DATAROOT="E:\datacleaning\nuscenes-trainval"
# (once) precompute the test-batch scenes the viewer plays:
E:\datacleaning\.venv\Scripts\python.exe -m datasets.export_scene_viz --split test --num 8
# start the dashboard:
E:\datacleaning\.venv\Scripts\python.exe -m app.server
```

Open <http://localhost:8000> → click **🛰️ nuScenes BEV fusion** (top-right), or go
straight to <http://localhost:8000/nuscenes>. On load it auto-runs the batch:

- **Bird's-eye view (ego frame):** the ego car sits at the bottom driving up; the
  shaded band is the forward-path corridor. Each dot is a pedestrian, coloured by
  the model's predicted crossing probability (green→amber→red), sized by LiDAR
  point density; a white ring marks a pedestrian currently inside the corridor.
- **Live metrics** accumulate across the batch at the model's calibrated
  threshold: ROC-AUC, precision, recall, and TP/FP/FN, plus a per-scene log
  ("caught 26/28, 4 FA").

`export_scene_viz.py` writes `app/static/nuscenes_scenes/*.json` (one per scene +
`index.json`), computed with the metadata-only KIN+LiDAR model, so it needs no
sensor blobs. Re-run the export with different `--split`/`--num`/`--scenes` to
change the batch. Example batch (8 test scenes): **ROC-AUC 0.991, P 0.70, R 0.67.**

## Optimization — "faster + real-time feasible"

### Inference (measured on this machine, RTX 4090 Laptop)

`python benchmark_fusion.py` — model-inference budget (`obs_len=16`, 26 fusion dims):

| Path | Latency | Throughput |
|------|---------|-----------|
| (a) NumPy LSTM, 1 window (deployed torch-free path) | 2.58 ms | 388 windows/s |
| (b) Torch CPU, batch 32 | 13.7 ms | 2,339 windows/s |
| (c) **Torch CUDA, batch 32** | **1.01 ms** | **31,778 windows/s** |

**Takeaway:** the deployed NumPy path is fine for a handful of pedestrians
(388 peds/s ≈ 38 FPS with 10 peds/frame). For dense scenes, the win is to
**batch all pedestrians in a frame into one CUDA call** — 32 pedestrians in
~1 ms (~30× the per-pedestrian NumPy loop). No TensorRT needed to hit real time;
measure before reaching for it.

The real per-frame cost is **feature extraction, not the model** — dominated by
RTMPose on the camera crop. Profile it on real data:

```bash
python benchmark_fusion.py --features 60
```

LiDAR density + geometry are read from the annotation (≈0 ms); radar sweeps are
tiny. Pose is the target for batching/caching.

### Training

- **GPU + AMP** are on by default (`--cpu` / `--no-amp` to disable). The LSTM is
  tiny and the whole feature set fits in memory, so training itself is seconds.
- The real cost is **feature extraction**, paid **once** via per-scene caching
  (`checkpoints/nusc_features/`). Re-runs and ablations are then instant.
- For `trainval`, download **keyframes only** — the default `NUSC_POSE_SAMPLING
  = "keyframe"` never touches the `sweeps/` blobs.

---

## Limitations & honest caveats

- **Derived label** (see top): a new task, not comparable to PIE/JAAD AUC.
- **Corridor label** approximates "steps into our path"; it does not use the HD
  map's crosswalk polygons yet (a natural next refinement — `NUSC_*` corridor
  params are the current proxy).
- **Pose between keyframes** is held piecewise-constant (2 Hz) unless you set
  `NUSC_POSE_SAMPLING="dense"` and download sweeps.
- **mini** cannot give trustworthy metrics — use it only to prove the pipeline
  runs; report numbers from `trainval`.
- The live mono-camera web UI (`app/server.py`) cannot feed LiDAR/radar, so a
  nuScenes fusion model runs via this offline path, not that UI.
