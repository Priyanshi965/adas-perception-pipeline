# 🚗 ADAS Sensor Data Cleaning & Behavior Analysis Pipeline

An end-to-end ML pipeline that ingests dashcam or surveillance video, detects and tracks road agents, classifies behaviors, scores danger, and produces annotated video output — all accessible through a live web UI.

---

## 📌 Overview

This project evolved from `intent_fusion.py`, a prototype script that ran YOLOv8 on `.MOV` dashcam clips to flag moving pedestrians as dangerous. The full pipeline is the production system built on that idea — modular, resumable, and multi-output.

**Input:** Raw dashcam video or JAAD dataset clips  
**Output:** Annotated MP4 video · `dataset.json` · `dataset.csv` · PASCAL VOC XML annotations

---

## 🏗️ Architecture

```
Input Video / JAAD Dataset
        │
        ▼
┌──────────────────────────────────────────┐
│         PipelineRunner (orchestrator)    │
│  · Runs 8 stages in sequence             │
│  · Checkpoints after each stage          │
│  · Supports resume on failure            │
└──────────────────────────────────────────┘
        │
        ▼
   1. input_handler      → validate video / JAAD XML
   2. frame_extractor    → extract JPG frames (every Nth)
   3. frame_cleaner      → remove blurry / dark / corrupt frames
   4. detector           → RT-DETR bounding boxes (transformer; YOLO selectable)
   5. tracker            → assign consistent track IDs (ByteTrack / IoU)
   6. pose_estimator     → 17-keypoint skeleton + body-language features
   7. behavior_analyzer  → classify motion (walking / crossing / running / driving / stopping)
   8. intent_predictor   → crossing-intent PREDICTION from pose + trajectory
   9. tagger             → compute danger_score + DANGER/SAFE scene tag
  10. exporter           → write dataset.json + dataset.csv
        │
        ▼
   visualizer.py         → render annotated MP4 (H.264)
   xml_exporter.py       → PASCAL VOC XML annotations
```

> **What changed vs the original pipeline.** The system no longer just *tracks*
> and reacts to motion — it *predicts* whether each pedestrian is about to cross,
> from body posture (lean, head/gaze turn, stance, gait, foot placement) fused
> with trajectory, using a leak-free observe→predict temporal model. The detector
> was upgraded from YOLOv8 to RT-DETR (a transformer detector). See
> **Crossing-Intent Prediction** below.

---

## 🔬 Pipeline Stages

| # | Stage | Description |
|---|-------|-------------|
| 1 | **input_handler** | Validates codec, dimensions, frame count; parses JAAD XML for intent labels |
| 2 | **frame_extractor** | Samples every Nth frame (default: 5), saves JPEGs to `output/frames/` |
| 3 | **frame_cleaner** | Drops frames failing blur (Laplacian < 100), darkness (mean < 40), or corruption (>95% solid) checks |
| 4 | **detector** | RT-DETR (transformer, NMS-free) by default; `yolo11`/`yolov8` selectable via `config.DETECTOR_BACKEND` |
| 5 | **tracker** | ByteTrack (default) or IoU fallback; JAAD mode uses ground-truth `ped_id` directly |
| 6 | **pose_estimator** | YOLO-pose 17-keypoint skeleton per pedestrian → body-language features (torso lean, head/gaze turn, stance, gait, foot placement) |
| 7 | **behavior_analyzer** | Classifies motion over a 7-frame rolling window: `stopping / walking / running / crossing / driving / driving_slow` |
| 8 | **intent_predictor** | **Predicts** crossing intent per frame from a temporal model over pose + trajectory; writes `intent`, `crossing_prob`, `intent_conf` |
| 9 | **tagger** | Computes `danger_score` (0.0–1.0) per object, proximity bonus (+0.25) for ped+vehicle within 150px, emits `DANGER/SAFE` scene tag |
| 10 | **exporter** | Writes `dataset.json` (per-frame) and `dataset.csv` (per-object per-frame) |

---

## 📦 Output Formats

### `dataset.json`
```json
{
  "frame_id": 42,
  "timestamp": "00:00:07.00",
  "file_path": "output/clean_frames/frame_0042.jpg",
  "scene_tag": "DANGER",
  "safety_reason": "Pedestrian crossing near vehicle",
  "objects": [
    {
      "track_id": "ped_003",
      "class": "pedestrian",
      "bbox": [410, 220, 60, 120],
      "behavior": "crossing",
      "danger_score": 0.95,
      "tags": ["#crossing"]
    }
  ]
}
```

### `dataset.csv`
Flattened version — one row per detected object per frame.

### XML
PASCAL VOC-style annotations, compatible with LabelImg and similar tools.

---

## 🌐 Web UI

Built with **FastAPI** + a custom animated frontend.

- Drag-and-drop multi-video upload queue
- Real-time log streaming via **Server-Sent Events (SSE)**
- Animated "Living Pipeline" — highlights the active stage with progress arcs
- In-browser annotated video playback (HTTP range streaming)
- Per-job result cards with download links for MP4, JSON, CSV, and XML

```bash
# Activate venv first
.venv\Scripts\Activate.ps1

python adas_pipeline/app/server.py
# → Open http://localhost:8000
```

---

## 🚀 Quick Start

### Web UI
```bash
.venv\Scripts\Activate.ps1
python adas_pipeline/app/server.py
```

### CLI — Single Video
```bash
python adas_pipeline/main.py --mode video --input path/to/video.mp4
```

### CLI — JAAD Dataset
```bash
python adas_pipeline/main.py --mode jaad --resume
```

### Local Preview
```bash
python adas_pipeline/preview.py
```

---

## ⚙️ Configuration

Key parameters in `config.py`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `FRAME_SAMPLE_INTERVAL` | `5` | Extract every Nth frame |
| `BLUR_THRESHOLD` | `100.0` | Laplacian variance cutoff |
| `DETECTION_CONFIDENCE` | `0.5` | YOLO minimum confidence |
| `USE_BYTETRACK` | `True` | ByteTrack vs IoU tracker |
| `DANGER_PROXIMITY_PX` | `150` | Ped–vehicle danger distance (px) |
| `CROSSING_HORIZONTAL_RATIO` | `0.55` | Horizontal motion fraction to classify as crossing |
| `DETECTOR_BACKEND` | `"rtdetr"` | Detector: `rtdetr` / `yolo11` / `yolov8` |
| `POSE_ENABLED` | `True` | Extract skeletons + body-language features |
| `INTENT_OBS_LEN` | `16` | Observation window (timeline steps) |
| `INTENT_TTE` | `15` | Predict a crossing within this many future steps |
| `INTENT_USE_POSE` | `True` | Fuse body-language features into the intent model |

---

## 🧠 Crossing-Intent Prediction

The heart of the upgrade: instead of flagging *current* motion, the model
**predicts** whether a pedestrian is about to step into the road.

**Task formulation (leak-free).** Following the JAAD/PIE convention, at timeline
step *t* the model observes only `[t−OBS_LEN+1 … t]` and is labelled by whether a
crossing occurs in the *future* window `[t+1 … t+TTE]`. The observation window
never overlaps the label horizon, so the model genuinely predicts rather than
describing the present. (The previous version labelled a whole track from its
full history — including future frames — and stamped one label on every frame;
that non-causal behaviour has been removed.)

**Features (per frame, `modules/intent_features.py`).**

| Group | Dims | Signals |
|-------|------|---------|
| Kinematics | 8 | normalised bbox centre/size, velocity, acceleration |
| Body language (pose) | 10 | torso lean, head turn, head pitch, body frontal-ness, stance width, step extension, leg lift, knee bend, arm extension, lower-body visibility |
| Aux (JAAD GT, optional) | 2 | `look`, `action` — off by default (not available at deploy time) |

Pose features are scale-normalised by bbox height and mirrored to ego-lateral
coordinates so left- and right-approaching pedestrians share one representation.

**Model.** 2-layer LSTM over the observation window → sigmoid. Trained in PyTorch,
exported to NumPy `.npz` (weights + normalisation stats + active-feature columns +
calibrated threshold) so inference needs no PyTorch. Model selection is by
validation ROC-AUC and the operating threshold is calibrated for max-F1 on the
validation split.

### Reproduce

```bash
# 1. Build features from JAAD (pose extraction is cached per video)
python -m datasets.build_jaad_features --videos 150            # pose + kinematics
python -m datasets.build_jaad_features --videos 150 --no-pose  # kinematics only (fast)

# 2. Train — pose fusion vs bbox-only baseline (same tracks → fair comparison)
python train_intent.py --videos 150 --pose      # body-language + trajectory
python train_intent.py --videos 150 --no-pose   # trajectory-only baseline

# 3. Evaluate the saved model (metrics + early-prediction curve + plots)
python evaluate.py --videos 150
```

> **Data scale matters.** JAAD track-level splits are small; on only ~40 videos
> the held-out set is ~8 tracks and metrics are noisy. Use ≥150 videos for stable
> numbers. Pose extraction runs on CPU here (~0.5s/frame) and is cached, so builds
> are incremental.

**Known limitations.**
- *Label = crossing onset.* A window is positive only if the pedestrian is *not
  yet* crossing and a crossing *begins* within the horizon — the ADAS-relevant
  event, not "currently mid-crossing." This deliberately lowers the positive rate
  and the headline accuracy versus a naive "any future crossing frame" label.
- *Train/deploy frame-rate.* Training timelines are built at ~10fps
  (`JAAD_TIMELINE_STRIDE=3`) while live video is sampled at
  `FRAME_SAMPLE_INTERVAL=5`. Pose features are instantaneous and transfer
  cleanly; the velocity/acceleration channels are per-step deltas and so differ
  in scale between train and deploy. Align these strides (or add Δt-normalised
  velocities) before trusting kinematic channels on live video.

## 🤖 Models

| File | Purpose |
|------|---------|
| `rtdetr-l.pt` | RT-DETR detector weights (auto-downloaded if missing) |
| `yolo11n.pt` / `yolov8n.pt` | Alternative detector backbones |
| `yolo11n-pose.pt` | YOLO-pose weights for 17-keypoint skeletons |
| `checkpoints/intent_model.npz` | Trained crossing-intent model (weights + norm stats + threshold) |

> Detector/pose weights live in `Pedistrian_intent_detection/`. All `*.pt`/`*.npz`
> files are gitignored. Legacy `.pth` classifiers in `models/` are superseded by
> the pose-fused intent model above.

---

## 📁 Input Modes

| Mode | Description |
|------|-------------|
| `--mode video` | Process a single uploaded video file |
| `--mode jaad` | Process JAAD dataset (XML annotations + optional video clips) |
| `--mode pie` | Placeholder — not yet implemented |

> The JAAD dataset lives at `Pedistrian_intent_detection/JAAD/` and is gitignored due to size. It contains dashcam clips with frame-level pedestrian crossing intent labels.

---

## 🔁 Checkpointing & Resume

Every stage writes a checkpoint to `adas_pipeline/checkpoints/`. Large intermediate data (frame records, track history) is stored in `.pkl` sidecar files to keep JSON checkpoints lightweight.

If the pipeline crashes mid-run:
```bash
python adas_pipeline/main.py --mode jaad --resume   # skips completed stages
python adas_pipeline/main.py --mode jaad             # fresh restart
```

---

## 🎬 Video Encoding

Three encoders attempted in waterfall order:

1. **PyAV** → true H.264 MP4, best browser compatibility
2. **imageio[ffmpeg]** → H.264 via bundled ffmpeg binary
3. **OpenCV mp4v** → MPEG-4 (plays after download; may not stream inline)

---

## 📂 Project Structure

```
adas_pipeline/
├── app/
│   ├── server.py             # FastAPI server
│   └── static/index.html     # Web UI
├── modules/
│   ├── input_handler.py
│   ├── frame_extractor.py
│   ├── frame_cleaner.py
│   ├── detector.py           # RT-DETR / YOLO backbones
│   ├── tracker.py
│   ├── pose_estimator.py     # 17-keypoint skeletons (stage + wrapper)
│   ├── body_language.py      # keypoints → posture features
│   ├── intent_features.py    # canonical feature layout + windowing
│   ├── intent_predictor.py   # crossing-intent inference stage
│   ├── behavior_analyzer.py
│   └── tagger.py
├── datasets/
│   ├── jaad_loader.py        # JAAD XML → structured tracks
│   └── build_jaad_features.py# offline feature/pose extraction (cached)
├── train_intent.py           # supervised training (pose vs baseline)
├── evaluate.py               # metrics + early-prediction curve + plots
├── exporter.py · visualizer.py · xml_exporter.py
├── main.py · preview.py · config.py
```

