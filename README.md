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
   4. detector           → YOLOv8n bounding boxes per frame
   5. tracker            → assign consistent track IDs (ByteTrack / IoU)
   6. behavior_analyzer  → classify motion (walking / crossing / running / driving / stopping)
   7. tagger             → compute danger_score + DANGER/SAFE scene tag
   8. exporter           → write dataset.json + dataset.csv
        │
        ▼
   visualizer.py         → render annotated MP4 (H.264)
   xml_exporter.py       → PASCAL VOC XML annotations
```

---

## 🔬 Pipeline Stages

| # | Stage | Description |
|---|-------|-------------|
| 1 | **input_handler** | Validates codec, dimensions, frame count; parses JAAD XML for intent labels |
| 2 | **frame_extractor** | Samples every Nth frame (default: 5), saves JPEGs to `output/frames/` |
| 3 | **frame_cleaner** | Drops frames failing blur (Laplacian < 100), darkness (mean < 40), or corruption (>95% solid) checks |
| 4 | **detector** | YOLOv8n inference — detects pedestrians (class 0) and vehicles at ≥0.5 confidence |
| 5 | **tracker** | ByteTrack (default) or IoU fallback; JAAD mode uses ground-truth `ped_id` directly |
| 6 | **behavior_analyzer** | Classifies motion over a 7-frame rolling window: `stopping / walking / running / crossing / driving / driving_slow` |
| 7 | **tagger** | Computes `danger_score` (0.0–1.0) per object, proximity bonus (+0.25) for ped+vehicle within 150px, emits `DANGER/SAFE` scene tag |
| 8 | **exporter** | Writes `dataset.json` (per-frame) and `dataset.csv` (per-object per-frame) |

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

---

## 🤖 Models

| File | Purpose |
|------|---------|
| `models/yolov8n.pt` | YOLOv8 nano weights (auto-downloaded if missing) |
| `models/checkpoint_latest.pth` | Latest pedestrian intent classifier checkpoint |
| `models/intent_classifier_best.pth` | Best-performing intent classifier |

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
│   ├── server.py          # FastAPI server
│   └── static/index.html  # Web UI
├── pipeline/
│   ├── input_handler.py
│   ├── frame_extractor.py
│   ├── frame_cleaner.py
│   ├── detector.py
│   ├── tracker.py
│   ├── behavior_analyzer.py
│   ├── tagger.py
│   └── exporter.py
├── visualizer.py
├── xml_exporter.py
├── main.py
├── preview.py
└── config.py
```

