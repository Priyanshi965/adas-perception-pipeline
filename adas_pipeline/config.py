"""
config.py — All tunable parameters in one place.
No hardcoded values in any other module.
"""

import os
import sys

# Windows consoles default to cp1252 and crash when logs contain non-ASCII.
# config is imported before logging is used everywhere, so fix stdout here once.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ─── Paths ────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_DIR = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames")
CLEAN_FRAMES_DIR = os.path.join(OUTPUT_DIR, "clean_frames")
ANNOTATIONS_DIR = os.path.join(OUTPUT_DIR, "annotations")
FINAL_DIR = os.path.join(OUTPUT_DIR, "final")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")

# JAAD dataset root (relative to this repo)
JAAD_ROOT = os.path.join(BASE_DIR, "..", "Pedistrian_intent_detection", "JAAD")
JAAD_ANNOTATIONS_DIR = os.path.join(JAAD_ROOT, "annotations")
JAAD_SPLIT_IDS_DIR = os.path.join(JAAD_ROOT, "split_ids")

# PIE dataset root — update this once PIE videos are downloaded
PIE_ROOT = os.path.join(BASE_DIR, "..", "Pedistrian_intent_detection", "PIE_dataset")

# ─── Frame Extraction ─────────────────────────────────────────────────────────

# Extract every Nth frame from video (1 = every frame, 5 = every 5th, etc.)
FRAME_SAMPLE_INTERVAL = 5

# Output frame image format
FRAME_FORMAT = "jpg"
FRAME_QUALITY = 95  # JPEG quality (1-100)

# ─── Frame Cleaning ───────────────────────────────────────────────────────────

# Laplacian variance — frames below this are considered blurry
BLUR_THRESHOLD = 100.0

# Mean grayscale brightness — frames below this are considered too dark
BRIGHTNESS_THRESHOLD = 40.0

# Saturation ratio — if >95% of pixels are pure black or pure white, skip frame
CORRUPTION_SATURATION_RATIO = 0.95

# ─── Detection ────────────────────────────────────────────────────────────────

# Detector backbone for the live-video path. Options:
#   "rtdetr"  → RT-DETR: transformer detector, NMS-free, stronger under
#               occlusion/crowding. The "more adaptable algorithm" upgrade.
#   "yolo11"  → YOLO11 (newer YOLO generation, better than v8).
#   "yolov8"  → original YOLOv8n (legacy / fastest).
# NOTE: JAAD mode uses ground-truth boxes and never runs the detector, so this
#       only affects video/deployment inference, not JAAD intent accuracy.
DETECTOR_BACKEND = "rtdetr"

# Weights per backbone. Missing files auto-download from the ultralytics hub.
DETECTOR_WEIGHTS = {
    "rtdetr": os.path.join(BASE_DIR, "..", "Pedistrian_intent_detection", "rtdetr-l.pt"),
    "yolo11": os.path.join(BASE_DIR, "..", "Pedistrian_intent_detection", "yolo11n.pt"),
    "yolov8": os.path.join(BASE_DIR, "..", "Pedistrian_intent_detection", "yolov8n.pt"),
}
DETECTOR_FALLBACK = {"rtdetr": "rtdetr-l.pt", "yolo11": "yolo11n.pt", "yolov8": "yolov8n.pt"}

# Legacy aliases (kept so older code / configs keep working)
YOLO_MODEL_PATH = DETECTOR_WEIGHTS["yolov8"]
YOLO_MODEL_FALLBACK = "yolov8n.pt"

# Minimum detection confidence (0.0–1.0)
DETECTION_CONFIDENCE = 0.5

# COCO class IDs we care about:
#   0 = person (pedestrian)
#   2 = car, 3 = motorcycle, 5 = bus, 7 = truck (vehicles)
PEDESTRIAN_CLASS_IDS = {0}
VEHICLE_CLASS_IDS = {2, 3, 5, 7}
TARGET_CLASS_IDS = PEDESTRIAN_CLASS_IDS | VEHICLE_CLASS_IDS

# ─── Tracking ─────────────────────────────────────────────────────────────────

# Use ByteTrack (built into ultralytics) — set False to use simple IoU tracker
USE_BYTETRACK = True

# Minimum IoU for simple tracker to match bboxes across frames
IOU_MATCH_THRESHOLD = 0.3

# Maximum frames to keep a track alive without a detection
MAX_TRACK_AGE = 10

# ─── Behavior Analysis ────────────────────────────────────────────────────────

# Rolling window size for smoothing displacement vectors (frames)
DISPLACEMENT_WINDOW = 7

# Displacement thresholds (pixels per frame)
STOP_THRESHOLD = 5       # Below this → stopping
WALK_THRESHOLD = 20      # Between STOP and this → walking / slow
# Above WALK_THRESHOLD   → running (pedestrian) or driving (vehicle)

# Minimum frames of near-zero displacement to classify as stopping
STOP_MIN_FRAMES = 3

# Horizontal displacement fraction to trigger "crossing" classification
# If abs(dx) / (abs(dx) + abs(dy) + 1e-6) > this value → crossing
CROSSING_HORIZONTAL_RATIO = 0.55

# ─── Tagging (Safety) ─────────────────────────────────────────────────────────

# Pixel distance between a crossing pedestrian and a vehicle to trigger DANGER
DANGER_PROXIMITY_PX = 150

# Sudden deceleration: vehicle goes from driving to stopping within N frames
SUDDEN_STOP_FRAMES = 3

# Fraction of frame width/height defining "periphery" for blind-spot detection
PERIPHERY_FRACTION = 0.1

# ─── JAAD Integration ─────────────────────────────────────────────────────────

# Which JAAD split to use: "default", "all_videos", or "high_visibility"
JAAD_SPLIT = "default"

# When using JAAD mode, process only this subset: "train", "val", "test", or "all"
JAAD_SUBSET = "all"

# Maximum number of JAAD videos to process (None = process all)
JAAD_MAX_VIDEOS = None

# ─── JAAD video properties (1920x1080 @ ~30fps, 600 frames per clip) ─────────

JAAD_FRAME_WIDTH  = 1920
JAAD_FRAME_HEIGHT = 1080
JAAD_FPS          = 29.97

# ─── Output ───────────────────────────────────────────────────────────────────

OUTPUT_JSON_NAME = "dataset.json"
OUTPUT_CSV_NAME = "dataset.csv"

# ─── Frame Cleaning (enhanced) ────────────────────────────────────────────────

# CLAHE contrast normalisation parameters
CLAHE_CLIP_LIMIT = 2.0          # max amplification per tile (2.0 = moderate boost)
CLAHE_TILE_SIZE  = (8, 8)       # adaptive tile grid for localised normalisation

# Perceptual-hash deduplication — dHash Hamming distance below this → duplicate
# Range 0–256 (16×16 bit hash).  6 ≈ <3% pixel change between consecutive frames.
DUPLICATE_HASH_THRESHOLD = 6

# ─── Pose Estimation (body language) ──────────────────────────────────────────

# Enable the pose stage (17-keypoint COCO skeletons for body-language features)
POSE_ENABLED = True

# Pose backend: "rtmpose" (top-down RTMPose via rtmlib, more accurate, GPU ONNX)
# or "yolo" (ultralytics YOLO-pose). Both emit 17 COCO keypoints, so downstream
# body_language features are identical — this only changes keypoint quality.
POSE_BACKEND = "rtmpose"

# RTMPose ONNX model (auto-downloaded + cached by rtmlib). rtmpose-m body7 @256x192
# is a strong accuracy/speed balance; input size must match the chosen model.
POSE_RTMPOSE_ONNX = ("https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
                     "onnx_sdk/rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip")
POSE_RTMPOSE_INPUT_SIZE = (192, 256)   # (w, h) as rtmlib expects

# Pose model weights (YOLO-pose fallback backend). Auto-downloads if missing.
POSE_MODEL_PATH = os.path.join(BASE_DIR, "..", "Pedistrian_intent_detection", "yolo11n-pose.pt")
POSE_MODEL_FALLBACK = "yolo11n-pose.pt"

# Person-detection confidence for the pose model
POSE_CONF_THRESHOLD = 0.35

# Per-keypoint confidence below which a joint is treated as missing
POSE_KPT_CONF_THR = 0.30

# Padding (fraction of box size) added around a GT box before cropping for pose
POSE_CROP_PAD = 0.15

# IoU to match a detected skeleton to a tracked pedestrian box (frame-level pose)
POSE_MATCH_IOU = 0.30

# ─── Intent Prediction ────────────────────────────────────────────────────────

# Path for trained model weights (.npz format for NumPy inference)
INTENT_MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "intent_model.npz")

# Tracks are downsampled to a fixed timeline before featurising. JAAD is ~30fps;
# stride 3 → a ~10fps timeline. All window sizes below are in *timeline steps*.
JAAD_TIMELINE_STRIDE = 3

# Observe→predict formulation (JAAD/PIE standard, prevents future leakage):
#   at timeline step t, observe [t-OBS_LEN+1 .. t]; label = does a crossing
#   frame occur within the next TTE steps [t+1 .. t+TTE].
INTENT_OBS_LEN = 16          # observation window (~1.6s at 10fps)
INTENT_TTE = 15              # predict a crossing within ~1.5s ahead
INTENT_SEQ_LEN = INTENT_OBS_LEN   # backward-compat alias

# Slide window start by this many timeline steps to make multiple samples/track
INTENT_SAMPLE_STRIDE = 2

# Feature composition: which channels feed the model
INTENT_USE_KINEMATICS = True   # 8 bbox trajectory features
INTENT_USE_POSE = True         # 10 body-language features from keypoints

# LSTM hidden dimension
INTENT_HIDDEN_SIZE = 64

# Probability threshold for "crossing" classification
INTENT_THRESHOLD = 0.5

# ─── nuScenes multimodal fusion (camera + LiDAR + radar) ──────────────────────
# The nuScenes extension reuses the crossing-intent architecture but derives its
# own label (nuScenes ships no crossing annotation): a pedestrian "crosses" when
# it enters the ego-vehicle forward-path corridor within the prediction horizon.
# See datasets/nuscenes_loader.py and datasets/build_nuscenes_features.py.

NUSC_DATAROOT = os.environ.get("NUSC_DATAROOT", os.path.join(BASE_DIR, "..", "nuscenes"))
# v1.0-mini | v1.0-trainval | v1.0-test. Override without editing:  set NUSC_VERSION=v1.0-trainval
NUSC_VERSION = os.environ.get("NUSC_VERSION", "v1.0-mini")
NUSC_MODEL_PATH = os.path.join(BASE_DIR, "checkpoints", "intent_model_nuscenes.npz")
NUSC_CACHE_DIR = os.path.join(BASE_DIR, "checkpoints", "nusc_features")

# 3D boxes are annotated at 2 Hz; we linearly interpolate the ego-frame track to
# this rate so INTENT_OBS_LEN / INTENT_TTE (in steps) keep their ~seconds meaning.
NUSC_TIMELINE_HZ = 10.0

# Forward-path corridor that defines a "crossing" (ego frame: x fwd, y left).
NUSC_CORRIDOR_HALF_WIDTH_M = 1.75   # |y| <= this ⇒ inside our lane/path
NUSC_CORRIDOR_LOOKAHEAD_M = 30.0    # only count entries within x ∈ [0, this]
NUSC_CORRIDOR_MIN_X_M = 0.0         # ignore peds behind the ego

# Radar: aggregate returns within this radius (m, BEV) of the pedestrian centre.
NUSC_RADAR_ASSOC_RADIUS_M = 2.5
# LiDAR point count is read from the annotation (num_lidar_pts) — no blob load
# needed; this cap only normalises the log-count channel.
NUSC_LIDAR_PTS_NORM = 500.0

# Pose sampling: "keyframe" holds the 2 Hz pose between keyframes (keyframes-only
# download works); "dense" samples the nearest camera sweep (needs sweeps blobs).
NUSC_POSE_SAMPLING = "keyframe"
# Normalisers for the geometric channels (metres) so inputs stay ~unit scale.
NUSC_RANGE_NORM_M = 60.0

# nuScenes instances disappear for stretches (occlusion / out of FOV) and then
# reappear with no annotations in between. Interpolating across such a gap would
# fabricate motion, so any timeline step further than this (seconds) from the
# nearest real keyframe is marked cross=-1 (excluded, never a prediction point).
# Keyframes are ~0.5 s apart; 0.75 s tolerates normal jitter but rejects real gaps.
NUSC_MAX_GAP_S = 0.75
