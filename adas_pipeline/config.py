"""
config.py — All tunable parameters in one place.
No hardcoded values in any other module.
"""

import os

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

# ─── Detection (YOLOv8) ───────────────────────────────────────────────────────

# Path to YOLO weights — uses the one already in the repo
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "..", "Pedistrian_intent_detection", "yolov8n.pt")

# Fallback: download yolov8n if not found locally
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
