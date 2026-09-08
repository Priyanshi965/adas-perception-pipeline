"""
nuscenes_loader.py — Load nuScenes pedestrian tracks for crossing-intent fusion.

nuScenes ships **no** crossing-intent annotation (unlike JAAD/PIE). This loader
therefore returns the raw *geometry* needed to (a) derive a leak-free crossing
label from each pedestrian's future 3D trajectory and (b) extract camera + LiDAR
+ radar features downstream. Label derivation and featurisation live in
datasets/build_nuscenes_features.py; this module stays pure "read + transform to
ego frame" so its geometry can be unit-tested without the heavy sensor blobs.

Coordinate convention (nuScenes ego frame): **x forward, y left, z up**, origin
at the ego rear axle at each keyframe timestamp. Every pedestrian position is
expressed in the ego frame *of that keyframe* — i.e. "where is the pedestrian
relative to us, right now" — which is exactly the frame in which the forward-path
crossing question is meaningful.

Boxes are annotated at 2 Hz (keyframes). We emit keyframes here; the builder
interpolates to config.NUSC_TIMELINE_HZ so the intent window sizes keep their
~seconds meaning.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("nuscenes_loader")

PED_CATEGORY_PREFIX = "human.pedestrian"
CAMERAS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]


@dataclass
class PoseRef:
    """Where to find this pedestrian in a camera for RTMPose (chosen per keyframe
    as the camera with the largest in-image projected box)."""
    cam_channel: str
    image_path: str
    bbox_xywh: List[float]      # 2D box in that camera's pixels
    area: float


@dataclass
class NuscKeyframe:
    sample_token: str
    timestamp: float            # seconds
    ann_token: str
    # Pedestrian pose in the EGO frame of this keyframe (metres / radians).
    x: float                    # forward
    y: float                    # left
    z: float                    # up (box centre height)
    yaw: float                  # heading relative to ego heading
    size: Tuple[float, float, float]   # (w, l, h)
    num_lidar_pts: int
    num_radar_pts: int
    moving: bool                # nuScenes attribute pedestrian.moving
    radar_sample_token: str     # sample token → builder loads radar clouds
    pose_ref: Optional[PoseRef] = None


@dataclass
class NuscPedTrack:
    instance_token: str
    scene_token: str
    scene_name: str
    frames: List[NuscKeyframe] = field(default_factory=list)

    @property
    def video_id(self) -> str:
        """Cache/grouping key compatible with feature_builder's per-'video' cache."""
        return self.scene_name

    @property
    def ped_id(self) -> str:
        return self.instance_token


# ─── ego-frame transform ──────────────────────────────────────────────────────

def _to_ego_frame(p_global: np.ndarray, box_quat, ego_trans: np.ndarray, ego_quat):
    """Transform a global-frame point + box orientation into the ego frame.

    Returns (x, y, z, yaw_rel) with x forward, y left, z up."""
    p_ego = ego_quat.inverse.rotate(p_global - ego_trans)
    yaw_rel = box_quat.yaw_pitch_roll[0] - ego_quat.yaw_pitch_roll[0]
    # wrap to [-pi, pi]
    yaw_rel = (yaw_rel + np.pi) % (2 * np.pi) - np.pi
    return float(p_ego[0]), float(p_ego[1]), float(p_ego[2]), float(yaw_rel)


# ─── pose-camera selection (one pass over the 6 cameras per keyframe) ──────────

def _pose_refs_for_sample(nusc, sample) -> Dict[str, PoseRef]:
    """Best camera + 2D box for every annotation in a keyframe.

    One get_sample_data call per camera (not per pedestrian) — for each camera we
    project all visible boxes once and keep, per annotation, the camera giving the
    largest in-image box. This is the pose-crop source for the camera modality.
    """
    from nuscenes.utils.geometry_utils import view_points, BoxVisibility

    best: Dict[str, PoseRef] = {}
    for cam in CAMERAS:
        cam_tok = sample["data"].get(cam)
        if cam_tok is None:
            continue
        path, boxes, K = nusc.get_sample_data(cam_tok, box_vis_level=BoxVisibility.ANY)
        for b in boxes:
            corners = view_points(b.corners(), K, normalize=True)[:2]  # (2, 8)
            x1, y1 = float(corners[0].min()), float(corners[1].min())
            x2, y2 = float(corners[0].max()), float(corners[1].max())
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            prev = best.get(b.token)
            if prev is None or area > prev.area:
                best[b.token] = PoseRef(cam, path, [x1, y1, x2 - x1, y2 - y1], area)
    return best


# ─── main loader ──────────────────────────────────────────────────────────────

def load_nuscenes_tracks(
    nusc,
    scene_tokens: Optional[List[str]] = None,
    with_pose: bool = True,
    min_frames: int = 4,
    category_prefix: str = PED_CATEGORY_PREFIX,
) -> List[NuscPedTrack]:
    """Build agent tracks (ego-frame keyframe geometry) for the given scenes.

    Args:
        nusc: an initialised nuscenes.NuScenes instance.
        scene_tokens: restrict to these scenes (default: all in the split).
        with_pose: also resolve a pose camera per keyframe (extra cost).
        min_frames: drop tracks shorter than this many keyframes.
        category_prefix: which category to load (default pedestrians; pass
            "vehicle." to load surrounding vehicles for the scene view).
    """
    from pyquaternion import Quaternion

    if scene_tokens is None:
        scene_tokens = [s["token"] for s in nusc.scene]

    tracks: Dict[str, NuscPedTrack] = {}

    for scene_tok in scene_tokens:
        scene = nusc.get("scene", scene_tok)
        sample_tok = scene["first_sample_token"]
        while sample_tok:
            sample = nusc.get("sample", sample_tok)
            ts = sample["timestamp"] / 1e6  # µs → s

            # ego pose from LIDAR_TOP (the canonical ego reference)
            lidar_sd = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
            ego = nusc.get("ego_pose", lidar_sd["ego_pose_token"])
            ego_trans = np.array(ego["translation"])
            ego_quat = Quaternion(ego["rotation"])

            pose_refs = _pose_refs_for_sample(nusc, sample) if with_pose else {}

            for ann_tok in sample["anns"]:
                ann = nusc.get("sample_annotation", ann_tok)
                if not ann["category_name"].startswith(category_prefix):
                    continue

                x, y, z, yaw = _to_ego_frame(
                    np.array(ann["translation"]), Quaternion(ann["rotation"]),
                    ego_trans, ego_quat,
                )
                moving = any(
                    nusc.get("attribute", a)["name"] == "pedestrian.moving"
                    for a in ann["attribute_tokens"]
                )
                kf = NuscKeyframe(
                    sample_token=sample_tok,
                    timestamp=ts,
                    ann_token=ann_tok,
                    x=x, y=y, z=z, yaw=yaw,
                    size=tuple(ann["size"]),          # (w, l, h)
                    num_lidar_pts=int(ann["num_lidar_pts"]),
                    num_radar_pts=int(ann["num_radar_pts"]),
                    moving=moving,
                    radar_sample_token=sample_tok,
                    pose_ref=pose_refs.get(ann_tok),
                )
                inst = ann["instance_token"]
                if inst not in tracks:
                    tracks[inst] = NuscPedTrack(inst, scene_tok, scene["name"])
                tracks[inst].frames.append(kf)

            sample_tok = sample["next"]

    out = []
    for t in tracks.values():
        t.frames.sort(key=lambda f: f.timestamp)
        if len(t.frames) >= min_frames:
            out.append(t)
    logger.info("Loaded %d nuScenes pedestrian tracks from %d scenes (min_frames=%d)",
                len(out), len(scene_tokens), min_frames)
    return out


# ─── official / reproducible scene splits ─────────────────────────────────────

def available_scene_tokens(nusc, require=("LIDAR_TOP", "CAM_FRONT")) -> List[str]:
    """Scenes whose keyframe blobs are actually present on disk.

    nuScenes-trainval metadata lists all 850 scenes, but a **partial** download
    (a few of the 10 blob files) only has some scenes' sensor files. Processing a
    scene whose blobs are missing would fail on the first image/point load, so
    filter to scenes whose first keyframe has the required sensor files present.
    """
    ok = []
    for s in nusc.scene:
        sample = nusc.get("sample", s["first_sample_token"])
        present = True
        for ch in require:
            tok = sample["data"].get(ch)
            if tok is None:
                present = False; break
            sd = nusc.get("sample_data", tok)
            if not os.path.exists(os.path.join(nusc.dataroot, sd["filename"])):
                present = False; break
        if present:
            ok.append(s["token"])
    logger.info("%d/%d scenes have keyframe blobs on disk (partial-download safe)",
                len(ok), len(nusc.scene))
    return ok


def scene_tokens_for_split(nusc, split: str) -> List[str]:
    """Return scene tokens for an official nuScenes split name.

    Uses nuscenes.utils.splits so train/val never leak into each other. For
    v1.0-mini the valid names are 'mini_train' / 'mini_val'; for v1.0-trainval,
    'train' / 'val'. Unknown names fall back to all scenes in the loaded db.
    """
    try:
        from nuscenes.utils import splits as nusplits
        names = getattr(nusplits, split, None)
        if names is None:
            logger.warning("Unknown split '%s'; using all loaded scenes", split)
            return [s["token"] for s in nusc.scene]
        wanted = set(names)
        return [s["token"] for s in nusc.scene if s["name"] in wanted]
    except Exception as e:  # pragma: no cover
        logger.warning("split lookup failed (%s); using all scenes", e)
        return [s["token"] for s in nusc.scene]
