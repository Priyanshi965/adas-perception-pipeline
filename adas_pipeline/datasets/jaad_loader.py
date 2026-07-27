"""
jaad_loader.py — Parse JAAD XML annotations into structured pedestrian tracks.

JAAD stores, per video, a set of <track label="pedestrian"> elements. Each track
is a time-ordered list of <box> elements carrying corner coords plus per-frame
behavioural <attribute>s: cross, look, action, hand_gesture, reaction, nod,
occlusion. Only tracks labelled "pedestrian" (not "ped"/"people") carry the
behavioural attributes we need — the others are unlabeled bystanders/crowds.

This loader is the single source of truth for boxes + labels; both the bbox-only
baseline and the pose-fused model build on the PedTrack objects it returns.
"""

import glob
import logging
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger("jaad_loader")

# Categorical → numeric maps (kept explicit so features are reproducible)
_CROSS = {"crossing": 1, "not-crossing": 0}
_LOOK = {"looking": 1, "not-looking": 0}
_ACTION = {"walking": 1, "standing": 0}
_OCCL = {"none": 0, "part": 1, "full": 2}


@dataclass
class PedFrame:
    frame: int
    bbox: List[float]              # [x, y, w, h]
    cross: Optional[int]           # 1 crossing / 0 not / None unknown
    look: int                      # 1 looking / 0 not
    action: int                    # 1 walking / 0 standing
    occlusion: int                 # 0 none / 1 part / 2 full


@dataclass
class PedTrack:
    ped_id: str
    video_id: str
    frames: List[PedFrame] = field(default_factory=list)

    def crossing_frames(self) -> int:
        return sum(1 for f in self.frames if f.cross == 1)

    def ever_crosses(self) -> bool:
        return any(f.cross == 1 for f in self.frames)


def _attrs(box) -> Dict[str, str]:
    return {a.get("name"): (a.text or "") for a in box.findall("attribute")}


def parse_video(xml_path: str) -> List[PedTrack]:
    """Parse one JAAD annotation XML into a list of pedestrian tracks."""
    video_id = os.path.splitext(os.path.basename(xml_path))[0]
    root = ET.parse(xml_path).getroot()
    tracks: List[PedTrack] = []

    for tr in root.findall(".//track"):
        if tr.get("label") != "pedestrian":
            continue  # only 'pedestrian' tracks carry behavioural attributes
        ped_id = None
        frames: List[PedFrame] = []
        for box in tr.findall("box"):
            if box.get("outside") == "1":
                continue  # box has left the frame — no valid pixels
            a = _attrs(box)
            if ped_id is None:
                ped_id = a.get("id") or f"{video_id}_{tr.get('label')}"
            try:
                xtl, ytl = float(box.get("xtl")), float(box.get("ytl"))
                xbr, ybr = float(box.get("xbr")), float(box.get("ybr"))
            except (TypeError, ValueError):
                continue
            frames.append(PedFrame(
                frame=int(box.get("frame")),
                bbox=[xtl, ytl, xbr - xtl, ybr - ytl],
                cross=_CROSS.get(a.get("cross")),
                look=_LOOK.get(a.get("look"), 0),
                action=_ACTION.get(a.get("action"), 0),
                occlusion=_OCCL.get(a.get("occlusion"), 0),
            ))
        if ped_id and frames:
            frames.sort(key=lambda f: f.frame)
            tracks.append(PedTrack(ped_id=f"{video_id}:{ped_id}", video_id=video_id, frames=frames))
    return tracks


def available_video_ids(annotations_dir: str) -> List[str]:
    """All video ids that have an annotation XML, sorted."""
    xmls = glob.glob(os.path.join(annotations_dir, "*.xml"))
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in xmls)


def load_tracks(
    annotations_dir: str,
    video_ids: Optional[List[str]] = None,
    require_crossing_label: bool = True,
) -> List[PedTrack]:
    """
    Load pedestrian tracks across a set of videos.

    Args:
        annotations_dir: JAAD annotations/ directory.
        video_ids:       subset of video ids (None = all available).
        require_crossing_label: keep only tracks that have at least one frame
                                with a known cross label.
    """
    if video_ids is None:
        video_ids = available_video_ids(annotations_dir)

    all_tracks: List[PedTrack] = []
    for vid in video_ids:
        xml = os.path.join(annotations_dir, f"{vid}.xml")
        if not os.path.exists(xml):
            continue
        for t in parse_video(xml):
            if require_crossing_label and not any(f.cross is not None for f in t.frames):
                continue
            all_tracks.append(t)

    n_cross = sum(1 for t in all_tracks if t.ever_crosses())
    logger.info(
        f"Loaded {len(all_tracks)} pedestrian tracks from {len(video_ids)} videos "
        f"({n_cross} contain crossing frames)"
    )
    return all_tracks
