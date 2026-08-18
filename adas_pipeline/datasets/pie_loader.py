"""
pie_loader.py — Parse the PIE dataset into the same PedTrack objects as JAAD.

PIE (Rasouli et al., ICCV 2019) is JAAD's larger successor: real dashcam video
with per-frame pedestrian behaviour labels *plus* an explicit intention label and
per-frame ego-vehicle OBD. This loader deliberately emits the **same**
`PedTrack`/`PedFrame` structure as `jaad_loader`, so the existing feature builder,
trainer, and predictor consume PIE with no change to the 20-dim feature layout.
(OBD ego-motion fusion is a separate, later step — see project notes.)

Design decisions (verified against the data, not assumed):

* Label mapping. PIE per-frame `behavior.cross` is {not-crossing:0, crossing:1,
  crossing-irrelevant:-1}. We map -1 -> None (unknown/excluded), exactly like
  JAAD's missing cross label. `crossing-irrelevant` means the pedestrian is
  crossing but *not in the ego-vehicle's path* — not a clean "will they step in
  front of us?" signal — so it is neither a prediction point nor a supervised
  label in `make_windows`. Measured impact on the 3-set subset: 3.4% of frames,
  2 of 231 tracks dropped. `look`/`action`/`occlusion` already share JAAD's
  scalar encoding, so they pass through unchanged.

* video_id carries the set. PIE clips/images are nested per set
  (`PIE_clips/set01/video_0001.mp4`, `images/set01/video_0001/NNNNN.png`), unlike
  JAAD's flat layout. We encode video_id as "set01/video_0001" so the frame
  reader can resolve the right folder; the feature builder sanitises the slash
  when forming cache filenames.

The heavy XML parsing (behaviour, attributes, OBD) is reused from PIE's own
`pie_data.PIE.generate_database()` (cached to a .pkl), rather than re-implemented.
"""

import logging
import os
import sys
from typing import Dict, List, Optional

# Reuse the exact dataclasses JAAD uses so downstream code is dataset-agnostic.
from datasets.jaad_loader import PedFrame, PedTrack

logger = logging.getLogger("pie_loader")

# Default locations (override via load_pie_tracks args).
PIE_DATA_ROOT = r"E:\datacleaning\PIE_data"
PIE_INTERFACE_DIR = r"E:\datacleaning\PIE\utilities"  # holds pie_data.py

# PIE per-frame cross scalar -> our PedFrame.cross. -1 (irrelevant) -> None.
_CROSS = {1: 1, 0: 0, -1: None}


def _get_pie_database(data_root: str, interface_dir: str) -> Dict:
    """Load PIE's annotation database dict (uses PIE's own cached parser)."""
    if interface_dir not in sys.path:
        sys.path.insert(0, interface_dir)
    from pie_data import PIE  # noqa: E402  (path injected above)

    imdb = PIE(data_path=data_root)
    return imdb.generate_database()


def _track_from_ped(setid: str, vid: str, pid: str, pd: Dict) -> Optional[PedTrack]:
    """Convert one PIE pedestrian annotation dict into a PedTrack (or None)."""
    frames_idx = pd["frames"]
    bboxes = pd["bbox"]
    occl = pd["occlusion"]
    beh = pd["behavior"]
    cross_b, look_b, action_b = beh["cross"], beh["look"], beh["action"]

    video_id = f"{setid}/{vid}"
    ped_frames: List[PedFrame] = []
    for i, fr in enumerate(frames_idx):
        xtl, ytl, xbr, ybr = bboxes[i]
        ped_frames.append(PedFrame(
            frame=int(fr),
            bbox=[xtl, ytl, xbr - xtl, ybr - ytl],   # -> [x, y, w, h] like JAAD
            cross=_CROSS.get(cross_b[i]),
            look=int(look_b[i]),
            action=int(action_b[i]),
            occlusion=int(occl[i]),
        ))
    if not ped_frames:
        return None
    ped_frames.sort(key=lambda f: f.frame)
    return PedTrack(ped_id=f"{video_id}:{pid}", video_id=video_id, frames=ped_frames)


def load_pie_tracks(
    data_root: str = PIE_DATA_ROOT,
    interface_dir: str = PIE_INTERFACE_DIR,
    set_ids: Optional[List[str]] = None,
    require_crossing_label: bool = True,
) -> List[PedTrack]:
    """
    Load PIE pedestrian tracks as PedTrack objects.

    Args:
        data_root:     PIE data root (contains annotations/, PIE_clips/, images/).
        interface_dir: folder containing PIE's pie_data.py.
        set_ids:       subset of sets (e.g. ["set01"]); None = all present.
        require_crossing_label: keep only tracks with >=1 known cross frame
                                (0 or 1) — mirrors jaad_loader.load_tracks.
    """
    db = _get_pie_database(data_root, interface_dir)
    sets = set_ids if set_ids is not None else sorted(db.keys())

    tracks: List[PedTrack] = []
    for setid in sets:
        if setid not in db:
            logger.warning("set %s not in database — skipping", setid)
            continue
        for vid in sorted(db[setid].keys()):
            for pid, pd in db[setid][vid]["ped_annotations"].items():
                t = _track_from_ped(setid, vid, pid, pd)
                if t is None:
                    continue
                if require_crossing_label and not any(f.cross is not None for f in t.frames):
                    continue
                tracks.append(t)

    n_cross = sum(1 for t in tracks if t.ever_crosses())
    logger.info(
        "Loaded %d PIE pedestrian tracks from sets %s (%d contain crossing frames)",
        len(tracks), sets, n_cross,
    )
    return tracks


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    ts = load_pie_tracks()
    n = len(ts)
    n_pos = sum(1 for t in ts if t.ever_crosses())
    lens = sorted(len(t.frames) for t in ts)
    print(f"tracks={n}  with-crossing={n_pos}  ({100*n_pos/max(n,1):.1f}% positive-track rate)")
    print(f"track length frames: min={lens[0]} median={lens[len(lens)//2]} max={lens[-1]}")
    # sanity: show one track
    t0 = ts[0]
    print(f"example: {t0.ped_id}  video={t0.video_id}  frames={len(t0.frames)}  "
          f"first_bbox={t0.frames[0].bbox}  cross_vals={sorted(set(f.cross for f in t0.frames), key=lambda x:(x is None, x))}")
