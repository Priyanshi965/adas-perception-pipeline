"""
export_scene_viz.py — Export nuScenes scenes as BEV JSON for the web viewer.

For each chosen scene, produces a bird's-eye-view timeline: per 10 Hz frame, every
pedestrian's ego-frame position, LiDAR point density, the derived corridor label,
and the trained model's per-frame crossing probability. The viewer
(app/static/nuscenes.html) animates this so you can watch the model flag
pedestrians as they approach the ego path.

Uses metadata-only features (no camera/radar blobs), so it runs on any scene the
metadata pack describes and matches the KIN+LiDAR model.

Usage (from adas_pipeline/, with NUSC_* env vars set):
  python -m datasets.export_scene_viz --num 6
  python -m datasets.export_scene_viz --scenes scene-0061 scene-0103
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import config
from datasets import nuscenes_loader as nl
from datasets import build_nuscenes_features as bnf
from modules import intent_features as ifeat
from modules.intent_predictor import IntentPredictor

OUT_DIR = os.path.join(ROOT, "app", "static", "nuscenes_scenes")


def _vehicle_class(catname):
    if any(k in catname for k in ("truck", "bus", "trailer", "construction")):
        return "truck"
    if any(k in catname for k in ("motorcycle", "bicycle")):
        return "bike"
    return "car"


def _add_vehicles(nusc, scene_tok, frames, t0, hz, n_master):
    """Interpolate surrounding vehicle boxes (position + heading) onto the master
    timeline and attach them per frame as `objs` for the 3D-box scene view."""
    vts = nl.load_nuscenes_tracks(nusc, [scene_tok], with_pose=False,
                                  min_frames=2, category_prefix="vehicle.")
    for t in vts:
        ts = np.array([f.timestamp for f in t.frames])
        if ts[-1] <= ts[0]:
            continue
        xs = np.array([f.x for f in t.frames]); ys = np.array([f.y for f in t.frames])
        yaw = np.array([f.yaw for f in t.frames])
        w = float(np.median([f.size[0] for f in t.frames]))
        ln = float(np.median([f.size[1] for f in t.frames]))
        cls = _vehicle_class(nusc.get("sample_annotation", t.frames[0].ann_token)["category_name"])
        n = int(round((ts[-1] - ts[0]) * hz)) + 1
        grid = ts[0] + np.arange(n) / hz
        gx = np.interp(grid, ts, xs); gy = np.interp(grid, ts, ys)
        gs = np.interp(grid, ts, np.sin(yaw)); gc = np.interp(grid, ts, np.cos(yaw))
        off = int(round((ts[0] - t0) * hz))
        for i in range(n):
            mi = off + i
            if mi < 0 or mi >= n_master or gx[i] < -6 or gx[i] > 62 or abs(gy[i]) > 20:
                continue
            frames[mi].setdefault("objs", []).append(dict(
                id=t.instance_token[:8],
                x=round(float(gx[i]), 2), y=round(float(gy[i]), 2),
                yaw=round(float(np.arctan2(gs[i], gc[i])), 2),
                w=round(w, 1), l=round(ln, 1), c=cls))


_MAP_CACHE = {}
def _get_map(nusc, scene_tok):
    """NuScenesMap for the scene's location (cached). None if map pack absent."""
    from nuscenes.map_expansion.map_api import NuScenesMap
    loc = nusc.get("log", nusc.get("scene", scene_tok)["log_token"])["location"]
    if loc not in _MAP_CACHE:
        try:
            _MAP_CACHE[loc] = NuScenesMap(dataroot=config.NUSC_DATAROOT, map_name=loc)
        except Exception as e:
            logging.getLogger("export").warning("map '%s' unavailable (%s)", loc, e)
            _MAP_CACHE[loc] = None
    return _MAP_CACHE[loc]


def _ego_trajectory(nusc, scene_tok, t0, hz, n_master):
    """Global ego (x,y,yaw) at each keyframe → interpolated to the master timeline."""
    from pyquaternion import Quaternion
    st = nusc.get("scene", scene_tok)["first_sample_token"]; poses = []
    while st:
        s = nusc.get("sample", st)
        sd = nusc.get("sample_data", s["data"]["LIDAR_TOP"])
        ep = nusc.get("ego_pose", sd["ego_pose_token"])
        poses.append((s["timestamp"] / 1e6, ep["translation"][0], ep["translation"][1],
                      Quaternion(ep["rotation"]).yaw_pitch_roll[0]))
        st = s["next"]
    ts = np.array([p[0] for p in poses]); xs = np.array([p[1] for p in poses])
    ys = np.array([p[2] for p in poses]); yw = np.array([p[3] for p in poses])
    grid = t0 + np.arange(n_master) / hz
    gx = np.interp(grid, ts, xs); gy = np.interp(grid, ts, ys)
    gyaw = np.arctan2(np.interp(grid, ts, np.sin(yw)), np.interp(grid, ts, np.cos(yw)))
    return xs, ys, gx, gy, gyaw


def _extract_map(nmap, xs, ys, ox, oy, margin=55):
    """Road geometry around the trajectory, in scene-origin-relative coords."""
    box = (float(xs.min()-margin), float(ys.min()-margin), float(xs.max()+margin), float(ys.max()+margin))
    def rel(coords): return [[round(x-ox,1), round(y-oy,1)] for x,y in coords]
    def polygons(layer, key):
        out=[]
        try: toks = nmap.get_records_in_patch(box, [layer], mode="intersect")[layer]
        except Exception: return out
        for t in toks:
            rec = nmap.get(layer, t)
            for pt in (rec[key] if isinstance(rec.get(key), list) else [rec[key]]):
                try:
                    p = nmap.extract_polygon(pt).simplify(0.4)
                    if p.exterior: out.append(rel(p.exterior.coords))
                except Exception: pass
        return out
    def lines(layer):
        out=[]
        try: toks = nmap.get_records_in_patch(box, [layer], mode="intersect")[layer]
        except Exception: return out
        for t in toks:
            try:
                l = nmap.extract_line(nmap.get(layer, t)["line_token"]).simplify(0.4)
                if len(l.coords)>=2: out.append(rel(l.coords))
            except Exception: pass
        return out
    return dict(drivable=polygons("drivable_area","polygon_tokens"),
                crossings=polygons("ped_crossing","polygon_token"),
                dividers=lines("lane_divider")+lines("road_divider"))


def export_scene(nusc, scene_tok, predictor):
    scene = nusc.get("scene", scene_tok)
    tracks = nl.load_nuscenes_tracks(nusc, [scene_tok], with_pose=False)
    if not tracks:
        return None

    # Master scene timeline (10 Hz) spanning all tracks.
    t0 = min(t.frames[0].timestamp for t in tracks)
    t1 = max(t.frames[-1].timestamp for t in tracks)
    hz = config.NUSC_TIMELINE_HZ
    n_master = int(round((t1 - t0) * hz)) + 1
    frames = [dict(t=round(i / hz, 2), peds=[]) for i in range(n_master)]

    n_pos = 0
    thr = predictor.threshold
    tp = fp = fn = tn = 0        # confusion at the model's operating threshold
    scores, labels = [], []      # raw window scores for batch-level ROC-AUC
    for t in tracks:
        interp = bnf._interpolate_track(t, hz)
        if interp is None:
            continue
        pos, lidpt, valid = interp["pos"], interp["lidpt"], interp["valid"]
        d = bnf.build_track(nusc, t, None, with_pose=False, with_radar=False)
        if d is None:
            continue
        probs = predictor.predict_timeline(d["feats"])   # per-frame crossing prob
        cross = d["cross"]
        if (cross == 1).any():
            n_pos += 1
        # Confusion on the SAME observe→predict windows the model is scored on.
        _, ys, ends = ifeat.make_windows(d["feats"], cross, config.INTENT_OBS_LEN,
                                         config.INTENT_TTE, config.INTENT_SAMPLE_STRIDE)
        for yv, e in zip(ys, ends):
            scores.append(float(probs[e])); labels.append(int(yv))
            pred = 1 if probs[e] >= thr else 0
            if yv == 1 and pred == 1: tp += 1
            elif yv == 1 and pred == 0: fn += 1
            elif yv == 0 and pred == 1: fp += 1
            else: tn += 1
        offset = int(round((t.frames[0].timestamp - t0) * hz))
        pid = t.ped_id[:8]
        for i in range(pos.shape[0]):
            mi = offset + i
            if mi < 0 or mi >= n_master or not valid[i]:
                continue
            frames[mi]["peds"].append(dict(
                id=pid,
                x=round(float(pos[i, 0]), 2),      # forward (m)
                y=round(float(pos[i, 1]), 2),      # left (m)
                p=round(float(probs[i]), 3),       # model crossing probability
                lid=int(lidpt[i]),                 # LiDAR points on the box
                lbl=int(cross[i]),                 # 1 in-corridor, 0 out, -1 excl
            ))

    _add_vehicles(nusc, scene_tok, frames, t0, hz, n_master)

    # real road geometry + ego trajectory (so roads/turns are visible)
    ego = mp = None
    nmap = _get_map(nusc, scene_tok)
    if nmap is not None:
        try:
            xs, ys, gx, gy, gyaw = _ego_trajectory(nusc, scene_tok, t0, hz, n_master)
            ox, oy = float(gx[0]), float(gy[0])
            mp = _extract_map(nmap, xs, ys, ox, oy)
            ego = [[round(float(gx[i]-ox),2), round(float(gy[i]-oy),2), round(float(gyaw[i]),3)]
                   for i in range(n_master)]
        except Exception as e:
            logging.getLogger("export").warning("map/ego extract failed: %s", e)

    return dict(
        scene=scene["name"],
        description=scene.get("description", ""),
        hz=hz,
        ego=ego, map=mp,
        n_frames=n_master,
        n_tracks=len(tracks),
        n_positive=n_pos,
        confusion=dict(tp=tp, fp=fp, fn=fn, tn=tn),
        _scores=scores, _labels=labels,   # popped before writing (for batch AUC)
        corridor=dict(half_width=config.NUSC_CORRIDOR_HALF_WIDTH_M,
                      min_x=config.NUSC_CORRIDOR_MIN_X_M,
                      lookahead=config.NUSC_CORRIDOR_LOOKAHEAD_M),
        range_m=config.NUSC_RANGE_NORM_M,
        threshold=round(float(predictor.threshold), 3),
        frames=frames,
    )


def _test_split_scene_names(seed=42):
    """Replicate the trainer's held-out TEST scene set (meta-only cache, fast)."""
    import train_intent as ti
    tracks = bnf.build_dataset(scene_tokens=None, with_pose=False, with_radar=False,
                               require_blobs=False)
    tracks = [t for t in tracks if (t["cross"] >= 0).any()]
    _, _, te_ids = ti.split_tracks_by_scene(tracks, seed)
    # test scenes, ranked by number of crossing-positive tracks
    from collections import Counter
    pos = Counter(); allc = Counter()
    for i in te_ids:
        nm = tracks[i].get("scene", tracks[i]["video_id"]); allc[nm] += 1
        if (tracks[i]["cross"] == 1).any():
            pos[nm] += 1
    return [nm for nm, _ in pos.most_common()]  # test scenes that contain crossings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=8, help="how many scenes to export")
    ap.add_argument("--scenes", nargs="*", help="explicit scene names to export")
    ap.add_argument("--split", choices=["test", "any"], default="test",
                    help="test = held-out test scenes (a genuine test batch); any = "
                         "highest-crossing scenes regardless of split")
    args = ap.parse_args()

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=config.NUSC_VERSION, dataroot=config.NUSC_DATAROOT, verbose=False)
    predictor = IntentPredictor(model_path=config.NUSC_MODEL_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)

    by_name = {s["name"]: s["token"] for s in nusc.scene}

    if args.scenes:
        chosen = [(nm, by_name[nm]) for nm in args.scenes if nm in by_name]
    elif args.split == "test":
        names = _test_split_scene_names()[: args.num]
        chosen = [(nm, by_name[nm]) for nm in names]
        print(f"held-out TEST scenes selected: {names}")
    else:
        ranked = []
        for nm, tok in by_name.items():
            trs = nl.load_nuscenes_tracks(nusc, [tok], with_pose=False)
            npos = 0
            for t in trs:
                it = bnf._interpolate_track(t, config.NUSC_TIMELINE_HZ)
                if it is None:
                    continue
                c = bnf._corridor_label(it["pos"]); c[~it["valid"]] = -1
                npos += int((c == 1).any())
            ranked.append((npos, nm, tok))
        ranked.sort(reverse=True)
        chosen = [(nm, tok) for _, nm, tok in ranked[: args.num]]

    index = []
    all_scores, all_labels = [], []
    for nm, tok in chosen:
        data = export_scene(nusc, tok, predictor)
        if data is None:
            continue
        all_scores += data.pop("_scores"); all_labels += data.pop("_labels")
        with open(os.path.join(OUT_DIR, f"{nm}.json"), "w") as fh:
            json.dump(data, fh, separators=(",", ":"))
        index.append(dict(scene=nm, n_frames=data["n_frames"],
                          n_tracks=data["n_tracks"], n_positive=data["n_positive"],
                          confusion=data["confusion"], description=data["description"]))
        print(f"exported {nm}: {data['n_frames']} frames, {data['n_tracks']} peds, "
              f"{data['n_positive']} crossing")

    auc = None
    if len(set(all_labels)) > 1:
        from sklearn.metrics import roc_auc_score
        auc = round(float(roc_auc_score(all_labels, all_scores)), 3)
    with open(os.path.join(OUT_DIR, "index.json"), "w") as fh:
        json.dump(dict(scenes=index, split=args.split, auc=auc,
                       model=os.path.basename(config.NUSC_MODEL_PATH)), fh, indent=2)
    print(f"\nwrote {len(index)} scenes + index.json (batch AUC={auc}) → {OUT_DIR}")


if __name__ == "__main__":
    main()
