"""
download_nuscenes.py — Fetch nuScenes from the public AWS S3 bucket (fastest way).

The dataset is NOT committed to this repo (license forbids redistribution + it is
tens of GB). This script pulls exactly what each task needs from the Registry of
Open Data bucket `s3://motional-nuscenes`, which is public — no nuScenes account
or credentials required — using the AWS CLI's multi-threaded, auto-resuming
transfer (much faster and more reliable than the browser download).

Prereq (one-time):   pip install awscli

Usage (from adas_pipeline/):
  python -m datasets.download_nuscenes meta            # 0.46 GB — trains FULL trainval (KIN+LiDAR), no blobs
  python -m datasets.download_nuscenes map             # 0.40 GB — HD-map road geometry for the dashboard
  python -m datasets.download_nuscenes mini            # 4.2 GB  — 10-scene sample (all sensors)
  python -m datasets.download_nuscenes keyframes 1     # 4.5 GB  — one trainval partition, KEYFRAMES only (adds camera/pose)
  python -m datasets.download_nuscenes blobs 1         # 31 GB   — one trainval partition, full sweeps (rarely needed)
  python -m datasets.download_nuscenes list            # browse the bucket

Recommended fast path for THIS project (real numbers, ~0.9 GB, minutes):
  python -m datasets.download_nuscenes meta
  python -m datasets.download_nuscenes map
  # then, from adas_pipeline/, with the trainval env pointed at ../nuscenes-trainval:
  #   python train_intent.py --dataset nuscenes --meta-only --lidar
"""
import os, subprocess, sys, tarfile, zipfile

BUCKET = "s3://motional-nuscenes/public/v1.0"
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # E:\datacleaning
TRAINVAL_DIR = os.path.join(ROOT, "nuscenes-trainval")   # meta / map / keyframes / blobs land here
MINI_DIR = os.path.join(ROOT, "nuscenes")                # mini lands here
DL = os.path.join(ROOT, "downloads")

# key  ->  (s3 filename, extract-destination dir)
def targets(kind, n=None):
    if kind == "meta":  return [("v1.0-trainval_meta.tgz", TRAINVAL_DIR)]
    if kind == "map":   return [("nuScenes-map-expansion-v1.3.zip", os.path.join(TRAINVAL_DIR, "maps"))]
    if kind == "mini":  return [("v1.0-mini.tgz", MINI_DIR)]
    if kind == "keyframes": return [(f"v1.0-trainval{int(n):02d}_keyframes.tgz", TRAINVAL_DIR)]
    if kind == "blobs":     return [(f"v1.0-trainval{int(n):02d}_blobs.tgz", TRAINVAL_DIR)]
    raise SystemExit(f"unknown target '{kind}'")


def aws(*args):
    return subprocess.call([sys.executable, "-m", "awscli", "s3", "--no-sign-request", *args])


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    kind = sys.argv[1]
    if kind == "list":
        aws("ls", BUCKET + "/"); return
    n = sys.argv[2] if len(sys.argv) > 2 else None
    os.makedirs(DL, exist_ok=True)
    for fname, dest in targets(kind, n):
        local = os.path.join(DL, fname)
        print(f"\n↓ downloading {fname} (resumable) → {local}")
        if aws("cp", f"{BUCKET}/{fname}", local) != 0:
            raise SystemExit("download failed (is awscli installed?  pip install awscli)")
        os.makedirs(dest, exist_ok=True)
        print(f"⇪ extracting into {dest} ...")
        if fname.endswith(".zip"):
            with zipfile.ZipFile(local) as z: z.extractall(dest)
        else:
            with tarfile.open(local) as t: t.extractall(dest)
        print(f"✓ {fname} ready")
    print("\nDone. See docs/nuscenes_fusion.md for the training/dashboard commands.")


if __name__ == "__main__":
    main()
