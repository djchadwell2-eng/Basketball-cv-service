"""TEST 7 phase B (gated protocol): OC-SORT + StrongSORT (boxmot, ISOLATED
.venv-boxmot) fed the exact raw yolov8m detections from phase A. Compares
fragmentation stats vs the cached ByteTrack baseline. Read-only; the real
tracks cache is never touched; nothing adopted.

Run:  .venv-boxmot/Scripts/python spikes/boxmot_probe.py [CLIP]
"""
from __future__ import annotations
import json, os, sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

CLIP_NAME = sys.argv[1] if len(sys.argv) > 1 else "TEST1"
DETS_JSON = os.path.join(_HERE, "out", f"{CLIP_NAME}_rawdets_person.json")


def stats(per_frame_ids):
    ids = {}
    counts = []
    for frame_ids in per_frame_ids:
        counts.append(len(frame_ids))
        for tid in frame_ids:
            ids[tid] = ids.get(tid, 0) + 1
    n = len(ids)
    return n, sum(counts) / max(len(counts), 1), sum(ids.values()) / max(n, 1)


def run_tracker(name, tracker, doc, video_path, needs_img=True):
    # boxmot's base tracker validates img as ndarray even for motion-only
    # trackers, so frames are always read and passed
    cap = cv2.VideoCapture(video_path)
    per_frame = []
    for i, fr in enumerate(doc["frames"]):
        ok, img = cap.read()
        if not ok:
            img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        d = fr["dets"]
        dets = (np.array([[x1, y1, x2, y2, c, 0.0] for x1, y1, x2, y2, c in d],
                         dtype=np.float32)
                if d else np.zeros((0, 6), dtype=np.float32))
        out = tracker.update(dets, img)
        arr = np.asarray(out) if out is not None else np.zeros((0, 7))
        # boxmot output rows: x1,y1,x2,y2,track_id,conf,cls[,det_idx]
        per_frame.append([int(row[4]) for row in arr] if len(arr) else [])
        if i % 100 == 0:
            print(f"  [{name}] ...{i}/{len(doc['frames'])}", flush=True)
    if cap is not None:
        cap.release()
    return per_frame


def main():
    doc = json.load(open(DETS_JSON, encoding="utf-8"))
    video_path = doc["subclip_path"]
    if not os.path.exists(video_path):
        raise SystemExit(f"subclip missing: {video_path} — re-run phase A first")

    def attempt(name, make_tracker):
        """Isolate each tracker: one failure must not lose the others'
        results (bitten once already). Stats print IMMEDIATELY per tracker."""
        try:
            tracker = make_tracker()
            pf = run_tracker(name, tracker, doc, video_path)
            n, mpf, life = stats(pf)
            print(f"\n  RESULT {name}: distinct_ids={n}  mean_tracks/frame={mpf:.1f}  "
                  f"mean_lifespan={life:.1f}", flush=True)
        except Exception as e:
            print(f"\n  RESULT {name}: NOT MEASURED — {type(e).__name__}: {e}",
                  flush=True)

    def make_ocsort():
        from boxmot.trackers.bbox.ocsort import OcSort
        return OcSort(min_conf=0.1)

    def make_strongsort():
        from boxmot.trackers.bbox.strongsort import StrongSort
        from boxmot.models.reid import ReIDModel     # broken in 19.0.0 wheel
        return StrongSort(reid_model=ReIDModel(), min_conf=0.1)

    print("[boxmot] OC-SORT (motion-only) ...")
    attempt("ocsort", make_ocsort)
    print("[boxmot] StrongSORT (appearance re-ID) ...")
    attempt("strongsort", make_strongsort)
    print("\n  (baseline ByteTrack cached: 122 / 28.0 / 105.8 — same span,")
    print("   same detector family; dets here re-generated at conf>=0.1)")


if __name__ == "__main__":
    main()
