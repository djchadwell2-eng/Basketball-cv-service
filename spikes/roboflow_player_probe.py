"""READ-ONLY probe (TEST 4, gated protocol): fine-tuned Player/Ref classes as
tracking input -- fragmentation + referee count vs the cached COCO baseline.

Phase A (hosted API): fetch Player + Ref detections for the clip's tracking
span. Phase B (local): run ultralytics' BYTETracker STANDALONE (same default
bytetrack.yaml as the committed pipeline) over the Player detections only,
then compare fragmentation stats against the cached yolov8m/COCO baseline
(same span, same tracker family). Also counts Ref detections and how many
BASELINE tracked boxes IoU-match a Ref (refs the COCO pipeline can't
distinguish today). Caveats logged, not hidden: detector confidence scales
differ between models, and the baseline tracked via model.track() internals
vs this standalone harness -- association params are identical but plumbing
is not byte-identical.

The real tracks cache is NEVER touched. Key from env (RF_KEY).

Run:  RF_KEY=... .venv/Scripts/python spikes/roboflow_player_probe.py [CLIP]
"""
from __future__ import annotations
import json, os, sys, time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
import clip_config

KEY = os.environ.get("RF_KEY")
WORKSPACE, PROJECT, VERSION = "roboflow-universe-projects", "basketball-players-fy4c2", 25

CLIP_NAME = sys.argv[1] if len(sys.argv) > 1 else "TEST1"
CLIP = getattr(clip_config, f"{CLIP_NAME}_CLIP")
DETS_JSON = os.path.join(_HERE, "out", f"{CLIP_NAME}_playerdets_roboflow.json")
_TMP = os.path.join(_HERE, "out", "_rf_player_frame.jpg")


def fetch_detections():
    from roboflow import Roboflow
    model = Roboflow(api_key=KEY).workspace(WORKSPACE).project(PROJECT).version(VERSION).model
    start, length = CLIP.tracking_span_start, CLIP.tracking_span_len
    cap = cv2.VideoCapture(CLIP.video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames_out = []
    print(f"[player_probe] fetching {CLIP_NAME} span {start}..{start + length} ...")
    for i in range(length):
        ok, img = cap.read()
        if not ok:
            break
        cv2.imwrite(_TMP, img)
        for attempt in range(3):
            try:
                r = model.predict(_TMP, confidence=1).json()
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
        players, refs = [], []
        for p in r["predictions"]:
            rec = {"cx": p["x"], "cy": p["y"], "w": p["width"], "h": p["height"],
                   "conf": p["confidence"]}
            if p["class"] == "Player":
                players.append(rec)
            elif p["class"] == "Ref":
                refs.append(rec)
        frames_out.append({"frame_index": start + i, "players": players, "refs": refs})
        if i % 30 == 0:
            print(f"  ...{i}/{length}  ({len(players)} players, {len(refs)} refs)", flush=True)
    cap.release()
    json.dump({"clip": CLIP_NAME, "span_start": start, "span_len": len(frames_out),
               "model": f"{PROJECT}_v{VERSION}", "frames": frames_out},
              open(DETS_JSON, "w", encoding="utf-8"), indent=2)
    print(f"[player_probe] wrote {DETS_JSON}")


class _Dets:
    """Minimal Results-like wrapper for BYTETracker.update: exposes conf,
    cls, xywh; supports len() and boolean-mask indexing."""
    def __init__(self, xywh, conf, cls):
        self.xywh, self.conf, self.cls = xywh, conf, cls
    def __len__(self):
        return len(self.conf)
    def __getitem__(self, m):
        return _Dets(self.xywh[m], self.conf[m], self.cls[m])


def track_and_compare():
    from ultralytics.utils import IterableSimpleNamespace, YAML
    from ultralytics.trackers.byte_tracker import BYTETracker
    import ultralytics
    yaml_path = os.path.join(os.path.dirname(ultralytics.__file__),
                             "cfg", "trackers", "bytetrack.yaml")
    cfg = IterableSimpleNamespace(**YAML.load(yaml_path))
    tracker = BYTETracker(cfg)

    doc = json.load(open(DETS_JSON, encoding="utf-8"))
    frames_tracked = []
    for fr in doc["frames"]:
        ps = fr["players"]
        if ps:
            xywh = np.array([[p["cx"], p["cy"], p["w"], p["h"]] for p in ps], dtype=np.float32)
            conf = np.array([p["conf"] for p in ps], dtype=np.float32)
            cls = np.zeros(len(ps), dtype=np.float32)
        else:
            xywh = np.zeros((0, 4), dtype=np.float32)
            conf = np.zeros(0, dtype=np.float32)
            cls = np.zeros(0, dtype=np.float32)
        out = tracker.update(_Dets(xywh, conf, cls))
        tracks = []
        if out is not None and len(out):
            for row in np.asarray(out):
                # rows: x1,y1,x2,y2,track_id,conf,cls[,det_idx]
                tracks.append({"track_id": int(row[4]),
                               "bbox": [float(row[0]), float(row[1]),
                                        float(row[2]), float(row[3])]})
        frames_tracked.append({"frame_index": fr["frame_index"], "tracks": tracks})

    def stats(frames):
        ids = {}
        per_frame = []
        for fr in frames:
            per_frame.append(len(fr["tracks"]))
            for t in fr["tracks"]:
                ids[t["track_id"]] = ids.get(t["track_id"], 0) + 1
        n = len(ids)
        return n, (sum(per_frame) / max(len(per_frame), 1)), (sum(ids.values()) / max(n, 1))

    baseline = json.load(open(CLIP.tracks_cache_path, encoding="utf-8"))
    b = stats(baseline["frames"])
    p = stats(frames_tracked)

    # refs: avg per frame; and baseline tracked boxes IoU-matching a Ref det
    def iou(a, b_):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b_
        ix = max(0, min(ax2, bx2) - max(ax1, bx1))
        iy = max(0, min(ay2, by2) - max(ay1, by1))
        inter = ix * iy
        ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / ua if ua > 0 else 0.0

    refs_by_frame = {fr["frame_index"]: fr["refs"] for fr in doc["frames"]}
    n_ref_dets = sum(len(v) for v in refs_by_frame.values())
    ref_matched_track_ids = set()
    for fr in baseline["frames"]:
        refs = refs_by_frame.get(fr["frame_index"], [])
        if not refs:
            continue
        ref_boxes = [[r["cx"] - r["w"] / 2, r["cy"] - r["h"] / 2,
                      r["cx"] + r["w"] / 2, r["cy"] + r["h"] / 2]
                     for r in refs if r["conf"] >= 0.4]
        for t in fr["tracks"]:
            for rb in ref_boxes:
                if iou(t["bbox"], rb) >= 0.5:
                    ref_matched_track_ids.add(t["track_id"])

    print("\n============ PLAYER/REF PROBE (%s) ============" % CLIP_NAME)
    print("                          COCO-person(cached)   RF-Player(standalone BYTE)")
    print(f"  distinct track_ids     {b[0]:>15}   {p[0]:>16}")
    print(f"  mean tracks/frame      {b[1]:>15.1f}   {p[1]:>16.1f}")
    print(f"  mean lifespan (fr)     {b[2]:>15.1f}   {p[2]:>16.1f}")
    print(f"  Ref detections total (all conf): {n_ref_dets} "
          f"({n_ref_dets / max(len(doc['frames']), 1):.1f}/frame avg)")
    print(f"  baseline track_ids IoU-matching a Ref det (conf>=0.4): "
          f"{len(ref_matched_track_ids)} of {b[0]}")
    print(f"  ref-matched baseline ids: {sorted(ref_matched_track_ids)}")
    print("  (caveats: conf scales differ between detectors; baseline tracked via")
    print("   model.track() plumbing vs this standalone harness, same yaml params)")


if __name__ == "__main__":
    if not os.path.exists(DETS_JSON):
        assert KEY, "RF_KEY env var required for the fetch phase"
        fetch_detections()
    else:
        print(f"[player_probe] reusing existing {DETS_JSON}")
    track_and_compare()
