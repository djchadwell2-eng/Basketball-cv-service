"""TEST 7 phase A (gated protocol): dump RAW yolov8m person detections for a
clip's tracking span to JSON, so third-party trackers (boxmot, in an
ISOLATED venv) can be fed the exact same detections. Main-venv script.

Run:  .venv/Scripts/python spikes/raw_person_dets.py [CLIP]
"""
from __future__ import annotations
import json, os, sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "phase2"))

import clip_config

CLIP_NAME = sys.argv[1] if len(sys.argv) > 1 else "TEST1"
CLIP = getattr(clip_config, f"{CLIP_NAME}_CLIP")
clip_config.ACTIVE_CLIP = CLIP          # BEFORE importing run_tracking (binds at import)

import run_tracking
import tracking as trk
from ultralytics import YOLO

OUT_JSON = os.path.join(_HERE, "out", f"{CLIP_NAME}_rawdets_person.json")
CONF = 0.1   # matches bytetrack.yaml track_low_thresh -- the lowest det the
             # committed tracker ever sees; each probe tracker gates above this


def main():
    start, length = CLIP.tracking_span_start, CLIP.tracking_span_len
    subclip, fps, n = run_tracking.extract_subclip(CLIP.video_path, start, length)
    print(f"[rawdets] {CLIP_NAME} span {start}..{start + length}: {n} frames -> {subclip}")
    model = YOLO(trk.MODEL_NAME)
    results = model.predict(source=subclip, classes=[trk.PERSON_CLASS],
                            imgsz=trk.IMG_SIZE, conf=CONF, stream=True, verbose=False)
    frames_out = []
    for i, r in enumerate(results):
        dets = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c in zip(xyxy, conf):
                dets.append([float(x1), float(y1), float(x2), float(y2), float(c)])
        frames_out.append({"frame_index": start + i, "dets": dets})
        if i % 30 == 0:
            print(f"  ...{i}/{n} ({len(dets)} dets)", flush=True)
    json.dump({"clip": CLIP_NAME, "span_start": start, "span_len": len(frames_out),
               "conf_floor": CONF, "detector": "yolov8m@1280",
               "subclip_path": subclip, "frames": frames_out},
              open(OUT_JSON, "w", encoding="utf-8"), indent=2)
    print(f"[rawdets] wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
