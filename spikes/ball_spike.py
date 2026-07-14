"""READ-ONLY probe: can stock YOLOv8m see the ball at all? Phase 5 step 1
(ROADMAP.md "Ball spike" -- measure BEFORE building anything).

Runs YOLOv8m (the SAME model already used for person detection, phase2/tracking.py)
filtered to COCO class 32 "sports ball", at a low confidence threshold and the same
imgsz=1280 as the validated person config. RAW per-frame detections only -- no
tracker, no persistence, no arc-fitting. Draws every box on every frame and writes
an overlay video + a per-frame JSON log, so flicker/false-positive rate can be
measured from data, not just eyeballed.

Touches nothing else: new file, no edits to tracking.py/run_tracking.py/any cache.

Run (background, CPU ~2s/frame => a 360-frame span is ~10-15 min):
    .venv/Scripts/python spikes/ball_spike.py
"""

from __future__ import annotations

import json
import os
import sys

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "phase2"))

import clip_config

# CLI: ball_spike.py [clip_name] [span_start] [span_len]. No args = exact
# original behavior (HARD, the user-identified ~35-45s shot span). Passing
# a clip_name REQUIRES span_start/span_len too (no per-clip auto-default --
# explicit beats hidden magic when adding a brand-new clip, e.g. TEST1).
# Guarded by __main__: no test imports this module today, but the same
# trap as hoop_anchor.py applies if one ever does (pytest's own argv would
# get misread as a clip name).
_is_main = __name__ == "__main__"
CLIP_NAME = sys.argv[1] if _is_main and len(sys.argv) > 1 else "HARD"
CLIP = getattr(clip_config, f"{CLIP_NAME}_CLIP")
clip_config.ACTIVE_CLIP = CLIP          # set BEFORE importing run_tracking (binds at import;
                                        # otherwise the temp subclip gets the wrong clip's name)

import run_tracking                      # reuse extract_subclip (same span-extraction as tracking)
import tracking as trk                    # reuse MODEL_NAME / IMG_SIZE (same model as person detection)

from ultralytics import YOLO
# User-identified shot attempt: HARD.mp4 ~35-45s (30fps -> frames 1050-1350).
# +/-30 frame buffer so the shot isn't cut off at either edge.
SPAN_START = int(sys.argv[2]) if _is_main and len(sys.argv) > 2 else 1020
SPAN_LEN = int(sys.argv[3]) if _is_main and len(sys.argv) > 3 else 360

BALL_CLASS = 32                           # COCO "sports ball"
CONF = 0.05                               # deliberately low -- we want to SEE the misses too
OUT_DIR = os.path.join(_HERE, "out")
OUT_VIDEO = os.path.join(OUT_DIR, f"{CLIP.name}_ball_spike_overlay.mp4")
OUT_JSON = os.path.join(OUT_DIR, f"{CLIP.name}_ball_spike_log.json")
BOX_COLOR = (0, 165, 255)                 # orange, easy to spot against court colors


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[ball_spike] extracting {CLIP.name} span {SPAN_START}..{SPAN_START + SPAN_LEN} "
          f"({SPAN_LEN / 30.0:.1f}s) ...")
    subclip, fps, n = run_tracking.extract_subclip(CLIP.video_path, SPAN_START, SPAN_LEN)
    print(f"[ball_spike] {n} frames -> {subclip}")

    model = YOLO(trk.MODEL_NAME)
    print(f"[ball_spike] running YOLOv8m (imgsz={trk.IMG_SIZE}, conf={CONF}, "
          f"class=sports ball) -- CPU, ~2s/frame ...")
    results = model.predict(source=subclip, classes=[BALL_CLASS], imgsz=trk.IMG_SIZE,
                             conf=CONF, stream=True, verbose=False)

    cap = cv2.VideoCapture(subclip)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    writer = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frames_out = []
    n_with_det = 0
    for i, result in enumerate(results):
        frame = result.orig_img.copy()
        frame_idx = SPAN_START + i
        dets = []
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for (x1, y1, x2, y2), c in zip(xyxy, confs):
                dets.append({"bbox": [float(x1), float(y1), float(x2), float(y2)],
                             "conf": float(c)})
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), BOX_COLOR, 2)
                cv2.putText(frame, f"{c:.2f}", (int(x1), max(0, int(y1) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, BOX_COLOR, 2)
        if dets:
            n_with_det += 1
        t_sec = frame_idx / fps
        cv2.putText(frame, f"f={frame_idx} t={t_sec:04.1f}s dets={len(dets)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        writer.write(frame)
        frames_out.append({"frame_index": frame_idx, "t_sec": round(t_sec, 2),
                            "detections": dets})
        if i % 30 == 0:
            print(f"  ...{i}/{n}  ({len(dets)} dets this frame)", flush=True)
    writer.release()

    doc = {"clip": CLIP.name, "span_start": SPAN_START, "span_len": len(frames_out),
           "fps": fps, "conf_threshold": CONF, "frames": frames_out}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

    total = len(frames_out)
    flicker_pct = 100.0 * (total - n_with_det) / max(total, 1)
    mean_dets = sum(len(fr["detections"]) for fr in frames_out) / max(total, 1)
    print("\n================ BALL SPIKE SUMMARY ================")
    print(f"  clip: {CLIP.name}  span: {SPAN_START}..{SPAN_START + total} "
          f"({SPAN_START / fps:.1f}s..{(SPAN_START + total) / fps:.1f}s)")
    print(f"  frames: {total}   frames w/ >=1 detection: {n_with_det} "
          f"({100 - flicker_pct:.1f}%)   zero-detection frames: {flicker_pct:.1f}%")
    print(f"  mean detections/frame: {mean_dets:.2f}")
    print(f"  overlay: {OUT_VIDEO}")
    print(f"  log:     {OUT_JSON}")
    print("  (raw detections only -- false-positive rate needs an eyeball pass on the overlay)")


if __name__ == "__main__":
    main()
