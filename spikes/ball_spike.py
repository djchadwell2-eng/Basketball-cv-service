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
if _is_main:
    # Standalone run: set BEFORE importing run_tracking (binds at import;
    # otherwise the temp subclip gets the wrong clip's name). NOT on plain
    # import -- run_clip/ball_stages import this module with ACTIVE_CLIP
    # already synced, and resetting it here would clobber that sync with
    # "HARD" (the argv default).
    clip_config.ACTIVE_CLIP = CLIP

import run_tracking                      # reuse extract_subclip (same span-extraction as tracking)
import tracking as trk                    # reuse MODEL_NAME / IMG_SIZE (same model as person detection)

from ultralytics import YOLO
# User-identified shot attempt: HARD.mp4 ~35-45s (30fps -> frames 1050-1350).
# +/-30 frame buffer so the shot isn't cut off at either edge.
SPAN_START = int(sys.argv[2]) if _is_main and len(sys.argv) > 2 else 1020
SPAN_LEN = int(sys.argv[3]) if _is_main and len(sys.argv) > 3 else 360
# imgsz override (DECISIONS 20): swept, 1280 is optimal -- do not re-run.
IMG_SIZE = int(sys.argv[4]) if _is_main and len(sys.argv) > 4 else trk.IMG_SIZE
# model override (DECISIONS 21): input-resolution lever is exhausted; MODEL
# CAPACITY is the untried axis. A bigger stock model (yolov8x.pt) has more
# capacity for small AND motion-blurred balls -- the two §20 failure modes.
# Bare filename resolves against repo root; default = the validated yolov8m.
_model_arg = sys.argv[5] if _is_main and len(sys.argv) > 5 else trk.MODEL_NAME
MODEL = _model_arg if os.path.isabs(_model_arg) else os.path.join(_ROOT, _model_arg)

BALL_CLASS = 32                           # COCO "sports ball"
CONF = 0.05                               # deliberately low -- we want to SEE the misses too
OUT_DIR = os.path.join(_HERE, "out")
# A non-default imgsz OR model writes to a SUFFIXED file so a measurement run
# never clobbers the canonical log the downstream pipeline reads (measure
# first, adopt only on the numbers -- same discipline as the reid probe, §11).
_parts = []
if IMG_SIZE != trk.IMG_SIZE:
    _parts.append(f"imgsz{IMG_SIZE}")
if MODEL != trk.MODEL_NAME:
    _parts.append(os.path.splitext(os.path.basename(MODEL))[0])
_SUFFIX = ("_" + "_".join(_parts)) if _parts else ""
OUT_VIDEO = os.path.join(OUT_DIR, f"{CLIP.name}_ball_spike_overlay{_SUFFIX}.mp4")
OUT_JSON = os.path.join(OUT_DIR, f"{CLIP.name}_ball_spike_log{_SUFFIX}.json")
BOX_COLOR = (0, 165, 255)                 # orange, easy to spot against court colors


def detect(clip, span_start, span_len, imgsz, model_path, out_json, out_video):
    """The detection loop, callable with explicit inputs/outputs (used by the
    CLI below with argv-derived values, and by ball_stages.py with ClipConfig
    values). Identical behavior to the original main() body."""
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    print(f"[ball_spike] extracting {clip.name} span {span_start}..{span_start + span_len} "
          f"({span_len / 30.0:.1f}s) ...")
    # NO TEMP COPY, same reason as run_tracking: this re-encoded the span to an
    # mp4 just to read it back, at a MEASURED 29.1 ms/frame, and the copy loses
    # 1-3 detections per frame against the source. Ball detection is stateless
    # per frame, so it simply reads the film.
    _cap = cv2.VideoCapture(clip.video_path)
    fps = _cap.get(cv2.CAP_PROP_FPS) or 30.0
    if span_start > 0:
        _cap.set(cv2.CAP_PROP_POS_FRAMES, span_start)
        if int(_cap.get(cv2.CAP_PROP_POS_FRAMES)) != span_start:
            _cap.release()
            _cap = cv2.VideoCapture(clip.video_path)
            for _ in range(span_start):
                _cap.grab()

    def _span_frames():
        for _ in range(span_len):
            ok, fr = _cap.read()
            if not ok:
                return
            yield fr
    n = span_len
    print(f"[ball_spike] reading {n} frames straight from the film")

    model = YOLO(model_path)
    # BALL_CLASS=32 is COCO's numbering (stock models). A fine-tuned model
    # carries its OWN class list (e.g. "Ball" at index 0), so resolve the id
    # from the loaded model's names; unchanged behavior for stock models.
    ball_class = next((i for i, n in model.names.items()
                       if str(n).lower() in ("ball", "sports ball")), BALL_CLASS)
    print(f"[ball_spike] running {os.path.basename(model_path)} (imgsz={imgsz}, conf={CONF}, "
          f"class={model.names[ball_class]}[{ball_class}]) -- CPU ...")
    def _results():
        for _fr in _span_frames():
            yield model.predict(_fr, classes=[ball_class], imgsz=imgsz,
                                conf=CONF, verbose=False)[0]
    results = _results()

    w = int(_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # An eyeball artifact, not an output -- see ball_stages.overlays_wanted.
    # Three of these on a whole game is ~48 GB into 32 GB of container disk.
    import ball_stages
    writer = (cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
              if ball_stages.overlays_wanted(span_len, "ball-detection") else None)

    frames_out = []
    n_with_det = 0
    for i, result in enumerate(results):
        frame = result.orig_img.copy()
        frame_idx = span_start + i
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
        if writer is not None:
            writer.write(frame)
        frames_out.append({"frame_index": frame_idx, "t_sec": round(t_sec, 2),
                            "detections": dets})
        if i % 30 == 0:
            print(f"  ...{i}/{n}  ({len(dets)} dets this frame)", flush=True)
    if writer is not None:
        writer.release()

    _cap.release()
    doc = {"clip": clip.name, "span_start": span_start, "span_len": len(frames_out),
           "fps": fps, "conf_threshold": CONF, "imgsz": imgsz,
           "model": os.path.basename(model_path), "frames": frames_out}
    # COMPACT, not indent=2. These are machine-read caches that reach a
    # gigabyte on a whole game, and MEASURED on real slices the pretty-printing
    # is 2.2x the tracks file and 3.0x the on-court one -- ~1.2 GB a game of
    # bytes that are literally spaces, written to and read back from a network
    # volume. The PARSED CONTENT is identical (checked, both files).
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"))

    total = len(frames_out)
    flicker_pct = 100.0 * (total - n_with_det) / max(total, 1)
    mean_dets = sum(len(fr["detections"]) for fr in frames_out) / max(total, 1)
    print("\n================ BALL SPIKE SUMMARY ================")
    print(f"  clip: {clip.name}  span: {span_start}..{span_start + total} "
          f"({span_start / fps:.1f}s..{(span_start + total) / fps:.1f}s)")
    print(f"  frames: {total}   frames w/ >=1 detection: {n_with_det} "
          f"({100 - flicker_pct:.1f}%)   zero-detection frames: {flicker_pct:.1f}%")
    print(f"  mean detections/frame: {mean_dets:.2f}")
    print(f"  overlay: {out_video}")
    print(f"  log:     {out_json}")
    print("  (raw detections only -- false-positive rate needs an eyeball pass on the overlay)")
    return doc


def main():
    detect(CLIP, SPAN_START, SPAN_LEN, IMG_SIZE, MODEL, OUT_JSON, OUT_VIDEO)


if __name__ == "__main__":
    main()
