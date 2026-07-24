"""Milestone 2 step 1: harvest the exact frames worth human-labeling.

Selection is TARGETED, not random (TEST 9/10 diagnosis): frames where the
ball is provably in flight (inside a hosted-model arc span) but the local
v2 weights have no confident detection, plus the spans DJ's eyeballs
flagged (shot B blind spot, shot A margin, the HARD rebound/dish
false-positive sequence). Writes full-res JPGs named {clip}_{frame:05d}.jpg
to spikes/out/label_harvest/ for upload to a Roboflow labeling project.

Read-only over videos + existing logs; writes images only.

Run:  .venv/Scripts/python spikes/harvest_label_frames.py
"""
from __future__ import annotations

import json
import os
import sys

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
import clip_config

OUT_DIR = os.path.join(_HERE, "out", "label_harvest")
CONF_FLOOR = 0.10

# Hand-picked spans (frame ranges, inclusive) from TEST 9/10 findings.
MANUAL_SPANS = {
    "TEST1": [(50, 80),     # shot A: v2 density 11/20, marginal 118px pass
              (310, 335)],  # shot B: v2 blind (4/13) -- the regression
    "HARD":  [(395, 445)],  # rebound->dish sequence v2 false-claimed as layup
}

# HARD flight-miss frames: inside a hosted-model arc (ball provably in
# flight) but v2 has no conf>=0.10 detection there.
HOSTED_LOG = {"HARD": "HARD_ball_spike_log_roboflow.json"}
V2_LOG = {"HARD": "HARD_ball_spike_log_ball_finetuned_v2_gpu.json",
          "TEST1": "TEST1_ball_spike_log_ball_finetuned_v2.json"}
ARCS_FILE = {"HARD": "HARD_ball_arcs.json"}   # committed arc spans (stock) --
# NOT used; hosted arcs are recomputed below from the hosted log via the
# same trajectory code, keeping the flight definition identical to TEST 9.


def _dets_by_frame(path, floor=CONF_FLOOR):
    doc = json.load(open(path, encoding="utf-8"))
    return {fr["frame_index"]: [d for d in fr["detections"] if d["conf"] >= floor]
            for fr in doc["frames"]}, doc


def hosted_arc_spans(clip):
    """Recompute hosted-model arc spans exactly as TEST 9 did."""
    sys.path.insert(0, _HERE)
    from ball_trajectory import build_chains, classify_chain
    doc = json.load(open(os.path.join(_HERE, "out", HOSTED_LOG[clip]), encoding="utf-8"))
    frames = [{"frame_index": fr["frame_index"],
               "detections": [d for d in fr["detections"] if d["conf"] >= CONF_FLOOR]}
              for fr in doc["frames"]]
    spans = []
    for c in build_chains(frames):
        r = classify_chain(c)
        for a in r["arcs"]:
            spans.append((a["start_frame"], a["end_frame"]))
    return spans


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    picks = {}   # clip -> sorted set of frames

    for clip, spans in MANUAL_SPANS.items():
        s = picks.setdefault(clip, set())
        for a, b in spans:
            s.update(range(a, b + 1))

    # HARD flight misses (hosted sees flight, v2 confidently blind)
    v2, _ = _dets_by_frame(os.path.join(_HERE, "out", V2_LOG["HARD"]))
    miss = set()
    for (a, b) in hosted_arc_spans("HARD"):
        for f in range(a, b + 1):
            if not v2.get(f):
                miss.add(f)
    picks["HARD"].update(miss)
    print(f"[harvest] HARD flight-miss frames (in hosted arc, v2 blind): {len(miss)}")

    total = 0
    for clip, frames in picks.items():
        CLIP = getattr(clip_config, f"{clip}_CLIP")
        cap = cv2.VideoCapture(CLIP.video_path)
        for f in sorted(frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                print(f"  [warn] {clip} frame {f}: read failed, skipped")
                continue
            out = os.path.join(OUT_DIR, f"{clip}_{f:05d}.jpg")
            cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            total += 1
        cap.release()
        print(f"[harvest] {clip}: {len(frames)} frames requested")
    print(f"[harvest] wrote {total} JPGs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
