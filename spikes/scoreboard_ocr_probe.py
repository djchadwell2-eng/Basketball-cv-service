"""Second pass, ENTIRELY SEPARATE from ball/player detection (DJ's idea,
2026-07-23): the scoreboard graphic sits at a FIXED screen position for
the whole clip (confirmed by eye, 3 widely-spaced frames). Reading it
independently gives ground-truth MAKE/MISS + team + timestamp for free,
using the broadcast's own official scoring -- and keeps this OCR work
out of the player/ball detector's way entirely (which is currently
sometimes confused by this exact graphic, per TEST 13's bonus finding).

Method: sample the fixed bottom-left corner region every SAMPLE_STRIDE
frames, run EasyOCR (digits only, reused from phase2/ocr_reader.py's
engine) on the whole corner crop, and pick the two TALLEST digit
detections per frame -- the score digits are rendered far larger than
the foul-count/period numbers on the same graphic, so height alone
separates them without needing hand-tuned sub-crop coordinates (robust
to minor per-clip graphic layout differences). Left-of-center detection
= home score, right-of-center = away score (matches the graphic's own
layout). A frame-to-frame INCREASE in either score = a made basket,
timestamped.

Read-only probe: prints/writes findings only, no cache writes.

Run:  .venv/Scripts/python spikes/scoreboard_ocr_probe.py HARD
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
from ocr_reader import _get_reader

SAMPLE_STRIDE = 15          # every 0.5s @ 30fps -- a make can't blip by faster than that
# A single basketball play can raise ONE team's score by at most 3 (a
# 3-pointer) or a made-shot-plus-and-1 free throw (2+1). Anything bigger in
# one "change" is an OCR misread (e.g. a foul-count digit merged into the
# score, or two unrelated digits from a different reading averaged into the
# mode), not a real single play -- found 2026-07-31 running
# dense_shot_score_match.py for real for the first time: it reported a
# "0-0 -> 5-0" change that the monotonicity guard alone let straight through.
MAX_PLAUSIBLE_JUMP = 3
CORNER_H_FRAC = (0.72, 1.0)  # bottom-left graphic region, generous bounds
CORNER_W_FRAC = (0.0, 0.22)
UPSCALE = 3


def _read_scores(crop_bgr):
    """Returns (home_score, away_score) as ints, or (None, None) if the two
    tallest digit detections aren't confidently readable this frame."""
    if crop_bgr is None or crop_bgr.size == 0:
        return None, None
    up = cv2.resize(crop_bgr, (crop_bgr.shape[1] * UPSCALE, crop_bgr.shape[0] * UPSCALE),
                    interpolation=cv2.INTER_CUBIC)
    dets = []
    for (box, txt, conf) in _get_reader().readtext(up, allowlist="0123456789"):
        if not txt.isdigit():
            continue
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        height = max(ys) - min(ys)
        cx = sum(xs) / len(xs)
        dets.append((height, cx, int(txt), float(conf)))
    if len(dets) < 2:
        return None, None
    dets.sort(key=lambda d: -d[0])          # tallest first = the big score digits
    top2 = dets[:2]
    top2.sort(key=lambda d: d[1])           # left-to-right = home, away
    (_, _, left_val, left_conf), (_, _, right_val, right_conf) = top2
    if left_conf < 0.15 or right_conf < 0.15:
        # near-zero confidence = OCR found nothing digit-shaped there at all;
        # a genuinely-correct score digit can still read low (0.11-0.5) on
        # blurry/compressed frames -- temporal voting below handles the rest.
        return None, None
    return left_val, right_val


def run_probe(clip_name, frame_start=0, frame_end=None, sample_stride=SAMPLE_STRIDE,
             window=7, min_votes=4, initial_home=None, initial_away=None,
             verbose=True):
    """Core score-reading loop, reusable at any stride/frame range/starting
    state. Returns (events, readings, fps). See module docstring + TEST 14
    (TEST_LOG.md) for the noise-handling rationale (sliding-window majority
    vote + monotonicity guard -- NOT a strict per-frame threshold, NOT a
    naive N-in-a-row streak; both were measured to fail)."""
    CLIP = getattr(clip_config, f"{clip_name}_CLIP")
    cap = cv2.VideoCapture(CLIP.video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_end = total if frame_end is None else min(frame_end, total)

    from collections import deque, Counter
    recent = deque(maxlen=window)
    readings = []
    confirmed_home, confirmed_away = initial_home, initial_away
    events = []

    # Seek ONCE to the window start, then read sequentially. Re-seeking
    # (cap.set(POS_FRAMES,...)) on every iteration forces a decode from the
    # nearest prior keyframe each time -- fine for stride=15 (measured
    # acceptable), but at stride=1 it turns a ~10 min job into an hours-long
    # one (caught by wall-clock: TEST 14 follow-up's dense pass never
    # finished in 25+ min before this fix).
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
    next_seq_frame = frame_start
    f = frame_start
    while f < frame_end:
        if f != next_seq_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)     # only when stride skips frames
        ok, frame = cap.read()
        next_seq_frame = f + 1
        if not ok:
            break
        h, w = frame.shape[:2]
        y0, y1 = int(h * CORNER_H_FRAC[0]), int(h * CORNER_H_FRAC[1])
        x0, x1 = int(w * CORNER_W_FRAC[0]), int(w * CORNER_W_FRAC[1])
        crop = frame[y0:y1, x0:x1]
        raw = _read_scores(crop)
        if raw != (None, None):
            recent.append(raw)
            mode_val, mode_count = Counter(recent).most_common(1)[0]
            home, away = mode_val
            monotonic_ok = (confirmed_home is None
                            or (home >= confirmed_home and away >= confirmed_away))
            plausible_jump = (confirmed_home is None
                              or (home - confirmed_home <= MAX_PLAUSIBLE_JUMP
                                  and away - confirmed_away <= MAX_PLAUSIBLE_JUMP))
            if (mode_count >= min_votes and mode_val != (confirmed_home, confirmed_away)
                    and monotonic_ok and plausible_jump):
                if confirmed_home is not None:
                    events.append({"frame": f, "t_sec": round(f / fps, 2),
                                   "from": [confirmed_home, confirmed_away], "to": [home, away]})
                    if verbose:
                        print(f"  SCORE CHANGE at f={f} ({f/fps:.1f}s): "
                              f"{confirmed_home}-{confirmed_away} -> {home}-{away}")
                confirmed_home, confirmed_away = home, away
                readings.append((f, home, away))
                recent.clear()
            elif mode_count >= min_votes and (not monotonic_ok or not plausible_jump):
                recent.clear()
        f += sample_stride
        if verbose and (f - frame_start) % (sample_stride * 20) == 0:
            print(f"  ...{f}/{frame_end}", flush=True)
    cap.release()
    return events, readings, fps


def main():
    clip_name = sys.argv[1] if len(sys.argv) > 1 else "HARD"
    CLIP = getattr(clip_config, f"{clip_name}_CLIP")
    cap = cv2.VideoCapture(CLIP.video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"[scoreboard] {clip_name}: {total} frames @ {fps:.1f}fps, "
          f"sampling every {SAMPLE_STRIDE} frames")

    events, readings, fps = run_probe(clip_name, 0, total, SAMPLE_STRIDE)

    out_json = os.path.join(_HERE, "out", f"{clip_name}_scoreboard_ocr.json")
    json.dump({"clip": clip_name, "sample_stride": SAMPLE_STRIDE,
               "n_readings": len(readings), "events": events,
               "readings": readings}, open(out_json, "w"), indent=2)

    final = readings[-1][1:] if readings else (None, None)
    print(f"\n[scoreboard] final confirmed score: {final[0]}-{final[1]} "
          f"({len(readings)} state-establishment(s) logged, i.e. initial lock + real changes)")
    print(f"[scoreboard] {len(events)} score-change events -> {out_json}")
    for e in events:
        print(f"  {e['t_sec']}s: {e['from']} -> {e['to']}")


if __name__ == "__main__":
    main()
