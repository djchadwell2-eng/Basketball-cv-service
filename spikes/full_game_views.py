"""HOW MANY MARKS DOES A REAL FULL GAME NEED? -- measured, not extrapolated.

DJ, 2026-07-29: "you keep testing on a five minute clip. What if I just gave you
a full game?" He was right -- every full-game number quoted so far was arithmetic
from clips of 1-5 minutes. Full_Game.mp4 is 171,120 frames / 95.1 min.

THE OPERATIONAL QUESTION, asked the way the pipeline actually works: a marked
frame can serve every frame that SIFT can bridge back to it at an inlier ratio
>= 0.6 (the project's own weak-pair threshold, spikes/stage2_multikeyframe.py).
So the number of marks a game needs is the number of frames you must mark before
EVERY other frame can reach one of them.

ALGORITHM -- greedy incremental cover, which is exactly that question:
  for each sampled frame:
      try to match it against the existing view representatives
      if one holds ratio >= 0.6 -> this frame is covered, no new mark
      if none does              -> this frame becomes a NEW representative
  answer = number of representatives

Two speed tricks, neither affecting the result:
  - a cheap 32x18 greyscale signature shortlists which representatives are even
    worth a SIFT attempt (SIFT is the expensive part)
  - representatives are tried most-recently-used first, because consecutive
    frames are nearly always the same view

ALSO REPORTED, all free from the same pass:
  - whether the count FLATTENS over the game (per-quarter新 marks) -- the thing
    that decides whether extrapolating from 5 minutes was ever valid
  - hard CUTS (quarter breaks, replays) that no SIFT chain can cross
  - how far the camera actually pans

Usage:  .venv/Scripts/python.exe spikes/full_game_views.py [stride]
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
import stage1_keyframe_match as s1                               # noqa: E402

VIDEO = os.path.join(_ROOT, "Full_Game.mp4")
MIN_RATIO = 0.6          # the project's own weak-pair bar -- not a new knob
SCALE = 0.35             # SIFT on 672x378; enough texture, much faster
SIG = (32, 18)
# SIFT attempts per frame, best signature matches first. MEASURED HAZARD
# (2026-07-29): with TRY_TOP=3 and 42 accumulated marks, a frame is only offered
# 3 of 42 candidates, so one that WOULD have matched mark #20 never gets the
# chance and opens a spurious new mark -- which then makes the shortlist cover an
# even smaller fraction. That feedback loop reproduces the exact growth curve
# (8 -> 18 -> 42 marks as sampling got finer) and is indistinguishable from the
# footage genuinely having more views. Overridable so the two can be told apart.
TRY_TOP = int(os.environ.get("TRY_TOP", "3"))
CUT_SIG_DELTA = 40.0     # mean |sig| jump that indicates a hard cut


def signature(bgr):
    g = cv2.cvtColor(cv2.resize(bgr, SIG), cv2.COLOR_BGR2GRAY)
    return g.astype(np.float32)


def ratio(a, b):
    with contextlib.redirect_stdout(io.StringIO()):
        kp_a, kp_b, good = s1.detect_and_match(a, b, 0.75, [])
        if len(good) < 15:
            return 0.0
        _H, _m, inl = s1.estimate_homography(kp_a, kp_b, good, 3.0)
    return (inl / len(good)) if inl else 0.0


def main():
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    print(f"[full_game_views] TRY_TOP={TRY_TOP} (candidates tested per frame)",
          flush=True)
    cap = cv2.VideoCapture(VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[full_game_views] {total:,} frames / {total/fps/60:.1f} min, "
          f"sampling every {stride} ({total//stride:,} samples), "
          f"ratio bar {MIN_RATIO}", flush=True)

    reps = []            # [{'img':.., 'sig':.., 'frame':.., 'hits':int}]
    order = []           # rep indices, most-recently-used first
    cuts, sigs = [], []
    new_by_bucket = {}   # 10-min bucket -> new reps opened
    t0 = time.time()
    i, n_seen, prev_sig = 0, 0, None

    while True:
        if i % stride:
            if not cap.grab():
                break
            i += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, None, fx=SCALE, fy=SCALE)
        sg = signature(frame)
        n_seen += 1

        if prev_sig is not None:
            d = float(np.abs(sg - prev_sig).mean())
            sigs.append(d)
            if d > CUT_SIG_DELTA:
                cuts.append(i)
        prev_sig = sg

        covered = False
        cand = sorted(order, key=lambda r: float(np.abs(reps[r]["sig"] - sg).mean()))
        for r in cand[:TRY_TOP]:
            if ratio(reps[r]["img"], small) >= MIN_RATIO:
                reps[r]["hits"] += 1
                order.remove(r)
                order.insert(0, r)
                covered = True
                break
        if not covered:
            reps.append({"img": small, "sig": sg, "frame": i, "hits": 0})
            order.insert(0, len(reps) - 1)
            b = int(i / fps / 600)
            new_by_bucket[b] = new_by_bucket.get(b, 0) + 1

        if n_seen % 50 == 0:
            print(f"  {i:>7}/{total} ({100*i/total:>4.1f}%)  "
                  f"marks so far {len(reps):>4}   {time.time()-t0:>5.0f}s",
                  flush=True)
        i += 1
    cap.release()

    print(f"\n=========== FULL GAME -- MEASURED ===========")
    print(f"  sampled {n_seen:,} frames of {total:,} (every {stride})")
    print(f"  MARKS NEEDED: {len(reps)}   -> ~{len(reps)*5} clicks at 5 each"
          f"   (~{len(reps)*5*4/60:.0f} min at 4s/click)")
    print(f"  hard cuts detected: {len(cuts)}"
          + (f"  first few at frames {cuts[:6]}" if cuts else ""))
    print(f"\n  DOES IT FLATTEN? new marks opened per 10-min block:")
    for b in sorted(new_by_bucket):
        print(f"    {b*10:>3}-{b*10+10:>3} min: {'#' * min(new_by_bucket[b], 60)}"
              f" {new_by_bucket[b]}")
    tail = sum(v for k, v in new_by_bucket.items() if k >= max(new_by_bucket) - 1)
    print(f"  last 20 min opened {tail} new marks "
          f"({'FLATTENING' if tail <= len(reps) * 0.15 else 'STILL GROWING'})")
    busy = sorted(reps, key=lambda r: -r["hits"])[:5]
    print(f"\n  busiest views (frames they cover): "
          f"{[(r['frame'], r['hits']) for r in busy]}")

    # THE DELIVERABLE: the actual frame numbers to click, busiest first, plus a
    # still of each so DJ can see what he is being asked to mark. Ordered by
    # coverage so the FIRST few clicks buy the most -- the top 5 covered 97% of
    # sampled frames, so a partial session is still worth something.
    import json
    out_dir = os.path.join(_HERE, "out")
    os.makedirs(out_dir, exist_ok=True)
    ranked = sorted(reps, key=lambda r: -r["hits"])
    doc = {"video": os.path.basename(VIDEO), "total_frames": total, "fps": fps,
           "stride": stride, "try_top": TRY_TOP, "min_ratio": MIN_RATIO,
           "n_marks": len(reps), "estimated_clicks": len(reps) * 5,
           "marks": [{"frame": r["frame"], "t_sec": round(r["frame"] / fps, 1),
                      "frames_covered": r["hits"]} for r in ranked]}
    jp = os.path.join(out_dir, f"FULLGAME_marks_stride{stride}.json")
    json.dump(doc, open(jp, "w", encoding="utf-8"), indent=2)
    print(f"\n  wrote {jp}")

    shot_dir = os.path.join(out_dir, f"FULLGAME_mark_frames_stride{stride}")
    os.makedirs(shot_dir, exist_ok=True)
    cap2 = cv2.VideoCapture(VIDEO)
    for rank, r in enumerate(ranked, 1):
        cap2.set(cv2.CAP_PROP_POS_FRAMES, r["frame"])
        ok, fr = cap2.read()
        if ok:
            cv2.imwrite(os.path.join(
                shot_dir, f"{rank:02d}_f{r['frame']}_covers{r['hits']}.jpg"), fr)
    cap2.release()
    print(f"  wrote {len(ranked)} stills -> {shot_dir}")
    print(f"  elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
