"""HOW MANY MARKED FRAMES DOES A FULL GAME NEED, so the chain actually holds?

THE QUESTION THIS FIXES (TEST 36): spikes/full_game_views.py picked 5 frames that
COVER the game, and calibration failed at 15.45 ft because two of them were 28
minutes apart and shared only 9 matching points. Coverage and CHAINABILITY are
different problems, and the cover algorithm optimises for frames being maximally
DIFFERENT -- exactly backwards for a chain.

THE RIGHT QUESTION: walk forward from the start and place the next marked frame at
the FURTHEST point that still matches the previous one at inlier ratio >= 0.6 (the
project's own weak-pair bar, spikes/stage2_multikeyframe.py). Same rule as
spikes/plan_keyframes.py -- which was built and validated in TEST 29 and then not
used on the full game.

WHY A SEPARATE SCRIPT: plan_keyframes calls extract_frames per probe, and that does
a frame-accurate SEQUENTIAL read from frame 0 every time. On a 171,120-frame file
that is hours. Here the video is read ONCE, caching every Nth frame small, and the
walk runs over the cache.

Answers DJ's actual question -- 8 frames or 30? -- with NO clicking required.

Usage:  .venv/Scripts/python.exe spikes/plan_fullgame_chain.py [video_name] [cache_stride]
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
import stage1_keyframe_match as s1                               # noqa: E402

VIDEO_NAME = sys.argv[1] if len(sys.argv) > 1 else "Full_Game.mp4"
VIDEO = os.path.join(_ROOT, VIDEO_NAME)
MIN_RATIO = 0.6           # the project's own weak-pair bar. Not a new knob.
SCALE = 0.35              # match at 672x378 -- same as the coverage sweep
_TAG = os.path.splitext(VIDEO_NAME)[0].upper()
OUT = os.path.join(_HERE, "out", f"{_TAG}_chain_plan.json")


def ratio(a, b):
    with contextlib.redirect_stdout(io.StringIO()):
        kp_a, kp_b, good = s1.detect_and_match(a, b, 0.75, [])
        if len(good) < 15:
            return 0.0
        _H, _m, inl = s1.estimate_homography(kp_a, kp_b, good, 3.0)
    return (inl / len(good)) if inl else 0.0


def main():
    stride = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    cap = cv2.VideoCapture(VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[chain] {total:,} frames / {total/fps/60:.1f} min -- caching every "
          f"{stride} ({total//stride} frames) in ONE pass", flush=True)

    cache, order = {}, []
    i, t0 = 0, time.time()
    while True:
        if i % stride:
            if not cap.grab():
                break
            i += 1
            continue
        ok, fr = cap.read()
        if not ok:
            break
        cache[i] = cv2.resize(fr, None, fx=SCALE, fy=SCALE)
        order.append(i)
        if len(order) % 100 == 0:
            print(f"  cached {len(order)} ({100*i/total:.0f}%)  "
                  f"{time.time()-t0:.0f}s", flush=True)
        i += 1
    cap.release()
    print(f"  cached {len(order)} frames in {time.time()-t0:.0f}s\n", flush=True)

    # ---- greedy chain: from each mark, jump as far as the match still holds ----
    marks = [order[0]]
    pairs = []
    pos = 0
    while pos < len(order) - 1:
        cur = order[pos]
        lo, hi = pos + 1, len(order) - 1
        best = None
        # binary search for the furthest cached frame still matching `cur`.
        # Match quality falls off monotonically with camera movement, so this is
        # sound and costs ~log2(N) SIFT calls instead of N.
        while lo <= hi:
            mid = (lo + hi) // 2
            r = ratio(cache[cur], cache[order[mid]])
            if r >= MIN_RATIO:
                best = (mid, r)
                lo = mid + 1
            else:
                hi = mid - 1
        if best is None:                    # even the very next cached frame fails
            r = ratio(cache[cur], cache[order[pos + 1]])
            best = (pos + 1, r)
            pairs.append({"a": cur, "b": order[pos + 1], "ratio": round(r, 3),
                          "weak": True})
            print(f"  !! {cur} -> {order[pos+1]} ratio {r:.3f} -- NO safe jump; "
                  f"the footage breaks here", flush=True)
        else:
            pairs.append({"a": cur, "b": order[best[0]], "ratio": round(best[1], 3),
                          "weak": False})
        pos = best[0]
        marks.append(order[pos])
        print(f"  mark {len(marks):>3}: frame {order[pos]:>7} "
              f"({order[pos]/fps/60:>5.1f} min)  jumped "
              f"{order[pos]-cur:>6} frames  ratio {pairs[-1]['ratio']:.3f}",
              flush=True)

    gaps = [p["b"] - p["a"] for p in pairs]
    weak = [p for p in pairs if p["weak"]]
    print(f"\n=========== CHAIN PLAN ===========")
    print(f"  MARKED FRAMES NEEDED: {len(marks)}")
    print(f"  -> ~{len(marks)*12} clicks at ~12 landmarks each"
          f"  (~{len(marks)*12*4/60:.0f} min at 4s/click)")
    print(f"  jump sizes: min {min(gaps)}  median {sorted(gaps)[len(gaps)//2]}  "
          f"max {max(gaps)} frames")
    print(f"  every adjacent pair holds ratio >= {MIN_RATIO}: "
          f"{'YES' if not weak else f'NO -- {len(weak)} weak link(s)'}")
    if weak:
        for p in weak:
            print(f"    weak: {p['a']} -> {p['b']}  ratio {p['ratio']}")
    print(f"  resolution: +/- {stride} frames ({stride/fps:.0f}s)")

    json.dump({"video": os.path.basename(VIDEO), "total_frames": total, "fps": fps,
               "cache_stride": stride, "min_ratio": MIN_RATIO,
               "n_marks": len(marks), "marks": marks, "pairs": pairs},
              open(OUT, "w", encoding="utf-8"), indent=2)
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
