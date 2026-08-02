"""Find a full-resolution-verified bridge chain across ONE broken gap.

WHY THIS EXISTS: the low-res planner (plan_fullgame_chain.py, SCALE=0.35) can
be wrong about which frames actually connect -- proven twice now (TEST 36's
bridge candidate scored 0.781 low-res / 0.056 full-res; Full_Game2's
165000->208800 scored 0.712 low-res / 0.590 full-res). A NAIVE FIX IS ALSO NOT
SAFE: the arithmetic midpoint of a broken gap is not the same as a frame the
camera agrees with -- Full_Game2's midpoint (187000) tested WORSE than the
original single link on BOTH sides (0.266 and 0.354). Frame-number proximity
is not view similarity (established TEST 37); guessing a bridge by eye repeats
that exact mistake.

METHOD: cache candidate frames across the gap at FULL resolution in ONE
sequential video pass, then binary-search-walk forward from the gap's start
using the project's own rule (furthest frame that still holds inlier ratio
>= 0.6, spikes/stage2_multikeyframe.WEAK_INLIER_RATIO) -- same algorithm as
plan_fullgame_chain.py, just at full scale and scoped to one gap so it stays
fast enough to run.

Usage:  .venv/Scripts/python.exe spikes/bridge_gap_fullres.py <video_name> <gap_start> <gap_end> [stride]
"""

from __future__ import annotations

import os
import sys
import time

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
import stage1_keyframe_match as s1                               # noqa: E402

MIN_RATIO = 0.6            # the project's own weak-pair bar, not a new knob.
SEED = 0                   # matches stage2_multikeyframe.py -- RANSAC is random;
                            # every pairwise call must reseed or results drift
                            # run to run (caught 2026-07-30: same 165000->208800
                            # pair scored 0.590 seeded vs 0.659 unseeded here)

# Kept in sync with verify_chain_fullres.py's per-video lookup by hand -- only
# two entries exist right now, not worth a shared module yet.
EXCLUDE_REGIONS = {
    "FULL_GAME": [(0.0, 830.0, 330.0, 1080.0)],
    "FULL_GAME2": [(0.0, 870.0, 340.0, 1080.0)],
}


def ratio(earlier, later, regions):
    """Inlier ratio for an adjacent pair, MEASURED EXACTLY AS THE CALIBRATION
    PIPELINE MEASURES IT.

    SIFT matching is NOT symmetric -- the ratio test runs over query->train
    matches, so swapping the two frames gives a different match set and a
    different inlier ratio. stage2_multikeyframe.adjacent_homographies (the
    function whose WEAK PAIR FLAG is the project's actual gate, and whose
    output feeds refit_keyframes) calls detect_and_match(LATER, EARLIER).

    Caught 2026-07-30: calling it (earlier, later) instead scored the same
    Full_Game2 pair 165000->208800 at 0.659 while the real gate scored it
    0.590 -- an optimistic false PASS on a pair that actually fails. Always
    mirror stage2's argument order here, or this tool measures a number the
    pipeline never uses.
    """
    cv2.setRNGSeed(SEED)
    kp_l, kp_e, good = s1.detect_and_match(later, earlier, 0.75, regions)
    if len(good) < 15:
        return 0.0
    _H, _m, inl = s1.estimate_homography(kp_l, kp_e, good, 3.0)
    return (inl / len(good)) if inl else 0.0


def main():
    video_name = sys.argv[1]
    gap_start = int(sys.argv[2])
    gap_end = int(sys.argv[3])
    stride = int(sys.argv[4]) if len(sys.argv) > 4 else 2000
    video_path = os.path.join(_ROOT, video_name)
    tag = os.path.splitext(video_name)[0].upper()
    regions = EXCLUDE_REGIONS.get(tag, [])

    need = sorted(set([gap_start] + list(range(gap_start, gap_end, stride)) + [gap_end]))
    print(f"[bridge] video: {video_path}")
    print(f"[bridge] gap: {gap_start} -> {gap_end} ({gap_end-gap_start:,} frames)")
    print(f"[bridge] exclude_regions: {regions}")
    print(f"[bridge] caching {len(need)} FULL-RESOLUTION candidates "
          f"(stride {stride}) in one pass...", flush=True)

    t0 = time.time()
    cap = cv2.VideoCapture(video_path)
    cache = {}
    it = iter(need)
    nxt, idx = next(it), 0
    while nxt is not None:
        if not cap.grab():
            break
        if idx == nxt:
            ok, fr = cap.retrieve()
            if ok:
                cache[idx] = fr
            nxt = next(it, None)
        idx += 1
    cap.release()
    missing = [i for i in need if i not in cache]
    print(f"[bridge] cached {len(cache)}/{len(need)} in {time.time()-t0:.0f}s"
          + (f"  MISSING: {missing}" if missing else ""), flush=True)
    order = [i for i in need if i in cache]

    # ---- same greedy chain walk as plan_fullgame_chain.py, at full res ----
    marks = [order[0]]
    pairs = []
    pos = 0
    while pos < len(order) - 1:
        cur = order[pos]
        lo, hi = pos + 1, len(order) - 1
        best = None
        while lo <= hi:
            mid = (lo + hi) // 2
            r = ratio(cache[cur], cache[order[mid]], regions)
            if r >= MIN_RATIO:
                best = (mid, r)
                lo = mid + 1
            else:
                hi = mid - 1
        if best is None:
            r = ratio(cache[cur], cache[order[pos + 1]], regions)
            best = (pos + 1, r)
            pairs.append({"a": cur, "b": order[pos + 1], "ratio": round(r, 3), "weak": True})
            print(f"  !! {cur} -> {order[pos+1]} ratio {r:.3f} -- NO safe jump "
                  f"even at stride {stride}; try a smaller stride", flush=True)
        else:
            pairs.append({"a": cur, "b": order[best[0]], "ratio": round(best[1], 3), "weak": False})
        pos = best[0]
        marks.append(order[pos])
        print(f"  mark {len(marks):>2}: frame {order[pos]:>7}  jumped "
              f"{order[pos]-cur:>6}  ratio {pairs[-1]['ratio']:.3f}", flush=True)

    print(f"\n=========== BRIDGE RESULT ===========")
    print(f"  bridge frames needed: {marks}")
    weak = [p for p in pairs if p["weak"]]
    print(f"  every link >= {MIN_RATIO}: {'YES' if not weak else f'NO -- {len(weak)} weak'}")
    if marks[-1] != gap_end:
        print(f"  NOTE: walk landed on {marks[-1]}, not exactly {gap_end} -- "
              f"the last hop ({marks[-1]} -> {gap_end}) still needs checking.")


if __name__ == "__main__":
    main()
