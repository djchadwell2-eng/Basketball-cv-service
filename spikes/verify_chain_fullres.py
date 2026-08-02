"""THE CHECK THAT GOT SKIPPED LAST TIME (TEST 36). Before asking DJ to click
ANYTHING, verify a candidate keyframe chain the REAL way: full-resolution
frames, scorebug/graphic masked, using the project's own guardrail function
(stage2_multikeyframe.adjacent_homographies, WEAK_INLIER_RATIO=0.6) --
not the 35%-scale proxy in plan_fullgame_chain.py that gave a confident wrong
answer twice (TEST 36, Part 3 error #2).

GENERALIZED 2026-07-30 to work on any game, not just Full_Game.mp4: reads the
candidate marks straight from the chain-plan JSON that
spikes/plan_fullgame_chain.py already wrote (spikes/out/{TAG}_chain_plan.json),
and looks up the graphic mask per video below (checked by eye each time -- a
DIFFERENT recording/streaming setup can put its overlay in a different spot,
so a mask must never be assumed, only confirmed).

Does NOT touch clips_config.py -- passes the candidate frame list straight
into the existing function. Zero risk to any already-clicked mark set.

Usage:  .venv/Scripts/python.exe spikes/verify_chain_fullres.py [video_name]
"""

from __future__ import annotations

import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)

import clips_config                                               # noqa: E402
clips_config.ACTIVE = "FULL_GAME"        # any entry -- only used as an import anchor
import stage2_multikeyframe as s2                                 # noqa: E402

VIDEO_NAME = sys.argv[1] if len(sys.argv) > 1 else "Full_Game.mp4"
VIDEO_PATH = os.path.join(_ROOT, VIDEO_NAME)
_TAG = os.path.splitext(VIDEO_NAME)[0].upper()

# Per-video graphic/scorebug mask, CHECKED BY EYE (spikes/out/*_frame*_check.jpg)
# each time a new recording is added -- never copied from another video on the
# assumption it's "probably the same corner".
EXCLUDE_REGIONS = {
    "FULL_GAME": [(0.0, 830.0, 330.0, 1080.0)],   # bottom-left scorebug
    "FULL_GAME2": [(0.0, 870.0, 340.0, 1080.0)],  # bottom-left video-player overlay
}.get(_TAG, [])

CHAIN_PLAN = os.path.join(_HERE, "out", f"{_TAG}_chain_plan.json")
OUT = os.path.join(_HERE, "out", f"{_TAG}_chain_verify_fullres.json")


def main():
    plan = json.load(open(CHAIN_PLAN, encoding="utf-8"))
    candidate = plan["marks"]
    print(f"[verify] video: {VIDEO_PATH}")
    print(f"[verify] exclude_regions (graphic mask): {EXCLUDE_REGIONS}"
          + ("  <-- WARNING: none on file for this video, check for one!"
             if not EXCLUDE_REGIONS else ""))
    print(f"[verify] candidate chain (from {CHAIN_PLAN}): {candidate}")
    print(f"[verify] weak-pair bar: ratio >= {s2.WEAK_INLIER_RATIO} (the project's "
          f"own rule, not a new one)\n")

    t0 = time.time()
    print("[verify] extracting frames at FULL resolution (frame-accurate, single "
          "sequential pass -- this is the slow, correct read, not a seek)...",
          flush=True)
    frames = s2.extract_frames(VIDEO_PATH, candidate)
    print(f"[verify] extracted {len(frames)} frames in {time.time()-t0:.0f}s\n",
          flush=True)

    fwd, stats, weak = s2.adjacent_homographies(frames, candidate, EXCLUDE_REGIONS)

    print("\n=========== FULL-RESOLUTION CHAIN CHECK ===========")
    for st in stats:
        flag = "  <-- WEAK" if st["ratio"] < s2.WEAK_INLIER_RATIO else "  OK"
        print(f"  {st['a']:>7} -> {st['b']:<7}  {st['matches']:>4} matches  "
              f"{st['inliers']:>4} inliers  ratio {st['ratio']:.3f}{flag}")

    verdict = "HOLDS -- every link >= 0.6 at full resolution" if not weak else \
              f"BROKEN -- {len(weak)} weak link(s), same failure mode as TEST 36"
    print(f"\n  VERDICT: {verdict}")

    json.dump({"video": VIDEO_NAME, "candidate": candidate,
               "exclude_regions": EXCLUDE_REGIONS,
               "weak_inlier_ratio_bar": s2.WEAK_INLIER_RATIO, "stats": stats,
               "weak": weak, "verdict": verdict},
              open(OUT, "w", encoding="utf-8"), indent=2)
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
