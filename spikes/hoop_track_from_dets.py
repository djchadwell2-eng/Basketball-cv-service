"""Build a hoop_track json from the fine-tuned model's own Hoop detections,
instead of carrying a hand-clicked rim through the calibration chain.

WHY THIS EXISTS. spikes/hoop_anchor.py carries two clicked rim pixels through
the pan using the calibration homographies -- which means shot classification
on a NEW clip normally waits on that clip being fully calibrated. TEST 3
measured the alternative (just detect the rim every frame) on the hosted model
and found it CLIP-DEPENDENT: decent on TEST1, effectively blind on HARD
(2/280 frames at conf>=0.4), so it was shelved as "a proposer, not a
replacement".

On TEST4 the same measurement comes out completely differently with v3:
exactly one hoop in 8247/9022 frames at conf>=0.40, median frame-to-frame
movement 0.5px, p99 21px, and ZERO jumps over 100px. That is a usable rim
track, so this clip's holdout gate does not have to wait for calibration.

This is deliberately NOT a general replacement for hoop_anchor -- it is only
valid on a clip where the above numbers have actually been checked. Run
hoop_det_quality() and look before trusting it anywhere new.

Frames with no confident detection get NO hoop position (null), matching
hoop_anchor's own honesty rule: a frame that cannot be resolved is absent,
never guessed from a neighbour.

Run:  .venv/Scripts/python spikes/hoop_track_from_dets.py TEST4
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

CONF = 0.40        # TEST 3's usable-confidence bar; below it the class is junk-flooded


def hoop_det_quality(frames):
    """The numbers that decide whether this approach is safe on a given clip."""
    per, jumps, prev = {}, [], None
    for f in frames:
        d = [x for x in f["detections"] if x["conf"] >= CONF]
        per[len(d)] = per.get(len(d), 0) + 1
        if not d:
            prev = None
            continue
        b = max(d, key=lambda x: x["conf"])["bbox"]
        c = (0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]))
        if prev:
            jumps.append(((c[0] - prev[0]) ** 2 + (c[1] - prev[1]) ** 2) ** 0.5)
        prev = c
    jumps.sort()
    q = lambda p: jumps[int(p * len(jumps))] if jumps else float("nan")
    return {"dets_per_frame": dict(sorted(per.items())),
            "jump_median": q(0.5), "jump_p99": q(0.99),
            "jump_over_100px": sum(1 for j in jumps if j > 100), "n_jumps": len(jumps)}


def main():
    clip = sys.argv[1] if len(sys.argv) > 1 else "TEST4"
    src = os.path.join(_HERE, "out", f"{clip}_hoop_dets_v3.json")
    doc = json.load(open(src, encoding="utf-8"))
    frames = doc["frames"]

    q = hoop_det_quality(frames)
    print(f"[hoop-track] {clip} detection quality at conf>={CONF}:")
    print(f"  dets/frame {q['dets_per_frame']}")
    print(f"  frame-to-frame jump: median {q['jump_median']:.1f}px  "
          f"p99 {q['jump_p99']:.1f}px  over-100px {q['jump_over_100px']}/{q['n_jumps']}")
    if q["jump_over_100px"] > 0.01 * max(q["n_jumps"], 1):
        print("  *** WARNING: unstable rim track -- do NOT use this for shot "
              "classification on this clip; fall back to hoop_anchor. ***")

    out = []
    n_ok = 0
    for f in frames:
        d = [x for x in f["detections"] if x["conf"] >= CONF]
        px = None
        if d:
            b = max(d, key=lambda x: x["conf"])["bbox"]
            px = [0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3])]
            n_ok += 1
        # ONE rim stream: the classifier tries "far" then "near" and keeps the
        # better candidate, so the visible rim goes in one slot and the other
        # stays null. The far/near LABEL is therefore meaningless on a clip
        # built this way -- it says "a rim", not "which end". That is fine for
        # a shot / not-a-shot gate and must not be read as court location.
        out.append({"frame_index": f["frame_index"],
                    "hoop_far_px": px, "hoop_near_px": None})

    dst = os.path.join(_HERE, "out", f"{clip}_hoop_track.json")
    json.dump({"clip": clip, "span_start": 0, "span_len": len(frames),
               "source": "v3 Hoop-class detections (NOT hoop_anchor/calibration)",
               "frames": out}, open(dst, "w", encoding="utf-8"))
    print(f"[hoop-track] rim resolved on {n_ok}/{len(frames)} frames "
          f"-> {os.path.basename(dst)}")


if __name__ == "__main__":
    main()
