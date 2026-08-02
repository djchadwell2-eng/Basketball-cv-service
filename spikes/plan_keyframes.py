"""SMART KEYFRAME SPACING -- tell DJ WHERE to click, and what it will COST,
before he clicks anything.

THE PROBLEM (TEST 27, 2026-07-29): marking every ~100 frames is a workflow
convention, never a measured requirement. Thinning proved it is far too dense
on calm footage (TEST1 tolerated 200-460 frame hops at 0.33 ft) and about right
on fast-panning footage (HARD was already marginal at 100 frames, 0.99 ft). A
fixed spacing therefore over-charges the easy clips and barely covers the hard
ones.

THE RULE, and it invents no new number: keep the largest jump whose SIFT match
back to the previous mark still holds an inlier ratio >= WEAK_PAIR_MIN. That
threshold (0.6) is not mine -- it is the weak-pair guardrail already coded in
spikes/stage2_multikeyframe.py, and it fired correctly on both clips in TEST 27
(HARD 600->1200 at 0.039, TEST1 120->580 at 0.317).

WHAT THIS IS NOT: it is not auto-calibration. DJ still clicks the landmarks. It
only decides WHICH FRAMES are worth clicking, and says so up front.

THE HONEST HEADLINE it exists to deliver: a quarter of a game is ~14,400
frames. At today's spacing that is ~144 marked frames, ~1,440 clicks. Even a
best case of 300-frame spacing is ~48 frames and ~480 clicks -- roughly 40
minutes of solid clicking. This tool does not make a quarter cheap. It makes
the price VISIBLE BEFORE the work, and never asks for a mark the footage does
not need.

Usage (one clip per process):
    .venv/Scripts/python.exe spikes/plan_keyframes.py TEST1
    .venv/Scripts/python.exe spikes/plan_keyframes.py TEST1 --start 120 --end 580
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The project's OWN weak-pair threshold (stage2_multikeyframe.py). Not a new
# knob -- if that guardrail moves, this moves with it.
WEAK_PAIR_MIN = 0.6

# Jumps to probe, largest first: keep the biggest that holds. Coarse on purpose
# -- the point is to find roughly how far the footage carries, not to squeeze
# out single frames, and every probe costs a SIFT match.
PROBE_JUMPS = (450, 350, 250, 175, 125, 90, 60, 40)

CLICKS_PER_MARK = 10          # observed: TEST1 58 marks/6 frames, HARD 59/7


def match_ratio(img_a, img_b, s1, exclude_regions):
    """RANSAC inlier ratio between two frames -- the project's own health
    measure for a keyframe pair. 0.0 when there is nothing to match.

    The matchers print per-call diagnostics unconditionally, which would bury
    the walk under hundreds of lines. Their output is swallowed HERE rather
    than by adding a flag to shared calibration code that other stages depend
    on -- the probe is this module's concern, not theirs."""
    import contextlib
    import io

    import cv2
    cv2.setRNGSeed(0)
    with contextlib.redirect_stdout(io.StringIO()):
        kp_a, kp_b, good = s1.detect_and_match(img_a, img_b, 0.75, exclude_regions)
        if len(good) < 10:
            return 0.0
        _H, _mask, inliers = s1.estimate_homography(kp_a, kp_b, good, 3.0)
    return (inliers / len(good)) if inliers else 0.0


def plan(video_path, start, end, exclude_regions, s1, s2,
         min_ratio=WEAK_PAIR_MIN, jumps=PROBE_JUMPS, verbose=True):
    """Greedy walk -> (marks, pairs). From the current mark, probe jumps
    largest-first and keep the first that holds min_ratio. If even the smallest
    jump fails, the footage is degenerate there: place the smallest jump anyway
    and FLAG it rather than silently pretending the pair is healthy."""
    cache = {}

    def frame(i):
        if i not in cache:
            cache[i] = s2.extract_frames(video_path, [i])[i]
        return cache[i]

    marks, pairs = [start], []
    cur = start
    while cur < end:
        chosen, chosen_ratio, flagged = None, 0.0, False
        for j in jumps:
            nxt = min(cur + j, end)
            if nxt <= cur:
                continue
            r = match_ratio(frame(cur), frame(nxt), s1, exclude_regions)
            if verbose:
                print(f"    probe {cur:>6} -> {nxt:<6} (+{nxt - cur:>4})  "
                      f"ratio {r:.3f}{'  OK' if r >= min_ratio else ''}",
                      flush=True)
            if r >= min_ratio:
                chosen, chosen_ratio = nxt, r
                break
        if chosen is None:                       # even the tightest jump fails
            chosen = min(cur + jumps[-1], end)
            chosen_ratio = match_ratio(frame(cur), frame(chosen), s1, exclude_regions)
            flagged = True
        pairs.append({"a": cur, "b": chosen, "gap": chosen - cur,
                      "ratio": round(chosen_ratio, 3), "weak": flagged})
        marks.append(chosen)
        cur = chosen
    return marks, pairs


def report(clip, marks, pairs, start, end, fps=30.0):
    gaps = [p["gap"] for p in pairs]
    weak = [p for p in pairs if p["weak"]]
    n_clicks = len(marks) * CLICKS_PER_MARK
    dur = (end - start) / fps
    print(f"\n=========== KEYFRAME PLAN -- {clip} ===========")
    print(f"  span {start}..{end}  ({dur:.1f}s at {fps:.0f}fps)")
    print(f"  PROPOSED MARKS: {len(marks)}   -> ~{n_clicks} clicks "
          f"(~{CLICKS_PER_MARK} landmarks each)")
    print(f"  spacing: min {min(gaps)}  median {sorted(gaps)[len(gaps) // 2]}  "
          f"max {max(gaps)} frames")
    fixed = int(round((end - start) / 100.0)) + 1
    delta = 100.0 * (len(marks) - fixed) / fixed
    word = "FEWER" if delta < 0 else ("MORE" if delta > 0 else "SAME")
    print(f"  vs today's every-100-frames convention: {fixed} marks "
          f"(~{fixed * CLICKS_PER_MARK} clicks)  =>  "
          f"{abs(delta):.0f}% {word} clicks")
    if weak:
        print(f"  !! {len(weak)} pair(s) could NOT reach ratio {WEAK_PAIR_MIN} "
              f"even at the tightest jump -- the footage is hard there "
              f"(fast pan, glare, or a cut). Calibration will be weak across:")
        for p in weak:
            print(f"       {p['a']}..{p['b']}  ratio {p['ratio']}")
    print(f"  marks: {marks}")
    return {"clip": clip, "marks": marks, "pairs": pairs,
            "estimated_clicks": n_clicks, "weak_pairs": len(weak)}


def main():
    import json
    args = sys.argv[1:]
    clip = args[0] if args and not args[0].startswith("--") else "TEST1"
    import clips_config
    clips_config.ACTIVE = clip
    import clip_config
    cfg = getattr(clip_config, f"{clip}_CLIP")
    clip_config.ACTIVE_CLIP = cfg

    import stage1_keyframe_match as s1
    import stage2_multikeyframe as s2

    start = cfg.tracking_span_start
    end = cfg.tracking_span_start + cfg.tracking_span_len - 1
    if "--start" in args:
        start = int(args[args.index("--start") + 1])
    if "--end" in args:
        end = int(args[args.index("--end") + 1])

    print(f"[plan_keyframes] {clip}: walking {start}..{end}, "
          f"keeping the largest jump with inlier ratio >= {WEAK_PAIR_MIN}")
    marks, pairs = plan(s2.VIDEO_PATH, start, end, s2.EXCLUDE_REGIONS, s1, s2)
    doc = report(clip, marks, pairs, start, end)

    out = os.path.join(_HERE, "out", f"{clip}_keyframe_plan.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
