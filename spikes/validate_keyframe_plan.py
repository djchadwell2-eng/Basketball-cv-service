"""DOES THE INLIER-RATIO RULE ACTUALLY PREDICT COURT ACCURACY?

plan_keyframes.py proposes marks by keeping the largest jump whose SIFT inlier
ratio holds >= 0.6. That is a hypothesis, and it FAILED its pre-stated check on
first run (TEST 27 said HARD needs marks roughly every 100 frames; the planner
happily jumped 450 frames on HARD at ratio 0.863). Either the ratio is not
measuring what accuracy needs, or the thinning subsets I picked by hand were
just unlucky. This decides it.

METHOD -- the planner is restricted to frames DJ has ALREADY CLICKED, so its
choice can be scored by the same true holdout as TEST 27 (refit without the
dropped marks, SIFT-hop from the nearest kept mark, project the held-back marks
against the rulebook). Proposing unclicked frames would be unfalsifiable: there
would be no ground truth at those frames to check against.

VERDICT RULE, stated before the run: the inlier rule is USEFUL only if the
subset it chooses scores as well as the hand-picked subsets of the same size in
TEST 27. If it chooses an equally-thin subset that scores WORSE, the ratio is
not a proxy for court accuracy and this whole approach is dead.

Usage:  .venv/Scripts/python.exe spikes/validate_keyframe_plan.py HARD
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "phase1")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                              # noqa: E402

import plan_keyframes as pk                                     # noqa: E402


def walk_existing(kf_list, frames, s1, exclude, min_ratio=pk.WEAK_PAIR_MIN):
    """The same greedy rule, but only allowed to land on keyframes DJ clicked.
    Returns (chosen, probe_log)."""
    chosen, log = [kf_list[0]], []
    i = 0
    while i < len(kf_list) - 1:
        pick = None
        for j in range(len(kf_list) - 1, i, -1):        # furthest first
            r = pk.match_ratio(frames[kf_list[i]], frames[kf_list[j]], s1, exclude)
            log.append((kf_list[i], kf_list[j], round(r, 3), r >= min_ratio))
            if r >= min_ratio:
                pick = j
                break
        if pick is None:
            pick = i + 1                                 # nothing holds: step one
        chosen.append(kf_list[pick])
        i = pick
    return chosen, log


def main():
    clip = sys.argv[1] if len(sys.argv) > 1 else "HARD"
    import clips_config
    clips_config.ACTIVE = clip
    import clip_config
    clip_config.ACTIVE_CLIP = getattr(clip_config, f"{clip}_CLIP")

    import stage1_keyframe_match as s1
    import stage2_multikeyframe as s2
    import stage3_optimize as s3
    import stage4_courtmap as s4
    import refit_keyframes as rk
    from keyframe_thinning_test import evaluate

    all_kf = list(s2.KEYFRAMES)
    all_marks = sum(len(s2.LANDMARKS.get(k, [])) for k in all_kf)
    frames = s2.extract_frames(s2.VIDEO_PATH, all_kf)

    print(f"\nVALIDATING THE INLIER RULE -- {clip}")
    print(f"  clicked keyframes: {all_kf}  ({all_marks} marks)")
    chosen, log = walk_existing(all_kf, frames, s1, s2.EXCLUDE_REGIONS)
    print(f"  probes (furthest-first, keep ratio >= {pk.WEAK_PAIR_MIN}):")
    for (a, b, r, ok) in log:
        print(f"    {a:>5} -> {b:<5}  ratio {r:.3f}{'   KEPT' if ok else ''}")
    print(f"  RULE CHOOSES: {chosen}  ({len(chosen)} of {len(all_kf)} marks)")

    res, n_used, fit_mean = evaluate(chosen, all_kf, frames, s1, s2, s3, s4, rk)
    saved = 100 * (1 - n_used / all_marks)
    print(f"\n  clicks used {n_used}/{all_marks} ({saved:.0f}% fewer)   "
          f"in-sample court fit {fit_mean:.2f} ft")
    worst = 0.0
    for d in sorted(res):
        r = res[d]
        if r.get("failed"):
            print(f"    kf {d:>5}: SIFT MATCH FAILED")
            worst = float("inf")
            continue
        worst = max(worst, r["mean_ft"])
        print(f"    kf {d:>5}: held-back marks land {r['mean_ft']:.2f} ft off "
              f"(max {r['max_ft']:.2f})  [{r['gap']} frames from kf {r['anchor_kf']}]")
    verdict = ("PASS -- the ratio rule picked a safe subset" if worst <= 0.5 else
               "MARGINAL" if worst <= 1.0 else "FAIL -- ratio does NOT predict accuracy")
    print(f"\n  WORST held-out keyframe: {worst:.2f} ft   => {verdict}")
    print(f"  (bar: 0.29 ft is the glued benchmark; 0.94 ft is what DJ called "
          f"BROKEN by eye)")


if __name__ == "__main__":
    main()
