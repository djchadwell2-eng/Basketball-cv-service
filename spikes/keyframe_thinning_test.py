"""HOW MUCH CLICKING IS ACTUALLY NECESSARY? -- a true holdout on the marks.

THE PROBLEM THIS ANSWERS (measured 2026-07-28): calibration clicking is the
binding constraint on this whole project, not accuracy.
    TEST1  58 clicked marks -> 15.4s of usable footage
    HARD   59 marks         -> 20.0s
    TEST2  70 marks         -> 12.0s
About ten clicks per three seconds. A 2-minute clip would cost ~360 clicks and
a full game ~5,700. Every stat the product wants needs LENGTH, and length is
gated by DJ's thumbs.

THE QUESTION: the live system already carries the court between marked frames
by SIFT-matching each frame to its NEAREST keyframe. Nobody has ever measured
how FAR one keyframe can carry. If a mark can cover 200 frames instead of 100,
the clicking halves.

THE TEST (a real holdout, not a self-check -- the LOO discipline the court-
dimension work established after the per-keyframe overfit):
    1. Refit using only a SUBSET of the keyframes. Dropping a keyframe from
       s2.KEYFRAMES automatically drops its clicked marks from the fit, so the
       held-back marks are genuinely unseen.
    2. For each DROPPED keyframe, do exactly what the live path does: SIFT-match
       its image to the nearest KEPT keyframe and compose the transform.
    3. Project that keyframe's OWN clicked marks to court feet and compare with
       the rulebook position of each landmark (stage4_courtmap.COURT_MODEL).
    -> "if DJ had not clicked this frame, how far off would the court be there?"

THE BAR, from this project's own history: TEST2 at 0.94 ft was judged BROKEN by
eye; 0.29 ft is the "glued" benchmark. So under ~0.5 ft is the target and
anything approaching 1 ft is a failure.

HONEST NOTE ON THE COMPARISON: the holdout number includes SIFT-hop error that
the shipped in-sample figure (TEST1: 0.15 ft) does not. It is an upper bound on
the cost of skipping a mark, not a like-for-like reading.

Usage (one clip per process):
    .venv/Scripts/python.exe spikes/keyframe_thinning_test.py TEST1
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "phase1")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _norm(H):
    return H / H[2, 2]


def evaluate(subset, all_kf, frames, s1, s2, s3, s4, rk):
    """Fit on `subset` only, then score every dropped keyframe's held-back
    marks. Returns (per_keyframe_errors, n_marks_used, court_fit_mean)."""
    s2.KEYFRAMES = list(subset)
    s2.REFERENCE_POS = None                     # -> n//2, always in range

    KF, ref_pos, Hs0, L0, tags, obs, corr = rk._setup()
    Hs, L, _res = rk._solve(KF, ref_pos, Hs0, L0, tags, obs, corr, rk.CORR_WEIGHT)
    H_court, _per, fit_mean, _fit_max = s4.compute_H_court(L, tags)

    kept = np.array(KF)
    out = {}
    for d in [k for k in all_kf if k not in subset]:
        k = int(kept[np.argmin(np.abs(kept - d))])      # nearest KEPT keyframe
        cv2.setRNGSeed(0)
        kp_k, kp_d, good = s1.detect_and_match(frames[k], frames[d],
                                               0.75, s2.EXCLUDE_REGIONS)
        H, mask, inl = s1.estimate_homography(kp_k, kp_d, good, 3.0)
        if H is None or not inl:
            out[d] = {"anchor_kf": k, "failed": True}
            continue
        T = _norm(Hs[KF.index(k)] @ H)                  # dropped px -> reference px
        M = H_court @ T                                 # dropped px -> court feet

        errs = []
        for (tag, x, y) in s2.LANDMARKS.get(d, []):
            if tag not in s4.COURT_MODEL:
                continue
            p = M @ np.array([float(x), float(y), 1.0])
            fx, fy = p[0] / p[2], p[1] / p[2]
            tx, ty = s4.COURT_MODEL[tag]
            errs.append(float(np.hypot(fx - tx, fy - ty)))
        out[d] = {"anchor_kf": k, "gap": abs(d - k), "n_marks": len(errs),
                  "mean_ft": float(np.mean(errs)) if errs else None,
                  "max_ft": float(np.max(errs)) if errs else None,
                  "inliers": int(inl)}
    n_used = sum(len(s2.LANDMARKS.get(k, [])) for k in subset)
    return out, n_used, fit_mean


def main():
    clip = sys.argv[1] if len(sys.argv) > 1 else "TEST1"
    import clips_config
    clips_config.ACTIVE = clip
    import clip_config
    clip_config.ACTIVE_CLIP = getattr(clip_config, f"{clip}_CLIP")

    import stage1_keyframe_match as s1
    import stage2_multikeyframe as s2
    import stage3_optimize as s3
    import stage4_courtmap as s4
    import refit_keyframes as rk

    all_kf = list(s2.KEYFRAMES)
    all_marks = sum(len(s2.LANDMARKS.get(k, [])) for k in all_kf)
    frames = s2.extract_frames(s2.VIDEO_PATH, all_kf)

    print(f"\nKEYFRAME THINNING TEST -- {clip}")
    print(f"  all keyframes: {all_kf}   total clicked marks: {all_marks}")
    print(f"  bar: under ~0.5 ft is fine; ~1 ft is the error DJ called BROKEN "
          f"on TEST2.\n")

    # Subsets to try, endpoints always kept (the span cannot extrapolate past
    # its ends). Declared here BEFORE any result is seen.
    ends = [all_kf[0], all_kf[-1]]
    mid = all_kf[1:-1]
    subsets = [("every 2nd (half the clicks)", ends[:1] + mid[1::2] + ends[1:]),
               ("ends + middle only",          ends[:1] + [mid[len(mid) // 2]] + ends[1:]),
               ("ends only (extreme)",         ends)]

    for label, sub in subsets:
        sub = sorted(set(sub))
        res, n_used, fit_mean = evaluate(sub, all_kf, frames, s1, s2, s3, s4, rk)
        saved = 100 * (1 - n_used / all_marks)
        print(f"--- {label}: keep {sub}")
        print(f"    clicks used {n_used}/{all_marks}  ({saved:.0f}% FEWER CLICKS)"
              f"   in-sample court fit {fit_mean:.2f} ft")
        worst = 0.0
        for d in sorted(res):
            r = res[d]
            if r.get("failed"):
                print(f"    kf {d:>4}: SIFT MATCH FAILED against kf {r['anchor_kf']}")
                worst = float("inf")
                continue
            worst = max(worst, r["mean_ft"])
            print(f"    kf {d:>4}: held-back marks land {r['mean_ft']:.2f} ft off "
                  f"(max {r['max_ft']:.2f})  [{r['n_marks']} marks, "
                  f"{r['gap']} frames from kf {r['anchor_kf']}, {r['inliers']} inliers]")
        verdict = ("PASS -- clicks can be cut here" if worst <= 0.5 else
                   "MARGINAL" if worst <= 1.0 else "FAIL -- too far off")
        print(f"    WORST held-out keyframe: {worst:.2f} ft   => {verdict}\n")


if __name__ == "__main__":
    main()
