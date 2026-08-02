"""WHICH landmark is dragging the court fit? Per-tag error, from the cached refit.

WHY: FULL_GAME2 came back at mean 0.38 ft / max 1.28 ft, against gym #1's
0.21 / 0.56 and a "glued" bar of 0.30. A max of 1.28 ft is past 0.94 ft, which
is what DJ judged BROKEN by eye. A single mis-click and a genuinely harder gym
produce the same summary number, so the summary cannot tell them apart -- the
per-landmark spread can.

Reads the cached .npz refit (no video re-read, no re-solve), so this is seconds.

Usage:  .venv/Scripts/python.exe spikes/diagnose_landmark_fit.py [CLIP_NAME]
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "phase1"))

import clips_config                                               # noqa: E402
clips_config.ACTIVE = sys.argv[1] if len(sys.argv) > 1 else "FULL_GAME2"

import stage2_multikeyframe as s2                                 # noqa: E402
import stage4_courtmap as s4                                      # noqa: E402
import refit_keyframes as rk                                      # noqa: E402


def main():
    KF, ref_pos, Hs, L, tags = rk.refit(use_cache=True)
    H_court, per, mean, mx = s4.compute_H_court(L, tags)

    print(f"\nCLIP: {clips_config.ACTIVE}")
    print(f"court fit: mean {mean:.2f} ft / max {mx:.2f} ft")
    print(f"bars: <=0.30 ft glued | 0.94 ft is what DJ called BROKEN by eye\n")

    print(f"{'landmark':<20}{'error ft':>10}   {'seen in frames':<30}")
    print("-" * 64)
    seen = {}
    for kf in KF:
        for (tag, x, y) in s2.LANDMARKS.get(kf, []):
            seen.setdefault(tag, []).append(kf)
    for tag in sorted(per, key=lambda t: -per[t]):
        n = len(seen.get(tag, []))
        flag = "  <-- WORST" if per[tag] == mx else ""
        print(f"{tag:<20}{per[tag]:>10.2f}   n={n}  {seen.get(tag, [])}{flag}")

    print(f"\nlandmarks clicked in only ONE frame (nothing cross-checks them):")
    lonely = [t for t in per if len(seen.get(t, [])) == 1]
    print(f"  {lonely if lonely else 'none'}")

    print(f"\nper-FRAME landmark counts (thin frames constrain the fit weakly):")
    for kf in KF:
        print(f"  frame {kf:>7}: {len(s2.LANDMARKS.get(kf, []))} marks")

    # --- WHAT-IF: refit the court homography without the uncross-checked tags.
    # NOT a fix and NOT a licence to delete inconvenient data -- the rule is
    # structural (a landmark clicked in ONE frame has nothing to check it
    # against, so an error there is indistinguishable from a mis-click), and it
    # is applied to every n=1 tag, not to whichever tag scored worst.
    if lonely:
        keep = [t for t in tags if t not in lonely]
        _H2, per2, mean2, mx2 = s4.compute_H_court(
            {t: L[t] for t in keep}, keep)
        print(f"\nWHAT-IF -- court homography refit WITHOUT the n=1 tag(s) {lonely}:")
        print(f"  mean {mean:.2f} -> {mean2:.2f} ft     max {mx:.2f} -> {mx2:.2f} ft")
        print(f"  (diagnostic only; the real fix is DJ re-checking that click, "
              f"or clicking it in a second frame so it is cross-checked)")


if __name__ == "__main__":
    main()
