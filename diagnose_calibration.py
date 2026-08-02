"""WHY DOES THE OVERLAY LOOK WRONG WHEN THE NUMBER LOOKS GOOD?

The reported "court fit" is measured on the CONSOLIDATED landmark positions --
one averaged position per landmark, after the optimiser has pulled every
keyframe's view of it together. That number can be excellent while an INDIVIDUAL
keyframe's transform is badly wrong, because the consolidation hides it: the
error lives in the disagreement between keyframes, not in the average.

The overlay renders each frame through whichever KEYFRAME it matches best. So a
single bad keyframe transform paints a wrecked court on every frame anchored to
it -- while frames anchored to good keyframes stay glued. That is exactly the
symptom: first and last fine, middles wrong.

This measures the thing that actually matters: for EACH keyframe, project ITS
OWN clicked marks through ITS OWN transform to court feet, and compare against
the court model. No averaging, no hiding.

Usage:  .venv/Scripts/python.exe diagnose_calibration.py <CLIP_NAME>
"""

from __future__ import annotations

import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (_ROOT, os.path.join(_ROOT, "spikes"), os.path.join(_ROOT, "phase1")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import clip_registry                                              # noqa: E402


def main():
    name = sys.argv[1]
    doc = clip_registry.load(name)

    import clips_config
    clips_config._merge_registry_clips()
    clips_config.CLIPS[name] = clip_registry.to_calibration_entry(doc)
    clips_config.ACTIVE = name
    clips_config._RESOLVED.pop(name, None)

    import stage2_multikeyframe as s2
    import stage3_optimize as s3
    import stage4_courtmap as s4
    import refit_keyframes as rk

    KF, ref_pos, Hs, L, tags = rk.refit(use_cache=True)
    H_court, per, mean_ft, max_ft = s4.compute_H_court(L, tags)

    print(f"\nCLIP {name}")
    print(f"reference keyframe: position {ref_pos} = frame {KF[ref_pos]}")
    print(f"REPORTED court fit (consolidated landmarks): "
          f"mean {mean_ft:.2f} ft / max {max_ft:.2f} ft")
    print("\nPER-KEYFRAME TRUTH -- each keyframe's own marks through its own "
          "transform:\n")
    print(f"{'keyframe':>10} {'marks':>6} {'mean ft':>9} {'max ft':>8}   verdict")
    print("-" * 62)

    worst = []
    for pos, kf in enumerate(KF):
        marks = s2.LANDMARKS.get(kf, [])
        errs = []
        for (tag, x, y) in marks:
            if tag not in s4.COURT_MODEL:
                continue
            # keyframe px -> reference px -> court feet
            rx, ry = s3.project(Hs[pos], x, y)
            p = H_court @ np.array([rx, ry, 1.0])
            if abs(p[2]) < 1e-12:
                continue
            fx, fy = p[0] / p[2], p[1] / p[2]
            tx, ty = s4.COURT_MODEL[tag]
            errs.append(((fx - tx) ** 2 + (fy - ty) ** 2) ** 0.5)
        if not errs:
            print(f"{kf:>10} {0:>6}        --       --   no usable marks")
            continue
        m, mx = float(np.mean(errs)), float(np.max(errs))
        verdict = ("GLUED" if m <= 0.30 else
                   "usable" if m < 0.94 else "*** BROKEN ***")
        print(f"{kf:>10} {len(errs):>6} {m:>9.2f} {mx:>8.2f}   {verdict}")
        worst.append((m, kf))

    worst.sort(reverse=True)
    if worst and worst[0][0] > 0.94:
        print(f"\nThe overlay will look wrong on every frame that anchors to "
              f"keyframe {worst[0][1]} (and any other BROKEN row).")
        print("The reported number does not see this, because it averages the "
              "keyframes together first.")


if __name__ == "__main__":
    main()
