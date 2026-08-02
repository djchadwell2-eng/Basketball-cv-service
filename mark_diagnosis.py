"""WHICH CLICK IS WRONG? -- pinpoint the bad mark, not just the bad frame.

WHY. When the court cannot be solved, the coach got a number ("off by 2.88 ft")
and no way to act on it. Naming the frame helped; naming the actual CLICK is
what they need, because on a 69-mark set they cannot tell which two of their
own points are the problem by looking.

HOW. Two cheap experiments, in the order the real failure occurred:

  1. SWAPPED PAIR. Try exchanging each UPPER/LOWER (and LEFT/RIGHT) pair within
     one frame. If the court suddenly solves, those two are the wrong way round.
     This is FIRST because it is the failure that actually happened: DJ clicked
     both half-court sideline points in the right places on frame 600 and simply
     assigned them to each other, and a swap took the same 69 marks from "no
     court fits" to 0.20 ft.
     A swap is invisible to leave-one-out: removing ONE of two exchanged points
     leaves the other still wrong, so the court still will not solve.

  2. ONE BAD POINT. Try removing each mark on its own. If the court then solves,
     that single click is misplaced.

Nothing here CHANGES a coach's marks. It reports what it found and lets them
decide -- a swap is easy to be sure about, but silently rewriting someone's
input is exactly the kind of confident-wrong behaviour this project refuses.
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (_ROOT, os.path.join(_ROOT, "spikes")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import court_detect                                               # noqa: E402

# Pairs a person can plausibly interchange, because they are the same feature at
# opposite ends of one line.
SWAPPABLE = [
    ("center_near", "center_far"),
    ("circle_bottom", "circle_top"),
    ("circle_left", "circle_right"),
    ("L_FT_near", "L_FT_far"),
    ("R_FT_near", "R_FT_far"),
    ("L_lane_base_near", "L_lane_base_far"),
    ("R_lane_base_near", "R_lane_base_far"),
    ("LB_side_near", "LB_side_far"),
    ("RB_side_near", "RB_side_far"),
]

# Plain English for the app. Mirrors components/CourtMarker.tsx, which is what
# the coach actually read when they placed the point.
LABEL = {
    "center_near": "Half-court line meets the UPPER sideline",
    "center_far": "Half-court line meets the LOWER sideline",
    "circle_bottom": "Center circle — UPPER edge",
    "circle_top": "Center circle — LOWER edge",
    "circle_left": "Center circle — LEFT edge",
    "circle_right": "Center circle — RIGHT edge",
    "center_logo": "Center of the center circle",
    "L_FT_near": "LEFT free-throw line — UPPER corner",
    "L_FT_far": "LEFT free-throw line — LOWER corner",
    "R_FT_near": "RIGHT free-throw line — UPPER corner",
    "R_FT_far": "RIGHT free-throw line — LOWER corner",
    "L_lane_base_near": "LEFT end line — UPPER corner of the key",
    "L_lane_base_far": "LEFT end line — LOWER corner of the key",
    "R_lane_base_near": "RIGHT end line — UPPER corner of the key",
    "R_lane_base_far": "RIGHT end line — LOWER corner of the key",
    "LB_side_near": "LEFT end line meets the UPPER sideline",
    "LB_side_far": "LEFT end line meets the LOWER sideline",
    "RB_side_near": "RIGHT end line meets the UPPER sideline",
    "RB_side_far": "RIGHT end line meets the LOWER sideline",
    "L_arc_top": "Top of the LEFT 3-point arc",
    "R_arc_top": "Top of the RIGHT 3-point arc",
}


def label(tag):
    return LABEL.get(tag, tag)


def _swap_in_frame(marks, a, b):
    return [((b if t == a else a if t == b else t), x, y) for (t, x, y) in marks]


def diagnose(landmarks):
    """Return the best explanation for an unsolvable mark set, or None.

    landmarks: {frame:int -> [(tag, x, y), ...]}
    """
    base = court_detect.identify(landmarks)
    if base["identified"]:
        return None

    # --- 1. a swapped pair -------------------------------------------------
    best = None
    for frame, marks in landmarks.items():
        tags = {t for (t, _x, _y) in marks}
        for a, b in SWAPPABLE:
            if a not in tags or b not in tags:
                continue
            trial = dict(landmarks)
            trial[frame] = _swap_in_frame(marks, a, b)
            r = court_detect.identify(trial)
            if r["identified"] and (best is None or r["error_ft"] < best["fit_ft"]):
                best = {
                    "kind": "swapped_pair",
                    "frame": frame,
                    "tags": [a, b],
                    "labels": [label(a), label(b)],
                    "fit_ft": round(float(r["error_ft"]), 2),
                    "message": (
                        f'"{label(a)}" and "{label(b)}" look swapped with each other '
                        f"on this frame — the two points are in the right places, but "
                        f"assigned to each other. Fixing them makes the whole court "
                        f"solve to {r['error_ft']:.2f} ft."
                    ),
                }
    if best:
        return best

    # --- 2. one misplaced point -------------------------------------------
    for frame, marks in landmarks.items():
        for (tag, _x, _y) in marks:
            trial = dict(landmarks)
            trial[frame] = [m for m in marks if m[0] != tag]
            r = court_detect.identify(trial)
            if r["identified"] and (best is None or r["error_ft"] < best["fit_ft"]):
                best = {
                    "kind": "bad_point",
                    "frame": frame,
                    "tags": [tag],
                    "labels": [label(tag)],
                    "fit_ft": round(float(r["error_ft"]), 2),
                    "message": (
                        f'"{label(tag)}" on this frame doesn\'t agree with your other '
                        f"marks. Re-place it (or delete it) and the rest solve to "
                        f"{r['error_ft']:.2f} ft."
                    ),
                }
    if best:
        return best

    # --- 3. a whole frame --------------------------------------------------
    for drop in sorted(landmarks):
        r = court_detect.identify({k: v for k, v in landmarks.items() if k != drop})
        if r["identified"] and (best is None or r["error_ft"] < best["fit_ft"]):
            best = {
                "kind": "bad_frame",
                "frame": drop,
                "tags": [],
                "labels": [],
                "fit_ft": round(float(r["error_ft"]), 2),
                "message": (
                    f"More than one point on this frame disagrees with the others. "
                    f"Without this frame the rest solve to {r['error_ft']:.2f} ft — "
                    f"check its points, especially any UPPER/LOWER pair."
                ),
            }
    return best


if __name__ == "__main__":
    import json
    import clip_registry
    doc = clip_registry.load(sys.argv[1])
    lm = {int(k): [(m[0], float(m[1]), float(m[2])) for m in v]
          for k, v in doc["landmarks"].items()}
    print(json.dumps(diagnose(lm), indent=2))
