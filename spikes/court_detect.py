"""Work out WHICH COURT a clip was filmed on, from the clicked landmarks alone.

Why this exists: TEST2 (Fairfield) is a 94-ft floor. Its clips_config entry was
copied from TEST1 and said 84 ft, so the engine spent every fit squeezing a
94-ft court into 84 and dragged all of DJ's marks ~10 ft out of place (mean
error 0.94 ft; the overlay visibly off the paint). Nothing in the system
noticed, because a wrong court still produces a plausible-looking number.

The fix is NOT to fit the court dimensions as free parameters. A court is not
an arbitrary shape -- there are only a few real ones -- and free-fitting throws
away that hard truth to chase click noise: solving TEST1's court freely returns
82.6 ft when it is really 84.0, which would bake a 1.4-ft error into every shot
location forever.

So instead: SCORE the handful of courts that actually exist, take the best, and
use its EXACT published dimensions. The measurement decides which court; the
rulebook supplies the numbers. If two courts score too close to call, or the
best one still fits badly, say so and refuse -- a clip that can't be identified
is a clip whose marks need another look, not a court to be invented.

Pure geometry: numpy + cv2 only, no video, no config, milliseconds.
"""

import cv2
import numpy as np

# NFHS/NCAA/NBA all put the basket 5.25 ft in from the baseline. 19.75 is the
# high-school / NCAA-women 3pt radius (arc apex 25.0 ft from the baseline).
HOOP_DX, R3 = 5.25, 19.75

# Every level plays on a 50-ft-wide floor with a 6-ft centre circle and the
# free-throw line 19 ft from the baseline. Only the LENGTH and the KEY WIDTH
# actually vary, which is why those are the two things worth identifying.
_HS_KEY = {"width": 50.0, "lane_y0": 19.0, "lane_y1": 31.0,
           "ft_x": 19.0, "circle_r": 6.0}
_WIDE_KEY = {"width": 50.0, "lane_y0": 17.0, "lane_y1": 33.0,
             "ft_x": 19.0, "circle_r": 6.0}

KNOWN_COURTS = [
    ("84 ft floor, 12 ft key -- standard high school",
     dict(_HS_KEY, length=84.0)),
    ("94 ft floor, 12 ft key -- full-size floor, high-school markings",
     dict(_HS_KEY, length=94.0)),
    ("84 ft floor, 16 ft key",
     dict(_WIDE_KEY, length=84.0)),
    ("94 ft floor, 16 ft key -- college / pro markings",
     dict(_WIDE_KEY, length=94.0)),
]

# A homography has 8 degrees of freedom, so 4 marks fit ANY court exactly and a
# 4-mark frame scores zero against every candidate -- it carries no evidence and
# would only dilute the frames that do.
MIN_MARKS_PER_FRAME = 5
# The runner-up must be at least this much worse before we call it. Measured
# separation is ~3x on both real clips, so 1.35 refuses only genuine ties.
DECISIVE_RATIO = 1.35
# Even the winner has to actually fit. Above this, no court is right and the
# marks are the thing to look at.
MAX_ERR_FT = 1.0


def court_model(dims):
    """tag -> (x, y) in court feet, for the given court dimensions.

    The single source of truth for what each clicked landmark MEANS; stage4
    builds its COURT_MODEL from this so the engine and the detector can never
    drift apart.
    """
    L, W = dims["length"], dims["width"]
    y0, y1, ft, r = (dims["lane_y0"], dims["lane_y1"],
                     dims["ft_x"], dims["circle_r"])
    cx, cy = L / 2.0, W / 2.0
    return {
        "LB_side_near": (0.0, 0.0), "LB_side_far": (0.0, W),
        "L_lane_base_near": (0.0, y0), "L_lane_base_far": (0.0, y1),
        "L_FT_near": (ft, y0), "L_FT_far": (ft, y1),
        "center_near": (cx, 0.0), "center_logo": (cx, cy), "center_far": (cx, W),
        "R_FT_near": (L - ft, y0), "R_FT_far": (L - ft, y1),
        "R_lane_base_near": (L, y0), "R_lane_base_far": (L, y1),
        "RB_side_near": (L, 0.0), "RB_side_far": (L, W),
        "circle_top": (cx, cy + r), "circle_bottom": (cx, cy - r),
        "circle_left": (cx - r, cy), "circle_right": (cx + r, cy),
        "L_arc_top": (HOOP_DX + R3, cy), "R_arc_top": (L - HOOP_DX - R3, cy),
    }


def score(landmarks, dims):
    """How well this court explains the marks: (mean_error_ft, marks, frames).

    Each keyframe is fit to its own marks and the leftover distance is measured
    in feet. Fitting per frame is deliberate here: the camera moves between
    keyframes, so a shared fit would need the SIFT chain, and this check has to
    stay cheap. It is only ever used to COMPARE courts against the same marks,
    never as the calibration itself.
    """
    model = court_model(dims)
    errs, frames = [], 0
    for marks in landmarks.values():
        marks = [m for m in marks if m[0] in model]
        if len(marks) < MIN_MARKS_PER_FRAME:
            continue
        src = np.array([[x, y] for (_, x, y) in marks], dtype=np.float64)
        dst = np.array([model[t] for (t, _, _) in marks], dtype=np.float64)
        H, _ = cv2.findHomography(src, dst, method=0)
        if H is None:
            continue
        proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
        errs.extend(np.linalg.norm(proj - dst, axis=1))
        frames += 1
    if not errs:
        return float("inf"), 0, 0
    return float(np.mean(errs)), len(errs), frames


def identify(landmarks, candidates=KNOWN_COURTS):
    """Which of the real courts is this? -> result dict.

    keys: identified (bool), name, dims, error_ft, runner_up, runner_up_error_ft,
          margin, marks, frames, reason (why not, when identified is False)
    """
    ranked = []
    for name, dims in candidates:
        err, marks, frames = score(landmarks, dims)
        ranked.append({"name": name, "dims": dims, "error_ft": err,
                       "marks": marks, "frames": frames})
    ranked.sort(key=lambda r: r["error_ft"])
    best = ranked[0]
    out = dict(best, all=ranked, identified=False, reason=None,
               runner_up=None, runner_up_error_ft=None, margin=None)

    if best["frames"] == 0:
        out["reason"] = (f"no keyframe has {MIN_MARKS_PER_FRAME}+ marks -- fewer "
                         f"than that fits any court exactly and proves nothing")
        return out
    if best["error_ft"] > MAX_ERR_FT:
        out["reason"] = (f"no known court fits: the closest is off by "
                         f"{best['error_ft']:.2f} ft (limit {MAX_ERR_FT:.2f}). "
                         f"The marks need another look, not a new court")
        return out

    second = ranked[1]
    out["runner_up"] = second["name"]
    out["runner_up_error_ft"] = second["error_ft"]
    out["margin"] = second["error_ft"] / max(best["error_ft"], 1e-9)
    if out["margin"] < DECISIVE_RATIO:
        out["reason"] = (f"too close to call between '{best['name']}' "
                         f"({best['error_ft']:.2f} ft) and '{second['name']}' "
                         f"({second['error_ft']:.2f} ft) -- the marks so far do "
                         f"not tell these courts apart. Mark a frame that shows "
                         f"a baseline and the halfway line together")
        return out

    out["identified"] = True
    return out


def report(result):
    """The identification as printable lines, for the calibration log."""
    lines = []
    if result["identified"]:
        lines.append(f"Court identified from the marks: {result['name']}")
        lines.append(f"  fits to {result['error_ft']:.2f} ft over "
                     f"{result['marks']} marks on {result['frames']} keyframes")
        lines.append(f"  next-best court '{result['runner_up']}' is "
                     f"{result['margin']:.1f}x worse "
                     f"({result['runner_up_error_ft']:.2f} ft) -- clear call")
    else:
        lines.append("Court NOT identified from the marks.")
        lines.append(f"  {result['reason']}")
    for r in result["all"]:
        lines.append(f"    {r['error_ft']:>6.2f} ft   {r['name']}")
    return lines
