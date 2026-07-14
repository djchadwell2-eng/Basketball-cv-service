"""Phase 5 step 5 -- MAKE/MISS, TIMEBOXED (ROADMAP Gate 4).

Simple geometric discriminators over data already produced by steps 2-4
(raw ball detections, claimed arcs, the carried hoop position). Outputs
are CANDIDATE labels feeding review -- candidate_make / candidate_miss /
unknown -- never a bare made/missed stat, per ROADMAP: "treat outcomes as
candidate labels feeding the review queue."

Two independent signals, checked in the window right after a shot's
closest hoop approach:
  MAKE evidence: RAW (unclaimed) detections in a narrow horizontal
    corridor around the hoop x, BELOW the hoop, with y increasing across
    the window -- the ball dropping through the net. Uses raw detections
    (not physics-claimed arcs) because a short net-drop may be only a few
    frames -- not enough for ball_trajectory's own arc-claim minimum.
  MISS evidence: a chain (claimed or not) whose earliest point in the
    window starts near the hoop and whose later points move clearly away
    -- a deflection (the exact section-15 rim-out shape, now read as
    signal instead of being an accidental second "shot attempt").
Both present -> unknown (conflict, never resolved by guessing which to
trust). Neither present -> unknown (no evidence). Exactly one -> that
candidate label.

GATE 4 (ROADMAP): if automatic outcome accuracy on eyeballed samples is
honestly <~85%, ship attempts+locations+reviewed-outcomes and move on --
do not chase make/miss for weeks. This module does not itself decide
Gate 4; main() reports whether there is even enough sample to evaluate it.
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

FALL_WINDOW_FRAMES = 15
BELOW_RIM_CORRIDOR_PX = 50.0
MIN_FALL_POINTS = 3
FALL_Y_JITTER_TOLERANCE_PX = 5.0     # small non-monotonicity allowed (detection noise)

DEFLECTION_NEAR_HOOP_PX = 100.0      # matches shot_attempts.HOOP_RADIUS_PX
DEFLECTION_ESCAPE_PX = 60.0


def below_rim_fall_evidence(raw_dets_by_frame, hoop_at, after_frame,
                            window=FALL_WINDOW_FRAMES, corridor=BELOW_RIM_CORRIDOR_PX,
                            min_points=MIN_FALL_POINTS):
    """raw_dets_by_frame: {frame: [{"bbox":[x1,y1,x2,y2]}, ...]}. Looks at
    frames (after_frame, after_frame+window] for detections within
    `corridor` px of the hoop's x and BELOW it (larger y = lower in image),
    then requires >= min_points such matches with non-decreasing y across
    frame order (a little jitter tolerance for detection noise)."""
    matches = []
    for f in range(after_frame + 1, after_frame + window + 1):
        h = hoop_at(f)
        if h is None:
            continue
        hx, hy = h
        for d in raw_dets_by_frame.get(f, []):
            x1, y1, x2, y2 = d["bbox"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if abs(cx - hx) <= corridor and cy > hy:
                matches.append((f, cy))
    if len(matches) < min_points:
        return None
    ys = [m[1] for m in matches]
    if not all(ys[i + 1] >= ys[i] - FALL_Y_JITTER_TOLERANCE_PX for i in range(len(ys) - 1)):
        return None
    return {"frames": [m[0] for m in matches], "n_points": len(matches)}


def deflection_evidence(chains, hoop_at, after_frame, window=FALL_WINDOW_FRAMES,
                        near_radius=DEFLECTION_NEAR_HOOP_PX, escape_px=DEFLECTION_ESCAPE_PX):
    """chains: [{"points": [[frame, cx, cy, conf], ...]}, ...] (raw chains,
    claimed or not -- a deflection may be too short to earn a physics
    claim). Finds a chain whose earliest point inside the search window
    starts within near_radius of the hoop, and whose LAST point in that
    chain has moved >= escape_px away from the (then-current) hoop
    position -- escaping the rim rather than dropping through it."""
    for chain in chains:
        pts = sorted(chain["points"], key=lambda p: p[0])
        start = next((p for p in pts if after_frame < p[0] <= after_frame + window), None)
        if start is None:
            continue
        f0, x0, y0 = start[0], start[1], start[2]
        h0 = hoop_at(f0)
        if h0 is None:
            continue
        d0 = ((x0 - h0[0]) ** 2 + (y0 - h0[1]) ** 2) ** 0.5
        if d0 > near_radius:
            continue
        later = [p for p in pts if p[0] >= f0]
        if len(later) < 2:
            continue
        fx, lx, ly = later[-1][0], later[-1][1], later[-1][2]
        h_last = hoop_at(fx)
        if h_last is not None:
            dist_last = ((lx - h_last[0]) ** 2 + (ly - h_last[1]) ** 2) ** 0.5
        else:
            dist_last = ((lx - x0) ** 2 + (ly - y0) ** 2) ** 0.5
        if dist_last >= escape_px:
            return {"chain_start_frame": f0, "start_dist_px": round(d0, 1),
                    "end_frame": fx, "end_dist_px": round(dist_last, 1)}
    return None


def classify_outcome(make_evidence, miss_evidence):
    if make_evidence and miss_evidence:
        return {"outcome": "unknown",
                "reason": "conflicting evidence (both fall-through and "
                          "deflection signals present)",
                "make_evidence": make_evidence, "miss_evidence": miss_evidence}
    if make_evidence:
        return {"outcome": "candidate_make", "evidence": make_evidence}
    if miss_evidence:
        return {"outcome": "candidate_miss", "evidence": miss_evidence}
    return {"outcome": "unknown", "reason": "no evidence either way"}


# ---------------------------------------------------------------- runner ----

def main():
    CLIP_NAME = "HARD"
    out_dir = os.path.join(_HERE, "out")

    sa_doc = json.load(open(os.path.join(out_dir, f"{CLIP_NAME}_shot_attempts.json"),
                            encoding="utf-8"))
    arcs_doc = json.load(open(os.path.join(out_dir, f"{CLIP_NAME}_ball_arcs.json"),
                              encoding="utf-8"))
    hoop_doc = json.load(open(os.path.join(out_dir, f"{CLIP_NAME}_hoop_track.json"),
                              encoding="utf-8"))
    log_doc = json.load(open(os.path.join(out_dir, f"{CLIP_NAME}_ball_spike_log.json"),
                             encoding="utf-8"))

    # TWO hoops (DECISIONS 18): use whichever hoop shot_attempts.py already
    # matched this shot to (recorded as "hoop": "far"/"near") -- avoids
    # re-guessing which basket is relevant when evaluating outcome evidence.
    def _hoop_lookup(key):
        by_frame = {r["frame_index"]: tuple(r[key]) for r in hoop_doc["frames"]
                   if r.get(key) is not None}
        return lambda f: by_frame.get(f)

    hoop_lookups = {"far": _hoop_lookup("hoop_far_px"), "near": _hoop_lookup("hoop_near_px")}
    raw_by_frame = {fr["frame_index"]: fr["detections"] for fr in log_doc["frames"]}
    chains = arcs_doc["chains"]

    results = []
    for a in sa_doc["attempts"]:
        if a["verdict"] != "shot_attempt":
            continue
        hoop_at = hoop_lookups[a.get("hoop", "far")]
        after = a["at_frame"]     # closest hoop approach -- outcome plays out AFTER this
        make_ev = below_rim_fall_evidence(raw_by_frame, hoop_at, after)
        miss_ev = deflection_evidence(chains, hoop_at, after)
        outcome = classify_outcome(make_ev, miss_ev)
        results.append({"start_frame": a["start_frame"], "end_frame": a["end_frame"],
                        "hoop": a.get("hoop", "far"),
                        "checked_after_frame": after, **outcome})

    n_total = len(results)
    n_evaluable_evidence = sum(1 for r in results if r["outcome"] != "unknown")
    out_json = os.path.join(out_dir, f"{CLIP_NAME}_shot_outcomes.json")
    json.dump({"clip": CLIP_NAME, "outcomes": results}, open(out_json, "w", encoding="utf-8"),
              indent=2)

    print(f"\n================ SHOT OUTCOMES ({CLIP_NAME}) ================")
    for r in results:
        print(f"  {r['start_frame']}..{r['end_frame']}  outcome={r['outcome']}  "
              f"{r.get('reason', '')}")
    print(f"  wrote {out_json}")
    print(f"\n  GATE 4 (ROADMAP): needs an eyeballed accuracy RATE (>=~85%) to decide "
          f"auto vs review-only.")
    print(f"  shot attempts this session: {n_total}  (1 with a user-verified real "
          f"outcome: rim-out MISS)")
    print(f"  n={n_total} CANNOT measure a rate -- Gate 4 is UNMEASURABLE this session, "
          f"not passed or failed.")
    print(f"  To get a real sample: a full-clip HARD ball-detection run (~90s clip, "
          f"~90min background) would surface every shot for eyeballed accuracy scoring.")


if __name__ == "__main__":
    main()
