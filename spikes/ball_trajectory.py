"""Phase 5 step 2 -- BALL TRAJECTORY LAYER over the step-1 spike detections.

Turns raw per-frame ball detections (spikes/ball_spike.py log) into honest
BALL-IN-FLIGHT claims. DECISIONS.md 13 measured the design constraints:
  - detector confidence CANNOT gate ball claims (the real shot arc never
    crossed conf 0.5) -> consume ALL detections, low conf included;
  - the real ball moves smoothly (~12 px/frame); glare junk is positionally
    static -> position CHAINING separates them where confidence can't;
  - the ball appears in short streaks -> SHORT parabolic fits, bounded gap
    tolerance, no long extrapolation.

Abstention-first, same as identity: a chain earns an ARC claim only when a
quadratic fit passes the physics gate (downward accel inside a measured
plausible band + tight residuals). Everything else stays visible as
no_claim / static_junk / too_short -- surfaced, never guessed, and NOTHING
here writes into team_events (the ball layer is a NEW layer, forever).

Run (reads the spike log, re-renders overlay from the source video):
    .venv/Scripts/python spikes/ball_trajectory.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "phase2"))   # run_tracking (overlay render)

# --- chaining gates (from the HARD spike measurements, DECISIONS 13) ---------
MAX_STEP_PX = 40.0        # per-frame center displacement gate (~3x measured 12)
MAX_GAP_FRAMES = 3        # a chain survives holes up to this many frames
MIN_CHAIN_LEN = 6         # fewer points = too little evidence for any claim
MIN_TRAVEL_PX = 30.0      # total travel below this = static junk (floor glare)

# --- physics gate (measured on the user-verified HARD arcs) ------------------
# y-accel on the real arcs: ~0.4-0.5 px/frame^2 (distant, high in frame) to
# ~1.1-2.6 (nearer camera). Band is deliberately wider but still refuses
# non-ballistic motion; image +y is DOWN, so gravity means accel_y > 0.
# ACCEL_Y_MIN raised 0.1 -> 0.3 after the first HARD run (user-eyeballed):
# camera-pan-dragged floor glare drifts horizontally and fit at 0.10-0.15,
# while every user-verified real arc measured >= 0.89. Near-flat "arcs"
# are abstained on, not claimed -- missing a super-distant flat arc is the
# accepted cost of never claiming glare.
ACCEL_Y_MIN = 0.3
ACCEL_Y_MAX = 3.0
ACCEL_X_MAX = 1.5         # mild x curvature allowed (pan/perspective), no more
RESIDUAL_MAX_PX = 3.0     # RMS ceiling for both axes' fits
MIN_FIT_LEN = 8           # minimum points in one parabolic segment
# A ballistic arc must actually TRAVEL vertically; near-flat segments have
# their curvature fit to noise. Measured on HARD: real arcs span 48-351 px
# of y; the one glare slice that squeaked past the accel band spanned 11.
MIN_Y_RANGE_PX = 25.0


def _centers(frames_doc):
    """Flatten a spike-log frames[] doc to (frame, cx, cy, conf) tuples."""
    out = []
    for fr in frames_doc:
        for d in fr["detections"]:
            x1, y1, x2, y2 = d["bbox"]
            out.append((fr["frame_index"], (x1 + x2) / 2.0, (y1 + y2) / 2.0,
                        d["conf"]))
    return out


def build_chains(frames_doc):
    """Greedy nearest-first association of detections into position chains.

    Active chains predict their next position (last point + velocity * dt);
    a detection may extend a chain only if it is within MAX_STEP_PX * dt of
    the prediction and the hole is <= MAX_GAP_FRAMES. Each (chain, frame)
    pair takes at most one detection; leftovers start new chains.
    """
    dets = _centers(frames_doc)
    by_frame = {}
    for frame, cx, cy, conf in dets:
        by_frame.setdefault(frame, []).append((cx, cy, conf))

    chains = []          # each: {"points": [(frame, cx, cy, conf), ...]}
    active = []          # indices into chains
    for frame in sorted(by_frame):
        # retire chains whose hole just exceeded the gap tolerance
        active = [ci for ci in active
                  if frame - chains[ci]["points"][-1][0] <= MAX_GAP_FRAMES + 1]
        # score every (active chain, detection) pair by prediction distance
        pairs = []
        for ci in active:
            pts = chains[ci]["points"]
            f0, x0, y0, _ = pts[-1]
            dt = frame - f0
            if len(pts) >= 2:
                f1, x1, y1, _ = pts[-2]
                vx = (x0 - x1) / max(f0 - f1, 1)
                vy = (y0 - y1) / max(f0 - f1, 1)
            else:
                vx = vy = 0.0
            px, py = x0 + vx * dt, y0 + vy * dt
            for di, (cx, cy, _c) in enumerate(by_frame[frame]):
                dist = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if dist <= MAX_STEP_PX * dt:
                    pairs.append((dist, ci, di))
        pairs.sort()
        used_c, used_d = set(), set()
        for dist, ci, di in pairs:
            if ci in used_c or di in used_d:
                continue
            used_c.add(ci)
            used_d.add(di)
            cx, cy, conf = by_frame[frame][di]
            chains[ci]["points"].append((frame, cx, cy, conf))
        for di, (cx, cy, conf) in enumerate(by_frame[frame]):
            if di not in used_d:
                chains.append({"points": [(frame, cx, cy, conf)]})
                active.append(len(chains) - 1)
    return chains


def _fit_segment(points):
    """Quadratic fit of cx(t) and cy(t). Returns (ok, info) where ok means
    the segment passes the physics gate: downward y-accel inside the band,
    only mild x curvature, tight residuals on both axes."""
    t = np.array([p[0] for p in points], dtype=float)
    t = t - t[0]
    x = np.array([p[1] for p in points], dtype=float)
    y = np.array([p[2] for p in points], dtype=float)
    cy2, cy1, cy0 = np.polyfit(t, y, 2)
    cx2, cx1, cx0 = np.polyfit(t, x, 2)
    ry = float(np.sqrt(np.mean((np.polyval([cy2, cy1, cy0], t) - y) ** 2)))
    rx = float(np.sqrt(np.mean((np.polyval([cx2, cx1, cx0], t) - x) ** 2)))
    accel_y = 2.0 * cy2                      # px/frame^2, +y = down = gravity
    accel_x = 2.0 * cx2
    ok = (ACCEL_Y_MIN <= accel_y <= ACCEL_Y_MAX
          and abs(accel_x) <= ACCEL_X_MAX
          and ry <= RESIDUAL_MAX_PX and rx <= RESIDUAL_MAX_PX
          and float(y.max() - y.min()) >= MIN_Y_RANGE_PX)
    info = {"accel_y": round(accel_y, 3), "accel_x": round(accel_x, 3),
            "rms_y": round(ry, 2), "rms_x": round(rx, 2),
            "fit_y": [float(cy2), float(cy1), float(cy0)],
            "fit_x": [float(cx2), float(cx1), float(cx0)]}
    return ok, info


# Robust fallback (DECISIONS 26): a denser detector (fine-tuned model) adds
# occasional junk members to otherwise-clean chains; one bad point fails the
# whole-window fit (found on TEST1's user-verified shot B: 11-pt chain with 3
# junk members -> no_claim). Bounded remedy, never "drop until it fits":
ROBUST_MAX_DROP_FRACTION = 0.25   # at most 25% of a chain's points dropped...
ROBUST_MAX_DROPS_ABS = 4          # ...and never more than 4 total. HARD cap:
                                  # dense fine-tuned logs produce 100+ point
                                  # chains where an uncapped 25% budget makes
                                  # the subset search combinatorial (measured:
                                  # a full-log analysis hung for 2.4 CPU-hours
                                  # before this cap). Physically consistent
                                  # too -- a chain needing >4 junk drops isn't
                                  # a trustworthy rescue; abstain instead.
ROBUST_MIN_KEPT = MIN_FIT_LEN     # and never below the ordinary fit minimum


def _robust_whole_chain_fit(pts):
    """Bounded combination search: try dropping every subset (smallest
    first) of the worst-residual candidate POOL, within the drop budget.
    Greedy and one-step-lookahead both pick wrong drops when junk distorts
    the fit they rank against (measured on the real shot-B chain: greedy
    lands at rms 3.2 vs the true junk set's 2.37). Chains are tiny and the
    pool is capped at budget+3 points, so exhausting subsets is at most a
    few hundred cheap polyfits. Deterministic: fewest drops win, ties by
    residual rank. Returns (kept_pts, info, dropped_frames) on a passing
    fit, else None. Only ever called when the ordinary growth loop found
    NO arcs, so it can add claims but never change existing ones."""
    import itertools
    import math
    if len(pts) < ROBUST_MIN_KEPT:
        return None            # §14 gate: <MIN_FIT_LEN points = too little
                               # evidence for ANY physics claim, robust or not
    ok, info = _fit_segment(pts)
    if ok:
        return list(pts), info, []
    max_drops = math.ceil(len(pts) * ROBUST_MAX_DROP_FRACTION)
    max_drops = min(max_drops, ROBUST_MAX_DROPS_ABS, len(pts) - ROBUST_MIN_KEPT)
    if max_drops < 1:
        return None
    # candidate pool: the budget+3 points with the largest residuals against
    # the full-chain fit (junk distorts that fit, but junk still ranks high)
    t0 = pts[0][0]
    cy2, cy1, cy0 = info["fit_y"]
    cx2, cx1, cx0 = info["fit_x"]
    def _resid(p):
        t = p[0] - t0
        rx = p[1] - (cx2 * t * t + cx1 * t + cx0)
        ry = p[2] - (cy2 * t * t + cy1 * t + cy0)
        return rx * rx + ry * ry
    pool = sorted(range(len(pts)), key=lambda i: -_resid(pts[i]))[:max_drops + 3]
    for size in range(1, max_drops + 1):
        for combo in itertools.combinations(pool, size):
            drop = set(combo)
            kept = [p for i, p in enumerate(pts) if i not in drop]
            t_ok, t_info = _fit_segment(kept)
            if t_ok:
                return kept, t_info, sorted(pts[i][0] for i in drop)
    return None


def classify_chain(chain):
    """Verdict for one chain: static_junk | too_short | arc | no_claim.
    An 'arc' carries the maximal passing sub-segments; a failing chain is
    kept visible (no_claim), never silently dropped. If the ordinary growth
    loop finds nothing, a BOUNDED robust whole-chain fit (drop worst-residual
    points, <=25%, physics gates unchanged) gets one more try -- arcs found
    that way carry n_dropped/dropped_frames so review sees the difference."""
    pts = chain["points"]
    if len(pts) < MIN_CHAIN_LEN:
        return {"verdict": "too_short", "arcs": []}
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    travel = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
    if travel < MIN_TRAVEL_PX:
        return {"verdict": "static_junk", "arcs": []}

    arcs = []
    i = 0
    while i + MIN_FIT_LEN <= len(pts):
        ok, _ = _fit_segment(pts[i:i + MIN_FIT_LEN])
        if not ok:
            i += 1
            continue
        j = i + MIN_FIT_LEN
        info_best = None
        while j <= len(pts):
            ok, info = _fit_segment(pts[i:j])
            if not ok:
                break
            info_best = info
            j += 1
        seg = pts[i:j - 1] if info_best else pts[i:i + MIN_FIT_LEN]
        if info_best is None:                       # can't happen, guard anyway
            i += 1
            continue
        arcs.append({"start_frame": seg[0][0], "end_frame": seg[-1][0],
                     "n_points": len(seg), **info_best})
        i = i + len(seg)

    if not arcs:
        robust = _robust_whole_chain_fit(pts)
        if robust is not None:
            kept, info, dropped = robust
            arcs.append({"start_frame": kept[0][0], "end_frame": kept[-1][0],
                         "n_points": len(kept), "n_dropped": len(dropped),
                         "dropped_frames": dropped, **info})

    return {"verdict": "arc" if arcs else "no_claim", "arcs": arcs}


# ---------------------------------------------------------------- runner ----

def _render_overlay(video_path, span_start, span_len, chains, results, out_path):
    """Overlay: claimed arcs bright green (fitted curve + live marker),
    no-claim chains dim gray, junk faint dots. Honest picture of everything."""
    import cv2
    import clip_config
    import run_tracking
    subclip, fps, _n = run_tracking.extract_subclip(video_path, span_start, span_len)
    cap = cv2.VideoCapture(subclip)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    GREEN, GRAY, FAINT = (0, 255, 0), (160, 160, 160), (80, 80, 80)

    arc_curves = []          # (start_f, end_f, [(f, x, y)...])
    for chain, res in zip(chains, results):
        for a in res["arcs"]:
            pts = [(f, x, y) for (f, x, y, _c) in chain["points"]
                   if a["start_frame"] <= f <= a["end_frame"]]
            arc_curves.append((a["start_frame"], a["end_frame"], pts))

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        f = span_start + i
        for chain, res in zip(chains, results):
            color = FAINT if res["verdict"] in ("static_junk", "too_short") else GRAY
            for (pf, x, y, _c) in chain["points"]:
                if pf <= f:
                    cv2.circle(frame, (int(x), int(y)), 3, color, -1)
        n_live = 0
        for (sf, ef, pts) in arc_curves:
            if f < sf:
                continue
            past = [(int(x), int(y)) for (pf, x, y) in pts if pf <= f]
            for a, b in zip(past, past[1:]):
                cv2.line(frame, a, b, GREEN, 3)
            if sf <= f <= ef and past:
                cv2.circle(frame, past[-1], 12, GREEN, 2)
                n_live += 1
        cv2.putText(frame, f"f={f} t={f/fps:04.1f}s arcs_live={n_live}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        writer.write(frame)
        i += 1
    writer.release()
    cap.release()


def main():
    import clip_config
    CLIP_NAME = sys.argv[1] if len(sys.argv) > 1 else "HARD"   # ball_trajectory.py [clip_name]
    CLIP = getattr(clip_config, f"{CLIP_NAME}_CLIP")
    clip_config.ACTIVE_CLIP = CLIP     # BEFORE run_tracking import inside render

    log_path = os.path.join(_HERE, "out", f"{CLIP.name}_ball_spike_log.json")
    doc = json.load(open(log_path, encoding="utf-8"))
    print(f"[trajectory] {doc['clip']} span {doc['span_start']}..+{doc['span_len']}, "
          f"{sum(len(fr['detections']) for fr in doc['frames'])} raw detections")

    chains = build_chains(doc["frames"])
    results = [classify_chain(c) for c in chains]
    counts = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    arcs = [(c, r) for c, r in zip(chains, results) if r["verdict"] == "arc"]

    out_json = os.path.join(_HERE, "out", f"{CLIP.name}_ball_arcs.json")
    json.dump({"clip": doc["clip"], "span_start": doc["span_start"],
               "span_len": doc["span_len"], "fps": doc["fps"],
               "params": {"MAX_STEP_PX": MAX_STEP_PX, "MAX_GAP_FRAMES": MAX_GAP_FRAMES,
                          "MIN_CHAIN_LEN": MIN_CHAIN_LEN, "MIN_TRAVEL_PX": MIN_TRAVEL_PX,
                          "ACCEL_Y_MIN": ACCEL_Y_MIN, "ACCEL_Y_MAX": ACCEL_Y_MAX,
                          "ACCEL_X_MAX": ACCEL_X_MAX, "RESIDUAL_MAX_PX": RESIDUAL_MAX_PX,
                          "MIN_FIT_LEN": MIN_FIT_LEN},
               "chains": [{"points": c["points"], **r}
                          for c, r in zip(chains, results)]},
              open(out_json, "w", encoding="utf-8"), indent=2)

    print("\n================ TRAJECTORY SUMMARY ================")
    print(f"  chains: {len(chains)}  verdicts: {counts}")
    for c, r in arcs:
        for a in r["arcs"]:
            fps = doc["fps"]
            print(f"  ARC {a['start_frame']}..{a['end_frame']} "
                  f"({a['start_frame']/fps:.1f}s-{a['end_frame']/fps:.1f}s) "
                  f"n={a['n_points']} accel_y={a['accel_y']} "
                  f"rms=({a['rms_x']},{a['rms_y']})")
    print(f"  wrote {out_json}")

    out_video = os.path.join(_HERE, "out", f"{CLIP.name}_ball_arcs_overlay.mp4")
    print("[trajectory] rendering overlay ...")
    _render_overlay(CLIP.video_path, doc["span_start"], doc["span_len"],
                    chains, results, out_video)
    print(f"  wrote {out_video}")


if __name__ == "__main__":
    main()
