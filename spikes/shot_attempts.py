"""Phase 5 step 3, parts B+C -- SHOT ATTEMPTS from claimed ball arcs.

A claimed arc (ball_trajectory.py; already physics-gated, DECISIONS 14) is
a SHOT ATTEMPT iff, at or after its apex (the descending half -- a ball
still rising hasn't reached the rim), it passes within HOOP_RADIUS_PX of
the carried hoop position (hoop_anchor.py, DECISIONS 15) AT THAT SAME
FRAME. Floor-level flight (dribbles) fails this by geometry alone -- the
hoop sits high in the frame, a dribble apex sits near the floor -- not by
a special case that has to be kept in sync.

Shooter = nearest tracked body (feet pixel) to the arc's FIRST claimed
point (release; DECISIONS 14 already logged this as an approximation --
the detector is blind right at true release), joined to that track's
identity_state from the merged player events. No identity data for that
frame (outside the tracks-cache span, or the track never got an identity
event) -> an honest review item, never a guessed shooter.

Nothing here writes into team_events or any existing artifact.
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "phase2"))

# HOOP_RADIUS_PX 100 -> 125 (DECISIONS 19, measured on ALL 43 arcs across
# both clips): real shots' closest approach (observed, or bounded forward
# extension of the fit) is <= 110px; every non-shot arc is >= 163px -- a
# clean separation gap, so 125 sits between them with margin on both sides.
# Also consistent with the anchor's own measured slack (DECISIONS 15: the
# carried rim marker sits ~20-40px off true center) + ball size. NOT a
# loosened threshold to chase a marginal case: the same radius also drives
# the ORIGIN gate, where BIGGER = stricter (rejects more
# deflection/continuation starts), and both known deflections (start 26-69px
# from the hoop) stay rejected.
HOOP_RADIUS_PX = 125.0

# ARRIVAL forward-extension (DECISIONS 19): on sparse-detection footage the
# tracker loses the ball near the rim (backboard/body clutter), so a claimed
# arc may cover ONLY the ascent -- the flight's actual arrival at the hoop is
# unobserved. The arc's own fitted quadratic (already physics-gated) predicts
# the missing remainder: extend it forward a bounded window, descending
# portion only (past the FIT's apex -- a rising ball hasn't arrived). Same
# claim-extension pattern as RELEASE_BACK_MAX_FRAMES, opposite direction; an
# arrival found this way is stamped "extrapolated" (vs "observed") so review
# always knows which evidence class it is. Bound: measured need on the
# user-confirmed truncated shot was +13 frames; 15 (~0.5s, about half a
# shot's flight time) adds a small margin without inviting long fantasy
# extrapolation.
ARRIVAL_FORWARD_MAX_FRAMES = 15

# RELEASE back-extrapolation (DECISIONS 15/16): the arc claim starts near
# APEX (detector blindness at true release, DECISIONS 14), so "nearest body
# to the arc's first point" picks whoever stands under the apex, not who
# released the ball. Instead extend the arc's own fit BACKWARD a bounded
# window and look for a tracked body the extrapolated ball position lands
# on -- a genuine CLAIM EXTENSION, so it stays bounded + gated and only ever
# produces a review HINT, never an auto-attribution.
RELEASE_BACK_MAX_FRAMES = 10
RELEASE_DIST_GATE_PX = 120.0

def apex_index(points):
    """Index of the highest point (MIN cy -- image y grows downward)."""
    return min(range(len(points)), key=lambda i: points[i][2])


def classify_shot(points, hoop_at, radius=HOOP_RADIUS_PX,
                  fit_x=None, fit_y=None,
                  max_forward=ARRIVAL_FORWARD_MAX_FRAMES):
    """points: [(frame, cx, cy), ...] of one claimed arc (chronological).
    hoop_at: callable frame -> (hx, hy) or None if no hoop position that
    frame. fit_x/fit_y: the arc's quadratic coefficients ([a,b,c] over
    t = frame - start_frame) enabling the bounded forward extension.

    A shot attempt = an arc that ARRIVES at the hoop region (comes within
    `radius`). ARRIVAL (DECISIONS 19): all observed points count; if none
    is within `radius`, the fit is extended forward up to `max_forward`
    frames, DESCENDING portion only (a rising ball hasn't arrived), hoop
    position required per predicted frame. An arrival found only via the
    extension is stamped "arrival": "extrapolated" (weaker evidence).

    ARRIVING vs LEAVING (DECISIONS 25, replaces the §18 hard origin gate):
    an arc that STARTS near the rim is a fresh close-range shot (LAYUP) only
    if it ARRIVES -- ends AT the rim (last point within `radius`, or the
    forward-extension carried a truncated arc to the rim). Otherwise it
    started near the rim and LEFT (a deflection/continuation: its closest
    point is its origin, then it moves away) and is rejected. An arc
    starting far that reaches the rim is an ordinary jump shot. shot_type
    ('layup'|'jumpshot') is a transparent heuristic from release distance,
    not a claim about shot form. Measured on real arcs from BOTH clips:
    layups end 13/25px from the rim, deflections end 131/167/374/384px."""
    f0, x0, y0 = points[0]
    h0 = hoop_at(f0)
    origin_dist = (((x0 - h0[0]) ** 2 + (y0 - h0[1]) ** 2) ** 0.5) if h0 is not None else None

    # closest approach to the rim: observed points, then (only if none reached)
    # a bounded forward extension of the fit (descending portion only).
    best = None            # (dist, frame)
    arrival = None
    for (f, x, y) in points:
        h = hoop_at(f)
        if h is None:
            continue
        d = ((x - h[0]) ** 2 + (y - h[1]) ** 2) ** 0.5
        if best is None or d < best[0]:
            best = (d, f)
            arrival = "observed"
    if (best is None or best[0] > radius) and fit_x is not None and fit_y is not None:
        ax, bx, cx0 = fit_x
        ay, by, cy0 = fit_y
        t_apex = (-by / (2 * ay)) if ay > 0 else 0.0
        f_start, f_end = points[0][0], points[-1][0]
        for k in range(0, max_forward + 1):
            f = f_end + k
            t = (f_end - f_start) + k
            if t < t_apex:
                continue                       # still rising per the fit
            h = hoop_at(f)
            if h is None:
                continue
            x = ax * t * t + bx * t + cx0
            y = ay * t * t + by * t + cy0
            d = ((x - h[0]) ** 2 + (y - h[1]) ** 2) ** 0.5
            if best is None or d < best[0]:
                best = (d, f)
                arrival = "extrapolated"

    if best is None or best[0] > radius:
        return {"verdict": "not_shot",
                "min_dist": round(best[0], 1) if best else None,
                "reason": "never reaches the hoop region"}

    # reaches the rim -- fresh shot (arriving) or deflection (starts near, leaves)?
    if origin_dist is not None and origin_dist <= radius:
        # released near the rim: a LAYUP ends AT the rim; a deflection LEAVES.
        # arrival=="extrapolated" means a truncated arc was carried to the rim
        # by the forward extension (arriving); otherwise use the last point.
        fN, xN, yN = points[-1]
        hN = hoop_at(fN)
        end_dist = (((xN - hN[0]) ** 2 + (yN - hN[1]) ** 2) ** 0.5) if hN is not None else None
        arrived = arrival == "extrapolated" or (end_dist is not None and end_dist <= radius)
        if not arrived:
            end_txt = f"{round(end_dist, 1)}px" if end_dist is not None else "?"
            return {"verdict": "not_shot", "min_dist": round(best[0], 1),
                    "reason": f"originates {round(origin_dist, 1)}px from the hoop "
                              f"and leaves it (ends {end_txt} away) -- a deflection/"
                              f"continuation, not a fresh release"}
        shot_type = "layup"
    else:
        shot_type = "jumpshot"

    return {"verdict": "shot_attempt", "min_dist": round(best[0], 1),
            "at_frame": best[1], "arrival": arrival, "shot_type": shot_type,
            "origin_dist": round(origin_dist, 1) if origin_dist is not None else None}


def point_to_bbox_dist(x, y, bbox):
    """Distance from (x,y) to the nearest point ON or IN bbox (0 if inside).
    Used for release matching -- the ball leaves the HANDS, not the feet, so
    bbox proximity (not feet-pixel distance) is the right test."""
    x1, y1, x2, y2 = bbox
    dx = max(x1 - x, 0.0, x - x2)
    dy = max(y1 - y, 0.0, y - y2)
    return (dx * dx + dy * dy) ** 0.5


def find_release(fit_x, fit_y, start_frame, tracks_by_frame,
                 max_back=RELEASE_BACK_MAX_FRAMES, dist_gate=RELEASE_DIST_GATE_PX):
    """Walk the arc's own quadratic fit BACKWARD from start_frame (t=0) up
    to max_back frames, looking for a tracked body whose bbox the
    extrapolated ball position lands on. Returns the CLOSEST match found in
    that window if it clears dist_gate; otherwise an honest
    no_confident_shooter -- never a guess past the gate or the frame bound."""
    ax, bx, cx0 = fit_x
    ay, by, cy0 = fit_y
    best = None
    for k in range(1, max_back + 1):
        t = -k
        f = start_frame - k
        x = ax * t * t + bx * t + cx0
        y = ay * t * t + by * t + cy0
        for tr in tracks_by_frame.get(f, []):
            d = point_to_bbox_dist(x, y, tr["bbox"])
            if best is None or d < best[0]:
                best = (d, f, tr["track_id"], x, y)
    if best is not None and best[0] <= dist_gate:
        return {"status": "found", "release_frame": best[1], "track_id": best[2],
                "dist_px": round(best[0], 1),
                "ball_xy": [round(best[3], 1), round(best[4], 1)]}
    return {"status": "no_confident_shooter", "checked_frames": max_back}


def nearest_track_feet(tracks, x, y):
    """tracks: [{"track_id":.., "bbox":[x1,y1,x2,y2]}, ...] for ONE frame.
    Returns (track_id, dist) for the closest feet pixel (bottom-center of
    the bbox, same convention as phase2/tracking.Track.feet_pixel), or
    None if the frame has no tracks."""
    best = None
    for t in tracks:
        x1, y1, x2, y2 = t["bbox"]
        fx, fy = (x1 + x2) / 2.0, y2
        d = ((fx - x) ** 2 + (fy - y) ** 2) ** 0.5
        if best is None or d < best[1]:
            best = (t["track_id"], d)
    return best


# ---------------------------------------------------------------- runner ----

def _load(path):
    return json.load(open(path, encoding="utf-8"))


def main():
    CLIP_NAME = sys.argv[1] if len(sys.argv) > 1 else "HARD"   # shot_attempts.py [clip_name]
    out_dir = os.path.join(_HERE, "out")

    arcs_doc = _load(os.path.join(out_dir, f"{CLIP_NAME}_ball_arcs.json"))
    hoop_doc = _load(os.path.join(out_dir, f"{CLIP_NAME}_hoop_track.json"))
    tracks_doc = _load(os.path.join(_ROOT, "phase2", "out", f"{CLIP_NAME}_tracks_raw.json"))
    events_doc = _load(os.path.join(_ROOT, "phase2", "out",
                                    f"{CLIP_NAME}_player_events_merged.json"))

    # TWO hoops (DECISIONS 18): each is its own hoop_at callable, reusing the
    # SAME already-tested classify_shot() unchanged -- try both, keep whichever
    # (if either) passes, preferring the closer one on a tie/both-pass.
    def _hoop_lookup(key):
        by_frame = {r["frame_index"]: tuple(r[key]) for r in hoop_doc["frames"]
                   if r.get(key) is not None}
        return lambda f: by_frame.get(f)

    hoop_at_far = _hoop_lookup("hoop_far_px")
    hoop_at_near = _hoop_lookup("hoop_near_px")
    hoop_by_frame = {r["frame_index"]: tuple(r["hoop_far_px"]) for r in hoop_doc["frames"]
                     if r.get("hoop_far_px") is not None}   # kept for the overlay render

    tracks_by_frame = {fr["frame_index"]: fr["tracks"] for fr in tracks_doc["frames"]}
    tracks_span = (tracks_doc["span_start"], tracks_doc["span_start"] + tracks_doc["span_len"])

    # identity_state per (frame, track_id): last event wins is fine here --
    # this reads a snapshot for shooter attribution, it does not stamp a stat.
    identity_by_ft = {}
    for e in events_doc["player_events"]:
        identity_by_ft[(e["frame"], e["track_id"])] = (e["identity_id"], e["identity_state"])

    results = []
    for chain in arcs_doc["chains"]:
        if chain["verdict"] != "arc":
            continue
        pts = [(p[0], p[1], p[2]) for p in chain["points"]]
        for a in chain["arcs"]:
            seg = [p for p in pts if a["start_frame"] <= p[0] <= a["end_frame"]]
            shot_far = classify_shot(seg, hoop_at_far, fit_x=a["fit_x"], fit_y=a["fit_y"])
            shot_near = classify_shot(seg, hoop_at_near, fit_x=a["fit_x"], fit_y=a["fit_y"])
            # prefer whichever PASSES; if both pass (shouldn't happen -- the
            # two hoops are far apart -- but stay honest if it ever does) or
            # both fail, prefer the smaller min_dist as the more informative report
            candidates = [(shot_far, "far"), (shot_near, "near")]
            passing = [c for c in candidates if c[0]["verdict"] == "shot_attempt"]
            if passing:
                passing.sort(key=lambda c: c[0]["min_dist"])
                shot, hoop_label = passing[0]
            else:
                candidates.sort(key=lambda c: (c[0]["min_dist"] is None, c[0]["min_dist"]))
                shot, hoop_label = candidates[0]
            rec = {"start_frame": a["start_frame"], "end_frame": a["end_frame"],
                   "accel_y": a["accel_y"], "hoop": hoop_label, **shot}
            if shot["verdict"] == "shot_attempt":
                # RELEASE: extrapolate the arc's own fit backward (DECISIONS
                # 16 -- "nearest body to the arc's first point" picks whoever
                # stands under the APEX, not the shooter; the first point is
                # already several frames past release per DECISIONS 14).
                rel = find_release(a["fit_x"], a["fit_y"], a["start_frame"], tracks_by_frame)
                if rel["status"] != "found":
                    rec["shooter"] = {"status": "no_confident_shooter",
                                      "reason": f"no tracked body within "
                                                f"{RELEASE_DIST_GATE_PX}px of the "
                                                f"back-extrapolated release path within "
                                                f"{RELEASE_BACK_MAX_FRAMES} frames"}
                elif not (tracks_span[0] <= rel["release_frame"] < tracks_span[1]):
                    rec["shooter"] = {"status": "no_identity_data",
                                      "reason": f"release frame {rel['release_frame']} "
                                                f"outside tracks cache span {tracks_span}",
                                      "release": rel}
                else:
                    tid = rel["track_id"]
                    ident = identity_by_ft.get((rel["release_frame"], tid))
                    if ident is None:
                        rec["shooter"] = {"status": "no_identity_data",
                                          "reason": f"track {tid} has no identity "
                                                    f"event at release frame "
                                                    f"{rel['release_frame']}",
                                          "release": rel}
                    else:
                        identity_id, identity_state = ident
                        status = ("attributed" if identity_state == "confirmed"
                                 else "review_item")
                        rec["shooter"] = {"status": status, "track_id": tid,
                                          "identity_id": identity_id,
                                          "identity_state": identity_state,
                                          "release": rel}
            results.append(rec)

    out_json = os.path.join(out_dir, f"{CLIP_NAME}_shot_attempts.json")
    json.dump({"clip": CLIP_NAME, "hoop_radius_px": HOOP_RADIUS_PX,
               "tracks_span": list(tracks_span), "attempts": results},
              open(out_json, "w", encoding="utf-8"), indent=2)

    n_shots = sum(1 for r in results if r["verdict"] == "shot_attempt")
    print(f"\n================ SHOT ATTEMPTS ({CLIP_NAME}) ================")
    print(f"  arcs evaluated: {len(results)}  shot attempts: {n_shots}")
    for r in results:
        if r["verdict"] != "shot_attempt":
            continue
        sh = r.get("shooter", {})
        rel = sh.get("release", {})
        print(f"  SHOT {r['start_frame']}..{r['end_frame']} hoop={r.get('hoop')} "
              f"min_dist={r['min_dist']}px at f={r['at_frame']}  "
              f"shooter={sh.get('status')} "
              f"track={rel.get('track_id', sh.get('track_id', ''))} "
              f"release_f={rel.get('release_frame', '')} "
              f"{sh.get('identity_state', sh.get('reason',''))}")
    print(f"  wrote {out_json}")

    hoop_near_by_frame = {r["frame_index"]: tuple(r["hoop_near_px"]) for r in hoop_doc["frames"]
                          if r.get("hoop_near_px") is not None}
    _render_overlay(CLIP_NAME, arcs_doc, hoop_by_frame, hoop_near_by_frame, results, out_dir)


def _render_overlay(clip_name, arcs_doc, hoop_by_frame, hoop_near_by_frame, results, out_dir):
    import cv2
    import clip_config
    import run_tracking
    CLIP = getattr(clip_config, f"{clip_name}_CLIP")
    clip_config.ACTIVE_CLIP = CLIP
    span_start, span_len = arcs_doc["span_start"], arcs_doc["span_len"]
    subclip, fps, _n = run_tracking.extract_subclip(CLIP.video_path, span_start, span_len)
    cap = cv2.VideoCapture(subclip)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = os.path.join(out_dir, f"{clip_name}_shot_attempts_overlay.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    RED, GREEN, GRAY = (0, 0, 255), (0, 255, 0), (150, 150, 150)
    MAGENTA, CYAN = (255, 0, 255), (255, 255, 0)

    shot_spans = {(r["start_frame"], r["end_frame"]): r for r in results
                  if r["verdict"] == "shot_attempt"}
    curves = []
    for chain in arcs_doc["chains"]:
        pts = {p[0]: (p[1], p[2]) for p in chain["points"]}
        for a in chain.get("arcs", []):
            key = (a["start_frame"], a["end_frame"])
            color = RED if key in shot_spans else GREEN
            seg = [(f, pts[f][0], pts[f][1]) for f in range(a["start_frame"], a["end_frame"] + 1)
                   if f in pts]
            curves.append((a["start_frame"], a["end_frame"], seg, color))

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        f = span_start + i
        hp = hoop_by_frame.get(f)
        if hp is not None:
            cv2.circle(frame, (int(hp[0]), int(hp[1])), int(HOOP_RADIUS_PX), MAGENTA, 2)
        hp_near = hoop_near_by_frame.get(f)
        if hp_near is not None:
            cv2.circle(frame, (int(hp_near[0]), int(hp_near[1])), int(HOOP_RADIUS_PX), CYAN, 2)
        live_shot = False
        for (sf, ef, seg, color) in curves:
            if f < sf:
                continue
            past = [(int(x), int(y)) for (pf, x, y) in seg if pf <= f]
            for a, b in zip(past, past[1:]):
                cv2.line(frame, a, b, color, 3)
            if sf <= f <= ef and past:
                cv2.circle(frame, past[-1], 12, color, 2)
                if color == RED:
                    live_shot = True
        label = f"f={f} t={f/fps:04.1f}s" + ("  SHOT ATTEMPT" if live_shot else "")
        cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255) if live_shot else (255, 255, 255), 2)
        writer.write(frame)
        i += 1
    writer.release()
    cap.release()
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
