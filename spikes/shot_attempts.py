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

HOOP_RADIUS_PX = 100.0   # measured slack: DECISIONS 15 marker sat ~20-40px off
                         # the true rim center even across a keyframe switch;
                         # this adds room for ball size + motion blur without
                         # reaching anywhere near a dribble's floor-level apex

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


def classify_shot(points, hoop_at, radius=HOOP_RADIUS_PX):
    """points: [(frame, cx, cy), ...] of one claimed arc (chronological).
    hoop_at: callable frame -> (hx, hy) or None if no hoop position that
    frame. Only points at-or-after the apex are checked (descending half,
    apex inclusive) -- a rising ball cannot have reached the rim yet.

    ORIGIN GATE (DECISIONS 18, KNOWN DEBT candidate fix (b), validated on
    real data): a real shot RELEASES away from the hoop and ARRIVES close;
    a rim deflection/continuation STARTS already close and moves away.
    Measured on all 4 arcs found so far: real shots start 285-338px from
    the hoop and end 19-55px away; the two known false positives start
    26-69px away and end 374-384px away -- a clean, consistent split. If
    the arc's FIRST point is within `radius` of the hoop AT THAT FRAME,
    reject it regardless of what happens later -- it did not originate as
    a fresh release. If the hoop position at the first frame is unknown,
    this gate is skipped (absence of data must not manufacture a
    rejection) and classification falls back to the apex-based check only.
    KNOWN TRADE-OFF, accepted not fixed: a genuine close-range shot (e.g.
    a layup released near the rim) would also fail this gate -- logged as
    an accepted limitation, not silently ignored."""
    f0, x0, y0 = points[0]
    h0 = hoop_at(f0)
    if h0 is not None:
        origin_dist = ((x0 - h0[0]) ** 2 + (y0 - h0[1]) ** 2) ** 0.5
        if origin_dist <= radius:
            return {"verdict": "not_shot", "min_dist": None,
                    "reason": f"originates within {radius}px of the hoop "
                              f"({round(origin_dist, 1)}px) -- a deflection/"
                              f"continuation, not a fresh release"}
    i0 = apex_index(points)
    best = None
    for (f, x, y) in points[i0:]:
        h = hoop_at(f)
        if h is None:
            continue
        d = ((x - h[0]) ** 2 + (y - h[1]) ** 2) ** 0.5
        if best is None or d < best[0]:
            best = (d, f, x, y)
    if best is not None and best[0] <= radius:
        return {"verdict": "shot_attempt", "min_dist": round(best[0], 1),
                "at_frame": best[1]}
    return {"verdict": "not_shot",
            "min_dist": round(best[0], 1) if best else None}


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
    CLIP_NAME = "HARD"
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
            shot_far = classify_shot(seg, hoop_at_far)
            shot_near = classify_shot(seg, hoop_at_near)
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
