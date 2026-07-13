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


def apex_index(points):
    """Index of the highest point (MIN cy -- image y grows downward)."""
    return min(range(len(points)), key=lambda i: points[i][2])


def classify_shot(points, hoop_at, radius=HOOP_RADIUS_PX):
    """points: [(frame, cx, cy), ...] of one claimed arc (chronological).
    hoop_at: callable frame -> (hx, hy) or None if no hoop position that
    frame. Only points at-or-after the apex are checked (descending half,
    apex inclusive) -- a rising ball cannot have reached the rim yet."""
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

    hoop_by_frame = {r["frame_index"]: tuple(r["hoop_px"]) for r in hoop_doc["frames"]
                     if r["hoop_px"] is not None}
    hoop_at = lambda f: hoop_by_frame.get(f)  # noqa: E731

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
            shot = classify_shot(seg, hoop_at)
            rec = {"start_frame": a["start_frame"], "end_frame": a["end_frame"],
                   "accel_y": a["accel_y"], **shot}
            if shot["verdict"] == "shot_attempt":
                rel_f, rel_x, rel_y = seg[0]
                if not (tracks_span[0] <= rel_f < tracks_span[1]):
                    rec["shooter"] = {"status": "no_identity_data",
                                      "reason": f"frame {rel_f} outside tracks "
                                                f"cache span {tracks_span}"}
                else:
                    nearest = nearest_track_feet(tracks_by_frame.get(rel_f, []), rel_x, rel_y)
                    if nearest is None:
                        rec["shooter"] = {"status": "no_identity_data",
                                          "reason": "no tracked bodies that frame"}
                    else:
                        tid, dist = nearest
                        ident = identity_by_ft.get((rel_f, tid))
                        if ident is None:
                            rec["shooter"] = {"status": "no_identity_data",
                                              "reason": f"track {tid} has no identity "
                                                        f"event at frame {rel_f}",
                                              "nearest_track_id": tid,
                                              "nearest_track_dist_px": round(dist, 1)}
                        else:
                            identity_id, identity_state = ident
                            status = ("attributed" if identity_state == "confirmed"
                                     else "review_item")
                            rec["shooter"] = {"status": status, "track_id": tid,
                                              "identity_id": identity_id,
                                              "identity_state": identity_state,
                                              "dist_px": round(dist, 1)}
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
        print(f"  SHOT {r['start_frame']}..{r['end_frame']} "
              f"min_dist={r['min_dist']}px at f={r['at_frame']}  "
              f"shooter={sh.get('status')} "
              f"{sh.get('identity_id', sh.get('nearest_track_id', ''))} "
              f"{sh.get('identity_state', sh.get('reason',''))}")
    print(f"  wrote {out_json}")

    _render_overlay(CLIP_NAME, arcs_doc, hoop_by_frame, results, out_dir)


def _render_overlay(clip_name, arcs_doc, hoop_by_frame, results, out_dir):
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
    RED, GREEN, GRAY, MAGENTA = (0, 0, 255), (0, 255, 0), (150, 150, 150), (255, 0, 255)

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
