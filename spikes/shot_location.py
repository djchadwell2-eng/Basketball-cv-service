"""Phase 5 step 4C -- SHOT LOCATION + shot chart.

Location = the hinted shooter's court-feet position AT THE ESTIMATED
RELEASE FRAME (spikes/shot_attempts.py's find_release), read from the
oncourt cache -- a free join against data the identity layer already
trusts (phase2/oncourt.py). Deliberately NOT "arc origin -> court feet":
the arc origin is an ELEVATED mid-air pixel, and pushing it through the
floor homography would smear it to the wrong point on the floor (the same
reason the hoop needed its own anchor, DECISIONS 15). The shooter's own
feet are the honest ground-truth point.

No shooter hint, or the shooter's own on-court classifier abstained (its
"on" flag is False) -> location_unknown. Trusting oncourt's own
abstention here is deliberate: it is already a validated signal (DECISIONS
9's ROI-mask seeding), not something this layer should second-guess.

Nothing here writes into team_events or any existing artifact.
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

COURT_LEN, COURT_WID = 84.0, 50.0     # HS dims, same as stage4_courtmap.py
LANE_Y0, LANE_Y1 = 19.0, 31.0
FT_X = 19.0
CIRCLE_R = 6.0
HOOP_DX = 5.25                        # basket center distance from baseline
R3 = 19.75                            # HS 3pt radius


def feet_to_px(fx, fy, scale):
    """Court feet -> flat-diagram pixels (top-down, no homography -- this
    is a DIAGRAM, not a video overlay)."""
    return (int(round(fx * scale)), int(round(fy * scale)))


def find_shot_location(attempt, oncourt_doc):
    """attempt: one record from {clip}_shot_attempts.json ('shot_attempt'
    verdict). oncourt_doc: the loaded {clip}_oncourt.json. Returns a dict
    with status 'located' (+ court_feet) or 'location_unknown' (+ reason)."""
    sh = attempt.get("shooter", {})
    rel = sh.get("release")
    if rel is None:
        return {"status": "location_unknown",
                "reason": f"no release estimate (shooter status: {sh.get('status')})"}
    frame, tid = rel["release_frame"], rel["track_id"]
    for fr in oncourt_doc["frames"]:
        if fr["frame_index"] != frame:
            continue
        rec = fr["tracks"].get(str(tid))
        if rec is None:
            return {"status": "location_unknown",
                    "reason": f"track {tid} has no oncourt record at frame {frame}"}
        if not rec["on"]:
            return {"status": "location_unknown",
                    "reason": f"track {tid} classified OFF-court at frame {frame} "
                              f"(trusting the oncourt classifier's own abstention)"}
        return {"status": "located", "court_feet": rec["court_feet"],
                "release_frame": frame, "track_id": tid}
    return {"status": "location_unknown",
            "reason": f"frame {frame} not present in the oncourt cache"}


def _court_polylines():
    L, W = COURT_LEN, COURT_WID
    polys = [
        [(0, 0), (L, 0), (L, W), (0, W), (0, 0)],
        [(L / 2, 0), (L / 2, W)],
        [(0, LANE_Y0), (FT_X, LANE_Y0), (FT_X, LANE_Y1), (0, LANE_Y1), (0, LANE_Y0)],
        [(L, LANE_Y0), (L - FT_X, LANE_Y0), (L - FT_X, LANE_Y1), (L, LANE_Y1), (L, LANE_Y0)],
    ]
    cx, cy = L / 2, W / 2
    circle = [(cx + CIRCLE_R * np.cos(t), cy + CIRCLE_R * np.sin(t))
              for t in np.linspace(0, 2 * np.pi, 48)]
    polys.append(circle)
    # two 3pt arcs (same geometry as spikes/stage6_arc_overlay.py)
    for hoop_x, ang0, ang1 in [(HOOP_DX, -90, 90), (L - HOOP_DX, 90, 270)]:
        arc = [(hoop_x + R3 * np.cos(np.radians(d)), cy + R3 * np.sin(np.radians(d)))
               for d in range(ang0, ang1 + 1, 4)]
        polys.append(arc)
    return polys


def _court_feet_to_diagram_px(fx, fy, scale):
    """Court feet -> diagram pixels, Y-FLIPPED to match the ALREADY-
    VALIDATED convention in phase1/stage3_heatmap.py (matplotlib
    origin='lower': near-sideline y=0 at the BOTTOM, far-sideline y=W at
    the TOP). cv2 images index row 0 at the top, so that convention means
    "diagram row" = (COURT_WID - fy) * scale, not fy * scale directly --
    plotting fy straight into image-space silently mirrors near/far
    top-to-bottom (caught by user eyeball on the real HARD shot)."""
    return feet_to_px(fx, COURT_WID - fy, scale)


def render_shot_chart(clip_name, located_shots, out_path, scale=10):
    W, H = int(COURT_LEN * scale), int(COURT_WID * scale)
    WOOD = (219, 233, 244)
    img = np.full((H, W, 3), WOOD, dtype=np.uint8)
    for poly in _court_polylines():
        pts = [_court_feet_to_diagram_px(x, y, scale) for (x, y) in poly]
        for a, b in zip(pts, pts[1:]):
            cv2.line(img, a, b, (120, 90, 40), 2)
    GREEN, ORANGE = (0, 160, 0), (0, 140, 255)
    for s in located_shots:
        color = GREEN if s["status"] == "confirmed" else ORANGE
        x, y = _court_feet_to_diagram_px(s["court_feet"][0], s["court_feet"][1], scale)
        cv2.circle(img, (x, y), 10, color, -1)
        cv2.circle(img, (x, y), 10, (0, 0, 0), 2)
        cv2.putText(img, s["status"], (x + 14, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1)
    cv2.putText(img, f"{clip_name} shot chart ({len(located_shots)} located)",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.imwrite(out_path, img)


def main():
    CLIP_NAME = "HARD"
    out_dir = os.path.join(_HERE, "out")
    sa_doc = json.load(open(os.path.join(out_dir, f"{CLIP_NAME}_shot_attempts.json"),
                            encoding="utf-8"))
    oncourt_doc = json.load(open(os.path.join(_ROOT, "phase2", "out",
                                              f"{CLIP_NAME}_oncourt.json"), encoding="utf-8"))

    results = []
    for a in sa_doc["attempts"]:
        if a["verdict"] != "shot_attempt":
            continue
        loc = find_shot_location(a, oncourt_doc)
        rec = {"start_frame": a["start_frame"], "end_frame": a["end_frame"],
               "shooter_status": a.get("shooter", {}).get("status"), **loc}
        results.append(rec)

    out_json = os.path.join(out_dir, f"{CLIP_NAME}_shot_locations.json")
    json.dump({"clip": CLIP_NAME, "locations": results},
              open(out_json, "w", encoding="utf-8"), indent=2)
    print(f"\n================ SHOT LOCATIONS ({CLIP_NAME}) ================")
    for r in results:
        print(f"  {r['start_frame']}..{r['end_frame']}  shooter={r['shooter_status']}  "
              f"{r['status']}  {r.get('court_feet', r.get('reason'))}")
    print(f"  wrote {out_json}")

    located = [{"status": r["shooter_status"], "court_feet": r["court_feet"]}
              for r in results if r["status"] == "located"]
    chart_path = os.path.join(out_dir, f"{CLIP_NAME}_shot_chart.png")
    render_shot_chart(CLIP_NAME, located, chart_path)
    print(f"  wrote {chart_path}")


if __name__ == "__main__":
    main()
