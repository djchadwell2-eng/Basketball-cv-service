"""EYEBALL DELIVERABLE for ball touches -- the video DJ actually judges.

Numbers cannot tell anyone whether "nearest to the ball" found the right girl.
This draws the answer on the footage so a human can watch twenty seconds and
say yes or no. Rendering is DELIBERATELY separate from spikes/ball_touch.py so
the measurement stays a pure, fast, testable function with no cv2 in it.

What you see, and what each colour MEANS:
    yellow dot        where the system thinks the ball is
    GREEN thick box   this girl is credited with a TOUCH right now, on the
                      evidence of THIS frame
    DARK GREEN box    still her touch, but the ball is not visible this frame --
                      the run is being BRIDGED across a detector dropout. Shown
                      distinctly so nobody mistakes a bridged frame for evidence
    ORANGE thick box  she looked like she had it, but the run was too short to
                      count -- this is the system THROWING SOMETHING AWAY, and
                      it is shown rather than hidden
    RED thin boxes    two girls too close to call -- CONTESTED, nobody credited
    grey thin boxes   everyone else being tracked
    top banner        the plain-English verdict for this exact frame

Usage (one clip per process -- clip_config.ACTIVE_CLIP binds at import):
    .venv/Scripts/python.exe spikes/render_ball_touches.py TEST1
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "phase2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ball_touch as bt                                          # noqa: E402
from local_weights_check import CONF_FLOOR                       # noqa: E402

BALL = (0, 255, 255)        # yellow
CREDITED = (0, 220, 0)      # green
BRIDGED = (0, 110, 0)       # dark green -- inside her touch, no evidence here
DROPPED = (0, 165, 255)     # orange
CONTESTED = (60, 60, 255)   # red
OTHER = (140, 140, 140)     # grey
WHITE = (255, 255, 255)


def _label(img, text, xy, color, scale=0.7, thick=2):
    import cv2
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thick + 3, cv2.LINE_AA)          # outline so it reads on any floor
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color,
                thick, cv2.LINE_AA)


def render(config, out_path=None, touches_path=None, exclude=()):
    import cv2

    out_dir = os.path.join(_ROOT, "spikes", "out")
    det = json.load(open(os.path.join(out_dir, f"{config.name}_ball_detections.json"),
                         encoding="utf-8"))
    touches_path = touches_path or os.path.join(
        out_dir, f"{config.name}_ball_touches.json")
    touch_doc = json.load(open(touches_path, encoding="utf-8"))
    tracks_doc = json.load(open(config.tracks_cache_path, encoding="utf-8"))
    # the per-frame verdicts must be recomputed with the SAME exclusions the
    # touches were built with, or the video disagrees with its own JSON.
    exclude = frozenset(exclude or touch_doc.get("excluded_tracks", ()))

    ball_by_frame = {fr["frame_index"]: bt.ball_position(fr["detections"], CONF_FLOOR)
                     for fr in det["frames"]}
    tracks_by_frame = {fr["frame_index"]: fr["tracks"] for fr in tracks_doc["frames"]}

    # frame -> the accepted touch covering it (only frames she was CREDITED on)
    touch_at = {}
    for t in touch_doc["touches"]:
        for f in range(t["start_frame"], t["end_frame"] + 1):
            touch_at[f] = t

    a, b = touch_doc["overlap_span"]
    fps = touch_doc.get("fps") or 30.0
    cap = cv2.VideoCapture(config.video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    for _ in range(a):
        cap.grab()

    out_path = out_path or os.path.join(out_dir, f"{config.name}_ball_touches_overlay.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for f in range(a, b + 1):
        ok, img = cap.read()
        if not ok:
            break
        tracks = tracks_by_frame.get(f, [])
        ball = ball_by_frame.get(f)
        verdict = bt.holder_at_frame(ball, tracks, exclude=exclude)

        touch = touch_at.get(f)
        # THIS frame's evidence vs the touch it sits inside. A frame the
        # detector missed is drawn as BRIDGED, never as evidence -- and never
        # as "no holder" either, which would hide a touch the system did claim.
        on_evidence = touch is not None and verdict.get("track_id") == touch["track_id"]
        highlight = {}
        if touch is not None and (on_evidence or verdict["status"] == "no_ball"):
            ident = touch["identity"]
            who = (f"#{ident['jersey_number']}" if ident["jersey_number"] is not None
                   else "unnamed body")
            if on_evidence:
                highlight[touch["track_id"]] = (CREDITED, f"{who} HAS THE BALL")
                banner, bcolor = (f"{who} HAS THE BALL  ({ident['status']})", CREDITED)
            else:
                highlight[touch["track_id"]] = (BRIDGED, f"{who} -- ball not visible")
                banner, bcolor = (f"still {who}'s touch -- ball NOT VISIBLE this "
                                  f"frame (bridged, not evidence)", BRIDGED)
        elif verdict["status"] == "held":
            highlight[verdict["track_id"]] = (DROPPED, "too brief -- dropped")
            banner, bcolor = ("looked like a hold, TOO BRIEF TO COUNT -- "
                              "thrown away", DROPPED)
        elif verdict["status"] == "contested":
            for tid in verdict["track_ids"]:
                highlight[tid] = (CONTESTED, "contested")
            banner, bcolor = ("CONTESTED -- two players too close to call, "
                              "nobody credited", CONTESTED)
        elif verdict["status"] == "no_ball":
            banner, bcolor = ("NO HOLDER -- cannot see the ball this frame", WHITE)
        elif verdict["status"] == "too_far":
            banner, bcolor = ("NO HOLDER -- ball is in the air, away from "
                              "everyone", WHITE)
        else:
            banner, bcolor = (f"NO HOLDER -- {verdict['status']}", WHITE)

        for t in tracks:
            x1, y1, x2, y2 = (int(v) for v in t["bbox"])
            color, text = highlight.get(t["track_id"], (OTHER, None))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 4 if text else 1)
            if text:
                _label(img, text, (x1, max(y1 - 10, 24)), color, 0.8, 2)

        if ball is not None:
            bx, by = int(ball[0]), int(ball[1])
            cv2.circle(img, (bx, by), 14, BALL, 3)
            cv2.circle(img, (bx, by), 3, BALL, -1)

        cv2.rectangle(img, (0, 0), (w, 56), (0, 0, 0), -1)
        _label(img, banner, (16, 38), bcolor, 0.9, 2)
        _label(img, f"f={f}  t={f / fps:.1f}s", (w - 260, 38), WHITE, 0.7, 2)
        writer.write(img)

    writer.release()
    cap.release()
    print(f"[render_ball_touches] wrote {out_path}")
    print(f"  frames {a}..{b}  ({(b - a + 1) / fps:.1f}s)  "
          f"{len(touch_doc['touches'])} touch(es) drawn in GREEN")
    return out_path


if __name__ == "__main__":
    import clip_config
    name = sys.argv[1] if len(sys.argv) > 1 else "TEST1"
    cfg = getattr(clip_config, f"{name}_CLIP")
    clip_config.ACTIVE_CLIP = cfg
    render(cfg)
