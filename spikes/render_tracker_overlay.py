"""Player-tracker plan item 2 support: render a WATCHABLE video from a
tracks json (any tracker variant) so a human can eyeball for ID SWITCHES
-- the failure fragmentation metrics (distinct ids, mean lifespan)
CANNOT see: a looser matcher (e.g. TEST 6's match_thresh=0.9) can silently
merge two DIFFERENT players into one track, which looks "better" on those
metrics while being confidently wrong. Each track_id gets a STABLE color
for the whole video (hashed from the id), so if a box's color/number
stays on one person the whole time, that ID never switched; if the same
color suddenly appears on a different person, that's a real switch to
flag.

Read-only: writes a new mp4 only, touches no cache.

Run:  .venv/Scripts/python spikes/render_tracker_overlay.py TEST1 spikes/out/TEST1_tracks_mt09.json
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "phase2"))
import clip_config


def _color_for_id(track_id):
    h = hashlib.md5(str(track_id).encode()).digest()
    # keep it bright/saturated enough to read against a wood floor
    return (100 + h[0] % 156, 100 + h[1] % 156, 100 + h[2] % 156)


def main():
    clip_name = sys.argv[1] if len(sys.argv) > 1 else "TEST1"
    tracks_json = sys.argv[2] if len(sys.argv) > 2 else \
        os.path.join(_HERE, "out", f"{clip_name}_tracks_mt09.json")

    CLIP = getattr(clip_config, f"{clip_name}_CLIP")
    clip_config.ACTIVE_CLIP = CLIP
    import run_tracking

    doc = json.load(open(tracks_json, encoding="utf-8"))
    span_start, span_len = doc["span_start"], doc["span_len"]
    tag = doc.get("tracker", "tracks")
    print(f"[render] {clip_name} span {span_start}..{span_start+span_len}, tracker={tag}")

    subclip, fps, n = run_tracking.extract_subclip(CLIP.video_path, span_start, span_len)
    by_frame = {fr["frame_index"]: fr["tracks"] for fr in doc["frames"]}

    cap = cv2.VideoCapture(subclip)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = os.path.join(_HERE, "out", f"{clip_name}_{tag}_overlay.mp4")
    # half speed so a fast-moving occlusion/relink moment is actually watchable
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps / 2.0, (w, h))

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        f = span_start + i
        for t in by_frame.get(f, []):
            tid = t["track_id"]
            x1, y1, x2, y2 = [int(v) for v in t["bbox"]]
            color = _color_for_id(tid)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            cv2.putText(frame, f"id={tid}", (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, f"f={f} t={f/fps:04.1f}s", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        writer.write(frame)
        i += 1
    writer.release()
    cap.release()
    print(f"[render] wrote {out_path} ({i} frames, played at half speed)")


if __name__ == "__main__":
    main()
