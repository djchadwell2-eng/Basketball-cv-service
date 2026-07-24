"""Player-tracker plan item 3 (tasks/todo.md): harvest CROWDED-PAINT frames
for a player-labeling session. Targeted, not random: DECISIONS §22 measured
ByteTrack shattering into 10 short fragments during a single layup because
the paint is crowded (rebounds/post-ups/cuts). This picks the frames where
that crowding is worst -- highest player-track density near the carried
hoop position -- plus the known layup spans (ground truth from TEST_LOG),
across both clips.

Read-only over cached tracks + hoop anchors + video; writes images only.

Run:  .venv/Scripts/python spikes/harvest_player_frames.py
"""
from __future__ import annotations

import json
import os
import sys

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
import clip_config

OUT_DIR = os.path.join(_HERE, "out", "label_harvest_players")
PAINT_RADIUS_PX = 350.0   # generous -- paint + surrounding traffic, not just the restricted area
TOP_N_DENSE = 60          # highest-density frames per clip, beyond the known spans

# Known layup/crowded spans (ground truth from TEST_LOG.md / DECISIONS §22)
KNOWN_SPANS = {
    "TEST1": [(165, 184), (242, 250), (581, 592)],
    "HARD":  [(395, 445), (350, 390), (1180, 1220)],
}


def _hoop_by_frame(clip):
    path = os.path.join(_HERE, "out", f"{clip}_hoop_track.json")
    if not os.path.exists(path):
        return {}
    doc = json.load(open(path, encoding="utf-8"))
    out = {}
    for r in doc["frames"]:
        pts = [r[k] for k in ("hoop_far_px", "hoop_near_px") if r.get(k) is not None]
        if pts:
            out[r["frame_index"]] = pts
    return out


def _density_by_frame(clip):
    path = os.path.join(_ROOT, "phase2", "out", f"{clip}_tracks_raw.json")
    doc = json.load(open(path, encoding="utf-8"))
    hoops = _hoop_by_frame(clip)
    density = {}
    for fr in doc["frames"]:
        f = fr["frame_index"]
        hp = hoops.get(f)
        if not hp:
            continue
        n = 0
        for t in fr["tracks"]:
            x1, y1, x2, y2 = t["bbox"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if any(((cx - hx) ** 2 + (cy - hy) ** 2) ** 0.5 <= PAINT_RADIUS_PX for hx, hy in hp):
                n += 1
        density[f] = n
    return density


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    picks = {}
    for clip, spans in KNOWN_SPANS.items():
        s = picks.setdefault(clip, set())
        for a, b in spans:
            s.update(range(a, b + 1))

        density = _density_by_frame(clip)
        top = sorted(density, key=lambda f: -density[f])[:TOP_N_DENSE]
        s.update(top)
        if density:
            print(f"[harvest] {clip}: top density frame has {density[top[0]]} tracks "
                  f"within {PAINT_RADIUS_PX:.0f}px of a hoop")

    total = 0
    for clip, frames in picks.items():
        CLIP = getattr(clip_config, f"{clip}_CLIP")
        cap = cv2.VideoCapture(CLIP.video_path)
        for f in sorted(frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                print(f"  [warn] {clip} frame {f}: read failed, skipped")
                continue
            out = os.path.join(OUT_DIR, f"{clip}_{f:05d}.jpg")
            cv2.imwrite(out, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            total += 1
        cap.release()
        print(f"[harvest] {clip}: {len(frames)} frames requested")
    print(f"[harvest] wrote {total} JPGs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
