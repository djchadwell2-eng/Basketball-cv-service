"""STAGE 2 -- honest loss + candidate recovery: print breaks + by-eye overlay.

Drives the identity state machine over the cached tracks. On occlusion a track
goes LOST (nothing attributed); on reappearance a position/motion relink makes it
a CANDIDATE (never CONFIRMED -- the set_confirmed lock holds). Prints every break
and renders an overlay video labelling each track's state per frame, with LOST
ghosts and CANDIDATE relinks visible.
"""

from __future__ import annotations

import json
import os
import sys

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))                          # repo root (clip_config)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "spikes"))

from clip_config import ACTIVE_CLIP as CLIP
import identity as idmod
from identity import IdentityState
from tracking import Track

TRACKS_JSON = CLIP.tracks_cache_path
OUT_DIR = os.path.join(_HERE, "out")
DISP_W, DISP_H = 1280, 720

_COLOR = {
    IdentityState.UNKNOWN: (180, 180, 180),    # gray  = abstain
    IdentityState.CANDIDATE: (0, 255, 255),    # yellow = probably-them, unverified
    IdentityState.CONFIRMED: (0, 255, 0),      # green  = certain (none yet)
    IdentityState.LOST: (0, 0, 255),           # red    = occluded ghost
}


def load(path):
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    frames = [(fr["frame_index"],
               [Track(t["track_id"], tuple(t["bbox"])) for t in fr["tracks"]])
              for fr in doc["frames"]]
    return frames, doc


def read_span_frames(video, start, length):
    cap = cv2.VideoCapture(video)
    for _ in range(start):
        cap.grab()
    imgs = {}
    for i in range(length):
        ok, fr = cap.read()
        if not ok:
            break
        imgs[start + i] = fr
    cap.release()
    return imgs


def draw(frame, machine, fidx):
    for ident in machine.active():
        x1, y1, x2, y2 = [int(v) for v in ident.last_bbox]
        col = _COLOR[ident.state]
        thick = 3 if ident.state == IdentityState.CANDIDATE else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), col, thick)
        lbl = f"t{ident.track_id}:{ident.state.value}"
        if ident.state == IdentityState.CANDIDATE:
            lbl += f" gap{ident.evidence.get('gap_frames','?')} d{ident.evidence.get('distance_px','?')}"
        cv2.putText(frame, lbl, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    for ident in machine.lost():                       # occluded ghosts
        cx, cy = [int(v) for v in ident.last_center()]
        gap = fidx - ident.last_seen_frame
        cv2.circle(frame, (cx, cy), 9, _COLOR[IdentityState.LOST], 2)
        cv2.putText(frame, f"t{ident.track_id}:LOST g{gap}", (cx + 8, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _COLOR[IdentityState.LOST], 1)
    cv2.putText(frame, f"f={fidx}  gray=unknown  red=LOST  yellow=CANDIDATE",
                (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def main():
    frames, doc = load(TRACKS_JSON)
    imgs = read_span_frames(CLIP.video_path, doc["span_start"], doc["span_len"])

    machine = idmod.IdentityStateMachine()
    writer = cv2.VideoWriter(os.path.join(OUT_DIR, f"{CLIP.name}_stage2_recovery.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (DISP_W, DISP_H))
    reappear_frames = set()
    for (fidx, tracks) in frames:
        machine.update(fidx, tracks)
        img = imgs.get(fidx)
        if img is not None:
            draw(img, machine, fidx)
            writer.write(cv2.resize(img, (DISP_W, DISP_H)))
    writer.release()

    # save a still at each candidate reappearance for close-up eyeballing
    for b in machine.breaks:
        if b.get("result") == "candidate":
            reappear_frames.add(b["reappeared_frame"])
    for fidx in sorted(reappear_frames)[:8]:
        if fidx in imgs:
            m2 = idmod.IdentityStateMachine()
            for (fi, tr) in frames:
                m2.update(fi, tr)
                if fi == fidx:
                    break
            snap = imgs[fidx].copy()
            draw(snap, m2, fidx)
            cv2.imwrite(os.path.join(OUT_DIR, f"{CLIP.name}_stage2_reappear_{fidx}.jpg"),
                        cv2.resize(snap, (DISP_W, DISP_H)))

    # --- print every break ---
    relinks = [b for b in machine.breaks if b.get("result") == "candidate"]
    ambigs = [b for b in machine.breaks if b.get("result", "").startswith("unknown")]
    print(f"span {doc['span_start']}..+{doc['span_len']}  "
          f"identities={len(machine.all_identities())}")
    print(f"\nCANDIDATE relinks (lost -> reappeared -> candidate, NEVER confirmed): "
          f"{len(relinks)}")
    for b in relinks:
        print(f"  track t{b['lost_track_id']} lost after f={b['lost_after_frame']} "
              f"-> gap {b['gap_frames']}f -> reappeared f={b['reappeared_frame']} as "
              f"t{b['reappeared_track_id']} at {b['reappeared_at']} "
              f"(d={b['distance_px']}px) => CANDIDATE")
    print(f"\nAMBIGUOUS reappearances (kept UNKNOWN, sent to review): {len(ambigs)}")
    for b in ambigs:
        print(f"  new t{b['reappeared_track_id']} at f={b['reappeared_frame']}: "
              f"{b['contenders']} lost tracks contend => UNKNOWN")

    still_lost = [i for i in machine.lost()]
    print(f"\nstill LOST at span end (never reappeared in-window): {len(still_lost)} "
          f"-> {[f't{i.track_id}' for i in still_lost][:12]}")

    from collections import Counter
    final = Counter(i.state.value for i in machine.all_identities())
    print(f"\nfinal identity states: {dict(final)}")
    print("CONFIRMED count:", final.get("confirmed", 0),
          "(must be 0 -- no seed/second signal exists yet)")
    print(f"\nsaved overlay: {CLIP.name}_stage2_recovery.mp4 (+ reappearance stills) in {OUT_DIR}")


if __name__ == "__main__":
    main()
