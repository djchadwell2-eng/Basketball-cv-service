"""STAGE 4 -- seeding (post-process, re-seed-on-loss) + coach review queue.

Seeding happens AFTER upload, not live. Model = re-seed at each window start:
establish identity for the tracks present (a stand-in for the coach clicking each
on-court player), track until a break, recover what continuity confidently gives
(CANDIDATE), and surface the rest. Seeding is the FIRST legitimate path to
CONFIRMED (provenance='seed') -- the first green, and ONLY from an explicit seed,
never from continuity.

Because NO second signal (jersey OCR) exists yet, every CANDIDATE and UNKNOWN goes
to the coach review queue as a one-click item. The queue is LONG by design: OCR
(the next step) is what shrinks it by safely auto-promoting candidates. We do NOT
shrink it by loosening the candidate bar or auto-confirming continuity.

NOTE (model honesty): this seeds EVERY track present at a window start, standing in
for the coach clicking the ~10 on-court players. Real on-court filtering / a click
UI is out of scope here; it does not change the mechanism being validated.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))                          # repo root (clip_config)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "spikes"))

from clip_config import ACTIVE_CLIP as CLIP
import windows as winmod
from identity import IdentityState
from tracking import Track

TRACKS_JSON = CLIP.tracks_cache_path
OUT_DIR = os.path.join(_HERE, "out")
QUEUE_JSON = os.path.join(OUT_DIR, f"{CLIP.name}_review_queue.json")
DEMO_WINDOW_SECONDS = CLIP.accumulation_window_seconds   # per-clip window (demo stand-in = 2.0s)


def load(path):
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    frames = [(fr["frame_index"],
               [Track(t["track_id"], tuple(t["bbox"])) for t in fr["tracks"]])
              for fr in doc["frames"]]
    return frames, doc


def flagged_frame(ident):
    if ident.state == IdentityState.CANDIDATE:
        return ident.evidence.get("reappeared_frame", ident.last_seen_frame)
    return ident._hist[0][0] if ident._hist else ident.last_seen_frame


def why(ident):
    ev = ident.evidence
    if ident.state == IdentityState.CANDIDATE:
        return (f"candidate: reappeared after {ev.get('gap_frames','?')}f occlusion, "
                f"{ev.get('distance_px','?')}px from motion prediction -- NOT "
                f"second-signal-verified")
    return f"unknown: {ev.get('reason','unidentified')}"


def main():
    frames, doc = load(TRACKS_JSON)
    span_start, fps = doc["span_start"], doc["fps"]
    win_frames = int(round(DEMO_WINDOW_SECONDS * fps))

    wid = winmod.WindowedIdentity(span_start, win_frames)
    seed_frames = {}                 # window -> (frame, [seeded track_ids])
    prev_win = None
    for (fidx, tracks) in frames:
        win = wid.update(fidx, tracks)
        if win != prev_win:          # window start = the (re-)seed point
            machine = wid.current_machine()
            seeded = []
            for t in tracks:
                machine.seed(t.track_id, label=f"w{win}_t{t.track_id}")   # -> CONFIRMED
                seeded.append(t.track_id)
            seed_frames[win] = (fidx, seeded)
            prev_win = win

    # --- per-window state + build the review queue ---
    queue = []
    print(f"clip={doc['clip']} span={span_start}..+{doc['span_len']}  "
          f"window={DEMO_WINDOW_SECONDS:.0f}s (demo; real stand-in ~15s)")
    for w, machine in sorted(wid.machines().items()):
        tally = Counter(i.state.value for i in machine.all_identities())
        sf, seeded = seed_frames[w]
        print(f"\nwindow {w}: seeded {len(seeded)} tracks at f={sf} -> CONFIRMED "
              f"(first legitimate green, provenance='seed')")
        print(f"  final states: {dict(tally)}")
        for ident in machine.all_identities():
            if ident.state in (IdentityState.CANDIDATE, IdentityState.UNKNOWN):
                queue.append({"window": w, "frame": flagged_frame(ident),
                              "track": ident.track_id, "state": ident.state.value,
                              "why": why(ident)})

    queue.sort(key=lambda q: (q["window"], q["frame"] or 0, q["track"] or 0))

    # --- confirmed came ONLY from seeds (safety) ---
    total_conf = sum(Counter(i.state.value for i in m.all_identities()).get("confirmed", 0)
                     for m in wid.machines().values())
    total_seeded = sum(len(s) for (_f, s) in seed_frames.values())
    print(f"\nCONFIRMED total = {total_conf} (all via provenance='seed'; seeds issued "
          f"= {total_seeded}). No continuity confirmation. second_signal still unbuilt.")

    # --- the review queue ---
    print(f"\n================ COACH REVIEW QUEUE ================")
    print(f"items needing one click: {len(queue)}")
    print("(window, frame, track, state, why)")
    for q in queue[:20]:
        print(f"  w{q['window']} f={q['frame']} t{q['track']} [{q['state']}]  {q['why']}")
    if len(queue) > 20:
        print(f"  ... and {len(queue)-20} more")
    n_cand = sum(1 for q in queue if q["state"] == "candidate")
    n_unk = sum(1 for q in queue if q["state"] == "unknown")
    print(f"\nqueue = {n_cand} candidates + {n_unk} unknowns.")
    print("This queue is LONG BY DESIGN -- with no second signal, every unverified")
    print("identity needs a human click. OCR (next step) shrinks it by SAFELY")
    print("auto-promoting candidates; we do NOT shrink it by loosening the bar.")

    with open(QUEUE_JSON, "w", encoding="utf-8") as f:
        json.dump({"clip": doc["clip"], "window_seconds_demo": DEMO_WINDOW_SECONDS,
                   "window_seconds_standin": 15.0, "items": queue}, f, indent=2)
    print(f"\nsaved review queue -> {QUEUE_JSON}")

    # --- a still per window start: the first green (seeded=CONFIRMED) ---
    cap = cv2.VideoCapture(CLIP.video_path)
    for w, (sf, seeded) in seed_frames.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(sf):
            cap.grab()
        ok, frame = cap.read()
        if not ok:
            continue
        # draw the seeded tracks green using their seed-frame boxes (all CONFIRMED).
        _fi, seed_tracks = frames[sf - span_start]
        by_id = {t.track_id: t.bbox for t in seed_tracks}
        for tid, bbox in by_id.items():
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"t{tid}:confirmed(seed)", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.putText(frame, f"WINDOW {w} SEED f={sf}: {len(seeded)} confirmed via seed",
                    (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(OUT_DIR, f"{CLIP.name}_stage4_seed_w{w}_f{sf}.jpg"),
                    cv2.resize(frame, (1280, 720)))
    cap.release()
    print(f"saved seed stills (first green) in {OUT_DIR}")


if __name__ == "__main__":
    main()
