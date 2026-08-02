"""Follow-up to TEST 16/19 + the real wiring (2026-07-31): the window
majority rule fixed HARD's 2 known false positives perfectly, but cost
TEST1 two real shots (314-327 "shot B" -- a KNOWN predicted risk -- and
571-589 "layup 3" -- a NEW regression). DJ asked to test every reasonable
variant before picking one, rather than guessing.

Cheap by design: pose inference is the expensive part, so this script pays
that cost ONCE per event, sampling every frame (not every 3rd) over a WIDE
window, and prints the full per-frame vote list. Every threshold rule below
is then just arithmetic over already-computed votes -- free to compare.

Read-only. Uses the v4 ball detections + hoop track already on disk from
the real HARD/TEST1 runs (spikes/out/{clip}_ball_detections.json / _hoop_track.json).

Run:  .venv/Scripts/python spikes/player_signal_experiment.py
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from local_weights_check import CONF_FLOOR
import pose_shot_check as psc

WIDE_WINDOW = 30     # 1.0s -- generous enough to compare narrower rules against
STEP = 1             # every frame, so any coarser sampling is a subset of this data

# Every candidate the real pipeline (v4 + player-signal filter) actually
# evaluated this session (spikes/out/{clip}_shot_attempts.json).
EVENTS = [
    ("HARD",  351,  374, "near", "REAL", "verified jump shot"),
    ("HARD", 1187, 1214, "far",  "REAL", "verified jump shot"),
    ("HARD",  403,  415, "near", "FAKE", "rebound caught -> dished out"),
    ("HARD", 1352, 1375, "far",  "FAKE", "cross-court pass"),
    ("TEST1",  57,   77, "far",  "REAL", "verified jump shot A"),
    ("TEST1", 166,  184, "far",  "REAL", "verified layup 1"),
    ("TEST1", 232,  248, "far",  "REAL", "verified layup 2"),
    ("TEST1", 314,  327, "far",  "REAL", "verified jump shot B (KNOWN short-flight risk)"),
    ("TEST1", 571,  589, "near", "REAL", "verified layup 3"),
    ("TEST1",  74,   98, "far",  "?",    "unverified marginal candidate, not ground truth"),
]


def _load_clip(clip):
    det = json.load(open(os.path.join(_HERE, "out", f"{clip}_ball_detections.json"),
                         encoding="utf-8"))
    frames = [{"frame_index": fr["frame_index"],
               "detections": [d for d in fr["detections"] if d["conf"] >= CONF_FLOOR]}
              for fr in det["frames"]]
    ball = psc.ball_center_by_frame(frames)
    hoop_doc = json.load(open(os.path.join(_HERE, "out", f"{clip}_hoop_track.json"),
                              encoding="utf-8"))
    hoop = {r["frame_index"]: r for r in hoop_doc["frames"]}
    import clip_config
    video = getattr(clip_config, f"{clip}_CLIP").video_path
    return ball, hoop, video


def main():
    from ultralytics import YOLO
    model = YOLO(psc.POSE_WEIGHTS)

    cache, rows = {}, []
    for clip, s, e, side, truth, desc in EVENTS:
        if clip not in cache:
            cache[clip] = _load_clip(clip)
        ball, hoop, video = cache[clip]

        votes = []
        for f in range(e, e + WIDE_WINDOW + 1, STEP):
            v = psc._ends_at_hand(model, video, ball, hoop, side, f)
            votes.append(v[0] if v else None)
        rows.append(dict(clip=clip, start=s, end=e, truth=truth, desc=desc, votes=votes))

    print(f"\n{'='*100}")
    print("PER-FRAME VOTES, end_frame .. end_frame+30 (every frame) -- raw data for every rule below")
    print(f"{'='*100}")
    for r in rows:
        v = r["votes"]
        print(f"{r['truth']:5s} {r['clip']:5s} {str(r['start'])+'-'+str(r['end']):11s} {r['desc']}")
        print("      " + " ".join((x or ".")[:1] for x in v)
              + "   (H=HAND, r=rim, .=no reading; frames end.." + str(r['end'] + WIDE_WINDOW) + ")")

    def score(rule_name, predict):
        print(f"\n--- {rule_name} ---")
        correct = total = 0
        for r in rows:
            if r["truth"] not in ("REAL", "FAKE"):
                pred = predict(r["votes"])
                print(f"  {r['clip']:5s} {r['start']}-{r['end']} (unverified)  predicted={pred}  {r['desc']}")
                continue
            pred = predict(r["votes"])
            want_shot = r["truth"] == "REAL"
            got_shot = pred != "HAND"       # "HAND" verdict = filter rejects it
            ok = got_shot == want_shot
            correct += ok
            total += 1
            print(f"  {'OK ' if ok else 'BAD'} {r['truth']:5s} {r['clip']:5s} "
                  f"{r['start']}-{r['end']:5d}  predicted={pred!s:5s}  {r['desc']}")
        print(f"  SCORE: {correct}/{total}")

    # RULE A (what's live right now): majority over end..end+15, step 3.
    score("RULE A -- current: majority vote, 0.5s window, step 3",
          lambda v: psc.window_majority([x for x in v[0:16:3] if x]))

    # RULE B: single frame at arrival only (the ORIGINAL TEST 16 rule, known
    # to flip on 3/10 events when the endpoint shifts -- included as the
    # no-window baseline for comparison).
    score("RULE B -- single frame at end_frame only, no window",
          lambda v: v[0])

    # RULE C: UNANIMOUS instead of majority -- only reject if EVERY vote in
    # the window says HAND (a rebounder grabbing the ball late no longer
    # flips the verdict; a pass/caught ball has no rim reading at all).
    score("RULE C -- unanimous HAND required (0.5s window, step 3)",
          lambda v: ("HAND" if [x for x in v[0:16:3] if x] and
                     all(x == "HAND" for x in v[0:16:3] if x) else "rim"))

    # RULE D: narrower window (0.2s, step 1) -- catches a ball caught
    # IMMEDIATELY (pass/rebound) without giving a made shot's rebounder
    # time to arrive and flip it.
    score("RULE D -- majority vote, 0.2s window (frames end..end+6, step 1)",
          lambda v: psc.window_majority([x for x in v[0:7] if x]))

    # RULE E: FIRST reading only, not majority -- does the ball touch the
    # rim/net at ANY point at or immediately after arrival, before checking
    # who ends up holding it. If it EVER reads "rim" in the first few
    # frames, call it a shot regardless of what happens after.
    score("RULE E -- ever-rim-in-first-6-frames wins (else majority of 0.5s)",
          lambda v: ("rim" if "rim" in [x for x in v[0:7] if x]
                     else psc.window_majority([x for x in v[0:16:3] if x])))

    out = os.path.join(_HERE, "out", "player_signal_experiment.json")
    json.dump(rows, open(out, "w", encoding="utf-8"), indent=1)
    print(f"\nraw votes -> {os.path.basename(out)}")


if __name__ == "__main__":
    main()
