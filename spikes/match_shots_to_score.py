"""Second half of DJ's scoreboard idea (2026-07-23): pair each VERIFIED shot
attempt with the scoreboard's own record to get real make/miss ground
truth, entirely independent of ball-trajectory guesswork.

Method: a verified shot is a MAKE if any score-change event (from
scoreboard_ocr_probe.py) lands within MATCH_WINDOW_SEC after the shot's
last observed frame; otherwise it's a MISS (no basket followed this
attempt within a reasonable window). Only checks VERIFIED shots (the
same hand-confirmed ground truth local_weights_check.py uses) -- this
tests the SCOREBOARD-MATCHING method in isolation, not entangled with any
open ball-detector accuracy question.

Read-only: prints the match table, writes nothing.

Run:  .venv/Scripts/python spikes/match_shots_to_score.py HARD
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

MATCH_WINDOW_SEC = 6.0      # generous: inbound/foul-shot delay after a make can add a beat

# Same verified spans local_weights_check.py's GROUND_TRUTH uses.
VERIFIED_SHOTS = {
    "HARD": [(351, 375, "near"), (1177, 1214, "far")],
    "TEST1": [(58, 77, "far"), (166, 184, "far"), (236, 250, "far"),
              (314, 327, "far"), (571, 589, "near")],
}


def main():
    clip_name = sys.argv[1] if len(sys.argv) > 1 else "HARD"
    sb_path = os.path.join(_HERE, "out", f"{clip_name}_scoreboard_ocr.json")
    sb = json.load(open(sb_path, encoding="utf-8"))
    fps = 30.0

    print(f"[match] {clip_name}: {len(sb['events'])} score-change event(s) on record")
    for e in sb["events"]:
        print(f"  {e['t_sec']}s  {e['from']} -> {e['to']}")
    print()

    for (start, end, hoop) in VERIFIED_SHOTS.get(clip_name, []):
        end_t = end / fps
        window_end = end_t + MATCH_WINDOW_SEC
        matches = [e for e in sb["events"] if end_t <= e["t_sec"] <= window_end]
        verdict = "MAKE (score changed nearby)" if matches else "MISS (no score change in window)"
        print(f"shot {start}-{end} ({hoop} hoop, ends {end_t:.1f}s, "
              f"window {end_t:.1f}-{window_end:.1f}s): {verdict}")
        for m in matches:
            print(f"    matched event: {m['t_sec']}s  {m['from']} -> {m['to']}")


if __name__ == "__main__":
    main()
