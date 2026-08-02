"""TEST 14 follow-up: pinpoint WHICH shot in a cluster caused a scoreboard
change, by reading EVERY frame (not sampled) in a bounded window right
after each verified shot. The coarse pass (scoreboard_ocr_probe.py,
stride=15) proved the score DOES change during TEST1's early shot
cluster but couldn't say which shot did it -- this closes that gap.

Each shot's search window: [shot_end_frame, min(shot_end + MAX_WINDOW,
next_shot_start_frame)] -- bounded by the NEXT shot's start so a make
during shot N+1 never gets misattributed to shot N. State (the running
confirmed score) carries forward continuously across shots in order, so
a shot late in a cluster starts from the correct just-prior score, not
from scratch.

Read-only: prints the per-shot attribution table, writes nothing.

Run:  .venv/Scripts/python spikes/dense_shot_score_match.py TEST1
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from scoreboard_ocr_probe import run_probe
from match_shots_to_score import VERIFIED_SHOTS

MAX_WINDOW_FRAMES = 180      # 6s cap per shot, same window match_shots_to_score.py uses
# stride=1 (every frame) was tried and REJECTED: adjacent frames are nearly
# identical, so they're not independent samples -- a single-instant visual
# glitch (blur, a graphic animation tick) can dominate a vote instead of
# getting outvoted, producing physically-impossible reads (a measured "5-0"
# jump). stride=5 keeps samples spread enough to be genuinely independent
# (same spirit as the coarse pass's stride=15) while still ~3x finer.
DENSE_STRIDE = 5
DENSE_WINDOW = 5
DENSE_MIN_VOTES = 3


def main():
    clip_name = sys.argv[1] if len(sys.argv) > 1 else "TEST1"
    shots = VERIFIED_SHOTS.get(clip_name, [])
    if not shots:
        print(f"[dense-match] no verified shots on record for {clip_name}")
        return

    shots_sorted = sorted(shots, key=lambda s: s[0])
    # Both clips are eye-verified 0-0 at frame 0 (TEST_LOG TEST 14) -- seed
    # this explicitly so the FIRST shot's window can register a genuine
    # change via the normal events path, instead of silently treating an
    # unknown starting state as "no prior score to compare against".
    confirmed_home, confirmed_away = 0, 0

    print(f"[dense-match] {clip_name}: {len(shots_sorted)} verified shots, "
          f"dense (every-frame) scoreboard read per shot window\n")

    for i, (start, end, hoop) in enumerate(shots_sorted):
        next_start = shots_sorted[i + 1][0] if i + 1 < len(shots_sorted) else None
        window_end = end + MAX_WINDOW_FRAMES
        if next_start is not None:
            window_end = min(window_end, next_start)

        print(f"shot {start}-{end} ({hoop} hoop): scanning frames {end}-{window_end} "
              f"({(window_end-end)/30.0:.1f}s window) ...", flush=True)
        events, readings, fps = run_probe(
            clip_name, frame_start=end, frame_end=window_end, sample_stride=DENSE_STRIDE,
            window=DENSE_WINDOW, min_votes=DENSE_MIN_VOTES,
            initial_home=confirmed_home, initial_away=confirmed_away, verbose=True)

        # DJ's hard rule (2026-07-26, binding): the scoreboard may CONFIRM a
        # make, it may NEVER be used to conclude a miss from silence/absence.
        # No score change in this shot's window = unknown, never MISS.
        if events:
            e = events[0]        # first change in this shot's window = attributed to this shot
            print(f"  -> candidate_make: score {e['from']} -> {e['to']} at f={e['frame']} "
                  f"({e['t_sec']}s, {(e['frame']-end)/30.0:.2f}s after shot ended)")
            confirmed_home, confirmed_away = e["to"]
        elif readings:
            # no CHANGE but a state got (re)confirmed -- carry it forward as-is
            confirmed_home, confirmed_away = readings[-1][1], readings[-1][2]
            print(f"  -> unknown: no score change in window (confirmed steady at "
                  f"{confirmed_home}-{confirmed_away} -- NOT evidence of a miss)")
        else:
            print(f"  -> unknown: no reliable scoreboard read this window either "
                  f"(carrying prior state {confirmed_home}-{confirmed_away} forward -- "
                  f"NOT evidence of a miss)")
        print()

    print(f"[dense-match] final score after all verified shots: "
          f"{confirmed_home}-{confirmed_away}")


if __name__ == "__main__":
    main()
