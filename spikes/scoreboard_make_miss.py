"""Scoreboard-based make/miss detection for any clip.

This is the production version of dense_shot_score_match.py — reads real
shot_attempts.json from the pipeline, performs dense scoreboard sampling
right after each detected shot, and returns structured make/miss results
that merge into measured_stats.json.

Used by ball_stages.py as a post-detection pass. Read-only (writes nothing;
returns data structure for measured_stats to serialize).

Call via: results = detect_makes_by_scoreboard(clip_name, shots_json_path, scoreboard_json_path)
"""
from __future__ import annotations

import json
import os
import sys
from typing import TypedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

from scoreboard_ocr_probe import run_probe

MAX_WINDOW_FRAMES = 180      # 6s cap per shot
DENSE_STRIDE = 5
DENSE_WINDOW = 5
DENSE_MIN_VOTES = 3


class ShotMakeResult(TypedDict):
    start_frame: int
    end_frame: int
    outcome: str  # "candidate_make", "unknown"
    score_from: list[int] | None
    score_to: list[int] | None
    score_change_frame: int | None
    score_change_time_sec: float | None


def detect_makes_by_scoreboard(
    clip_name: str,
    shots_json_path: str,
    scoreboard_json_path: str,
) -> list[ShotMakeResult]:
    """Read shot_attempts and scoreboard, return make/miss verdicts.

    shots_json_path: path to {clip}_shot_attempts.json from ball_stages
    scoreboard_json_path: path to {clip}_scoreboard_ocr.json from scoreboard_ocr_probe

    Returns list of {start_frame, end_frame, outcome, score_from, score_to, ...}.
    Outcome is "candidate_make" (score changed) or "unknown" (no change, per DJ's rule).
    NEVER "miss" — silence is never proof of a miss.
    """
    if not os.path.exists(shots_json_path):
        return []
    if not os.path.exists(scoreboard_json_path):
        return []

    try:
        with open(shots_json_path) as f:
            shots_doc = json.load(f)
    except Exception as e:
        print(f"[scoreboard_make_miss] ERROR reading shots: {e}")
        return []

    shots = [
        (a["start_frame"], a["end_frame"], a.get("hoop", "far"))
        for a in shots_doc.get("attempts", [])
        if a.get("verdict") == "shot_attempt"
    ]
    if not shots:
        return []

    shots_sorted = sorted(shots, key=lambda s: s[0])
    confirmed_home, confirmed_away = 0, 0
    results = []

    for i, (start, end, hoop) in enumerate(shots_sorted):
        next_start = shots_sorted[i + 1][0] if i + 1 < len(shots_sorted) else None
        window_end = end + MAX_WINDOW_FRAMES
        if next_start is not None:
            window_end = min(window_end, next_start)

        # Dense scoreboard sample in this shot's post-flight window
        events, readings, fps = run_probe(
            clip_name,
            frame_start=end,
            frame_end=window_end,
            sample_stride=DENSE_STRIDE,
            window=DENSE_WINDOW,
            min_votes=DENSE_MIN_VOTES,
            initial_home=confirmed_home,
            initial_away=confirmed_away,
            verbose=False
        )

        # DJ's rule: scoreboard confirms makes, never proves misses.
        score_from = None
        score_to = None
        score_change_frame = None
        score_change_time = None
        outcome = "unknown"

        if events:
            e = events[0]
            outcome = "candidate_make"
            score_from = list(e["from"])
            score_to = list(e["to"])
            score_change_frame = e["frame"]
            score_change_time = e["t_sec"]
            confirmed_home, confirmed_away = e["to"]
        elif readings:
            confirmed_home, confirmed_away = readings[-1][1], readings[-1][2]

        results.append(
            ShotMakeResult(
                start_frame=start,
                end_frame=end,
                outcome=outcome,
                score_from=score_from,
                score_to=score_to,
                score_change_frame=score_change_frame,
                score_change_time_sec=score_change_time,
            )
        )

    return results


if __name__ == "__main__":
    # Standalone: print results for a clip (for testing)
    clip_name = sys.argv[1] if len(sys.argv) > 1 else "TEST1"
    out_dir = os.path.join(_HERE, "out")
    shots_path = os.path.join(out_dir, f"{clip_name}_shot_attempts.json")
    sb_path = os.path.join(out_dir, f"{clip_name}_scoreboard_ocr.json")

    results = detect_makes_by_scoreboard(clip_name, shots_path, sb_path)
    print(f"\n[scoreboard_make_miss] {clip_name}: {len(results)} shots")
    for r in results:
        print(f"  {r['start_frame']}-{r['end_frame']}: {r['outcome']}", end="")
        if r["score_from"]:
            print(f" ({r['score_from'][0]}-{r['score_from'][1]} → {r['score_to'][0]}-{r['score_to'][1]})", end="")
        print()
