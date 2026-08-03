"""Fast make/miss detection using Gemma - optimized for speed.

Strategy: Read only 2-3 key frames per shot instead of many.
- Frame 1: ~0.5s after shot (catch late score update)
- Frame 2: ~1.5s after shot (if score not changed yet)
- Stop if score changes

This reduces API calls from 6-10 per shot down to 2-3.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from typing import TypedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)


class ShotMakeResult(TypedDict):
    start_frame: int
    end_frame: int
    outcome: str
    score_from: list[int] | None
    score_to: list[int] | None
    score_change_frame: int | None
    score_change_time_sec: float | None


SCOREBOARD_PROMPT = (
    "Look at this basketball scoreboard. What is the home score (the LEFT "
    "number) and the away score (the RIGHT number)? Answer ONLY in this "
    "format, no other text: Home: X Away: X\n"
    "If you cannot read it, answer: Home: ? Away: ?"
)


def _reread(client, image_data):
    """Read the same crop again -> (home, away), or None if it cannot be read.

    Used only to confirm a detected score change (see the call site). Any
    failure returns None, which the caller treats as disagreement -- an
    unconfirmable change must not become a confirmed make."""
    try:
        r = client.models.generate_content(
            model="gemma-4-26b-a4b-it",
            contents=[SCOREBOARD_PROMPT,
                      {"inline_data": {"mime_type": "image/jpeg",
                                       "data": image_data}}])
        h, a = _parse_response(r.text.strip())
        return None if h is None or a is None else (h, a)
    except Exception:
        return None


def is_scoring_play(last_home, last_away, home, away):
    """Could basketball have produced this change in the score?

    WHY THIS EXISTS (found 2026-08-02 on TEST1). The reader used to treat ANY
    difference from the previous reading as a made basket. On a real run that
    produced "MAKE [0,0]->[1,0]" (a one-point field goal) and, worse,
    "MAKE [1,0]->[0,0]" -- a score going DOWN being confirmed as a basket. A
    scoreboard number cannot decrease, so that was never a make; it was the
    vision model misreading a small, low-contrast crop.

    That failure mode is the one this project forbids everywhere else: the
    make/miss layer's entire promise is "the scoreboard CONFIRMS makes"
    (DJ's rule), and a confirmation built on a misread is worse than no answer.

    A real made basket moves exactly ONE team's score UP by 1, 2 or 3 (free
    throw, field goal, three). Anything else -- a decrease, both teams moving at
    once, or a jump of 4+ -- is the board being misread, and is refused.

    NOTE this cannot catch every misread: a wrong reading that happens to look
    like a legal +2 still passes. It removes the impossible ones, which is a
    floor, not a guarantee.
    """
    dh, da = home - last_home, away - last_away
    if dh and da:
        return False                     # both teams cannot score at once
    delta = dh or da
    return delta in (1, 2, 3)            # up by a legal amount; 0 handled by caller


def detect_makes_by_gemma_fast(
    clip_name: str,
    shots_json_path: str,
    api_key: str,
    fps: float = 60.0,
) -> list[ShotMakeResult]:
    """Fast make/miss detection - only reads key frames."""
    import cv2
    import clip_config

    try:
        import google.genai
    except ImportError:
        print("[gemma_fast] ERROR: google.genai not installed")
        return []

    if not os.path.exists(shots_json_path):
        return []

    try:
        with open(shots_json_path) as f:
            shots_doc = json.load(f)
    except Exception as e:
        print(f"[gemma_fast] ERROR reading shots: {e}")
        return []

    shots = [
        (a["start_frame"], a["end_frame"], a.get("hoop", "far"))
        for a in shots_doc.get("attempts", [])
        if a.get("verdict") == "shot_attempt"
    ]
    if not shots:
        return []

    # Get the clip's video
    clip_config_map = {
        "TEST1": clip_config.TEST1_CLIP,
        "TEST2": clip_config.TEST2_CLIP,
        "HARD": clip_config.HARD_CLIP,
    }

    if clip_name not in clip_config_map:
        print(f"[gemma_fast] ERROR: Unknown clip {clip_name}")
        return []

    clip = clip_config_map[clip_name]
    cap = cv2.VideoCapture(clip.video_path)
    if not cap.isOpened():
        print(f"[gemma_fast] ERROR: Could not open {clip.video_path}")
        return []

    client = google.genai.Client(api_key=api_key)

    shots_sorted = sorted(shots, key=lambda s: s[0])
    confirmed_home, confirmed_away = 0, 0
    results = []

    print(f"[gemma_fast] Detecting makes for {len(shots_sorted)} shots...")

    for i, (start, end, hoop) in enumerate(shots_sorted):
        # Key frames to read: 30 frames (0.5s), 90 frames (1.5s) after shot
        key_frames = [end + 30, end + 90]

        score_from = None
        score_to = None
        score_change_frame = None
        score_change_time = None
        outcome = "unknown"

        last_home, last_away = confirmed_home, confirmed_away
        found_score_change = False
        errors = []                  # why a frame could not be read, if it could not
        misreads = []                # score moves basketball cannot produce
        unstable = []                # changes a second read would not confirm

        print(f"  Shot {i+1}/{len(shots_sorted)}: ", end="", flush=True)

        for frame_num in key_frames:
            if found_score_change:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ok, frame = cap.read()
            if not ok:
                continue

            # Crop scoreboard
            h, w = frame.shape[:2]
            crop = frame[int(h * 0.72) : int(h * 1.0), int(w * 0.0) : int(w * 0.22)]

            # Send to Gemma
            _, jpg = cv2.imencode(".jpg", crop)
            image_data = base64.standard_b64encode(jpg.tobytes()).decode("utf-8")

            try:
                response = client.models.generate_content(
                    model="gemma-4-26b-a4b-it",
                    contents=[
                        # See SCOREBOARD_PROMPT for why the wording matters.
                        SCOREBOARD_PROMPT,
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_data,
                            }
                        },
                    ],
                )
                text = response.text.strip()

                home, away = _parse_response(text)

                if home is not None and away is not None:
                    if (home, away) == (last_home, last_away):
                        last_home, last_away = home, away
                        print(".", end="", flush=True)
                    elif is_scoring_play(last_home, last_away, home, away):
                        # CONFIRM BEFORE BELIEVING IT. A make is declared off a
                        # SINGLE reading of one frame, and the readings are not
                        # stable on the frames that matter: TEST1 frame 274 --
                        # the frame that decides this very shot -- read as 1, 1
                        # and 12 across three attempts (2026-08-02). The
                        # impossible-move guard cannot help, because 1 and 2 are
                        # both legal scores.
                        # So the deciding frame is read once more and the change
                        # only counts if the second read AGREES. Cost is one
                        # extra call per DETECTED CHANGE (a handful per game),
                        # not per frame -- the cheapest possible place to spend
                        # it, since a disagreement here is the difference
                        # between a confirmed make and a fabricated one.
                        second = _reread(client, image_data)
                        if second != (home, away):
                            unstable.append(
                                f"f{frame_num} read {[home, away]} then "
                                f"{list(second) if second else 'unreadable'}")
                            print("~", end="", flush=True)
                            continue
                        score_from = [last_home, last_away]
                        score_to = [home, away]
                        score_change_frame = frame_num
                        score_change_time = frame_num / fps
                        outcome = "candidate_make"
                        confirmed_home, confirmed_away = home, away
                        found_score_change = True
                        print(f"MAKE {score_from}->{score_to}")
                    else:
                        # The number moved in a way basketball cannot produce,
                        # so it is a MISREAD OF THE BOARD, not a basket. Ignored
                        # rather than confirmed -- see is_scoring_play.
                        misreads.append(f"{[last_home, last_away]}->{[home, away]}")
                        print("?", end="", flush=True)

            except Exception as e:
                # NEVER swallow this. It used to bind `e` and print a bare "E",
                # so a run that read nothing at all looked identical to one
                # that read the board fine and simply saw no score change --
                # and "unknown" is a legitimate answer here, which is exactly
                # what made the difference invisible (2026-08-02). The reasons
                # are collected and reported once per shot rather than per
                # frame, so a rate limit does not spam a hundred lines.
                errors.append(f"{type(e).__name__}: {e}")
                print("E", end="", flush=True)

        if not found_score_change:
            if unstable:
                # The most dangerous case, so it is named plainly: we SAW a
                # legal-looking score change and could not reproduce it on a
                # second read. Before this check that would have shipped as a
                # confirmed make.
                print(f"unknown  <- score change NOT REPRODUCIBLE on a second "
                      f"read, so not confirmed: {'; '.join(unstable[:2])}")
            elif misreads:
                print(f"unknown  <- {len(misreads)} impossible score move(s) "
                      f"REFUSED (board misread, not a basket): "
                      f"{'; '.join(misreads[:2])}")
            elif errors:
                # Say WHY there is no answer: a board we could not reach is not
                # the same fact as a board that showed no points.
                seen = []
                for msg in errors:
                    if msg not in seen:
                        seen.append(msg)
                print(f"unknown  <- {len(errors)} read(s) FAILED, not "
                      f"'no score change': {'; '.join(seen[:2])}")
            else:
                print("unknown")

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

    cap.release()
    return results


def _parse_response(response: str) -> tuple[int | None, int | None]:
    """Parse Gemma response."""
    home = None
    away = None

    parts = response.split()
    for i, part in enumerate(parts):
        if part == "Home:" and i + 1 < len(parts):
            try:
                home = int(parts[i + 1])
            except (ValueError, IndexError):
                pass
        elif part == "Away:" and i + 1 < len(parts):
            try:
                away = int(parts[i + 1])
            except (ValueError, IndexError):
                pass

    return home, away


if __name__ == "__main__":
    import os

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set")
        sys.exit(1)

    clip_name = "TEST1"
    shots_path = os.path.join(_HERE, "out", f"{clip_name}_shot_attempts.json")

    results = detect_makes_by_gemma_fast(clip_name, shots_path, api_key)
    print(f"\n[gemma_fast] Results for {clip_name}:")
    for r in results:
        print(f"  {r['start_frame']}-{r['end_frame']}: {r['outcome']}", end="")
        if r["score_from"]:
            print(f" ({r['score_from']} -> {r['score_to']})", end="")
        print()
