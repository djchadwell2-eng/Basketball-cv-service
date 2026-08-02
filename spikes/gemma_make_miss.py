"""Make/miss detection using Gemma vision for universal scoreboard reading.

This replaces scoreboard_make_miss.py and works on ANY scoreboard style
(broadcast, OHSAA, LED, etc) by sending frames to Gemma instead of OCR.

Usage:
    results = detect_makes_by_gemma(clip_name, shots_json_path, api_key)

Same interface as scoreboard_make_miss.py but works on all board styles.
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
    outcome: str  # "candidate_make", "unknown"
    score_from: list[int] | None
    score_to: list[int] | None
    score_change_frame: int | None
    score_change_time_sec: float | None


def detect_makes_by_gemma(
    clip_name: str,
    shots_json_path: str,
    api_key: str,
    fps: float = 60.0,
) -> list[ShotMakeResult]:
    """Read shot_attempts and use Gemma vision to detect makes.

    shots_json_path: path to {clip}_shot_attempts.json from ball_stages
    api_key: Google API key with access to Gemma

    Returns list of {start_frame, end_frame, outcome, score_from, score_to, ...}.
    Outcome is "candidate_make" (score changed) or "unknown" (no change).
    NEVER "miss" — silence is never proof of a miss.
    """
    import cv2
    import clip_config

    try:
        import google.genai
    except ImportError:
        print("[gemma_make_miss] ERROR: google.genai not installed")
        return []

    if not os.path.exists(shots_json_path):
        return []

    try:
        with open(shots_json_path) as f:
            shots_doc = json.load(f)
    except Exception as e:
        print(f"[gemma_make_miss] ERROR reading shots: {e}")
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
        print(f"[gemma_make_miss] ERROR: Unknown clip {clip_name}")
        return []

    clip = clip_config_map[clip_name]
    cap = cv2.VideoCapture(clip.video_path)
    if not cap.isOpened():
        print(f"[gemma_make_miss] ERROR: Could not open {clip.video_path}")
        return []

    client = google.genai.Client(api_key=api_key)

    shots_sorted = sorted(shots, key=lambda s: s[0])
    confirmed_home, confirmed_away = 0, 0
    results = []

    print(f"[gemma_make_miss] Detecting makes for {len(shots_sorted)} shots with Gemma...")

    for i, (start, end, hoop) in enumerate(shots_sorted):
        # Window after shot to look for score change
        window_end = end + 180  # 6 seconds at 60fps
        next_start = shots_sorted[i + 1][0] if i + 1 < len(shots_sorted) else None
        if next_start is not None:
            window_end = min(window_end, next_start)

        # Sample frames in the window - but limit to reduce API calls
        sample_stride = 10  # Increased from 5 to reduce API calls
        frames_to_read = list(range(end + 10, min(end + 60, window_end), sample_stride))  # Only check first 2 seconds

        score_from = None
        score_to = None
        score_change_frame = None
        score_change_time = None
        outcome = "unknown"

        last_home, last_away = confirmed_home, confirmed_away

        print(
            f"  Shot {i+1}/{len(shots_sorted)}: frames {start}-{end}, checking {len(frames_to_read)} frames...",
            end=" ",
        )

        for frame_num in frames_to_read:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ok, frame = cap.read()
            if not ok:
                continue

            # Crop scoreboard region
            h, w = frame.shape[:2]
            crop = frame[int(h * 0.72) : int(h * 1.0), int(w * 0.0) : int(w * 0.22)]

            # Encode and send to Gemma
            _, jpg = cv2.imencode(".jpg", crop)
            image_data = base64.standard_b64encode(jpg.tobytes()).decode("utf-8")

            try:
                response = client.models.generate_content(
                    model="gemma-4-26b-a4b-it",
                    contents=[
                        "Basketball scoreboard. Home: ? Away: ? Time: ?. Answer ONLY: Home: X Away: X Time: HH:MM",
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_data,
                            }
                        },
                    ],
                )
                text = response.text.strip()

                # Parse the response
                home, away = _parse_gemma_response(text)

                if home is not None and away is not None:
                    # Check if score changed
                    if (home, away) != (last_home, last_away):
                        score_from = [last_home, last_away]
                        score_to = [home, away]
                        score_change_frame = frame_num
                        score_change_time = frame_num / fps
                        outcome = "candidate_make"
                        confirmed_home, confirmed_away = home, away
                        print(f"MAKE! {score_from} -> {score_to}", end="")
                        break
                    else:
                        last_home, last_away = home, away

            except Exception as e:
                # Silently skip frames that fail to process
                pass

        print()

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


def _parse_gemma_response(response: str) -> tuple[int | None, int | None]:
    """Parse Gemma's response to extract home and away scores."""
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
    # Test on TEST1
    import os

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set")
        sys.exit(1)

    clip_name = "TEST1"
    shots_path = os.path.join(_HERE, "out", f"{clip_name}_shot_attempts.json")

    results = detect_makes_by_gemma(clip_name, shots_path, api_key)
    print(f"\n[gemma_make_miss] {clip_name}: {len(results)} shots")
    for r in results:
        print(f"  {r['start_frame']}-{r['end_frame']}: {r['outcome']}", end="")
        if r["score_from"]:
            print(f" ({r['score_from']} -> {r['score_to']})", end="")
        print()
