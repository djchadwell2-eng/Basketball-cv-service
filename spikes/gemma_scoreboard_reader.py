"""Universal scoreboard reader using Google's Gemma vision model.

This reads ANY scoreboard style (broadcast-overlay, OHSAA, LED, etc.) by sending
frames to Gemma's vision model and asking it to read the score.

Usage:
    results = read_scoreboard_frames(clip_name, frame_numbers, api_key)

Works on TEST1, TEST2, and any other scoreboard style that Gemma can see.
"""
from __future__ import annotations

import base64
import os
import sys
from typing import NamedTuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)


class ScoreReading(NamedTuple):
    frame: int
    home: int | None
    away: int | None
    time_str: str | None
    raw_response: str


def read_scoreboard_frames(
    clip_name: str,
    frame_numbers: list[int],
    api_key: str,
) -> list[ScoreReading]:
    """Read scoreboard from specific frames using Gemma vision.

    Args:
        clip_name: Name of the clip (e.g., "TEST1")
        frame_numbers: List of frame numbers to read
        api_key: Google API key with access to Gemma

    Returns:
        List of ScoreReading with parsed scores
    """
    import cv2
    import clip_config

    try:
        import google.genai
    except ImportError:
        print("ERROR: google.genai not installed. Run: pip install google-genai")
        return []

    # Get the clip's video
    clip_config_map = {
        "TEST1": clip_config.TEST1_CLIP,
        "TEST2": clip_config.TEST2_CLIP,
        "HARD": clip_config.HARD_CLIP,
    }

    if clip_name not in clip_config_map:
        print(f"ERROR: Unknown clip {clip_name}")
        return []

    clip = clip_config_map[clip_name]
    cap = cv2.VideoCapture(clip.video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open {clip.video_path}")
        return []

    client = google.genai.Client(api_key=api_key)
    results = []

    print(f"Reading {len(frame_numbers)} frames with Gemma...")

    for frame_num in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ok, frame = cap.read()
        if not ok:
            print(f"  Frame {frame_num}: could not read")
            continue

        # Crop the scoreboard region (upper left corner)
        h, w = frame.shape[:2]
        crop = frame[int(h * 0.72) : int(h * 1.0), int(w * 0.0) : int(w * 0.22)]

        # Encode to base64
        _, jpg = cv2.imencode(".jpg", crop)
        image_data = base64.standard_b64encode(jpg.tobytes()).decode("utf-8")

        # Ask Gemma to read it
        try:
            response = client.models.generate_content(
                model="gemma-4-26b-a4b-it",
                contents=[
                    "Look at this basketball scoreboard. What is the home score (left), away score (right), and game time? Answer ONLY in this format, no other text: Home: X Away: X Time: HH:MM\nIf you cannot read it, answer: Home: ? Away: ? Time: ?",
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_data,
                        }
                    },
                ],
            )
            text = response.text.strip()
            print(f"  Frame {frame_num}: {text}")

            # Parse the response
            home, away, time_str = _parse_score_response(text)

            results.append(
                ScoreReading(
                    frame=frame_num,
                    home=home,
                    away=away,
                    time_str=time_str,
                    raw_response=text,
                )
            )
        except Exception as e:
            print(f"  Frame {frame_num}: ERROR - {e}")

    cap.release()
    return results


def _parse_score_response(response: str) -> tuple[int | None, int | None, str | None]:
    """Parse Gemma's response to extract scores and time."""
    home = None
    away = None
    time_str = None

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
        elif part == "Time:" and i + 1 < len(parts):
            time_str = parts[i + 1]

    return home, away, time_str


if __name__ == "__main__":
    # Test: read frames from TEST1 and TEST2
    import os

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set")
        sys.exit(1)

    clips_to_test = [
        ("TEST1", [200, 300]),
        ("TEST2", [200, 300]),
    ]

    for clip_name, frames in clips_to_test:
        print(f"\n{clip_name}:")
        results = read_scoreboard_frames(clip_name, frames, api_key)
        for r in results:
            print(f"  Frame {r.frame}: {r.home}-{r.away} ({r.time_str})")
