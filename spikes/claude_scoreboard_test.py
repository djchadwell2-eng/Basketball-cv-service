"""Proof-of-concept: can Claude's vision read scorebug crops that broke the
OCR reader? Tests two different scoreboard styles using Claude.

This is a READ-ONLY test -- sends 2 image crops to Claude, asks
"what score and time do you see?", logs the answers. Budget estimate:
~20-40 cents if both run (Claude vision is cheaper than Gemini).

Run:  .venv/Scripts/python spikes/claude_scoreboard_test.py
"""
from __future__ import annotations

import base64
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")


def test_scoreboard_with_claude(image_path: str, label: str):
    """Send an image crop to Claude, ask it to read the scoreboard."""
    try:
        from anthropic import Anthropic
    except ImportError:
        print(f"ERROR: anthropic not installed. Run: pip install anthropic")
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(f"ERROR: ANTHROPIC_API_KEY not set in environment")
        return None

    if not os.path.exists(image_path):
        print(f"  !! {label}: image not found at {image_path}")
        return None

    # Create client
    client = Anthropic(api_key=api_key)

    # Encode image
    image_data = encode_image_to_base64(image_path)

    prompt = """Look at this scoreboard image. Tell me:
1. What is the home team's score (the left/top number)?
2. What is the away team's score (the right/bottom number)?
3. What is the game clock time shown (if visible)?

Answer in a simple format:
Home: X
Away: X
Time: HH:MM or "not visible"

If you cannot read the scoreboard, just say "Cannot read"."""

    print(f"\n--- {label} ---")
    print(f"  sending {os.path.getsize(image_path)} bytes to Claude...")

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        answer = response.content[0].text
        print(f"  Claude says:\n{answer}")
        return answer
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\nERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("If you only have a Gemini key, run the gemini test instead.")
        print("Otherwise, set ANTHROPIC_API_KEY in your environment.\n")
        return

    # The two test crops we have on disk
    tests = [
        (os.path.join(_HERE, "out", "TEST2_frame200_cornercrop.jpg"),
         "TEST2 OHSAA-style (the one OCR couldn't read)"),
        (os.path.join(_HERE, "out", "TEST1_frame200_cornercrop.jpg"),
         "TEST1 broadcast-overlay (the one OCR DID read, sanity check)"),
    ]

    print("\nClaude Scoreboard Reader Test")
    print("=" * 60)
    print("Sending 2 scoreboard crops to Claude for vision reading.")
    print("This is a proof-of-concept, not building a real pipeline yet.\n")

    # Generate TEST1 crop for sanity check if it doesn't exist
    test1_crop = os.path.join(_HERE, "out", "TEST1_frame200_cornercrop.jpg")
    if not os.path.exists(test1_crop):
        print("Generating TEST1 corner crop for sanity check...")
        import cv2
        import clip_config
        cap = cv2.VideoCapture(clip_config.TEST1_CLIP.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 200)
        ok, frame = cap.read()
        if ok:
            h, w = frame.shape[:2]
            y0, y1 = int(h * 0.72), int(h * 1.0)
            x0, x1 = int(w * 0.0), int(w * 0.22)
            crop = frame[y0:y1, x0:x1]
            cv2.imwrite(test1_crop, crop)
            print(f"  saved {test1_crop}")
        cap.release()

    results = []
    for image_path, label in tests:
        if os.path.exists(image_path):
            result = test_scoreboard_with_claude(image_path, label)
            results.append((label, result))
        else:
            print(f"\n--- {label} ---")
            print(f"  file not found, skipping")
            results.append((label, None))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for label, result in results:
        status = "[OK]" if result else "[FAIL]"
        print(f"{status}: {label}")

    print("\nDone. Check results above to see if Claude can read these styles.")


if __name__ == "__main__":
    main()
