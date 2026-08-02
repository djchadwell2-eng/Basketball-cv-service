"""Quick proof-of-concept: can Gemini read scorebug crops that broke the
OCR reader? Tests two different scoreboard styles using the google.genai SDK.

This is a READ-ONLY test -- sends 2 image crops to Gemini, asks
"what score and time do you see?", logs the answers. Budget estimate:
~30-50 cents if both run.

Run:  .venv/Scripts/python spikes/gemini_scoreboard_test.py
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


def test_scoreboard_with_gemini(image_path: str, label: str):
    """Send an image crop to Gemini, ask it to read the scoreboard."""
    try:
        import google.genai
    except ImportError:
        print(f"ERROR: google-genai not installed. Run: pip install google-genai")
        return None

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(f"ERROR: GEMINI_API_KEY not set")
        return None

    if not os.path.exists(image_path):
        print(f"  !! {label}: image not found at {image_path}")
        return None

    # Create client with new SDK
    client = google.genai.Client(api_key=api_key)

    # Encode image
    image_data = encode_image_to_base64(image_path)

    prompt = """Look at this scoreboard image. Tell me:
1. What is the home team's score (the left number)?
2. What is the away team's score (the right number)?
3. What is the game clock time shown (if visible)?

Answer in a simple format:
Home: X
Away: X
Time: HH:MM or "not visible"

If you cannot read the scoreboard, just say "Cannot read"."""

    print(f"\n--- {label} ---")
    print(f"  sending {os.path.getsize(image_path)} bytes to Gemini...")

    # Try multiple model names since availability varies by account
    models_to_try = ["gemini-2.0-flash-exp", "gemini-1.5-pro", "gemini-1.5-flash"]

    for model_name in models_to_try:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    {
                        "role": "user",
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": image_data
                                }
                            },
                            {
                                "text": prompt
                            }
                        ]
                    }
                ]
            )
            answer = response.text
            print(f"  Gemini ({model_name}) says:\n{answer}")
            return answer
        except Exception as e:
            error_msg = str(e)
            if "NOT_FOUND" in error_msg or "not found" in error_msg or "not available" in error_msg:
                print(f"  {model_name}: not available on this account, trying next...")
                continue
            else:
                print(f"  ERROR with {model_name}: {e}")
                return None

    print(f"  ERROR: No working Gemini model found for this account")
    return None


def main():
    os.environ["GEMINI_API_KEY"] = os.environ.get("GEMINI_API_KEY") or ""

    # The two test crops we have on disk
    tests = [
        (os.path.join(_HERE, "out", "TEST2_frame200_cornercrop.jpg"),
         "TEST2 OHSAA-style (the one OCR couldn't read)"),
        (os.path.join(_HERE, "out", "TEST1_frame200_cornercrop.jpg"),
         "TEST1 broadcast-overlay (the one OCR DID read, sanity check)"),
    ]

    print("\nGemini Scoreboard Reader Test")
    print("=" * 60)
    print("Sending 2 scoreboard crops to Gemini for vision reading.")
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
            result = test_scoreboard_with_gemini(image_path, label)
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

    print("\nDone. Check results above to see if Gemini can read these styles.")


if __name__ == "__main__":
    main()
