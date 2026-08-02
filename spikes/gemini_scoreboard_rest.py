"""Gemini scoreboard reader using REST API directly (bypasses SDK quirks).

Tests reading two scoreboard crops: the OHSAA-style one that broke OCR,
and the broadcast-overlay style that OCR handled.

Run:  .venv/Scripts/python spikes/gemini_scoreboard_rest.py
"""
from __future__ import annotations

import base64
import json
import os
import requests

_HERE = os.path.dirname(__file__)

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def test_with_gemini(image_path: str, label: str):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(f"ERROR: GEMINI_API_KEY not set")
        return None

    if not os.path.exists(image_path):
        print(f"  !! {label}: file not found")
        return None

    print(f"\n--- {label} ---")
    print(f"  sending {os.path.getsize(image_path)} bytes to Gemini (2.5-flash)...")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"

    image_data = encode_image_to_base64(image_path)

    payload = {
        "contents": [
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
                        "text": """Look at this scoreboard image. Tell me:
1. Home team score (left number)?
2. Away team score (right number)?
3. Game clock time (if visible)?

Answer:
Home: X
Away: X
Time: HH:MM or "not visible"

If you cannot read it: "Cannot read"."""
                    }
                ]
            }
        ]
    }

    try:
        resp = requests.post(url, json=payload, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            if "candidates" in data and data["candidates"]:
                text = data["candidates"][0]["content"]["parts"][0]["text"]
                print(f"  Gemini says:\n{text}")
                return text
            else:
                print(f"  ERROR: No response from Gemini")
                return None
        else:
            print(f"  ERROR {resp.status_code}: {resp.text[:300]}")
            return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def main():
    os.environ["GEMINI_API_KEY"] = os.environ.get("GEMINI_API_KEY") or ""

    tests = [
        (os.path.join(_HERE, "out", "TEST2_frame200_cornercrop.jpg"),
         "TEST2 OHSAA-style (OCR couldn't read this)"),
        (os.path.join(_HERE, "out", "TEST1_frame200_cornercrop.jpg"),
         "TEST1 broadcast-overlay (OCR could read this)"),
    ]

    # Generate TEST1 crop if needed
    test1_crop = os.path.join(_HERE, "out", "TEST1_frame200_cornercrop.jpg")
    if not os.path.exists(test1_crop):
        print("Generating TEST1 corner crop...")
        import cv2
        import sys
        sys.path.insert(0, os.path.dirname(_HERE))
        import clip_config
        cap = cv2.VideoCapture(clip_config.TEST1_CLIP.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 200)
        ok, frame = cap.read()
        if ok:
            h, w = frame.shape[:2]
            crop = frame[int(h*0.72):int(h*1.0), int(w*0.0):int(w*0.22)]
            cv2.imwrite(test1_crop, crop)
            print(f"  saved {test1_crop}")
        cap.release()

    print("\nGemini Scoreboard Reader Test (REST API)")
    print("=" * 60)

    results = []
    for image_path, label in tests:
        result = test_with_gemini(image_path, label)
        results.append((label, result))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for label, result in results:
        status = "[OK]" if result else "[FAIL]"
        print(f"{status}: {label}")


if __name__ == "__main__":
    main()
