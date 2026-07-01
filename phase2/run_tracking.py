"""Run YOLOv8m@1280 + ByteTrack on a span of the clip and cache the raw tracks.

Caching keeps the (slow, CPU) detection out of the identity stages: they read the
JSON. The span is a sub-clip written to a temp file so ByteTrack only processes the
frames we care about (it always starts at frame 0 of its source). Original frame
indices are preserved in the output as span_start + i.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))                          # repo root (clip_config)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "spikes"))

import tracking
from clip_config import ACTIVE_CLIP as CLIP

SPAN_START = CLIP.tracking_span_start        # per-clip ByteTrack span (was 300)
SPAN_LEN = CLIP.tracking_span_len            # (was 120)
OUT_JSON = CLIP.tracks_cache_path


def extract_subclip(video_path, start, length):
    """Write frames [start, start+length) to a temp mp4; return (path, fps)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    for _ in range(start):
        cap.grab()
    tmp = os.path.join(tempfile.gettempdir(), f"{CLIP.name}_span_{start}_{length}.mp4")
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = 0
    while n < length:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        n += 1
    writer.release()
    cap.release()
    return tmp, fps, n


def main():
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    video = CLIP.video_path
    print(f"Extracting span {SPAN_START}..{SPAN_START + SPAN_LEN} of {CLIP.name} ...")
    subclip, fps, n = extract_subclip(video, SPAN_START, SPAN_LEN)
    print(f"  wrote {n} frames -> {subclip}")

    print("Running YOLOv8m@1280 + ByteTrack (CPU, ~2s/frame)...")
    frames_out = []
    for i, (_idx, _img, tracks) in enumerate(tracking.iter_tracks(subclip)):
        frames_out.append({
            "frame_index": SPAN_START + i,
            "tracks": [{"track_id": t.track_id, "bbox": list(t.bbox)} for t in tracks],
        })
        if i % 20 == 0:
            print(f"  ...{i}/{n}  ({len(tracks)} tracks)")

    doc = {"clip": CLIP.name, "span_start": SPAN_START, "span_len": n, "fps": fps,
           "frames": frames_out}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    n_ids = len({t["track_id"] for fr in frames_out for t in fr["tracks"]})
    print(f"\nwrote {len(frames_out)} frames, {n_ids} distinct track_ids -> {OUT_JSON}")


if __name__ == "__main__":
    main()
