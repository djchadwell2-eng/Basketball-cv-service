"""READ-ONLY probe: does a BIGGER detector (yolov8x) reduce player-track
FRAGMENTATION vs the cached yolov8m baseline? (DECISIONS 23)

User question: "can we use YOLOv8x for the player tracker, not just the ball?
Better tracked players is huge, right?" Fragmentation drives review clicks
(§10: clicks scale with fragments, not game length) AND was the third layup-
detection signal failure (§22). So IF a bigger detector relinks players
through occlusion, it's high value. But detection quality may not be the
tracking bottleneck -- association through occlusion is (§11 re-ID failed on
identical uniforms). This MEASURES it instead of guessing.

Same span, same tracker (bytetrack.yaml), only the DETECTOR changes
(yolov8m -> yolov8x), so any delta is attributable to detector capacity.
Writes a SEPARATE json; the real tracks cache (the user's labels/queue
resolutions depend on its ids) is NEVER touched -- adoption is a later
decision on these numbers, exactly like the re-ID probe (§11).

Run (background, CPU, bigger model => slow):
    .venv/Scripts/python spikes/player_detector_probe.py [CLIP_NAME]
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "phase2"))

import clip_config

CLIP_NAME = sys.argv[1] if len(sys.argv) > 1 else "TEST1"
CLIP = getattr(clip_config, f"{CLIP_NAME}_CLIP")
clip_config.ACTIVE_CLIP = CLIP          # set BEFORE importing run_tracking (binds at import)

import run_tracking                      # reuse the exact same span extraction
import tracking

BIG_MODEL = os.path.join(_ROOT, "yolov8x.pt")
OUT_JSON = os.path.join(_HERE, "out", f"{CLIP_NAME}_tracks_yolov8x.json")


def _stats(frames):
    """(distinct ids, mean tracks/frame, mean track lifespan in frames)."""
    ids = {}
    per_frame = []
    for fr in frames:
        per_frame.append(len(fr["tracks"]))
        for t in fr["tracks"]:
            ids[t["track_id"]] = ids.get(t["track_id"], 0) + 1
    n = len(ids)
    return (n, sum(per_frame) / max(len(per_frame), 1),
            sum(ids.values()) / max(n, 1))


def main():
    assert os.path.exists(BIG_MODEL), f"missing {BIG_MODEL} (download yolov8x.pt first)"
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

    start, length = CLIP.tracking_span_start, CLIP.tracking_span_len
    print(f"[probe] extracting {CLIP_NAME} span {start}..{start + length} ...")
    subclip, fps, n = run_tracking.extract_subclip(CLIP.video_path, start, length)
    print(f"[probe] {n} frames -> {subclip}")

    print(f"[probe] tracking with yolov8x + ByteTrack (CPU, slow)...")
    frames_out = []
    for i, (_idx, _img, tracks) in enumerate(
            tracking.iter_tracks(subclip, model_name=BIG_MODEL)):
        frames_out.append({
            "frame_index": start + i,
            "tracks": [{"track_id": t.track_id, "bbox": list(t.bbox)} for t in tracks],
        })
        if i % 30 == 0:
            print(f"  ...{i}/{n}  ({len(tracks)} tracks)", flush=True)

    doc = {"clip": CLIP_NAME, "detector": "yolov8x", "span_start": start,
           "span_len": len(frames_out), "fps": fps, "frames": frames_out}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"[probe] wrote {OUT_JSON}")

    baseline = json.load(open(CLIP.tracks_cache_path, encoding="utf-8"))
    b_ids, b_mean, b_life = _stats(baseline["frames"])
    p_ids, p_mean, p_life = _stats(frames_out)
    print(f"\n============ PLAYER-DETECTOR PROBE ({CLIP_NAME}) ============")
    print(f"                       yolov8m(cached)   yolov8x")
    print(f"  distinct track_ids   {b_ids:>13}   {p_ids:>9}   "
          f"({'FEWER=better' if p_ids < b_ids else 'no improvement'})")
    print(f"  mean tracks/frame    {b_mean:>13.1f}   {p_mean:>9.1f}")
    print(f"  mean lifespan (fr)   {b_life:>13.1f}   {p_life:>9.1f}   "
          f"({'LONGER=better' if p_life > b_life else 'no improvement'})")
    if p_ids:
        print(f"  fragmentation ratio  {b_ids / p_ids:.2f}x "
              f"({'fewer fragments with yolov8x' if p_ids < b_ids else 'no gain'})")
    print("  (baseline cache untouched; adoption is a separate decision)")


if __name__ == "__main__":
    main()
