"""READ-ONLY probe: BoT-SORT + re-ID fragmentation vs the cached ByteTrack baseline.

Answers the DECISIONS §10 scaling question: human clicks scale with tracker
FRAGMENTATION (TEST1 baseline: 122 distinct track_ids over 461 frames), so if
an appearance-embedding tracker relinks bodies through occlusions, the whole
review workload shrinks by the same factor.

This script tracks the SAME TEST1 span with BoT-SORT (phase2/botsort_reid.yaml,
with_reid: True) and writes a SEPARATE json (spikes/out/TEST1_tracks_botsort.json).
It NEVER touches the real tracks cache — the user's labels, queue resolutions,
and downstream caches stay valid. Adoption is a separate decision made on the
numbers this prints.

Run (background, CPU ~2s/frame => ~20 min):
    .venv/Scripts/python spikes/reid_fragment_probe.py [tracker_yaml] [out_json]
(defaults: phase2/botsort_reid.yaml -> spikes/out/TEST1_tracks_botsort.json)
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

CLIP = clip_config.TEST1_CLIP
clip_config.ACTIVE_CLIP = CLIP          # set BEFORE importing run_tracking (binds at import)

import run_tracking                      # reuse the exact same span extraction
import tracking

TRACKER_YAML = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(_ROOT, "phase2", "botsort_reid.yaml")
OUT_JSON = sys.argv[2] if len(sys.argv) > 2 else \
    os.path.join(_HERE, "out", "TEST1_tracks_botsort.json")


def _stats(frames):
    """(distinct ids, mean tracks/frame, mean track lifespan in frames)."""
    ids = {}
    per_frame = []
    for fr in frames:
        per_frame.append(len(fr["tracks"]))
        for t in fr["tracks"]:
            ids.setdefault(t["track_id"], 0)
            ids[t["track_id"]] += 1
    n_ids = len(ids)
    mean_per_frame = sum(per_frame) / max(len(per_frame), 1)
    mean_lifespan = sum(ids.values()) / max(n_ids, 1)
    return n_ids, mean_per_frame, mean_lifespan


def main():
    assert os.path.exists(TRACKER_YAML), TRACKER_YAML
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

    print(f"[probe] extracting TEST1 span {CLIP.tracking_span_start}.."
          f"{CLIP.tracking_span_start + CLIP.tracking_span_len} ...")
    subclip, fps, n = run_tracking.extract_subclip(
        CLIP.video_path, CLIP.tracking_span_start, CLIP.tracking_span_len)
    print(f"[probe] {n} frames -> {subclip}")

    print("[probe] tracking with BoT-SORT + re-ID (CPU, ~2s/frame)...")
    frames_out = []
    for i, (_idx, _img, tracks) in enumerate(
            tracking.iter_tracks(subclip, tracker_config=TRACKER_YAML)):
        frames_out.append({
            "frame_index": CLIP.tracking_span_start + i,
            "tracks": [{"track_id": t.track_id, "bbox": list(t.bbox)} for t in tracks],
        })
        if i % 20 == 0:
            print(f"  ...{i}/{n}  ({len(tracks)} tracks)", flush=True)

    doc = {"clip": CLIP.name,
           "tracker": os.path.splitext(os.path.basename(TRACKER_YAML))[0],
           "span_start": CLIP.tracking_span_start, "span_len": len(frames_out),
           "fps": fps, "frames": frames_out}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    print(f"[probe] wrote {len(frames_out)} frames -> {OUT_JSON}")

    with open(CLIP.tracks_cache_path, encoding="utf-8") as f:
        baseline = json.load(f)

    b_ids, b_mean, b_life = _stats(baseline["frames"])
    p_ids, p_mean, p_life = _stats(frames_out)
    print("\n================ FRAGMENTATION PROBE (TEST1) ================")
    print(f"                       ByteTrack(cached)   BoT-SORT+reID")
    print(f"  distinct track_ids   {b_ids:>12}   {p_ids:>13}")
    print(f"  mean tracks/frame    {b_mean:>12.1f}   {p_mean:>13.1f}")
    print(f"  mean lifespan (fr)   {b_life:>12.1f}   {p_life:>13.1f}")
    if p_ids:
        print(f"  fragmentation ratio  {b_ids / p_ids:.2f}x fewer fragments")
    print("  (baseline cache untouched; adoption is a separate decision)")


if __name__ == "__main__":
    main()
