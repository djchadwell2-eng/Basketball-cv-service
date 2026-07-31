"""serverless_handler.py -- RunPod Serverless entry point (Phase 1 proof).

Runs the SAME thing analyze_clip.py runs locally (run_clip -> measured_stats
-> export_span) for a clip that is ALREADY set up -- its caches ship inside
this image, so the container needs no separate database/storage fetch for
this first proof. Phase 2 (a brand-new coach upload) is a different,
larger task -- see tasks/todo.md.

The only thing that needs patching at runtime is video_path: the repo's
ClipConfigs hardcode DJ's local Windows path, but the source video ships
inside this image at a Linux-friendly path instead (see Dockerfile).

Job input:  {"clip": "TEST1"}   (defaults to TEST1 if omitted)
Returns:    the same {clip}_measured_stats.json contract the web app's
            Measured Stats page already reads, plus timing/ok fields.
"""
from __future__ import annotations

import os
import sys
import time
import traceback

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
for _p in ("spikes", "phase1", "phase2"):
    sys.path.insert(0, os.path.join(_ROOT, _p))

VIDEO_DIR = os.path.join(_ROOT, "video_assets")

# clip name -> bundled video filename. Only clips whose video shipped inside
# this image can run (Phase 1 proof scope).
BUNDLED_VIDEOS = {"TEST1": "Test1.mp4", "HARD": "HARD.mp4"}


def _patch_video_path(clip_name):
    fname = BUNDLED_VIDEOS.get(clip_name)
    if not fname:
        return
    real_path = os.path.join(VIDEO_DIR, fname)
    if not os.path.exists(real_path):
        print(f"[serverless_handler] WARNING: bundled video not found at {real_path}")
        return
    import clip_config
    getattr(clip_config, f"{clip_name}_CLIP").video_path = real_path
    import clips_config as cc
    if clip_name in cc.CLIPS:
        cc.CLIPS[clip_name]["video_path"] = real_path
    print(f"[serverless_handler] video_path patched -> {real_path}")


def run_analysis(clip_name: str) -> dict:
    _patch_video_path(clip_name)

    import clip_config
    config = getattr(clip_config, f"{clip_name}_CLIP", None)
    if config is None:
        raise ValueError(f"no ClipConfig named {clip_name}_CLIP")

    print(f"[serverless_handler] STAGE run_clip (full pipeline) clip={clip_name}", flush=True)
    import run_clip
    run_clip.run(config)

    print("[serverless_handler] STAGE measured_stats bundle", flush=True)
    import measured_stats
    stats = measured_stats.generate(clip_name)

    try:
        print("[serverless_handler] STAGE export_span (vision-pass clip)", flush=True)
        import export_span
        export_span.export(clip_name)
    except Exception as e:  # non-fatal -- the numbers already exist without it
        print(f"[serverless_handler] export_span failed (non-fatal): {e}")

    return stats


def handler(job):
    job_input = job.get("input", {}) if isinstance(job, dict) else {}
    clip_name = job_input.get("clip", "TEST1")
    t0 = time.time()
    try:
        stats = run_analysis(clip_name)
    except Exception as e:
        return {
            "ok": False,
            "clip": clip_name,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
    return {
        "ok": True,
        "clip": clip_name,
        "seconds": round(time.time() - t0, 1),
        "measured_stats": stats,
    }


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
