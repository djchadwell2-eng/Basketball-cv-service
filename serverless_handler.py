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


VOLUME_ROOT = os.environ.get("RUNPOD_VOLUME_ROOT", "/runpod-volume")


def _install_uploaded_clip(clip_name: str, doc: dict, span=None) -> None:
    """Make a COACH'S game (clips/<NAME>.json from the browser) a real clip here.

    The container ships with the hand-built baselines only. A game set up in the
    browser arrives as its config document in the job input, and its film is
    already on the mounted network volume -- so the whole install is: point the
    config at the mounted film, drop it in clips/, and reload the two config
    modules that read that directory.

    The reload matters: a warm worker has already imported both modules, and
    their registry entries are built ONCE at import. Without it, a second job on
    the same worker would run the FIRST job's clip.
    """
    import importlib
    import clip_registry

    doc = dict(doc)
    key = doc.get("volume_key")
    if key:
        mounted = os.path.join(VOLUME_ROOT, key)
        if not os.path.exists(mounted):
            raise FileNotFoundError(
                f"film not on the volume at {mounted} -- upload_film.py must run "
                f"before the job (volume mounted at {VOLUME_ROOT}: "
                f"{os.path.isdir(VOLUME_ROOT)})")
        doc["video_path"] = mounted

    if span:
        doc["tracking_span_start"], doc["tracking_span_len"] = int(span[0]), int(span[1])
    elif not doc.get("tracking_span_len"):
        # Nobody has said WHAT to analyse, so analyse the game. The browser
        # setup only ever fills in the calibration half of a config, and the
        # honest default for "analyse my game" is the whole game -- 171k frames
        # is ~31 min of detection here, inside the endpoint's 3-hour cap.
        import cv2
        cap = cv2.VideoCapture(doc["video_path"])
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if total <= 0:
            raise ValueError(f"could not read a frame count from {doc['video_path']}")
        doc["tracking_span_start"], doc["tracking_span_len"] = 0, total
        print(f"[serverless_handler] no span set -- defaulting to the whole film "
              f"({total} frames)", flush=True)

    clip_registry.save(clip_name, doc)
    print(f"[serverless_handler] installed clip {clip_name}: "
          f"video={doc.get('video_path')} span="
          f"{doc.get('tracking_span_start')}..+{doc.get('tracking_span_len')}", flush=True)

    import clip_config
    importlib.reload(clip_config)
    import clips_config
    importlib.reload(clips_config)


def _build_caches(config) -> None:
    """Track the span and decide who is on court -- the two caches run_clip
    REFUSES to run without. On a baked-in baseline these ship in the image; a
    coach's game has never been tracked before, so they are built here. This is
    the part the GPU exists for (1.44 s/frame on DJ's laptop, 0.011 here)."""
    import cache_tracks
    import cache_oncourt

    if not os.path.exists(config.tracks_cache_path):
        print("[serverless_handler] STAGE tracking (building tracks cache) ...", flush=True)
        cache_tracks.cache(config)
    else:
        print("[serverless_handler] tracks cache present -- reusing", flush=True)

    oncourt_path = os.path.join(_ROOT, "phase2", "out", f"{config.name}_oncourt.json")
    if not os.path.exists(oncourt_path):
        print("[serverless_handler] STAGE on-court cache ...", flush=True)
        cache_oncourt.cache(config)
    else:
        print("[serverless_handler] on-court cache present -- reusing", flush=True)


def run_analysis(clip_name: str, doc: dict | None = None, span=None) -> dict:
    if doc:
        _install_uploaded_clip(clip_name, doc, span)
    else:
        _patch_video_path(clip_name)

    # get_clip, not getattr(...f"{clip}_CLIP"): the attribute form only ever
    # finds the hand-written baselines, so every coach upload failed here.
    import clip_config
    config = clip_config.get_clip(clip_name)
    if config is None:
        raise ValueError(
            f"no clip {clip_name} -- not a built-in, and no usable clips/{clip_name}.json "
            f"(a registry clip needs a roster AND a tracking span)")

    if doc:
        _build_caches(config)

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


def speedtest(frames: int = 60) -> dict:
    """How fast does THIS GPU run the detector? Measured, not assumed.

    The whole GPU plan is sized off one number -- seconds per frame for
    YOLOv8m@1280 -- and the local machine measures 1.44 s/frame with no CUDA at
    all. Guessing the speedup would mean sizing chunking, cost and job length
    off a number nobody checked, so the endpoint reports its own.

    Deliberately NOT the pipeline: no caches, no calibration, just the
    detection cost that dominates a full game.
    """
    import cv2
    import torch
    from ultralytics import YOLO

    dev = {
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    video = os.path.join(VIDEO_DIR, BUNDLED_VIDEOS["TEST1"])
    if not os.path.exists(video):
        return {"error": f"no bundled video at {video}", **dev}

    cap = cv2.VideoCapture(video)
    imgs = []
    while len(imgs) < frames:
        ok, f = cap.read()
        if not ok:
            break
        imgs.append(f)
    cap.release()
    if not imgs:
        return {"error": "could not read frames", **dev}

    model = YOLO(os.path.join(_ROOT, "yolov8m.pt"))
    model.predict(imgs[0], imgsz=1280, classes=[0], verbose=False)   # warm up
    t0 = time.time()
    for f in imgs[1:]:
        model.predict(f, imgsz=1280, classes=[0], verbose=False)
    per = (time.time() - t0) / max(1, len(imgs) - 1)

    full_game_frames = 171120                    # DJ's 95.1 min game at 30 fps
    return {
        **dev,
        "frames_timed": len(imgs) - 1,
        "seconds_per_frame": round(per, 4),
        "frame_hw": [imgs[0].shape[0], imgs[0].shape[1]],
        "laptop_seconds_per_frame": 1.44,        # measured on DJ's machine
        "speedup_vs_laptop": round(1.44 / per, 1) if per else None,
        "full_game_hours": round(full_game_frames * per / 3600, 2),
    }


def handler(job):
    job_input = job.get("input", {}) if isinstance(job, dict) else {}

    # A measure-only job: answers "is the GPU fast enough to build on?" without
    # needing a real clip, a video upload, or any caches.
    if job_input.get("mode") == "speedtest":
        t0 = time.time()
        try:
            out = speedtest(int(job_input.get("frames", 60)))
        except Exception as e:
            return {"ok": False, "mode": "speedtest", "error": str(e),
                    "traceback": traceback.format_exc()}
        return {"ok": "error" not in out, "mode": "speedtest",
                "seconds": round(time.time() - t0, 1), **out}

    # A look at the mounted volume: proves the film landed where the job will
    # expect it, without spending GPU minutes to find out.
    if job_input.get("mode") == "volume":
        root = VOLUME_ROOT
        out = {"mounted": os.path.isdir(root), "root": root, "entries": []}
        for base, _dirs, files in os.walk(root):
            for f in files:
                p = os.path.join(base, f)
                try:
                    out["entries"].append({"key": os.path.relpath(p, root),
                                           "gb": round(os.path.getsize(p) / 1e9, 2)})
                except OSError:
                    pass
            if len(out["entries"]) > 200:
                break
        return {"ok": out["mounted"], "mode": "volume", **out}

    clip_name = job_input.get("clip", "TEST1")
    doc = job_input.get("config")          # the browser-set-up game's own config
    span = job_input.get("span")           # optional [start, len] override
    t0 = time.time()
    try:
        stats = run_analysis(clip_name, doc, span)
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
