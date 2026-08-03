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
import re
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
PROGRESS_DIR = os.path.join(VOLUME_ROOT, "progress")


def progress(clip: str, line: str) -> None:
    """Say where the job is up to, on the SHARED VOLUME.

    A serverless worker's stdout goes to the RunPod console, which the app
    cannot read -- so a long job looks identical to a hung one from here. The
    volume is visible to any other job, so a one-line-per-stage file is the
    only progress a caller can actually see. It also survives the worker.
    """
    print(f"[serverless_handler] {line}", flush=True)
    try:
        os.makedirs(PROGRESS_DIR, exist_ok=True)
        with open(os.path.join(PROGRESS_DIR, f"{clip}.log"), "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {line}\n")
    except OSError:
        pass                        # progress must never break the analysis


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


def _cache_covers(path: str, config) -> bool:
    """Does this cache file describe THIS clip and THIS span?

    "The file exists" is not the same question, and getting them confused cost
    a whole parallel run: workers are reused between jobs, and a cache path
    carries only the clip name -- not the span -- so slice 7 arrived on a warm
    worker holding slice 2's cache, decided there was nothing to do, and
    reported success in 0 seconds having copied the WRONG FRAMES as its own
    output. Six of ten slices did this. Nothing crashed; the merge would simply
    have produced a game that never happened.
    """
    if not os.path.exists(path):
        return False
    try:
        import json
        with open(path, encoding="utf-8") as fh:
            head = json.load(fh)
    except (OSError, ValueError):
        return False
    return (head.get("clip") == config.name
            and head.get("span_start") == config.tracking_span_start
            and head.get("span_len") == config.tracking_span_len)


def _build_caches(config) -> None:
    """Track the span and decide who is on court -- the two caches run_clip
    REFUSES to run without. On a baked-in baseline these ship in the image; a
    coach's game has never been tracked before, so they are built here. This is
    the part the GPU exists for (1.44 s/frame on DJ's laptop, 0.011 here)."""
    import cache_tracks
    import cache_oncourt
    import torch

    progress(config.name, f"device: cuda={torch.cuda.is_available()} "
                          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ONLY'}")

    if not _cache_covers(config.tracks_cache_path, config):
        progress(config.name, f"STAGE tracking {config.tracking_span_len} frames ...")
        t = time.time()
        cache_tracks.cache(config)
        progress(config.name, f"STAGE tracking done in {time.time() - t:.0f}s")
    else:
        progress(config.name, "tracks cache already covers this span -- reusing")

    oncourt_path = os.path.join(_ROOT, "phase2", "out", f"{config.name}_oncourt.json")
    if not _cache_covers(oncourt_path, config):
        progress(config.name, "STAGE on-court cache ...")
        t = time.time()
        cache_oncourt.cache(config)
        progress(config.name, f"STAGE on-court done in {time.time() - t:.0f}s")
    else:
        progress(config.name, "on-court cache already covers this span -- reusing")


CHUNK_DIR = os.path.join(VOLUME_ROOT, "chunks")


def _chunk_paths(clip: str, index: int):
    d = os.path.join(CHUNK_DIR, clip)
    return (os.path.join(d, f"{index:03d}_tracks.json"),
            os.path.join(d, f"{index:03d}_oncourt.json"))


def run_chunk(clip: str, doc: dict, start: int, length: int, index: int) -> dict:
    """ONE SLICE of a game, on one worker.

    The expensive stages -- tracking every body and working out where the
    camera points -- look at one frame at a time and never look back, so they
    split cleanly: ten workers on a tenth of the game each finish in a tenth of
    the time, for the same money (RunPod bills by the second).

    Only the caches are built here. The identity stages are NOT run per slice:
    who a player is has to be decided over the whole game, not ten times over.

    Results go to the shared volume, because this worker is about to disappear.
    """
    _install_uploaded_clip(clip, doc, (start, length))
    import clip_config
    cfg = clip_config.get_clip(clip)
    if cfg is None:
        raise ValueError(f"chunk {index}: clip {clip} not usable")

    progress(clip, f"CHUNK {index}: frames {start}..{start + length}")
    t0 = time.time()
    _build_caches(cfg)

    # Check what is about to be handed over is actually THIS slice's frames.
    # The belt to _cache_covers' braces: a slice that publishes the wrong span
    # corrupts the merged game silently, and silence is the failure mode that
    # costs a day.
    oncourt_src = os.path.join(_ROOT, "phase2", "out", f"{clip}_oncourt.json")
    for p in (cfg.tracks_cache_path, oncourt_src):
        if not _cache_covers(p, cfg):
            raise ValueError(
                f"chunk {index}: {os.path.basename(p)} does not cover "
                f"{start}..+{length} -- refusing to publish it")

    import shutil
    tracks_dst, oncourt_dst = _chunk_paths(clip, index)
    os.makedirs(os.path.dirname(tracks_dst), exist_ok=True)
    shutil.copy(cfg.tracks_cache_path, tracks_dst)
    shutil.copy(oncourt_src, oncourt_dst)
    # Written to a NETWORK volume by a worker that is about to vanish. Confirm
    # the bytes are actually there and readable before saying "done" -- slices
    # 0 and 1 of the first parallel run reported success and left no file.
    for p in (tracks_dst, oncourt_dst):
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            raise IOError(f"chunk {index}: {p} did not land on the volume")

    dt = time.time() - t0
    progress(clip, f"CHUNK {index}: done in {dt:.0f}s")
    return {"index": index, "start": start, "length": length,
            "seconds": round(dt, 1), "tracks": tracks_dst}


def merge_chunks(clip: str, doc: dict, chunks: list) -> dict:
    """Glue the slices back together, then analyse the whole game once.

    TRACK IDS ARE PER-SLICE. Each worker's tracker counted from one, so slice
    3's "player 4" and slice 4's "player 4" are strangers. Ids are therefore
    offset per slice, which keeps them distinct and honest: a player who
    crosses a seam becomes TWO tracked identities rather than one identity
    silently stitched from two different people. Over-merging would invent a
    player who was never on the floor; splitting one only costs a name, and
    naming is a step the coach already has.
    """
    import json
    starts = [c["start"] for c in chunks]
    full_start = min(starts)
    full_len = max(c["start"] + c["length"] for c in chunks) - full_start

    _install_uploaded_clip(clip, doc, (full_start, full_len))
    import clip_config
    cfg = clip_config.get_clip(clip)

    merged_tracks, merged_oncourt = [], []
    for c in sorted(chunks, key=lambda c: c["start"]):
        tp, op = _chunk_paths(clip, c["index"])
        if not (os.path.exists(tp) and os.path.exists(op)):
            raise FileNotFoundError(f"slice {c['index']} missing on the volume ({tp})")
        offset = (c["index"] + 1) * 1_000_000
        with open(tp, encoding="utf-8") as fh:
            tdoc = json.load(fh)
        # Third check, at the point of use: a slice file that describes the
        # wrong frames must never be glued into the game.
        got = (tdoc.get("span_start"), tdoc.get("span_len"))
        if got != (c["start"], c["length"]):
            raise ValueError(
                f"slice {c['index']} covers {got[0]}..+{got[1]}, expected "
                f"{c['start']}..+{c['length']} -- refusing to merge")
        for fr in tdoc["frames"]:
            for t in fr["tracks"]:
                t["track_id"] += offset
            merged_tracks.append(fr)
        with open(op, encoding="utf-8") as fh:
            odoc = json.load(fh)
        for fr in odoc["frames"]:
            fr["tracks"] = {str(int(k) + offset): v for k, v in (fr.get("tracks") or {}).items()}
            merged_oncourt.append(fr)

    merged_tracks.sort(key=lambda f: f["frame_index"])
    merged_oncourt.sort(key=lambda f: f["frame_index"])
    head = {"clip": clip, "span_start": full_start, "span_len": full_len}
    with open(cfg.tracks_cache_path, "w", encoding="utf-8") as fh:
        json.dump({**head, "frames": merged_tracks}, fh)
    with open(os.path.join(_ROOT, "phase2", "out", f"{clip}_oncourt.json"), "w",
              encoding="utf-8") as fh:
        json.dump({**head, "frames": merged_oncourt}, fh)

    progress(clip, f"MERGE: {len(chunks)} slices -> {len(merged_tracks)} frames "
                   f"({full_start}..{full_start + full_len})")
    return run_analysis(clip, doc, (full_start, full_len), caches_ready=True)


def run_analysis(clip_name: str, doc: dict | None = None, span=None,
                 caches_ready: bool = False) -> dict:
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

    if doc and not caches_ready:
        _build_caches(config)

    progress(clip_name, "STAGE run_clip (calibration -> events -> identity -> box score)")
    t = time.time()
    import run_clip
    run_clip.run(config)
    progress(clip_name, f"STAGE run_clip done in {time.time() - t:.0f}s")

    progress(clip_name, "STAGE measured_stats bundle")
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


def image_sha() -> str:
    """The commit this container was built from -- see the Dockerfile."""
    try:
        with open(os.path.join(_ROOT, "IMAGE_SHA"), encoding="utf-8") as fh:
            return fh.read().strip()[:12]
    except OSError:
        return "unknown"


def handler(job):
    job_input = job.get("input", {}) if isinstance(job, dict) else {}

    # Answers "is this worker running the code I think it is?" in one job.
    if job_input.get("mode") == "version":
        return {"ok": True, "mode": "version", "image": image_sha(),
                "modes": ["chunk", "merge", "exec", "subsample", "anchorbench",
                          "progress", "volume", "speedtest", "version"]}

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
    # ONE SLICE of a game. Ten of these run at once, on ten workers.
    if job_input.get("mode") == "chunk":
        clip = job_input.get("clip", "")
        t0 = time.time()
        try:
            out = run_chunk(clip, job_input["config"], int(job_input["start"]),
                            int(job_input["length"]), int(job_input["index"]))
            return {"ok": True, "mode": "chunk", "image": image_sha(), **out}
        except Exception as e:
            return {"ok": False, "mode": "chunk", "index": job_input.get("index"),
                    "seconds": round(time.time() - t0, 1),
                    "error": str(e), "traceback": traceback.format_exc()}

    # The slices back into one game, then the identity stages over all of it.
    if job_input.get("mode") == "merge":
        clip = job_input.get("clip", "")
        t0 = time.time()
        try:
            stats = merge_chunks(clip, job_input["config"], job_input["chunks"])
            return {"ok": True, "mode": "merge", "clip": clip,
                    "seconds": round(time.time() - t0, 1), "measured_stats": stats}
        except Exception as e:
            return {"ok": False, "mode": "merge", "clip": clip,
                    "error": str(e), "traceback": traceback.format_exc()}

    # RUN AN EXPERIMENT THAT IS NOT IN THIS IMAGE.
    #
    # Every measurement so far cost ~20 minutes of overhead before it measured
    # anything: edit, commit, wait for GitHub to build a multi-gigabyte image,
    # repoint the template, cold start. That is 2-3 experiments a day, and DJ
    # is trying to ship. Experiment code now lives on the shared volume and is
    # simply run -- upload takes a second, so the cycle is the experiment
    # itself plus a warm worker.
    #
    # The file defines run(**kwargs) and returns a JSON-able dict. Only used
    # for measurement code on DJ's own volume; anything that earns its keep
    # gets committed into the image properly afterwards.
    if job_input.get("mode") == "exec":
        name = job_input.get("script", "")
        if not re.match(r"^[A-Za-z0-9_-]+$", name or ""):
            return {"ok": False, "mode": "exec", "error": "bad script name"}
        path = os.path.join(VOLUME_ROOT, "experiments", f"{name}.py")
        if not os.path.exists(path):
            return {"ok": False, "mode": "exec", "error": f"no experiment at {path}"}
        t0 = time.time()
        try:
            if job_input.get("config"):
                _install_uploaded_clip(job_input.get("clip", ""), job_input["config"],
                                       job_input.get("span"))
            ns = {"__name__": "experiment", "__file__": path}
            with open(path, encoding="utf-8") as fh:
                exec(compile(fh.read(), path, "exec"), ns)
            out = ns["run"](**(job_input.get("args") or {}))
            return {"ok": True, "mode": "exec", "script": name, "image": image_sha(),
                    "seconds": round(time.time() - t0, 1), "result": out}
        except Exception as e:
            return {"ok": False, "mode": "exec", "script": name,
                    "seconds": round(time.time() - t0, 1),
                    "error": str(e), "traceback": traceback.format_exc()}

    # Can we anchor every Nth frame instead of every frame? The step that
    # decides whether 30 minutes a game is reachable at all.
    if job_input.get("mode") == "subsample":
        clip = job_input.get("clip", "")
        try:
            if job_input.get("config"):
                _install_uploaded_clip(clip, job_input["config"], job_input.get("span"))
            import gpu_anchor_bench
            out = gpu_anchor_bench.subsample(clip, job_input.get("starts", [600]),
                                             int(job_input.get("frames", 300)))
            return {"ok": "error" not in out, "mode": "subsample", "image": image_sha(), **out}
        except Exception as e:
            return {"ok": False, "mode": "subsample", "error": str(e),
                    "traceback": traceback.format_exc()}

    # Can the GPU do the CAMERA ANCHOR too? Speed AND agreement in feet against
    # the CPU path -- see spikes/gpu_anchor_bench.py. Nothing local has CUDA, so
    # this measurement can only happen here.
    if job_input.get("mode") == "anchorbench":
        clip = job_input.get("clip", "")
        doc = job_input.get("config")
        try:
            if doc:
                _install_uploaded_clip(clip, doc, job_input.get("span"))
            import gpu_anchor_bench
            out = gpu_anchor_bench.bench(clip, int(job_input.get("start", 0)),
                                         int(job_input.get("frames", 20)),
                                         job_input.get("cpu_frames"))
            return {"ok": "error" not in out, "mode": "anchorbench", **out}
        except Exception as e:
            return {"ok": False, "mode": "anchorbench", "error": str(e),
                    "traceback": traceback.format_exc()}

    # Read a running job's progress file. Cheap, no GPU work -- but it needs a
    # SECOND worker slot, since the job it is reporting on is holding the first.
    if job_input.get("mode") == "progress":
        clip = job_input.get("clip", "")
        p = os.path.join(PROGRESS_DIR, f"{clip}.log")
        try:
            with open(p, encoding="utf-8") as fh:
                lines = fh.read().splitlines()
        except OSError:
            lines = []
        return {"ok": True, "mode": "progress", "clip": clip,
                "exists": os.path.exists(p), "lines": lines[-40:]}

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
