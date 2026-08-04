"""exp_tail_timing.py -- time THE TAIL: the stages that run once over the whole
game, on one machine, after the parallel work is done.

WHY. Splitting across ten GPUs fixes the per-frame stages (detection, camera
anchor). It does nothing for the stages that decide WHO each player is -- those
have to see the whole game, so they run once, on one worker, and nobody has
ever timed them. They are the last unknown between "25 minutes" on paper and a
real full-game run: they could add two minutes or be the next wall.

Also records PEAK MEMORY. The merge step loads the whole game's tracking into
RAM; at 171,120 frames that is roughly a gigabyte of JSON, and the honest worry
is that it dies before it is slow.

Run at two span lengths and compare: a stage whose time doubles with the frames
scales with game length, one that barely moves is a fixed cost. That is what
decides whether the tail matters at 171,120 frames.
"""

import gc
import os
import sys
import time


def _peak_mb():
    """Peak RSS in MB, best effort (Linux worker)."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return None


def run(span_start=600, span_len=900, build=True):
    for p in ("/app", "/app/spikes", "/app/phase1", "/app/phase2"):
        if p not in sys.path:
            sys.path.insert(0, p)

    import clip_config
    import clips_config
    clip = "Full_Game_9eb8bf2a"
    cfg = clip_config.get_clip(clip)
    if cfg is None:
        return {"error": "clip not installed"}
    clip_config.ACTIVE_CLIP = cfg
    clips_config.ACTIVE = clip

    out = {"span_start": cfg.tracking_span_start, "span_len": cfg.tracking_span_len,
           "stages": [], "peak_mb_start": _peak_mb()}

    # The caches the tail reads. Timed separately -- they are the PARALLEL part
    # and are not what this experiment is about.
    if build:
        import cache_tracks
        import cache_oncourt
        t = time.time()
        cache_tracks.cache(cfg)
        out["cache_tracks_s"] = round(time.time() - t, 1)
        t = time.time()
        cache_oncourt.cache(cfg)
        out["cache_oncourt_s"] = round(time.time() - t, 1)

    # The tail, in the order run_clip runs it.
    stages = [
        ("team_events", "stage2_generate_events"),
        ("team_stats", "stage3_team_stats"),
        ("windows", "stage3_windows"),
        ("seed_queue", "stage4_seed_queue"),
        ("player_events", "stage5_player_events"),
        ("ocr_confirm", "stage6_ocr_confirm"),
        ("retro_merge", "stage7_merge"),
        ("box_score", "stage8_box_score"),
    ]
    for label, mod_name in stages:
        t = time.time()
        try:
            mod = __import__(mod_name)
            mod.main()
            err = None
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
        gc.collect()
        out["stages"].append({"stage": label, "seconds": round(time.time() - t, 1),
                              "peak_mb": round(_peak_mb() or 0, 1), "error": err})
        if err:
            break

    out["peak_mb_end"] = round(_peak_mb() or 0, 1)
    out["tail_seconds"] = round(sum(s["seconds"] for s in out["stages"]), 1)
    # What the same tail would cost over a whole game IF it scales with frames.
    if cfg.tracking_span_len:
        out["tail_minutes_full_game_if_linear"] = round(
            out["tail_seconds"] * 171120 / cfg.tracking_span_len / 60, 1)
    return out
