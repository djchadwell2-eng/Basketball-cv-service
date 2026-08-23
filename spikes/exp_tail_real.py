"""exp_tail_real.py -- the identity tail, on REAL slices, on the real machine.

WHAT THIS SETTLES. Everything about the tail's size so far comes from a stand-in
cache built on this laptop. It matched the real slice 0's statistics (~34 bodies
a frame, ~5,112 track ids a slice) but it is still made up, and the number that
now matters most is one only real film can give: HOW MANY JERSEY CROPS does the
tail actually want to read? On the stand-in it was 28,649 per slice, which at
EasyOCR's pace is hours per slice and blows the 180-minute job cap.

SO THE READER IS STUBBED, deliberately. Every crop is still selected and cut --
that is the part whose cost we are measuring -- but the read itself is skipped
and counted instead. Paying for twelve hours of OCR to learn a number we can
count in twenty minutes would be the expensive way to be thorough.

THE BALL LAYER IS OFF, also deliberately. The clip now carries rims and a
whole-game ball span, and run_clip would fire the shot layer: a full-game
re-encode, three full-game overlay videos, and a rim tracker still on the CPU
path. That does not fit the container's 32 GB of disk, never mind the clock.
Shots need their own work first; this run is the box-score road.

Run:  .venv/Scripts/python.exe lab.py spikes/exp_tail_real.py --args '{...}'
      (the caller passes the clip document in args, so the clip file on disk is
      never touched and the ball span can be cleared for this run alone)
"""

import gc
import json
import os
import sys
import time

PER_SLICE = 17112          # a ten-way split of DJ's 171,120-frame game


def _peak_gb():
    """Peak resident memory, from the kernel rather than a guess."""
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return round(int(line.split()[1]) / 1e6, 2)
    except Exception:
        pass
    return None


def run(config=None, slices=8, **_kwargs):
    for p in ("/app", "/app/spikes", "/app/phase1", "/app/phase2"):
        if p not in sys.path:
            sys.path.insert(0, p)
    import serverless_handler as sh

    if not config:
        return {"error": "no clip config passed in args"}
    clip = config.get("name")
    n = int(slices)
    chunks = [{"index": i, "start": i * PER_SLICE, "length": PER_SLICE} for i in range(n)]
    span = (0, n * PER_SLICE)

    out = {"clip": clip, "slices": n, "frames": span[1], "image": sh.image_sha(),
           "stages": []}

    # The ball layer OFF for this run only -- the copy we install, not the file.
    cfg_doc = dict(config)
    cfg_doc["ball_span_len"] = 0
    cfg_doc["ball_span_start"] = 0

    t0 = time.time()
    sh._install_uploaded_clip(clip, cfg_doc, span)
    import clip_config
    cfg = clip_config.get_clip(clip)
    if cfg is None:
        return {"error": f"clip {clip} not usable after install"}
    if cfg.ball_span_len:
        return {"error": "ball layer still armed -- refusing (see module docstring)"}

    # --- merge the slices already paid for -------------------------------
    head = {"clip": clip, "span_start": span[0], "span_len": span[1]}
    oncourt_out = os.path.join("/app", "phase2", "out", f"{clip}_oncourt.json")
    t = time.time()
    n_frames = sh.merge_streamed(clip, chunks, "tracks", cfg.tracks_cache_path, head)
    sh.merge_streamed(clip, chunks, "oncourt", oncourt_out, head)
    out["merge_seconds"] = round(time.time() - t, 1)
    out["merged_frames"] = n_frames
    out["merged_tracks_mb"] = round(os.path.getsize(cfg.tracks_cache_path) / 1e6, 1)
    out["merged_oncourt_mb"] = round(os.path.getsize(oncourt_out) / 1e6, 1)
    out["peak_gb_after_merge"] = _peak_gb()

    # --- the reader, stubbed and COUNTED ---------------------------------
    import ocr_reader
    counter = {"attempts": 0}

    def _counted(crop, roster_numbers):
        counter["attempts"] += 1
        return []                      # "no confident read" -- the common real answer

    ocr_reader.read_jersey = _counted

    # --- the tail, in the order run_clip runs it -------------------------
    # BOTH clip selectors, through run_clip's own guard. The stage modules bind
    # clip_config.ACTIVE_CLIP AT IMPORT and it defaults to TEST1, so setting only
    # clips_config.ACTIVE leaves every stage pointed at TEST1's Windows video
    # path -- which is exactly how the first attempt at this died, in
    # stage4_seed_queue, "could not open video: C:\\Users\\djcha\\Downloads\\
    # Test1.mp4". _sync_and_guard sets both and fails loud if they disagree.
    import run_clip
    run_clip._sync_and_guard(cfg)
    stages = [("team_events", "stage2_generate_events"),
              ("team_stats", "stage3_team_stats"),
              ("windows", "stage3_windows"),
              ("seed_queue", "stage4_seed_queue"),
              ("player_events", "stage5_player_events"),
              ("ocr_confirm", "stage6_ocr_confirm"),
              ("retro_merge", "stage7_merge"),
              ("box_score", "stage8_box_score")]
    for label, mod_name in stages:
        gc.collect()
        t = time.time()
        err = None
        try:
            __import__(mod_name).main()
        except Exception as e:
            import traceback
            err = f"{type(e).__name__}: {e}"
            out["traceback"] = traceback.format_exc()[-1800:]
        out["stages"].append({"stage": label, "seconds": round(time.time() - t, 1),
                              "peak_gb": _peak_gb(), "error": err})
        if err:
            break

    out["ocr_attempts_real"] = counter["attempts"]
    out["tail_seconds"] = round(sum(s["seconds"] for s in out["stages"]), 1)
    out["total_seconds"] = round(time.time() - t0, 1)
    out["peak_gb"] = _peak_gb()

    # What the whole game would look like, from what this actually did.
    if n:
        scale = 10.0 / n
        out["projected_full_game"] = {
            "tail_minutes": round(out["tail_seconds"] * scale / 60, 1),
            "peak_gb": round((out["peak_gb"] or 0) * scale, 1),
            "ocr_attempts": int(counter["attempts"] * scale),
        }

    # The box score, if it got that far -- the point of the whole exercise.
    box = os.path.join("/app", "phase2", "out", f"{clip}_box_score.json")
    if os.path.exists(box):
        try:
            with open(box, encoding="utf-8") as fh:
                doc = json.load(fh)
            out["box_score_players"] = len(doc.get("players", doc.get("box_score", [])))
        except Exception:
            pass
    return out
