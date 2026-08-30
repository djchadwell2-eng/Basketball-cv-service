"""Run the pipeline on the windows where we KNOW a basket happened, and score it.

THE POINT. As of 2026-08-19 the whole system had 8 detected shots with exactly
ONE confirmed outcome, so shooting %, make/miss accuracy and shot-detection
recall were all uncomputable. A full-game run is ~18 days on CPU, so the answer
was never "run the game".

scoreboard_timeline.py found 47 baskets in a real game for 32 minutes of CPU.
refine_basket_times.py pinned each one to under a second. This runs the
expensive machinery ONLY on those seconds -- a few thousand frames instead of
171,120 -- and every one of them is a moment we already know the answer for.

WHAT IT CAN AND CANNOT SCORE
  shot DETECTION recall  : of N known baskets, how many did we find an arc for?
  MAKE/MISS agreement    : when we called a make, did the board agree?
  SHOOTER attribution    : is the credited girl the one who shot it? (still needs
                           an eyeball -- the board cannot tell us WHO)
  shooting PERCENTAGE    : NOT from this alone. Every window here is a MAKE, so
                           the sample is all makes by construction. A percentage
                           needs misses too, which the scoreboard cannot point
                           to. Recorded here so nobody reads a 100% off this.

REQUIRES RIM CLICKS. The shot layer needs ClipConfig.hoop_anchors -- two rim
pixels, which no calibration landmark can substitute for (the rim is ten feet
off the floor plane). Without them this still measures touches, possessions and
naming, and says loudly that shots were skipped.

Usage:
    .venv/Scripts/python spikes/run_basket_windows.py Full_Game_9eb8bf2a \
        --windows spikes/out/FULL_GAME_basket_windows.json --max 20
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "phase1"), os.path.join(_ROOT, "phase2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def run_one(base, start, frames, has_rims):
    """One window, end to end. Returns what the pipeline said about it."""
    import cache_oncourt
    import cache_tracks
    import clip_config
    import run_clip

    config = dataclasses.replace(
        base, tracking_span_start=start, tracking_span_len=frames,
        event_frames=range(start, start + frames, max(1, frames // 40)),
        render_sample_frames=range(start, start + frames, max(1, frames // 12)))
    ball_cfg = dataclasses.replace(config, ball_span_start=start,
                                   ball_span_len=frames)
    run_clip._sync_and_guard(config)

    cache_tracks.cache(config)
    cache_oncourt.cache(config)

    # Identity, so touches can carry a name.
    import stage3_windows, stage4_seed_queue, stage5_player_events
    import stage6_ocr_confirm, stage7_merge
    for mod in (stage3_windows, stage4_seed_queue, stage5_player_events,
                stage6_ocr_confirm, stage7_merge):
        mod.main()

    import ball_stages as bs
    det = bs.stage_ball_detect(ball_cfg)
    out = {"detections": det}

    if has_rims:
        hoop = bs.stage_hoop_anchor(ball_cfg)
        arcs = bs.stage_ball_trajectory(ball_cfg, det)
        sa = bs.stage_shot_attempts(ball_cfg, arcs, hoop, det)
        out["shot_attempts"] = sa
        doc = json.load(open(sa, encoding="utf-8"))
        out["shots"] = [(a["start_frame"], a["end_frame"])
                        for a in doc.get("attempts", [])
                        if a.get("verdict") == "shot_attempt"]
    else:
        out["shots"] = None                       # skipped, not zero

    touches = bs.stage_ball_touches(ball_cfg, det)
    tdoc = json.load(open(touches, encoding="utf-8"))
    out["n_touches"] = len(tdoc.get("touches", []))
    out["named_touches"] = sum(
        1 for t in tdoc["touches"]
        if (t.get("identity") or {}).get("jersey_number") is not None)

    poss = bs.stage_team_possessions(ball_cfg, touches)
    if poss and os.path.exists(poss):
        pdoc = json.load(open(poss, encoding="utf-8"))
        out["n_possessions"] = len(pdoc["possessions"])
        out["possession_teams"] = [p["team"] for p in pdoc["possessions"]]
    else:
        out["n_possessions"] = 0
        out["possession_teams"] = []
    return out


def _worker():
    """Run ONE window and print its result as JSON. Called as a subprocess.

    ONE WINDOW PER PROCESS, and this is not tidiness -- it is required. Half
    this pipeline binds its span at IMPORT time: phase2/run_tracking.py reads
    SPAN_START/SPAN_LEN from ACTIVE_CLIP as module-level constants, so the
    second `import run_tracking` in a process is a no-op and silently re-tracks
    the FIRST window's frames forever.
    That is exactly what happened on 2026-08-22: windows 2-7 of the overnight
    run were scored against window 1's tracks. The ball detections moved (they
    take the span as an argument) but the bodies did not, so the two never
    overlapped and every touch count came back 0. The shot numbers survived --
    they never touch the tracks -- which is why it looked like a touch-layer
    problem rather than a harness bug.
    A fresh process rebinds every module. DJ's own note for this repo has said
    "one clip per process" all along.
    """
    clip, start, frames, has_rims = (sys.argv[2], int(sys.argv[3]),
                                     int(sys.argv[4]), sys.argv[5] == "1")
    import clip_config
    base = clip_config.get_clip(clip)
    res = run_one(base, start, frames, has_rims)
    print("@@RESULT@@" + json.dumps(res))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--one-window":
        return _worker()
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", help="the PIPELINE clip name (has teams + roster)")
    ap.add_argument("--windows", required=True, help="the *_basket_windows.json")
    ap.add_argument("--max", type=int, default=20)
    ap.add_argument("--max-uncertainty", type=float, default=2.5,
                    help="skip baskets whose moment is pinned no better than this")
    args = ap.parse_args()

    import clip_config

    base = clip_config.get_clip(args.clip)
    if base is None:
        raise SystemExit(f"no pipeline config for {args.clip!r}")
    has_rims = bool(base.hoop_anchors)
    if not has_rims:
        print("!" * 70)
        print("NO RIM CLICKS for this game, so THE SHOT LAYER IS SKIPPED.")
        print("Touches, possessions and naming are still measured; shot")
        print("detection and make/miss are NOT, and will read as 'skipped'")
        print("rather than as zero. To enable them:")
        print(f"    .venv/Scripts/python.exe spikes/make_rim_clicker.py {args.clip}")
        print("    (open the page, mark both rims, Download, then --save it)")
        print("!" * 70, flush=True)

    doc = json.load(open(args.windows, encoding="utf-8"))
    allw = doc["baskets"]
    # ONLY WINDOWS THAT WILL ACTUALLY CONTAIN THEIR SHOT. The bisection narrows
    # about half the baskets to under a second and leaves the rest at the full
    # 10-20s sample gap, because the board was unreadable at the probe point.
    # Running a 5-second window against a 20-second uncertainty spends an hour
    # of CPU on film that probably does not hold the shot -- which would then
    # read as a detection MISS and quietly understate recall. Those are skipped,
    # and counted, rather than silently scored.
    usable = [w for w in allw if w["change_uncertainty_s"] <= args.max_uncertainty]
    skipped = len(allw) - len(usable)
    usable.sort(key=lambda w: w["change_uncertainty_s"])
    wins = usable[:args.max]
    print(f"{len(allw)} refined basket(s): {len(usable)} pinned to "
          f"<={args.max_uncertainty}s, {skipped} too loose to aim at (skipped, "
          f"NOT counted as misses)", flush=True)
    results, t0 = [], time.time()
    print(f"{len(wins)} window(s), {sum(w['run_frames'] for w in wins)} frames total",
          flush=True)

    for i, w in enumerate(wins):
        print(f"\n{'=' * 70}\nWINDOW {i + 1}/{len(wins)}  "
              f"basket {w['score_from']}->{w['score_to']} ({w['points']}pt "
              f"{w['team']})  frames {w['run_start_frame']}.."
              f"{w['run_start_frame'] + w['run_frames']}  (t+{time.time() - t0:.0f}s)"
              f"\n{'=' * 70}", flush=True)
        rec = {k: w[k] for k in ("basket_index", "points", "team", "score_from",
                                 "score_to", "change_uncertainty_s",
                                 "run_start_frame", "run_frames")}
        try:
            proc = subprocess.run(
                [sys.executable, os.path.abspath(__file__), "--one-window",
                 args.clip, str(w["run_start_frame"]), str(w["run_frames"]),
                 "1" if has_rims else "0"],
                cwd=_ROOT, capture_output=True, text=True, timeout=7200)
            sys.stdout.write(proc.stdout)
            line = next((l for l in proc.stdout.splitlines()
                         if l.startswith("@@RESULT@@")), None)
            if line is None:
                raise RuntimeError(
                    f"worker produced no result (exit {proc.returncode}); "
                    f"last stderr: {proc.stderr.strip()[-400:]}")
            rec.update(json.loads(line[len("@@RESULT@@"):]))
            rec["ok"] = True
        except Exception as e:
            # One bad window must not cost the whole night.
            rec["ok"] = False
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"  WINDOW FAILED: {rec['error']}", flush=True)
        results.append(rec)
        # Written after EVERY window, so an interrupted night still leaves data.
        json.dump({"clip": args.clip, "windows_file": args.windows,
                   "has_rims": has_rims, "results": results},
                  open(os.path.join(_HERE, "out",
                                    f"{args.clip}_basket_window_results.json"),
                       "w", encoding="utf-8"), indent=2)

    ok = [r for r in results if r.get("ok")]
    print(f"\n{'=' * 70}\nSCORECARD\n{'=' * 70}")
    print(f"  windows run: {len(ok)}/{len(results)}   "
          f"wall clock {time.time() - t0:.0f}s")
    if has_rims:
        found = [r for r in ok if r.get("shots")]
        print(f"  SHOT DETECTION RECALL: {len(found)}/{len(ok)} known baskets had "
              f"an arc detected  ({100.0 * len(found) / max(1, len(ok)):.0f}%)")
    else:
        print(f"  SHOT DETECTION: skipped (no rim clicks)")
    print(f"  touches: {sum(r.get('n_touches', 0) for r in ok)} total, "
          f"{sum(r.get('named_touches', 0) for r in ok)} with a jersey number")
    print(f"  possessions: {sum(r.get('n_possessions', 0) for r in ok)} total")
    print("\n  NOTE: every window here is a MAKE by construction, so this "
          "measures RECALL, never a shooting percentage.")


if __name__ == "__main__":
    main()
