"""Run the POSSESSION chain on a slice of a real full game.

WHY THIS EXISTS. Everything the possession work was built and measured on is a
15-40 second clip: 2-4 possessions each. Nobody has ever seen what this layer
does over continuous play. Three things are unknown and only length can answer
them:
  - the possession COUNT (a real game has ~70; we have never counted past 4)
  - whether the OUT-OF-BOUNDS rule ever fires (all three test clips have ZERO
    off-court touches, so DJ's restart rule has never run on real film)
  - whether jersey naming holds up over minutes rather than seconds

WHAT IT RUNS, and what it deliberately skips. Tracks -> on-court -> ball ->
touches -> possessions. It does NOT run the shot layer: that needs human-clicked
rim pixels (DECISIONS 19) which this game does not have yet, and shots are not
what is unproven here.

THE COST, measured on this machine 2026-08-04: the camera anchor is 5.62 s per
frame on CPU, which is 11 DAYS for a whole game and is why only short clips have
ever been possible. So this takes a SLICE and is honest about it. Time scales
linearly with --frames: about 6 minutes of wall clock per 60 frames of video.

Usage:
    .venv/Scripts/python spikes/full_game_possession_run.py CLIP --frames 300
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "spikes"), os.path.join(_ROOT, "phase1"),
           os.path.join(_ROOT, "phase2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--start", type=int, default=None,
                    help="first frame (default: the clip's configured span start)")
    ap.add_argument("--frames", type=int, default=300)
    args = ap.parse_args()

    import clip_config
    import run_clip

    base = clip_config.get_clip(args.clip)
    if base is None:
        raise SystemExit(f"no pipeline config for {args.clip!r}")

    start = args.start if args.start is not None else base.tracking_span_start
    # TWO CONFIGS, and the reason matters.
    # ClipConfig.validate() refuses a ball span without human-clicked rim pixels
    # -- correctly, because the SHOT layer places shots against those rims and a
    # made-up rim would put shots at made-up spots on the floor. This run does
    # not do shots. Touches and possessions need the ball and the bodies, not
    # the rims.
    # So: `config` (no ball span) is what gets validated and drives tracking and
    # the on-court cache; `ball_cfg` carries the span for the ball stages, which
    # do not validate. Nothing is fabricated -- the shot layer simply is not run.
    config = dataclasses.replace(
        base,
        tracking_span_start=start, tracking_span_len=args.frames,
        event_frames=range(start, start + args.frames, max(1, args.frames // 40)),
        render_sample_frames=range(start, start + args.frames, max(1, args.frames // 12)),
    )
    ball_cfg = dataclasses.replace(config, ball_span_start=start,
                                   ball_span_len=args.frames)
    run_clip._sync_and_guard(config)          # BOTH selectors, or fail loud
    reused = []

    def _step(name):
        print(f"\n{'=' * 70}\n{name}  (t+{time.time() - t0:.0f}s)\n{'=' * 70}", flush=True)

    t0 = time.time()
    print(f"clip={config.name} frames {start}..{start + args.frames - 1} "
          f"({args.frames} frames, {args.frames / 30.0:.0f}s of video)", flush=True)

    _step("TRACKING (ByteTrack over the span)")
    import cache_tracks
    if os.path.exists(config.tracks_cache_path):
        doc = json.load(open(config.tracks_cache_path, encoding="utf-8"))
        fresh = (doc.get("span_start") == start and doc.get("span_len") == args.frames
                 and doc.get("clip") == config.name)
    else:
        fresh = False
    if fresh:
        print("  tracks cache already matches this span -- reusing", flush=True)
        reused.append("tracks")
    else:
        cache_tracks.cache(config)

    _step("ON-COURT (the camera anchor -- THE expensive step on CPU)")
    import cache_oncourt
    import oncourt
    onc_path = os.path.join(_ROOT, "phase2", "out", f"{config.name}_oncourt.json")
    stale = True
    if os.path.exists(onc_path):
        d = json.load(open(onc_path, encoding="utf-8"))
        stale = not (d.get("frames") and d["frames"][0]["frame_index"] == start
                     and len(d["frames"]) == args.frames)
    if stale:
        cache_oncourt.cache(config)
    else:
        print("  on-court cache already matches this span -- reusing", flush=True)
        reused.append("on-court")
    oncourt.load_checked(config)

    # The touch layer joins the ball to a NAMED body, so it needs the identity
    # chain's merged events. Possessions themselves only need the track and its
    # jersey COLOUR -- but touches are built on identity, so the chain runs.
    _step("IDENTITY CHAIN (windows -> seeding -> events -> OCR -> merge)")
    import stage3_windows, stage4_seed_queue, stage5_player_events
    import stage6_ocr_confirm, stage7_merge
    for mod in (stage3_windows, stage4_seed_queue, stage5_player_events,
                stage6_ocr_confirm, stage7_merge):
        print(f"\n--- {mod.__name__} ---", flush=True)
        mod.main()

    _step("BALL DETECTION")
    import ball_stages as bs
    det_json = bs.stage_ball_detect(ball_cfg)

    _step("BALL TOUCHES (who has it)")
    touches_json = bs.stage_ball_touches(ball_cfg, det_json)

    _step("TEAM POSSESSIONS (whose ball, from when to when)")
    poss_json = bs.stage_team_possessions(ball_cfg, touches_json)

    # --- the three unknowns this run exists to answer ----------------------
    print(f"\n{'=' * 70}\nWHAT WE LEARNED\n{'=' * 70}")
    mins = args.frames / 30.0 / 60.0
    if poss_json and os.path.exists(poss_json):
        doc = json.load(open(poss_json, encoding="utf-8"))
        poss = doc["possessions"]
        ends = {}
        for p in poss:
            ends[p["ended_by"]] = ends.get(p["ended_by"], 0) + 1
        print(f"  possessions: {len(poss)} in {mins:.1f} min "
              f"-> {len(poss) / mins:.1f}/min, so ~{len(poss) / mins * 32:.0f} in a "
              f"32-minute game (a real one is roughly 70)")
        print(f"  ended by: {ends}")
        oob = ends.get("out_of_bounds", 0)
        print(f"  OUT OF BOUNDS fired {oob} time(s)" +
              ("  <- FIRST TIME EVER on real film" if oob else
               "  <- still never seen; the rule remains unproven"))
        secs = {}
        for p in poss:
            secs[p["team"]] = round(secs.get(p["team"], 0) + p["seconds"], 1)
        print(f"  seconds by team: {secs}")
    else:
        print("  NO POSSESSIONS -- the layer abstained; see the reason above.")
    # WALL CLOCK IS ONLY HONEST ON A COLD RUN. Tracking, the on-court cache and
    # the ball detections are reused when the span already matches, so a rerun
    # reports a fraction of a second per frame and would project a nonsense
    # full-game figure. Say which kind of run it was instead of quoting a number
    # that flatters itself.
    dt = time.time() - t0
    print(f"\n  wall clock: {dt:.0f}s for {args.frames} frames "
          f"= {dt / args.frames:.2f}s per frame")
    if reused:
        print(f"  ^ REUSED cached {', '.join(reused)} -- NOT a cold-run cost and "
              f"NOT usable to project a full game.")
    else:
        print(f"  cold run: a 95-minute game (171,120 frames) would take "
              f"{dt / args.frames * 171120 / 3600:.0f} hours on this CPU")
    print("  MEASURED COLD on this CPU (2026-08-04), per frame: tracking 3.1s + "
          "camera anchor 3.6s + ball detection 2.5s = ~9.2s/frame, "
          "i.e. ~440 hours (18 days) for a full game.")


if __name__ == "__main__":
    main()
