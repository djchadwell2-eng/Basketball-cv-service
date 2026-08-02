"""FLICKER CHECK -- DJ's question, measured before any safeguard is written.

DJ (2026-07-28): "if a girl puts up a shot but it flickers to another girl,
does the one it flickers to get credited with the shot?"

A FLICKER has a precise signature: the credited holder goes A -> B -> A. The
ball did not really change hands twice in half a second; one of B's frames is
an occlusion artifact, with B's body passing between the camera and the ball.

TWO SEPARATE HARMS, measured separately because they need different answers:

  HARM 1 -- B STEALS A TOUCH (the one DJ asked about).
     Only possible if B's flicker run reaches MIN_TOUCH_FRAMES. Below that the
     existing floor already deletes it. So the question is entirely empirical:
     how LONG do real flickers last on this footage?

  HARM 2 -- A's TOUCH GETS SPLIT IN HALF (quieter, and I had not flagged it).
     build_touches ends a run the instant a different holder appears. So even
     a 2-frame flicker that is itself discarded still CUTS A's touch in two.
     That costs credited time and can leave a shot with no touch before it.

Also cross-references flickers against claimed shot arcs: a flicker far from
any shot is a coverage nuisance, a flicker in the moments before a release is
the thing that would misattribute a shot.

Usage (one clip per process):
    .venv/Scripts/python.exe spikes/flicker_check.py TEST1
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "phase2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ball_touch as bt                                          # noqa: E402
from local_weights_check import CONF_FLOOR                       # noqa: E402

# A flicker near a shot is the dangerous kind. "Near" = within this many frames
# before the arc's first point -- the window in which the last touch would be
# read as the shooter. Matches shooter_compare's MAX_BACK_FRAMES.
NEAR_SHOT_FRAMES = 60


def _load(p):
    return json.load(open(p, encoding="utf-8"))


def credited_runs(verdicts):
    """[(holder, start_frame, end_frame, n_credited_frames)] over frames that
    actually named a holder. Frames with no holder do not start or end a run --
    same view build_touches takes."""
    runs = []
    for f in sorted(verdicts):
        tid = verdicts[f].get("track_id")
        if tid is None:
            continue
        if runs and runs[-1][0] == tid:
            runs[-1][2], runs[-1][3] = f, runs[-1][3] + 1
        else:
            runs.append([tid, f, f, 1])
    return [tuple(r) for r in runs]


def flickers(runs):
    """Runs whose neighbours on BOTH sides are the same OTHER holder: A-B-A."""
    out = []
    for i in range(1, len(runs) - 1):
        prev, cur, nxt = runs[i - 1], runs[i], runs[i + 1]
        if prev[0] == nxt[0] and cur[0] != prev[0]:
            out.append({"holder": cur[0], "start": cur[1], "end": cur[2],
                        "frames": cur[3], "between": prev[0],
                        "span": cur[2] - cur[1] + 1})
    return out


def main():
    clip = sys.argv[1] if len(sys.argv) > 1 else "TEST1"
    import clip_config
    cfg = getattr(clip_config, f"{clip}_CLIP")
    clip_config.ACTIVE_CLIP = cfg
    from roster import load_ref_tracks

    out_dir = os.path.join(_ROOT, "spikes", "out")
    det = _load(os.path.join(out_dir, f"{clip}_ball_detections.json"))
    tracks = _load(cfg.tracks_cache_path)
    excl = load_ref_tracks(os.path.join(_ROOT, "phase2", "out",
                                        f"{clip}_decisions.json"))
    ball = {fr["frame_index"]: bt.ball_position(fr["detections"], CONF_FLOOR)
            for fr in det["frames"]}
    tbf = {fr["frame_index"]: fr["tracks"] for fr in tracks["frames"]}
    verdicts = {f: bt.holder_at_frame(ball[f], tbf[f], exclude=excl)
                for f in sorted(set(ball) & set(tbf))}
    fps = det.get("fps") or 30.0

    runs = credited_runs(verdicts)
    fl = flickers(runs)

    print(f"\nFLICKER CHECK -- {clip}")
    print(f"  credited runs: {len(runs)}   flickers (A->B->A): {len(fl)}")
    if not fl:
        print("  no flickers found.")
        return

    lens = Counter(f["frames"] for f in fl)
    print(f"\n  how long does a flicker last? (credited frames)")
    for n in sorted(lens):
        print(f"    {n} frame(s) ({n / fps * 1000:>4.0f}ms): {lens[n]}"
              + ("   <-- SURVIVES the 6-frame floor" if n >= bt.MIN_TOUCH_FRAMES else ""))
    survivors = [f for f in fl if f["frames"] >= bt.MIN_TOUCH_FRAMES]
    print(f"\n  HARM 1 -- flickers long enough to STEAL A TOUCH: "
          f"{len(survivors)} of {len(fl)}")
    if not survivors:
        print(f"    NONE. Every flicker on this clip is shorter than "
              f"MIN_TOUCH_FRAMES={bt.MIN_TOUCH_FRAMES}, so the existing floor "
              f"already deletes all of them.")
    for s in survivors:
        print(f"    f{s['start']}..{s['end']}  t{s['holder']} "
              f"({s['frames']} frames) interrupting t{s['between']}")

    # HARM 2: how many of A's runs were cut by a flicker?
    print(f"\n  HARM 2 -- real touches SPLIT by a flicker that was itself "
          f"discarded: {len(fl) - len(survivors)}")
    print(f"    each one ends the holder's run early and starts a new one, "
          f"costing credited time even though the flicker itself is dropped.")

    # near a shot?
    sa_path = os.path.join(out_dir, f"{clip}_shot_attempts.json")
    if os.path.exists(sa_path):
        arcs = [r["start_frame"] for r in _load(sa_path)["attempts"]
                if r["verdict"] == "shot_attempt"]
        near = [(f, a) for f in fl for a in arcs
                if 0 <= a - f["end"] <= NEAR_SHOT_FRAMES]
        print(f"\n  flickers within {NEAR_SHOT_FRAMES} frames BEFORE a claimed "
              f"shot (the window that decides the shooter): {len(near)}")
        for (f, a) in near:
            danger = ("WOULD STEAL THE SHOT" if f["frames"] >= bt.MIN_TOUCH_FRAMES
                      else "too short to steal it")
            print(f"    flicker f{f['start']}..{f['end']} t{f['holder']} "
                  f"({f['frames']}f) before arc at f{a}  -- {danger}")


if __name__ == "__main__":
    main()
