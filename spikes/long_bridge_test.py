"""THE LONG-WAIT TEST -- DJ's rule, built so he can judge whether it is RIGHT.

DJ's rule (2026-07-27): "the girl last seen with the ball has the ball until
proven otherwise", with a ceiling of 15 SECONDS (his number, from basketball).

WHY THIS SCRIPT EXISTS AND THE EARLIER SWEEP DID NOT SETTLE IT: TEST 21's sweep
measured HOW MUCH time would be filled in. It never measured whether the
filling-in is CORRECT -- and that is the whole question. If she really did have
the ball, the filled seconds are TRUE and the short setting was UNDERCOUNTING
her. A table of quantities cannot tell those apart. A human watching the box
stay on (or wander off) the right girl can.

So this writes a SUFFIXED artifact pair -- the canonical
{clip}_ball_touches.json / .mp4 are never clobbered -- and prints the honest
before/after so the cost in inferred time is visible next to the benefit.

BE CLEAR ABOUT WHAT 15s MEANS ON THESE CLIPS: TEST1's answerable window is
15.4s and HARD's is 20.0s, so a 15s ceiling is EFFECTIVELY NO LIMIT here. That
makes this the WORST CASE on purpose. If the box still follows the right girl
under no limit at all, the rule is safe at any smaller number. If it wanders,
we will see exactly where and why -- which is more useful than a middling
setting that hides the failure.

Usage (one clip per process):
    .venv/Scripts/python.exe spikes/long_bridge_test.py TEST1
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "phase2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ball_touch as bt                                          # noqa: E402
from local_weights_check import CONF_FLOOR                       # noqa: E402

CEILING_SECONDS = 15.0        # DJ's number, 2026-07-27. From basketball.
TAG = "gap15s"


def _load(p):
    return json.load(open(p, encoding="utf-8"))


def run(config):
    out_dir = os.path.join(_ROOT, "spikes", "out")
    det = _load(os.path.join(out_dir, f"{config.name}_ball_detections.json"))
    tracks = _load(config.tracks_cache_path)
    oncourt = _load(os.path.join(_ROOT, "phase2", "out",
                                 f"{config.name}_oncourt.json"))
    events = _load(os.path.join(_ROOT, "phase2", "out",
                                f"{config.name}_player_events_merged.json"))
    reg_path = os.path.join(_ROOT, "phase2", "out",
                            f"{config.name}_ocr_confirms.json")
    registry = _load(reg_path) if os.path.exists(reg_path) else None

    from roster import load_ref_tracks
    excl = load_ref_tracks(os.path.join(_ROOT, "phase2", "out",
                                        f"{config.name}_decisions.json"))

    fps = det.get("fps") or 30.0
    ceiling = int(round(CEILING_SECONDS * fps))

    common = dict(conf_floor=CONF_FLOOR, registry_doc=registry,
                  exclude_tracks=excl)
    now = bt.analyze(det, tracks, oncourt, events, **common)
    long_ = bt.analyze(det, tracks, oncourt, events,
                       max_gap=ceiling, **common)

    span_s = now["overlap_frames"] / fps
    print(f"\nTHE LONG-WAIT TEST -- {config.name}")
    print(f"  answerable window: {span_s:.1f}s "
          f"({now['overlap_frames']} frames)")
    print(f"  ceiling: {CEILING_SECONDS:.0f}s = {ceiling} frames"
          + (f"  <-- LONGER THAN THE WHOLE WINDOW, so this is "
             f"'no limit at all' on this clip"
             if ceiling >= now["overlap_frames"] else ""))

    def tot(r, key):
        return sum(t[key] for t in r["touches"])

    print(f"\n  {'':<18}{'touches':>9}{'SEEN s':>9}{'FILLED s':>10}{'% filled':>10}")
    for label, r in (("now (0.27s wait)", now), (f"DJ's rule ({CEILING_SECONDS:.0f}s)", long_)):
        obs, inf = tot(r, "observed_seconds"), tot(r, "inferred_seconds")
        pct = 100 * inf / max(obs + inf, 1e-9)
        print(f"  {label:<18}{len(r['touches']):>9}{obs:>9.1f}{inf:>10.1f}{pct:>9.0f}%")

    print(f"\n  what MERGED (two touches under the old rule -> one under DJ's):")
    old_spans = {(t["start_frame"], t["end_frame"]) for t in now["touches"]}
    for t in long_["touches"]:
        inside = [s for s in old_spans
                  if s[0] >= t["start_frame"] and s[1] <= t["end_frame"]]
        n = t["identity"]["jersey_number"]
        who = f"#{n}" if n is not None else "unnamed"
        flag = "" if t["on_court"] else "   <- OFF_COURT"
        mark = "  <-- MERGED" if len(inside) > 1 else ""
        print(f"    f{t['start_frame']:>5}..{t['end_frame']:<5} track "
              f"{t['track_id']:>4}  {t['observed_seconds']:>4.1f}s seen +"
              f"{t['inferred_seconds']:>5.1f}s filled  {who:<8}"
              f"{mark}{flag}")

    out_json = os.path.join(out_dir, f"{config.name}_ball_touches_{TAG}.json")
    json.dump({"clip": config.name, "conf_floor": CONF_FLOOR,
               "note": f"DJ's until-proven-otherwise rule, {CEILING_SECONDS:.0f}s "
                       f"ceiling. NOT the canonical artifact -- a test.",
               "params": {"MAX_GAP_FRAMES": ceiling,
                          "HOLD_GATE_BODY_FRAC": bt.HOLD_GATE_BODY_FRAC,
                          "MARGIN_BODY_FRAC": bt.MARGIN_BODY_FRAC,
                          "MIN_TOUCH_FRAMES": bt.MIN_TOUCH_FRAMES},
               **long_}, open(out_json, "w", encoding="utf-8"), indent=2)
    print(f"\n  wrote {out_json}")

    import render_ball_touches as rr
    out_mp4 = os.path.join(out_dir, f"{config.name}_ball_touches_overlay_{TAG}.mp4")
    rr.render(config, out_path=out_mp4, touches_path=out_json, exclude=excl)
    print(f"\n  WATCH THIS and judge ONE thing: through the DARK GREEN "
          f"stretches\n  (ball not visible), does the box stay on the RIGHT "
          f"girl?")


if __name__ == "__main__":
    import clip_config
    name = sys.argv[1] if len(sys.argv) > 1 else "TEST1"
    cfg = getattr(clip_config, f"{name}_CLIP")
    clip_config.ACTIVE_CLIP = cfg
    run(cfg)
