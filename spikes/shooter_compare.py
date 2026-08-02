"""WHO SHOT IT -- two methods, scored head to head. NOTHING ADOPTED.

DJ's proposal (2026-07-27): credit a shot to the last player actually SEEN
holding the ball. The pipeline already answers this question a different way,
and the point of this script is to put them side by side rather than assume the
new one is better.

    TODAY (shot_attempts.find_release)
        Extend the arc's own parabola BACKWARD up to 10 frames, then credit
        whichever tracked body's box the predicted ball position lands on
        within 120px. A guess about WHERE BODIES WERE STANDING.

    PROPOSED (ball_touch.shooter_from_touches)
        The last recorded TOUCH before the arc started. A memory of who was
        WATCHED HOLDING THE BALL.

HONEST LIMIT, STATED UP FRONT: the project's ground truth
(local_weights_check.GROUND_TRUTH) records WHICH arcs are real shots -- it does
NOT record WHO TOOK THEM. So this script CANNOT declare a winner on its own.
What it can do is find the DISAGREEMENTS and hand them to DJ, whose answer then
becomes the shooter ground truth that does not exist yet. Agreement between two
methods is weak evidence (they can be wrong together); disagreement is where
the information is.

Also reports coverage honestly: the proposed method abstains whenever no touch
was recorded before the arc, which on this footage is often.

Usage (one clip per process):
    .venv/Scripts/python.exe spikes/shooter_compare.py HARD
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
from local_weights_check import GROUND_TRUTH                     # noqa: E402

# How stale a touch may be and still speak to a shot. Basketball reason, not a
# fitted number: a player who last held the ball two seconds ago has had time
# to pass it, so the memory stops being evidence. Declared BEFORE the run.
MAX_BACK_FRAMES = 60


def _load(p):
    return json.load(open(p, encoding="utf-8"))


def verified_shot(clip, start, end):
    """Is this arc one DJ confirmed is a real shot? (Spans are recorded per
    clip; match generously since arc bounds shift slightly by model.)"""
    for (a, b, _hoop) in GROUND_TRUTH.get(clip, {}).get("shots", []):
        if abs(a - start) <= 12 and abs(b - end) <= 12:
            return True
    return False


def main():
    clip = sys.argv[1] if len(sys.argv) > 1 else "HARD"
    out = os.path.join(_ROOT, "spikes", "out")
    sa = _load(os.path.join(out, f"{clip}_shot_attempts.json"))
    touch_doc = _load(os.path.join(out, f"{clip}_ball_touches.json"))
    touches = touch_doc["touches"]
    tracks_span = tuple(sa["tracks_span"])

    attempts = [r for r in sa["attempts"] if r["verdict"] == "shot_attempt"]
    print(f"WHO SHOT IT -- {clip}: {len(attempts)} claimed shot attempt(s), "
          f"{len(touches)} recorded touch(es)")
    print(f"  tracks span {tracks_span[0]}..{tracks_span[1]} -- outside it "
          f"NEITHER method can answer (no player data at all)")
    print(f"  MAX_BACK_FRAMES = {MAX_BACK_FRAMES} (~{MAX_BACK_FRAMES / 30:.0f}s; "
          f"declared before the run)\n")

    hdr = (f"{'arc':>13} {'DJ-verified':>12} {'TODAY (standing)':>18} "
           f"{'PROPOSED (held it)':>20} {'verdict':>12}")
    print(hdr)
    print("-" * len(hdr))

    agree = disagree = only_today = only_proposed = neither = 0
    rows = []
    for r in attempts:
        a, b = r["start_frame"], r["end_frame"]
        in_span = tracks_span[0] <= a < tracks_span[1]
        gt = "YES" if verified_shot(clip, a, b) else "-"

        sh = r.get("shooter", {})
        today = (f"t{sh.get('track_id')}" if sh.get("status") in
                 ("attributed", "review_item") else sh.get("status", "-"))

        t = bt.shooter_from_touches(touches, a, MAX_BACK_FRAMES)
        if t is None:
            prop = "no touch recorded"
        else:
            n = t["identity"]["jersey_number"]
            prop = f"t{t['track_id']}" + (f" (#{n})" if n is not None else "")

        today_has = today.startswith("t")
        prop_has = prop.startswith("t")
        if not in_span:
            # Outside the tracks span neither method has player data, so this
            # row is not evidence about either -- and must not be TALLIED as
            # if it were (it was, until 2026-07-27).
            verdict = "(outside span)"
        elif today_has and prop_has:
            same = today.split()[0] == prop.split()[0]
            verdict = "AGREE" if same else "DISAGREE <<"
            agree, disagree = agree + same, disagree + (not same)
        elif today_has:
            verdict, only_today = "only TODAY", only_today + 1
        elif prop_has:
            verdict, only_proposed = "only PROPOSED", only_proposed + 1
        else:
            verdict, neither = "neither", neither + 1
        rows.append((f"{a}..{b}", gt, today, prop, verdict))

    for row in rows:
        print(f"{row[0]:>13} {row[1]:>12} {row[2]:>18} {row[3]:>20} {row[4]:>12}")

    print(f"\n  agree {agree}   DISAGREE {disagree}   "
          f"only today {only_today}   only proposed {only_proposed}   "
          f"neither {neither}")
    print("\n  NO WINNER IS DECLARED HERE. The ground truth records which arcs")
    print("  are real shots, not WHO TOOK THEM -- so the DISAGREE rows above")
    print("  are the ones for DJ to adjudicate. His answers become the shooter")
    print("  ground truth this project does not have yet.")


if __name__ == "__main__":
    main()
