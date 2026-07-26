"""IDENTITY HEALTH REPORT -- the before/after safety net for identity changes.

DJ asked the right question: "is there a way to do the relinking fix and then
check whether it made more problems, before committing to it?" This is that
check. It reads only shipped artifacts, changes nothing, and answers four
questions -- run it before a change, run it after, compare.

The reason this can be trusted: DJ's own track labels in {clip}_decisions.json
are GROUND TRUTH. He looked at a crop and said "that is #23". So an identity
chain that spans two tracks he labelled with DIFFERENT numbers is wrong by
construction -- one identity cannot be two people. That gives an objective
correctness score no amount of tuning can game.

CORRECTNESS comes first and outranks coverage. DJ: "I don't want a lost player
as long as it's correct." A change that raises coverage while raising
wrong_chain_seconds is a REGRESSION, however good the headline looks.

Coverage is measured against what is actually READABLE -- bodies on court,
referees excluded -- not against an assumed 10 players per frame. Players walk
off a panning camera and substitute; 55% of HARD's frames hold fewer than 10
players, so a "x10" denominator sets a target the footage cannot contain.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "out")
_CONFIRMED = ("confirmed", "confirmed_retroactive")


def _load(clip, name, required=True):
    path = os.path.join(OUT, f"{clip}_{name}.json")
    if not os.path.exists(path):
        if required:
            raise SystemExit(f"missing {path}")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def collect(clip):
    """All four measures for a clip, as a plain dict (so tests can assert on it)."""
    merged = _load(clip, "player_events_merged")
    oncourt = _load(clip, "oncourt")
    decisions = _load(clip, "decisions", required=False) or {}
    box = _load(clip, "box_score", required=False)
    queue = _load(clip, "review_queue", required=False) or {}

    events = merged["player_events"]
    fps = float(oncourt.get("fps") or 30.0)
    labels = {int(k): v for k, v in decisions.get("track_labels", {}).items()}
    numbered = {k: v for k, v in labels.items() if isinstance(v, int)}
    refs = {k for k, v in labels.items() if v == "ref"}

    on_by_frame = {fr["frame_index"]: {int(t) for t, r in fr["tracks"].items() if r["on"]}
                   for fr in oncourt["frames"]}

    # --- 1. CORRECTNESS: identities that span two DIFFERENT labelled players --
    chains = defaultdict(lambda: defaultdict(int))
    for e in events:
        chains[(e["window"], e["identity_id"])][int(e["track_id"])] += 1
    wrong, judgeable, wrong_frames = [], 0, 0
    for key, tracks in chains.items():
        nums = {numbered[t] for t in tracks if t in numbered}
        if nums:
            judgeable += 1
        if len(nums) >= 2:
            wrong_frames += sum(tracks.values())
            wrong.append({"window": key[0], "identity_id": key[1],
                          "numbers": sorted(nums), "frames": sum(tracks.values()),
                          "tracks": dict(sorted(tracks.items(), key=lambda r: -r[1]))})
    wrong.sort(key=lambda r: -r["frames"])

    # --- 2. COVERAGE against what is readable --------------------------------
    readable = sum(1 for e in events
                   if int(e["track_id"]) in on_by_frame.get(e["frame"], ())
                   and int(e["track_id"]) not in refs)
    named_frames = (sum(p["seconds_total"] for p in box["players"]) * fps) if box else 0.0

    # --- 3. Is DJ's OWN clicking reaching the box score? ---------------------
    on_labelled = [e for e in events if int(e["track_id"]) in numbered]
    credited = sum(1 for e in on_labelled if e["identity_state"] in _CONFIRMED)

    # --- 4. LEGALITY: a team cannot have 6 players on the floor --------------
    #   Counted on NAMED confirmed identities only; a violation means either an
    #   unremarked substitution or a live misattribution -- either way, look.
    teams = {}
    if box:
        for p in box["players"]:
            if not str(p.get("team", "")).startswith("AMBIGUOUS"):
                teams[(p["team"], p["number"])] = p["team"]
    over = 0
    if box:
        num_of = {}
        for r in (_load(clip, "ocr_confirms", required=False) or {}).get("identities", []):
            if r.get("roster_number") is not None:
                num_of[(r["window"], r["identity_id"])] = r["roster_number"]
        per_frame = defaultdict(lambda: defaultdict(set))
        team_of_num = {n: t for (t, n) in teams}
        for e in events:
            if e["identity_state"] not in _CONFIRMED:
                continue
            n = num_of.get((e["window"], e["identity_id"]))
            t = team_of_num.get(n)
            if t:
                per_frame[e["frame"]][t].add(n)
        over = sum(1 for f, d in per_frame.items() for t, s in d.items() if len(s) > 5)

    return {
        "clip": clip, "fps": fps,
        "wrong_chains": len(wrong), "judgeable_chains": judgeable,
        "wrong_frames": wrong_frames, "wrong_seconds": round(wrong_frames / fps, 1),
        "worst": wrong[:5],
        "readable_frames": readable, "named_frames": round(named_frames),
        "pct_of_readable": round(100.0 * named_frames / readable, 1) if readable else 0.0,
        "labelled_frames": len(on_labelled), "labelled_credited": credited,
        "pct_your_clicks_used": round(100.0 * credited / len(on_labelled), 1) if on_labelled else 0.0,
        "queue_items": len(queue.get("items", []) if isinstance(queue, dict) else queue),
        "illegal_frames": over,
        "human_labels": {"numbered": len(numbered), "ref": len(refs)},
    }


def render(r, target_pct=80.0):
    L = []
    L.append(f"===== {r['clip']} =====")
    L.append("  CORRECTNESS (outranks coverage -- these are provably wrong)")
    L.append(f"    identity chains covering 2+ DIFFERENT players you labelled: "
             f"{r['wrong_chains']} of {r['judgeable_chains']} checkable")
    L.append(f"    time credited to the WRONG player: {r['wrong_seconds']}s "
             f"({r['wrong_frames']} frames)")
    for w in r["worst"][:3]:
        L.append(f"      window {w['window']} identity {w['identity_id']}: "
                 f"one identity covering players {w['numbers']} over {w['frames']} frames")
    if r["illegal_frames"]:
        L.append(f"    frames showing MORE THAN 5 players on one team: "
                 f"{r['illegal_frames']}  (impossible -- misattribution or a substitution)")
    L.append("  COVERAGE (of what is actually readable: on court, referees excluded)")
    hit = "MEETS TARGET" if r["pct_of_readable"] >= target_pct else "below target"
    L.append(f"    identified: {r['pct_of_readable']}% of {round(r['readable_frames'] / r['fps'])}s "
             f"readable   [target {target_pct:.0f}% -- {hit}]")
    L.append("  YOUR CLICKS")
    L.append(f"    frames on players you labelled that actually reach the box score: "
             f"{r['pct_your_clicks_used']}%  ({r['labelled_credited']}/{r['labelled_frames']})")
    L.append(f"    review queue: {r['queue_items']} items   "
             f"(you labelled {r['human_labels']['numbered']} tracks, "
             f"{r['human_labels']['ref']} refs)")
    return "\n".join(L)


def compare(before, after):
    """Did a change help or hurt? Correctness first -- a coverage gain that costs
    correctness is a REGRESSION, not a trade."""
    L = ["", "===== BEFORE -> AFTER ====="]
    verdict = []
    for b, a in zip(before, after):
        dw = a["wrong_seconds"] - b["wrong_seconds"]
        dc = a["pct_of_readable"] - b["pct_of_readable"]
        dk = a["pct_your_clicks_used"] - b["pct_your_clicks_used"]
        L.append(f"  {b['clip']}")
        L.append(f"    wrong-player time  {b['wrong_seconds']:6.1f}s -> {a['wrong_seconds']:6.1f}s "
                 f"({dw:+.1f}s)   {'WORSE' if dw > 0.05 else 'better' if dw < -0.05 else 'same'}")
        L.append(f"    coverage           {b['pct_of_readable']:6.1f}% -> {a['pct_of_readable']:6.1f}% ({dc:+.1f})")
        L.append(f"    your clicks used   {b['pct_your_clicks_used']:6.1f}% -> {a['pct_your_clicks_used']:6.1f}% ({dk:+.1f})")
        if dw > 0.05:
            verdict.append(f"{b['clip']}: REGRESSION -- more time on the wrong player")
    L.append("")
    L.append("  VERDICT: " + ("; ".join(verdict) if verdict
                              else "no correctness regression on any clip"))
    return "\n".join(L)


def main():
    clips = sys.argv[1:] or ["HARD", "TEST1"]
    reports = [collect(c) for c in clips]
    for r in reports:
        print(render(r))
    snap = os.path.join(OUT, "identity_report.json")
    with open(snap, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2)
    print(f"\nsaved {snap}  (re-run after a change and use compare())")


if __name__ == "__main__":
    main()
