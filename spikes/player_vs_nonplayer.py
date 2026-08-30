"""Can COLOUR tell a player from a referee or a coach?

WHY THIS MATTERS. Five-on-court exclusion works by counting: if exactly five of a
team are on court, four are CONFIRMED and one body is unnamed, that body is
forced to be the missing lineup member. The counting is only sound if the pool
contains PLAYERS ONLY. A referee standing in it gets confidently named, and a
confident wrong name is worse than a hundred abstentions.
DJ's build order, 2026-08-29: "Player-vs-non-player first. Exclusion with refs
in the pool will confidently name a referee."

WHAT IS ALREADY CLOSED, so this does not retry it:
  MOTION. Measured by DJ 2026-08-29 -- only 10% of tracks cover 25+ ft, median
  9.6 ft, and a coach walking the bench covered 26 ft. Dead.

THE HYPOTHESIS UNDER TEST (DJ's): colour. phase2/color_tiebreak.py already
builds TEAM COLOUR CENTROIDS from crops the system already trusts, measured on
this footage under this lighting. A body matching NEITHER team is not a player.

THE HONEST RISK, WRITTEN DOWN FIRST. On 2026-08-03 I measured that a torso crop
CANNOT be matched to an absolute colour: white and green kits both came back
muddy grey (~(121,97,109) and (82,93,101)). classify_team survives that because
it asks a RELATIVE question -- which of two centroids is nearer, by a margin.
"Far from BOTH" is much closer to the absolute question that already failed. So
this is a measurement, not a plan. It may simply not separate.

GROUND TRUTH IS FREE. phase2/out/{clip}_decisions.json holds human track labels
made months ago: jersey numbers for players, "ref"/"bench" for non-players.
Across TEST1/HARD/TEST2 that is 21 non-players against ~46 players, already on
disk, no new video processing beyond reading a few frames per track.

LEAVE-ONE-OUT, because otherwise this cheats. A player's own crops help build
her team's centroid, so testing her against it would flatter the result. Each
player is scored against centroids rebuilt WITHOUT her.

THE NUMBER THAT DECIDES IT is not "how many refs were caught". It is HOW MANY
REAL PLAYERS GET WRONGLY REFUSED -- a ref wrongly kept is a nuisance the rest of
the gate can survive, but a player wrongly refused deletes real floor time and
breaks the count that exclusion depends on.

Usage:
    .venv/Scripts/python.exe spikes/player_vs_nonplayer.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "phase2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                  # noqa: E402

import color_tiebreak as ct                                         # noqa: E402
import ocr_reader                                                   # noqa: E402
import stage2_multikeyframe as s2                                   # noqa: E402

CLIPS = ("TEST1", "HARD", "TEST2")
MIN_BOX_H = 90         # same floor stage6 uses -- below this a crop is unusable
CROPS_PER_TRACK = 6    # spread across her life, not six pictures of one pose
NON_PLAYER = {"ref", "bench"}


def _labels(clip):
    """{track_id: label} from the human decisions file. Labels are jersey number
    strings for players, 'ref'/'bench' for non-players."""
    p = os.path.join(_ROOT, "phase2", "out", f"{clip}_decisions.json")
    if not os.path.exists(p):
        return {}
    doc = json.load(open(p, encoding="utf-8"))
    raw = doc.get("track_labels") or doc.get("labels") or {}
    out = {}
    for tid, v in raw.items():
        # Values are ints for players ({"2": 3}) and strings for non-players
        # ({"1": "ref"}). The first version only handled strings and silently
        # dropped EVERY player, leaving nothing to build a centroid from.
        if isinstance(v, bool) or v is None:
            continue
        if isinstance(v, int):
            lab = str(v)
        elif isinstance(v, str):
            lab = v
        elif isinstance(v, dict):
            lab = v.get("kind")
        else:
            lab = None
        if lab and lab != "None":
            out[int(tid)] = lab
    return out


def _pick_frames(boxes, n=CROPS_PER_TRACK):
    """Up to n frames SPREAD ACROSS the track's life, biggest boxes preferred
    within each slice. Ten crops from one second is ten pictures of one pose --
    the mistake DJ measured on 30% of players (handoff 2026-08-29)."""
    usable = [(f, bb) for (f, bb) in boxes if (bb[3] - bb[1]) >= MIN_BOX_H]
    if not usable:
        return []
    usable.sort()
    step = max(1, len(usable) // n)
    picked = []
    for i in range(0, len(usable), step):
        chunk = usable[i:i + step]
        picked.append(max(chunk, key=lambda fb: fb[1][3] - fb[1][1]))
        if len(picked) >= n:
            break
    return picked


def track_colours(clip, wanted):
    """{track_id: mean BGR} for the wanted tracks, from real crops."""
    import clip_config
    cfg = clip_config.get_clip(clip)
    trk = json.load(open(cfg.tracks_cache_path, encoding="utf-8"))
    boxes = defaultdict(list)
    for fr in trk["frames"]:
        for t in fr["tracks"]:
            if t["track_id"] in wanted:
                boxes[t["track_id"]].append((fr["frame_index"], t["bbox"]))

    picks = {tid: _pick_frames(bb) for tid, bb in boxes.items()}
    need = sorted({f for p in picks.values() for (f, _b) in p})
    if not need:
        return {}
    imgs = dict(s2.iter_frames(cfg.video_path, need))
    out = {}
    for tid, p in picks.items():
        sigs = []
        for (f, bb) in p:
            im = imgs.get(f)
            if im is None:
                continue
            crop = ocr_reader.jersey_crop(im, bb)
            if crop is not None and crop.size:
                sigs.append(ct.crop_color_signature(crop))
        if sigs:
            out[tid] = tuple(np.mean(sigs, axis=0))
    return out


def team_of_number(clip, number):
    """The team wearing this number, or None if it is AMBIGUOUS.

    HARD lists #3 and #23 on BOTH rosters. Assigning such a number to whichever
    team happens to be first would feed one team's centroid with the other
    team's crops -- quietly wrecking the very thing being measured. Ambiguous
    numbers are dropped from centroid building instead."""
    import clip_config
    cfg = clip_config.get_clip(clip)
    hits = [t.name for t in cfg.teams if number in t.numbers]
    return hits[0] if len(hits) == 1 else None


def main():
    rows = []
    for clip in CLIPS:
        labs = _labels(clip)
        if not labs:
            print(f"{clip}: no decisions file, skipped")
            continue
        cols = track_colours(clip, set(labs))
        players, nonplayers = {}, {}
        for tid, lab in labs.items():
            if tid not in cols:
                continue
            if lab in NON_PLAYER:
                nonplayers[tid] = (lab, cols[tid])
            elif lab.isdigit():
                team = team_of_number(clip, int(lab))
                if team:
                    players[tid] = (team, cols[tid])

        by_team = defaultdict(list)
        for tid, (team, c) in players.items():
            by_team[team].append((tid, c))
        if len(by_team) < 2:
            print(f"{clip}: only {len(by_team)} team(s) labelled, cannot build "
                  f"two centroids -- skipped")
            continue

        def centroids(exclude_tid=None):
            return {t: tuple(np.mean([c for (i, c) in v if i != exclude_tid], axis=0))
                    for t, v in by_team.items()
                    if any(i != exclude_tid for (i, _c) in v)}

        # LEAVE-ONE-OUT for players; non-players never contribute, so the full
        # centroids are already honest for them.
        full = centroids()
        for tid, (team, c) in players.items():
            cs = centroids(exclude_tid=tid)
            if len(cs) < 2:
                continue
            d = sorted(ct._dist(c, v) for v in cs.values())
            rows.append((clip, tid, "player", team, d[0], d[1]))
        for tid, (lab, c) in nonplayers.items():
            d = sorted(ct._dist(c, v) for v in full.values())
            rows.append((clip, tid, lab, "-", d[0], d[1]))
        print(f"{clip}: {len(players)} players, {len(nonplayers)} non-players measured")

    if not rows:
        raise SystemExit("no data")

    pl = [r for r in rows if r[2] == "player"]
    npl = [r for r in rows if r[2] in NON_PLAYER]
    print(f"\n{'=' * 68}\nDISTANCE TO NEAREST TEAM COLOUR CENTROID\n{'=' * 68}")

    def stats(name, xs):
        v = sorted(x[4] for x in xs)
        if not v:
            return
        print(f"  {name:<22} n={len(v):<3} min {v[0]:6.1f}  median "
              f"{v[len(v) // 2]:6.1f}  p90 {v[int(.9 * (len(v) - 1))]:6.1f}  "
              f"max {v[-1]:6.1f}")
    stats("PLAYERS", pl)
    stats("NON-PLAYERS", npl)

    # The only question that matters: is there a cut that keeps players and
    # drops non-players? Sweep it and report BOTH error directions.
    print(f"\n  a body FARTHER than the cut is called 'not a player':")
    print(f"  {'cut':>6}  {'refs caught':>12}  {'PLAYERS LOST':>13}")
    best = None
    for cut in range(10, 140, 10):
        caught = sum(1 for r in npl if r[4] > cut)
        lost = sum(1 for r in pl if r[4] > cut)
        print(f"  {cut:>6}  {caught:>4}/{len(npl):<7}  {lost:>4}/{len(pl):<8}"
              + ("   <- zero players lost" if lost == 0 else ""))
        if lost == 0 and (best is None or caught > best[1]):
            best = (cut, caught)
    print()
    if best and best[1] > 0:
        print(f"  BEST SAFE CUT: {best[0]} -> catches {best[1]}/{len(npl)} "
              f"non-players with ZERO players lost.")
    else:
        print("  NO SAFE CUT EXISTS: every threshold that catches a non-player "
              "also throws away a real player. Colour does NOT separate them, "
              "and exclusion needs a different gate.")


if __name__ == "__main__":
    main()
