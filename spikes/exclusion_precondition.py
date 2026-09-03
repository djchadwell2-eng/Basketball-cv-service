"""How often is five-on-court exclusion actually ALLOWED to fire?

THE RULE IT MEASURES (DJ's, and the whole safety of the idea rests on it):
assign a name only when the arithmetic is EXACTLY FORCED --
    exactly 5 of that team on court, no more, no fewer
    exactly 4 of them named
    exactly 1 unnamed
Anything else and she stays unknown.

WHY MEASURE BEFORE BUILDING. A previous pass measured this UN-SCOPED -- counting
unnamed bodies across BOTH teams at once -- and got "exactly one unnamed" in 16%
of TEST1 frames, with the precondition met 0 of 3 times at the actual relink
moments. But exclusion is a PER-TEAM argument: five Milford players on court,
four named, so the fifth is the missing Milford girl. Both teams pooled together
is ten bodies and answers a question nobody asked.
So the un-scoped 16% is not the number that decides this. This is.

KILL NUMBER, written down first: if per-team scoping does not lift "exactly one
unnamed on this team" above roughly 40% of frames, exclusion fires too rarely to
change the order of magnitude, and the naming plan needs a different idea.

TEAM COMES FROM JERSEY COLOUR, and that is not a free assumption -- it is
BLOCKER 2 of the exclusion plan, so this measures it too. The method is
touch_teams': cluster the on-court bodies' torso colours into two groups
MEASURED ON THIS FOOTAGE (never matched against a typed colour name, which was
refuted 2026-08-03), then use the roster colours only to decide which cluster is
which team. Accuracy is scored against the human labels as a free byproduct.

NAMED means what the pipeline means by it: an identity a human labelled. Those
labels are in {clip}_decisions.json.

Usage:
    .venv/Scripts/python.exe spikes/exclusion_precondition.py
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "phase2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                  # noqa: E402

import color_tiebreak as ct                                         # noqa: E402
import ocr_reader                                                   # noqa: E402
import stage2_multikeyframe as s2                                   # noqa: E402
import touch_teams as tt                                            # noqa: E402

CLIPS = ("TEST1", "HARD", "TEST2")
MIN_BOX_H = 90
CROPS_PER_TRACK = 6
NON_PLAYER = {"ref", "bench"}


def labels_of(clip):
    p = os.path.join(_ROOT, "phase2", "out", f"{clip}_decisions.json")
    if not os.path.exists(p):
        return {}
    raw = json.load(open(p, encoding="utf-8")).get("track_labels") or {}
    out = {}
    for tid, v in raw.items():
        if isinstance(v, bool) or v is None:
            continue
        lab = str(v) if isinstance(v, int) else (v if isinstance(v, str) else None)
        if lab and lab != "None":
            out[int(tid)] = lab
    return out


def oncourt_by_frame(clip):
    """{frame: {track_id}} for bodies the classifier put ON the court."""
    p = os.path.join(_ROOT, "phase2", "out", f"{clip}_oncourt.json")
    doc = json.load(open(p, encoding="utf-8"))
    return {fr["frame_index"]: {int(t) for t, i in fr["tracks"].items() if i["on"]}
            for fr in doc["frames"]}


def team_by_track(clip, wanted):
    """{track_id: team_name} from jersey colour, measured on this footage."""
    import clip_config
    cfg = clip_config.get_clip(clip)
    trk = json.load(open(cfg.tracks_cache_path, encoding="utf-8"))
    boxes = defaultdict(list)
    for fr in trk["frames"]:
        for t in fr["tracks"]:
            if t["track_id"] in wanted and (t["bbox"][3] - t["bbox"][1]) >= MIN_BOX_H:
                boxes[t["track_id"]].append((fr["frame_index"], t["bbox"]))

    picks = {}
    for tid, bb in boxes.items():
        bb.sort()
        step = max(1, len(bb) // CROPS_PER_TRACK)
        picks[tid] = bb[::step][:CROPS_PER_TRACK]        # spread across her life
    need = sorted({f for v in picks.values() for (f, _b) in v})
    if not need:
        return {}, None
    imgs = dict(s2.iter_frames(cfg.video_path, need))
    colours = {}
    for tid, v in picks.items():
        sigs = [ct.crop_color_signature(ocr_reader.jersey_crop(imgs[f], bb))
                for (f, bb) in v if imgs.get(f) is not None]
        sigs = [s for s in sigs if s]
        if sigs:
            colours[tid] = tuple(np.mean(sigs, axis=0))

    refs = tt.refs_from_teams(cfg.teams)
    assigned, reason, _detail = tt.team_of_tracks(colours, refs) if refs else ({}, "no colours", None)
    return assigned, reason


def main():
    for clip in CLIPS:
        labs = labels_of(clip)
        if not labs:
            print(f"{clip}: no labels, skipped\n")
            continue
        onc = oncourt_by_frame(clip)
        on_tracks = {t for v in onc.values() for t in v}
        teams, reason = team_by_track(clip, on_tracks)
        if reason:
            print(f"{clip}: team assignment ABSTAINED -- {reason}\n")
            continue

        # --- free byproduct: how accurate IS the colour team assignment? -----
        import clip_config
        cfg = clip_config.get_clip(clip)
        num_team = {}
        for t in cfg.teams:
            for n in t.numbers:
                num_team.setdefault(n, []).append(t.name)
        right = wrong = 0
        for tid, lab in labs.items():
            if not lab.isdigit() or tid not in teams:
                continue
            truth = num_team.get(int(lab), [])
            if len(truth) != 1:            # dual-roster number: unscoreable
                continue
            right += (teams[tid] == truth[0])
            wrong += (teams[tid] != truth[0])
        acc = f"{right}/{right + wrong}" if (right + wrong) else "n/a"

        named = {tid for tid, lab in labs.items()
                 if lab.isdigit()}                      # a human named her
        nonplayers = {tid for tid, lab in labs.items() if lab in NON_PLAYER}

        # --- the precondition, per team, per frame ---------------------------
        hits = Counter()
        per_team_counts = Counter()
        for f, on in sorted(onc.items()):
            on = on - nonplayers                         # refs are not players
            for team in {teams[t] for t in on if t in teams}:
                mine = [t for t in on if teams.get(t) == team]
                n_named = sum(1 for t in mine if t in named)
                n_un = len(mine) - n_named
                per_team_counts[len(mine)] += 1
                if len(mine) == 5 and n_named == 4 and n_un == 1:
                    hits["FORCED"] += 1
                elif len(mine) == 5 and n_un == 1:
                    hits["5 on court, 1 unnamed, but <4 named"] += 1
                elif n_un == 1:
                    hits[f"1 unnamed but {len(mine)} on court"] += 1
                else:
                    hits[f"{n_un} unnamed"] += 1

        total = sum(hits.values())
        print(f"=== {clip} ===")
        print(f"  colour team assignment vs human labels: {acc}")
        print(f"  team-frames examined: {total}")
        print(f"  bodies per team per frame: "
              f"{dict(sorted(per_team_counts.items()))}")
        forced = hits['FORCED']
        print(f"  EXACTLY FORCED (5 on court, 4 named, 1 unnamed): "
              f"{forced} = {100.0 * forced / total:.1f}%" if total else "  no data")
        for k, v in hits.most_common(6):
            if k != "FORCED":
                print(f"     {k:<38} {v:>6} ({100.0 * v / total:.1f}%)")
        print()


if __name__ == "__main__" and "--relinks" not in sys.argv:
    main()


# ---------------------------------------------------------------------------
# THE METRIC THAT ACTUALLY MATTERS
#
# "% of frames where the precondition holds" is the wrong question, and the
# 40% kill number was set against it. Exclusion does not need to fire every
# frame -- it needs to fire ONCE, at the moment a name would otherwise be lost,
# to carry that name across the break. A rule that holds in 15% of frames could
# still cover most relinks, or none of them; the frame rate cannot tell you.
#
# So this asks the real question: AT THE MOMENT A RELINK IS NEEDED -- a track
# carrying a known number dies, and the same number reappears on a NEW track --
# is the arithmetic forced for that team right then?
# ---------------------------------------------------------------------------

def relink_moments(clip, labs, onc, teams, named, nonplayers):
    """Real relinks: the same jersey number on two tracks that never coexist."""
    import clip_config
    cfg = clip_config.get_clip(clip)
    trk = json.load(open(cfg.tracks_cache_path, encoding="utf-8"))
    life = defaultdict(list)
    for fr in trk["frames"]:
        for t in fr["tracks"]:
            life[t["track_id"]].append(fr["frame_index"])
    span = {t: (min(v), max(v)) for t, v in life.items()}

    by_num = defaultdict(list)
    for tid, lab in labs.items():
        if lab.isdigit() and tid in span:
            by_num[lab].append(tid)

    out = []
    for num, tids in by_num.items():
        tids.sort(key=lambda t: span[t][0])
        for a, b in zip(tids, tids[1:]):
            # NOT co-alive -- two tracks alive at once are two different girls
            # sharing a number (dual rosters), not one girl fragmented.
            if span[a][1] < span[b][0]:
                out.append((num, a, b, span[b][0]))
    return out


def check_relinks():
    import clip_config
    for clip in CLIPS:
        labs = labels_of(clip)
        if not labs:
            continue
        onc = oncourt_by_frame(clip)
        on_tracks = {t for v in onc.values() for t in v}
        teams, reason = team_by_track(clip, on_tracks)
        if reason:
            continue
        named = {tid for tid, lab in labs.items() if lab.isdigit()}
        nonplayers = {tid for tid, lab in labs.items() if lab in NON_PLAYER}
        moments = relink_moments(clip, labs, onc, teams, named, nonplayers)

        forced = 0
        detail = []
        for (num, a, b, birth) in moments:
            on = onc.get(birth, set()) - nonplayers
            team = teams.get(b) or teams.get(a)
            if team is None:
                detail.append((num, birth, "no team for either track"))
                continue
            mine = [t for t in on if teams.get(t) == team]
            n_named = sum(1 for t in mine if t in named)
            n_un = len(mine) - n_named
            ok = (len(mine) == 5 and n_named == 4 and n_un == 1)
            forced += ok
            detail.append((num, birth,
                           f"{len(mine)} on court, {n_named} named, {n_un} unnamed"
                           + ("  <- FORCED" if ok else "")))
        print(f"=== {clip}: {len(moments)} real relink moment(s) ===")
        for (num, birth, why) in detail:
            print(f"   #{num:<4} new track at f{birth:<7} {why}")
        if moments:
            print(f"   exclusion would have fired: {forced}/{len(moments)}")
        print()


if __name__ == "__main__" and "--relinks" in sys.argv:
    check_relinks()
