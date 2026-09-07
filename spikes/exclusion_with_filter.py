"""Does the five-on-court count read FIVE once referees and coaches are removed?

THE CHAIN THAT LEADS HERE, all measured:
  - exclusion fired 0 of 9 at real relink moments, because the on-court count
    per team read 1 to 9 bodies with SIX and SEVEN common (2026-09-03)
  - not the geometry margin (median body sits 0.00 ft outside the lines, and
    known bench tracks sit 0.16-0.51 ft out, inside the slack a real player
    needs), and not duplicate detections (99.8% of the excess do not overlap an
    already-counted player). Referees, coaches and bench bodies really are
    standing inside the court rectangle.
  - colour cannot tell them apart (no threshold catches a referee without
    deleting a real player) and neither can motion (a coach walked 26 ft)
  - ASKING THE VISION MODEL DOES: under RULE B -- only a UNANIMOUS "REFEREE" or
    "COACH" may delete a body, so OTHER ("cannot tell") and split votes always
    keep her -- it deleted ZERO real players on the labelled sample and caught
    two non-players the human labels themselves had missed.

So this applies that filter to EVERY on-court body, not just the labelled ones,
and re-asks the only question that matters: at the moment a name would otherwise
be lost, is the arithmetic forced?

WHY THE ASYMMETRY IS SAFE. A referee wrongly kept leaves the count one too high,
and the gate refuses on anything but exactly five -- an abstention, which costs a
click. A real player wrongly deleted turns a five into a FOUR, and a four with
one unnamed body FORCES A WRONG NAME. Rule B is chosen for that reason, not for
its accuracy score.

Votes are cached per clip so re-running the arithmetic never costs an API call.

Usage:
    .venv/Scripts/python.exe spikes/exclusion_with_filter.py
"""

from __future__ import annotations

import base64
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "phase2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2                                                          # noqa: E402

import ask_is_player as aip                                         # noqa: E402
import stage2_multikeyframe as s2                                   # noqa: E402

CLIPS = ("TEST1", "HARD", "TEST2")
CROPS_PER_TRACK = 3


def classify_all(clip, track_ids):
    """{track_id: (verdict, agreeing, asked)} for EVERY on-court track."""
    cache = os.path.join(_HERE, "out", f"{clip}_is_player_all.json")
    if os.path.exists(cache):
        d = json.load(open(cache, encoding="utf-8"))
        print(f"  reusing {len(d)} cached verdicts (delete {os.path.basename(cache)} "
              f"to re-ask)")
        return {int(k): tuple(v) for k, v in d.items()}

    import clip_config
    import env_local
    env_local.load()
    import google.genai
    client = google.genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    cfg = clip_config.get_clip(clip)
    trk = json.load(open(cfg.tracks_cache_path, encoding="utf-8"))
    boxes = defaultdict(list)
    for fr in trk["frames"]:
        for t in fr["tracks"]:
            if t["track_id"] in track_ids and (t["bbox"][3] - t["bbox"][1]) >= aip.MIN_BOX_H:
                boxes[t["track_id"]].append((fr["frame_index"], t["bbox"]))
    picks = {}
    for tid, bb in boxes.items():
        bb.sort()
        step = max(1, len(bb) // CROPS_PER_TRACK)
        picks[tid] = bb[::step][:CROPS_PER_TRACK]
    need = sorted({f for v in picks.values() for (f, _b) in v})
    if not need:
        return {}
    imgs = dict(s2.iter_frames(cfg.video_path, need))
    jobs = []
    for tid, v in picks.items():
        for (f, bb) in v:
            im = imgs.get(f)
            if im is None:
                continue
            crop = aip.body_crop(im, bb)
            if crop is None or crop.size == 0:
                continue
            big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            ok, jpg = cv2.imencode(".jpg", big)
            if ok:
                jobs.append((tid, base64.standard_b64encode(jpg.tobytes()).decode()))
    print(f"  asking about {len(jobs)} crops from {len(picks)} on-court tracks",
          flush=True)
    with ThreadPoolExecutor(max_workers=aip.WORKERS) as ex:
        res = list(ex.map(lambda j: aip.ask_once(client, j[1]), jobs))
    votes = defaultdict(list)
    for (tid, _b), v in zip(jobs, res):
        if v:
            votes[tid].append(v)
    out = {}
    for tid, vs in votes.items():
        top, n = Counter(vs).most_common(1)[0]
        out[tid] = (top, n, len(vs))
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump({str(k): list(v) for k, v in out.items()}, open(cache, "w"), indent=1)
    return out


def is_non_player(verdict):
    """RULE B. Only a UNANIMOUS referee/coach may delete a body. An unclassified
    track is KEPT -- missing evidence must never remove a player."""
    if verdict is None:
        return False
    top, n, asked = verdict
    return top in ("REFEREE", "COACH") and n == asked


def main():
    sys.argv = ["x"]
    import exclusion_precondition as ep

    for clip in CLIPS:
        labs = ep.labels_of(clip)
        if not labs:
            continue
        print(f"=== {clip} ===")
        onc = ep.oncourt_by_frame(clip)
        on_tracks = {t for v in onc.values() for t in v}
        verdicts = classify_all(clip, on_tracks)
        dropped = {t for t in on_tracks if is_non_player(verdicts.get(t))}
        print(f"  {len(on_tracks)} on-court tracks -> {len(dropped)} refused as "
              f"referee/coach, {len(on_tracks) - len(dropped)} kept as players")

        # TEAMS FOR EVERY on-court track, INCLUDING the refused ones. Computing
        # them only for the survivors made the BEFORE and AFTER counts identical
        # -- a refused track had no team, so it was already absent from both.
        # The filter has to be applied in the counting, not in the input.
        teams, reason = ep.team_by_track(clip, on_tracks)
        if reason:
            print(f"  team assignment ABSTAINED: {reason}\n")
            continue
        named = {t for t, l in labs.items() if l.isdigit()}

        # --- the count, before and after the filter -------------------------
        for label, drop in (("BEFORE (bodies only)", set()),
                            ("AFTER  (players only)", dropped)):
            counts = Counter()
            for f, on in onc.items():
                on = on - drop
                for team in {teams[t] for t in on if t in teams}:
                    counts[len([t for t in on if teams.get(t) == team])] += 1
            tot = sum(counts.values())
            five = counts.get(5, 0)
            over = sum(v for k, v in counts.items() if k > 5)
            print(f"  {label}: exactly five {100.0 * five / max(tot,1):5.1f}%   "
                  f"MORE than five {100.0 * over / max(tot,1):5.1f}%   "
                  f"spread {dict(sorted(counts.items()))}")

        # --- and the only question that matters -----------------------------
        moments = ep.relink_moments(clip, labs, onc, teams, named, dropped)
        forced = 0
        for (num, a, b, birth) in moments:
            on = onc.get(birth, set()) - dropped
            team = teams.get(b) or teams.get(a)
            if team is None:
                continue
            mine = [t for t in on if teams.get(t) == team]
            n_named = sum(1 for t in mine if t in named)
            ok = (len(mine) == 5 and n_named == 4 and len(mine) - n_named == 1)
            forced += ok
            print(f"     #{num:<4} f{birth:<7} {len(mine)} on court, {n_named} named"
                  + ("   <- FORCED" if ok else ""))
        if moments:
            print(f"  EXCLUSION WOULD FIRE: {forced}/{len(moments)}")
        print()


if __name__ == "__main__":
    main()
