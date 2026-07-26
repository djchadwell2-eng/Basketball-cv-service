"""ONE-OFF MIGRATION -- re-key the coach's queue clicks from identity_id to
track_id, so they survive code changes.

Why this exists. A queue click was filed as (window, identity_id). identity_id
is just a creation counter (identity.py `_next_id`), so ANY change to how
identities are created renumbers it -- and the only staleness guard checks
window boundaries (stage7_merge.load_queue_resolutions), which do not move when
the identity logic changes. On 2026-07-25 the relink rule changed and the guard
passed: 11 of DJ's 15 HARD clicks and 7 of his 10 TEST1 clicks silently landed
on a DIFFERENT body, naming ~20.5s of HARD's box score and ~11.5s of TEST1's
from clicks that had moved. Nothing failed loudly.

track_id does not have this problem -- it comes from the tracker, not from our
numbering. It is why DJ's 22 Part-1 track labels survived every code change
since July while his queue clicks did not. And since the relink fix, an
identity maps to exactly ONE track for life (measured: HARD 283/283,
TEST1 151/151), so (window, track_id) identifies the same thing (window,
identity_id) used to -- but stably.

What this does: for each saved click, look up the identity it was made against
in the PRE-CHANGE artifacts (phase2/out/_baseline_.../) -- i.e. what the coach
was actually looking at -- and record that track_id.

A click made on an identity that spanned SEVERAL tracks (one of the bad merges
the relink fix removed) is NOT migrated. The coach saw crops from a chain that
covered two or three different people; which one they meant is genuinely
unknowable, and guessing would be exactly the silent misattribution this whole
exercise is about. Those are marked needs_review and refused at load.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, "out")


def identity_tracks(events_path):
    """(window, identity_id) -> {track_id: frames} from a player_events doc."""
    with open(events_path, encoding="utf-8") as f:
        doc = json.load(f)
    events = doc.get("player_events", doc)
    out = defaultdict(lambda: defaultdict(int))
    for e in events:
        out[(e["window"], e["identity_id"])][int(e["track_id"])] += 1
    return out


def migrate(resolutions, baseline_tracks):
    """Add track_id to each resolution using the identity->track map the coach
    actually clicked against. Returns (migrated, report_rows)."""
    out, rows = [], []
    for r in resolutions:
        key = (r["window"], r["identity_id"])
        tracks = baseline_tracks.get(key, {})
        rec = dict(r)
        if len(tracks) == 1:
            rec["track_id"] = int(next(iter(tracks)))
            rec["rekeyed_from_identity_id"] = r["identity_id"]
            status = "migrated"
        elif len(tracks) > 1:
            rec["needs_review"] = (
                "clicked on an identity that spanned "
                f"{len(tracks)} different tracks {sorted(tracks)} -- which "
                "player was meant cannot be recovered; re-click this one")
            status = "AMBIGUOUS -- needs re-click"
        else:
            rec["needs_review"] = ("no matching identity in the pre-change "
                                   "artifacts; cannot recover the track")
            status = "NOT FOUND"
        out.append(rec)
        rows.append((r["window"], r["identity_id"], r["number"],
                     rec.get("track_id"), sorted(tracks), status))
    return out, rows


def run(clip, baseline_dir, write=False):
    dec_path = os.path.join(OUT, f"{clip}_decisions.json")
    with open(dec_path, encoding="utf-8") as f:
        dec = json.load(f)
    res = dec.get("queue_resolutions", [])
    if not res:
        print(f"{clip}: no queue resolutions"); return
    base = identity_tracks(os.path.join(baseline_dir,
                                        f"{clip}_player_events_merged.json"))
    migrated, rows = migrate(res, base)

    print(f"===== {clip}: {len(res)} saved clicks =====")
    for w, i, n, tid, tracks, status in rows:
        tgt = f"-> track {tid}" if tid is not None else f"tracks {tracks}"
        print(f"   w{w} id{i} = #{n:<7} {tgt:<22} {status}")
    ok = sum(1 for r in migrated if "track_id" in r)
    print(f"   {ok} recovered, {len(migrated) - ok} need re-clicking")

    if write:
        dec["queue_resolutions"] = migrated
        # Stamp what these clicks were keyed against, so a future reader can
        # tell migrated clicks from raw identity_id ones at a glance.
        dec["resolution_key"] = "track_id"
        with open(dec_path, "w", encoding="utf-8") as f:
            json.dump(dec, f, indent=2)
        print(f"   WROTE {dec_path}")
    else:
        print("   (dry run -- pass --write to save)")


def main():
    baseline = os.path.join(OUT, "_baseline_20260725")
    write = "--write" in sys.argv
    clips = [a for a in sys.argv[1:] if not a.startswith("-")] or ["HARD", "TEST1"]
    if not os.path.isdir(baseline):
        raise SystemExit(f"missing pre-change artifacts at {baseline} -- "
                         "without them the clicks cannot be recovered")
    for c in clips:
        run(c, baseline, write=write)


if __name__ == "__main__":
    main()
