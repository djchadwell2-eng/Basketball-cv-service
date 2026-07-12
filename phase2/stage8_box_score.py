"""STAGE 8 -- the JERSEY-KEYED box score (the output a coach actually reads).

Aggregates the MERGED player_events by jersey number ACROSS windows (numbers
are the stable player key; identity_ids are per-window internals). Counts live
`confirmed` and `confirmed_retroactive` seconds separately and jointly, and
joins each confirmed frame to its court position (already computed in the
on-court cache) for per-player ZONE time.

HONESTY RULES (enforced here, tested in tests/test_box_score.py):
  * Confirmed identities WITHOUT a jersey number are SURFACED as an "unnamed"
    bucket -- real floor time that needs a coach click to become a name. Never
    dropped, never guessed.
  * A number on BOTH rosters (e.g. HARD's #3) is flagged team-AMBIGUOUS until
    the jersey-color tiebreak exists.
  * Candidate/unknown events go to review counts, never to a player line.
  * These are presence-seconds over a short clip -- not game stats. Say so.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))                          # repo root (clip_config)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "phase1"))  # zones
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "spikes"))  # zones' court dims

import zones as Z
from clip_config import ACTIVE_CLIP as CLIP

MERGED_JSON = os.path.join(_HERE, "out", f"{CLIP.name}_player_events_merged.json")
OCR_JSON = os.path.join(_HERE, "out", f"{CLIP.name}_ocr_confirms.json")
OUT_JSON = os.path.join(_HERE, "out", f"{CLIP.name}_box_score.json")
OUT_CSV = os.path.join(_HERE, "out", f"{CLIP.name}_box_score.csv")

_CONFIRMED = ("confirmed", "confirmed_retroactive")


def build_box_score(merged_doc, registry, number_teams, fps, oncourt_doc):
    """Pure aggregation: merged events + identity registry + roster team map +
    on-court positions -> the box-score document."""
    numbers = {(r["window"], r["identity_id"]): r["roster_number"]
               for r in registry}
    pos = {}
    for fr in oncourt_doc.get("frames", []):
        for tid_s, info in fr.get("tracks", {}).items():
            pos[(fr["frame_index"], int(tid_s))] = tuple(info["court_feet"])

    # number -> frame -> [(state, track_id, window), ...]  A frame claimed by
    # 2+ identities for the SAME number = one body is wrong (splice/mislabel):
    # DISPUTED -- excluded from the line, surfaced, never counted.
    claims = defaultdict(dict)
    unnamed = {"identities": set(), "frames": 0}
    review = {"candidate_events": 0, "unknown_events": 0}

    for e in merged_doc["player_events"]:
        st = e["identity_state"]
        if st not in _CONFIRMED:
            if st == "candidate":
                review["candidate_events"] += 1
            elif st == "unknown":
                review["unknown_events"] += 1
            continue
        # The merge stamp (OCR agree or human resolution) is the most specific
        # name an event carries; the registry hypothesis is the fallback.
        n = (e.get("merge") or {}).get("number")
        if n is None:
            n = numbers.get((e["window"], e["identity_id"]))
        if n is None:                              # confirmed but nameless:
            unnamed["identities"].add((e["window"], e["identity_id"]))
            unnamed["frames"] += 1                 # surface, never drop
            continue
        claims[n].setdefault(e["frame"], []).append(
            (st, e["track_id"], e["window"]))

    players = []
    for n, by_frame in claims.items():
        sole = {f: c[0] for f, c in by_frame.items() if len(c) == 1}
        n_disputed = sum(1 for c in by_frame.values() if len(c) > 1)
        live = sum(1 for (st, _t, _w) in sole.values() if st == "confirmed")
        retro = len(sole) - live
        windows = {w for (_s, _t, w) in sole.values()}
        zones, unpositioned = defaultdict(int), 0
        for f, (_st, tid, _w) in sole.items():
            xy = pos.get((f, tid))
            if xy is None:
                unpositioned += 1
            else:
                zones[Z.zone_of(*xy)] += 1
        teams = number_teams.get(n, [])
        team = (teams[0] if len(teams) == 1
                else f"AMBIGUOUS ({' / '.join(teams)})" if teams else "off-roster?")
        top = max(zones, key=zones.get) if zones else "-"
        players.append({
            "number": n, "team": team,
            "seconds_live": live / fps,
            "seconds_retro": retro / fps,
            "seconds_total": (live + retro) / fps,
            "disputed_seconds": n_disputed / fps,
            "windows_present": len(windows),
            "zone_seconds": {z: c / fps for z, c in sorted(zones.items())},
            "top_zone": top,
            "unpositioned_frames": unpositioned})
    players.sort(key=lambda r: -r["seconds_total"])

    return {"clip": merged_doc.get("clip"),
            "note": "presence-seconds over this clip's tracked span -- "
                    "not full-game stats",
            "players": players,
            "unnamed_confirmed": {"identities": len(unnamed["identities"]),
                                  "seconds_total": unnamed["frames"] / fps},
            "review": review}


def main():
    for path, hint in ((MERGED_JSON, "run stage7_merge first"),
                       (OCR_JSON, "run stage6 first")):
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} -- {hint}.")
    merged = json.load(open(MERGED_JSON, encoding="utf-8"))
    ocr = json.load(open(OCR_JSON, encoding="utf-8"))
    if "identities" not in ocr:
        raise SystemExit(f"{OCR_JSON} has no identities registry -- re-run stage6.")
    import oncourt
    onc = oncourt.load_checked(CLIP)
    tdoc = json.load(open(CLIP.tracks_cache_path, encoding="utf-8"))
    fps = tdoc["fps"]

    number_teams = {}
    for t in CLIP.teams:
        for n in t.numbers:
            number_teams.setdefault(n, []).append(t.name)

    doc = build_box_score(merged, ocr["identities"], number_teams, fps, onc)

    span_s = tdoc["span_len"] / fps
    print(f"BOX SCORE -- {doc['clip']}  ({span_s:.0f}s tracked span; "
          f"{doc['note']})")
    print(f"\n  {'#':>3} {'team':<28} {'total_s':>8} {'live_s':>7} {'retro_s':>8} "
          f"{'windows':>8}  top zone")
    for r in doc["players"]:
        tz = (f"{r['top_zone']} ({r['zone_seconds'].get(r['top_zone'], 0):.1f}s)"
              if r["top_zone"] != "-" else "-")
        disp = (f"  [!] {r['disputed_seconds']:.1f}s DISPUTED (same number in "
                f"2 places -- not counted)" if r["disputed_seconds"] else "")
        print(f"  {r['number']:>3} {r['team']:<28} {r['seconds_total']:>8.1f} "
              f"{r['seconds_live']:>7.1f} {r['seconds_retro']:>8.1f} "
              f"{r['windows_present']:>8}  {tz}{disp}")
    u = doc["unnamed_confirmed"]
    print(f"\n  UNNAMED confirmed floor time: {u['identities']} identities, "
          f"{u['seconds_total']:.1f}s  <- real players awaiting a coach click "
          f"(click-seeding, next unit)")
    rv = doc["review"]
    print(f"  review (never counted): {rv['candidate_events']} candidate + "
          f"{rv['unknown_events']} unknown event-frames")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["number", "team", "seconds_total", "seconds_live",
                    "seconds_retro", "disputed_seconds", "windows_present",
                    "top_zone"])
        for r in doc["players"]:
            w.writerow([r["number"], r["team"], f"{r['seconds_total']:.2f}",
                        f"{r['seconds_live']:.2f}", f"{r['seconds_retro']:.2f}",
                        f"{r['disputed_seconds']:.2f}",
                        r["windows_present"], r["top_zone"]])
    print(f"\nsaved box score -> {OUT_JSON}  (+ CSV: {os.path.basename(OUT_CSV)})")


if __name__ == "__main__":
    main()
