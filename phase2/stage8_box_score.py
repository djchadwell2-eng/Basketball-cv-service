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
  * A number on BOTH rosters (e.g. HARD's #3) is split into separate per-team
    lines via the jersey-color tiebreak (phase2/color_tiebreak.py) wherever
    color evidence resolves it; whatever can't be resolved stays flagged
    team-AMBIGUOUS -- never a guess. Disputed (simultaneous conflicting-claim)
    seconds are never attributed to a team either, since the identity conflict
    itself is unresolved -- they surface on their own AMBIGUOUS row.
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

import roster
import zones as Z
from clip_config import ACTIVE_CLIP as CLIP

MERGED_JSON = os.path.join(_HERE, "out", f"{CLIP.name}_player_events_merged.json")
OCR_JSON = os.path.join(_HERE, "out", f"{CLIP.name}_ocr_confirms.json")
OUT_JSON = os.path.join(_HERE, "out", f"{CLIP.name}_box_score.json")
OUT_CSV = os.path.join(_HERE, "out", f"{CLIP.name}_box_score.csv")

_CONFIRMED = ("confirmed", "confirmed_retroactive")


def build_box_score(merged_doc, registry, number_teams, fps, oncourt_doc,
                    identity_team=None):
    """Pure aggregation: merged events + identity registry + roster team map +
    on-court positions -> the box-score document.

    identity_team: optional {(window, identity_id): team_name} -- the color
    tiebreak's resolved verdicts for identities credited under a dual-roster
    number (phase2/color_tiebreak.py). Absence of a key means unresolved:
    that identity's frames stay in the AMBIGUOUS bucket, exactly as if this
    argument were never passed (default behavior is unchanged)."""
    identity_team = identity_team or {}
    numbers = {(r["window"], r["identity_id"]): r["roster_number"]
               for r in registry}
    pos = {}
    for fr in oncourt_doc.get("frames", []):
        for tid_s, info in fr.get("tracks", {}).items():
            pos[(fr["frame_index"], int(tid_s))] = tuple(info["court_feet"])

    # number -> frame -> [(state, track_id, window, identity_id), ...]  A
    # frame claimed by 2+ identities for the SAME number = one body is wrong
    # (splice/mislabel): DISPUTED -- excluded from the line, surfaced, never
    # counted, and (for dual-roster numbers) never team-attributed either,
    # since the identity conflict itself is unresolved.
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
            (st, e["track_id"], e["window"], e["identity_id"]))

    players = []
    for n, by_frame in claims.items():
        sole = {f: c[0] for f, c in by_frame.items() if len(c) == 1}
        n_disputed = sum(1 for c in by_frame.values() if len(c) > 1)
        teams_for_n = number_teams.get(n, [])
        ambiguous_label = (f"AMBIGUOUS ({' / '.join(teams_for_n)})"
                           if len(teams_for_n) > 1 else None)

        # team_label -> {frame: (state, track_id, window)}
        groups = defaultdict(dict)
        if ambiguous_label is None:
            label = teams_for_n[0] if teams_for_n else "off-roster?"
            for f, (st, tid, w, _iid) in sole.items():
                groups[label][f] = (st, tid, w)
        else:
            for f, (st, tid, w, iid) in sole.items():
                label = identity_team.get((w, iid), ambiguous_label)
                groups[label][f] = (st, tid, w)
            if n_disputed:
                # Disputed frames belong to no identity we trust -- surface
                # them on the AMBIGUOUS row even if every SOLE frame resolved
                # cleanly to a team (never silently drop them).
                groups.setdefault(ambiguous_label, {})

        for team_label, group_frames in groups.items():
            live = sum(1 for (st, _t, _w) in group_frames.values() if st == "confirmed")
            retro = len(group_frames) - live
            windows = {w for (_s, _t, w) in group_frames.values()}
            zones, unpositioned = defaultdict(int), 0
            for f, (_st, tid, _w) in group_frames.items():
                xy = pos.get((f, tid))
                if xy is None:
                    unpositioned += 1
                else:
                    zones[Z.zone_of(*xy)] += 1
            top = max(zones, key=zones.get) if zones else "-"
            # A single-team number has exactly one row -- disputed seconds
            # go there, as before. A dual-roster number splits into several
            # rows; disputed seconds go ONLY on the AMBIGUOUS row (we don't
            # know which team a disputed/conflicting identity belongs to).
            row_disputed = n_disputed if ambiguous_label in (None, team_label) else 0
            players.append({
                "number": n, "team": team_label,
                "seconds_live": live / fps,
                "seconds_retro": retro / fps,
                "seconds_total": (live + retro) / fps,
                "disputed_seconds": row_disputed / fps,
                "windows_present": len(windows),
                "zone_seconds": {z: c / fps for z, c in sorted(zones.items())},
                "top_zone": top,
                "unpositioned_frames": unpositioned})
    players.sort(key=lambda r: -r["seconds_total"])

    # READABLE player time: every track-frame that is on court and is not a
    # referee the human flagged -- i.e. real player time that COULD carry a
    # jersey number. This is the only honest denominator for "how much of the
    # individual tracking do we actually have", and it belongs here, computed
    # once from the same on-court data the box score is built from, so the app
    # cannot invent a second, friendlier one. Measuring against
    # named+unnamed-CONFIRMED instead reads 53.6% on HARD where the truth is
    # 36.7%, because it silently drops every candidate/unknown frame -- which
    # is exactly the player time we FAILED to identify.
    refs = roster.ref_tracks()
    readable = sum(1 for fr in oncourt_doc["frames"]
                   for tid, rec in fr["tracks"].items()
                   if rec["on"] and int(tid) not in refs)

    return {"clip": merged_doc.get("clip"),
            "note": "presence-seconds over this clip's tracked span -- "
                    "not full-game stats",
            "players": players,
            "readable_seconds": round(readable / fps, 1),
            "unnamed_confirmed": {"identities": len(unnamed["identities"]),
                                  "seconds_total": unnamed["frames"] / fps},
            "review": review}


CROP_SAMPLES_PER_IDENTITY = 6   # frames sampled per identity for color classification


def _identity_occurrences(merged_doc, registry):
    """(window, identity_id, number) -> [(frame, track_id), ...] occurrences,
    for every CONFIRMED/CONFIRMED_RETROACTIVE event. Pure, no I/O -- mirrors
    build_box_score's own number resolution (merge stamp wins over registry)
    so the two never disagree.

    Keyed by the TRIPLE, not just (window, identity_id): one identity can
    legitimately carry more than one number if different portions of its
    span were separately merge-stamped (e.g. a Part-1 track label plus a
    later, different Part-2 queue resolution on the same track). Collapsing
    to one number per identity would silently drop one claim-group's frames
    from color sampling instead of resolving them -- each (window, identity,
    number) claim-group is sampled and classified on its OWN frames only."""
    numbers = {(r["window"], r["identity_id"]): r["roster_number"]
               for r in registry}
    occurrences = defaultdict(list)
    for e in merged_doc["player_events"]:
        if e["identity_state"] not in _CONFIRMED:
            continue
        n = (e.get("merge") or {}).get("number")
        if n is None:
            n = numbers.get((e["window"], e["identity_id"]))
        if n is None:
            continue
        occurrences[(e["window"], e["identity_id"], n)].append(
            (e["frame"], e["track_id"]))
    return occurrences


def _spread_sample(occs, cap):
    """Evenly spaced subset of an identity's occurrences, at most `cap` --
    a few frames spread across the span beat many frames bunched together."""
    occs = sorted(set(occs))
    if len(occs) <= cap:
        return occs
    step = len(occs) / cap
    return [occs[int(i * step)] for i in range(cap)]


def _resolve_ambiguous_teams(config, merged, registry, number_teams, tdoc):
    """Color tiebreak for every (window, identity, number) claim-group
    credited under a dual-roster number (phase2/color_tiebreak.py). Returns
    {(window, identity_id): team_name} for claim-groups that resolved;
    claim-groups NOT in the returned dict stay AMBIGUOUS in the box score --
    color evidence too close to call is never treated as a guess. Zero-cost
    (no video read) when the roster has no dual-team numbers, e.g. TEST1."""
    ambiguous_numbers = {n for n, teams in number_teams.items() if len(teams) > 1}
    if not ambiguous_numbers:
        return {}

    occurrences = _identity_occurrences(merged, registry)
    ref_triples = {k for k in occurrences if k[2] not in ambiguous_numbers}
    amb_triples = {k for k in occurrences if k[2] in ambiguous_numbers}
    if not amb_triples:
        return {}

    ref_samples = defaultdict(list)                  # team -> [(frame, track_id), ...]
    for key in ref_triples:
        team = number_teams[key[2]][0]
        ref_samples[team].extend(_spread_sample(occurrences[key], CROP_SAMPLES_PER_IDENTITY))
    amb_samples = {key: _spread_sample(occurrences[key], CROP_SAMPLES_PER_IDENTITY)
                   for key in amb_triples}

    need_frames = {f for picks in ref_samples.values() for f, _t in picks}
    need_frames |= {f for picks in amb_samples.values() for f, _t in picks}
    if not need_frames:
        return {}

    bbox_at = {}
    for fr in tdoc["frames"]:
        for t in fr["tracks"]:
            bbox_at[(fr["frame_index"], t["track_id"])] = t["bbox"]

    import cv2
    frames_img = {}
    cap = cv2.VideoCapture(config.video_path)
    it = iter(sorted(need_frames))
    nxt, idx = next(it, None), 0
    while nxt is not None and cap.grab():
        if idx == nxt:
            ok, img = cap.retrieve()
            if ok:
                frames_img[idx] = img
            nxt = next(it, None)
        idx += 1
    cap.release()

    import ocr_reader

    def crops_for(picks):
        out = []
        for f, tid in picks:
            bb = bbox_at.get((f, tid))
            img = frames_img.get(f)
            if bb is not None and img is not None:
                out.append(ocr_reader.jersey_crop(img, bb))
        return out

    import color_tiebreak
    crops_by_team = {team: crops_for(picks) for team, picks in ref_samples.items()}
    centroids = color_tiebreak.build_team_centroids(crops_by_team)
    print(f"[color tiebreak] reference crops: "
          + ", ".join(f"{t}={len(c)}" for t, c in crops_by_team.items())
          + f" -> {len(centroids)} team centroid(s)")

    identity_team = {}
    for (w, iid, _n), picks in amb_samples.items():
        team = color_tiebreak.classify_identity(crops_for(picks), centroids)
        if team:
            identity_team[(w, iid)] = team
    n_resolved = len(identity_team)
    print(f"[color tiebreak] {n_resolved}/{len(amb_samples)} ambiguous "
          f"claim-group(s) resolved by jersey color "
          f"({len(amb_samples) - n_resolved} stay AMBIGUOUS -- "
          f"color evidence too close to call)")
    return identity_team


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

    identity_team = _resolve_ambiguous_teams(CLIP, merged, ocr["identities"],
                                             number_teams, tdoc)

    doc = build_box_score(merged, ocr["identities"], number_teams, fps, onc,
                          identity_team)

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
