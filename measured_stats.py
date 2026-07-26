"""measured_stats.py -- bundle the CV pipeline's trustworthy outputs into
ONE web-ready JSON per clip (Phase 7 slice A1: the "measured stats"
contract the web app's new Measured tab reads).

Combines three existing artifacts, nothing recomputed from video:
  phase2/out/{clip}_box_score.json        floor-time + per-player zones
  spikes/out/{clip}_shot_locations.json   shooter court-feet at release
  spikes/out/{clip}_shot_attempts.json    shot_type + hoop per attempt

HONEST BOUNDARIES baked into the contract (so the UI can't overpromise):
  - make/miss is NOT included -- Gate 4 unpassed, so shooting % would be
    confident-wrong (meta.make_miss_available = false);
  - box-score seconds are PRESENCE over the tracked span, not full-game
    stats (meta.box_score_note carries the caveat);
  - only shots whose shooter was located get a court position + zone; the
    rest are COUNTED (shots_unlocated), never charted at a guessed spot.

The one derived number is the SHOT DISTRIBUTION by zone (% of attempts
behind vs inside the arc) -- the first actionable spatial stat, computed
from the located shots' court positions with the same 3pt geometry the
shot chart uses.

Usage:
    .venv/Scripts/python measured_stats.py HARD
"""

from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))

# Court geometry -- mirrors spikes/shot_location.py (kept local so this
# module and its tests stay dependency-light: no cv2/numpy import).
# COURT_LEN is only the DEFAULT. The real length varies by gym (TEST2's floor
# is 94 ft; HARD and TEST1 are configured 84) and it moves the far basket by
# 10 ft, so generate() passes the clip's OWN length in rather than assuming
# this one. Never read a court length from here -- ask the clip.
COURT_LEN, COURT_WID = 84.0, 50.0
HOOP_DX = 5.25                       # basket center distance from baseline
THREE_RADIUS_FT = 19.75              # HS 3pt radius
PAINT_RADIUS_FT = 8.0                # within this of the rim = "paint"/at-rim
                                     # (a distance proxy for the lane, honest
                                     # simplification for a v1 zone split)

_ZONES = ("three", "midrange", "paint")


def baskets(court_len=COURT_LEN):
    """The two rim positions in court feet, for a floor of this length."""
    return [(HOOP_DX, COURT_WID / 2.0), (court_len - HOOP_DX, COURT_WID / 2.0)]


def _dist(ax, ay, bx, by):
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def classify_zone(court_x, court_y, court_len=COURT_LEN):
    """(zone, dist_to_nearest_basket_ft) for a shooter's court-feet spot.
    Nearest basket = the basket being attacked (a shooter is far closer to
    the rim they shoot at than the other one), so a right-side shot is
    measured against the right rim, not read as a 70ft heave."""
    dist = min(_dist(court_x, court_y, bx, by) for (bx, by) in baskets(court_len))
    if dist > THREE_RADIUS_FT:
        zone = "three"
    elif dist <= PAINT_RADIUS_FT:
        zone = "paint"
    else:
        zone = "midrange"
    return zone, round(dist, 1)


def shot_distribution(zones):
    """Percentages by zone + the headline behind-vs-inside-the-arc split
    (the exact shape DJ's north-star goal cares about). Empty-safe."""
    n = len(zones)
    counts = {z: zones.count(z) for z in _ZONES}
    pct = {z: (round(100.0 * counts[z] / n, 1) if n else 0.0) for z in _ZONES}
    return {"n": n, "counts": counts, "pct": pct,
            "pct_three": pct["three"],
            "pct_two": round(pct["midrange"] + pct["paint"], 1) if n else 0.0}


_BOX_FIELDS = ("number", "team", "seconds_total", "seconds_live", "seconds_retro",
               "windows_present", "zone_seconds", "top_zone", "disputed_seconds")


def is_ambiguous(team):
    """stage8 marks a number it could not attribute to one team by prefixing
    the team label 'AMBIGUOUS (...)'. Surfaced as a flag here so the app can
    show it as an unresolved conflict instead of a player who barely played."""
    return str(team or "").startswith("AMBIGUOUS")


def tracking_coverage(box_doc):
    """How much of the tracked player time actually carries a jersey number.

    stage8 deliberately keeps confirmed-but-numberless identities in an
    'unnamed' bucket -- "real floor time that needs a coach click to become a
    name. Never dropped, never guessed." This contract used to forward only the
    named players, which silently threw that bucket away: on HARD it is 61.7 s
    across 15 identities, MORE tracked time than all the named players combined.
    The app then showed 11 players as if that were everyone, and the AI narrated
    the same partial picture as if it were complete -- exactly the
    confident-wrong failure the CV side refuses to commit.
    """
    unnamed = box_doc.get("unnamed_confirmed") or {}
    named_s = sum(float(p.get("seconds_total") or 0.0)
                  for p in box_doc.get("players", []))
    unnamed_s = float(unnamed.get("seconds_total") or 0.0)
    # Denominator = READABLE player time (on court, referees excluded), computed
    # by stage8 from the on-court cache. NOT named+unnamed: that drops every
    # candidate/unknown frame, which is precisely the player time we failed to
    # identify, and reads 53.6% on HARD where the honest figure is 36.7%. One
    # number, one definition, shared with phase2/identity_report.py.
    readable = float(box_doc.get("readable_seconds") or 0.0)
    return {
        "named_seconds": round(named_s, 1),
        "unnamed_seconds": round(unnamed_s, 1),
        "unnamed_identities": int(unnamed.get("identities") or 0),
        "readable_seconds": round(readable, 1),
        "pct_of_readable_named": round(100.0 * named_s / readable, 1) if readable else None,
        "review_backlog": box_doc.get("review") or {},
    }


def build_measured_stats(clip, box_doc, loc_doc, att_doc, court_len=COURT_LEN):
    """Assemble the web-ready contract from the three loaded docs."""
    # attempt lookup by frame span -> shot_type / hoop for a located shot
    att_by_span = {(a["start_frame"], a["end_frame"]): a
                   for a in att_doc.get("attempts", [])
                   if a.get("verdict") == "shot_attempt"}

    shots = []
    unlocated = 0
    for loc in loc_doc.get("locations", []):
        if loc.get("status") != "located":
            unlocated += 1
            continue
        cx, cy = loc["court_feet"]
        zone, dist = classify_zone(cx, cy, court_len)
        att = att_by_span.get((loc["start_frame"], loc["end_frame"]), {})
        shots.append({
            "start_frame": loc["start_frame"], "end_frame": loc["end_frame"],
            "court_x": cx, "court_y": cy, "zone": zone, "dist_ft": dist,
            "shot_type": att.get("shot_type"), "hoop": att.get("hoop"),
            "shooter_status": loc.get("shooter_status"),
        })

    box_score = [dict({k: p.get(k) for k in _BOX_FIELDS},
                      ambiguous=is_ambiguous(p.get("team")))
                 for p in box_doc.get("players", [])]

    return {
        "clip": clip,
        "meta": {
            "make_miss_available": False,
            "box_score_note": box_doc.get("note", ""),
            "court": {"length_ft": court_len, "width_ft": COURT_WID,
                      "hoop_dx_ft": HOOP_DX, "three_radius_ft": THREE_RADIUS_FT},
        },
        "tracking": tracking_coverage(box_doc),
        "box_score": box_score,
        "shots": shots,
        "shots_unlocated": unlocated,
        "shot_distribution": shot_distribution([s["zone"] for s in shots]),
    }


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def court_length_for(clip):
    """The clip's real floor length. Imported lazily so this module stays
    dependency-light for callers that only want classify_zone()."""
    sys.path.insert(0, os.path.join(_ROOT, "spikes"))
    import clips_config
    clips_config.ACTIVE = clip
    return float(clips_config.active()["court"]["length"])


def generate(clip):
    """Load the three artifacts for `clip`, build the web contract, write
    {clip}_measured_stats.json, and return it. Callable by analyze_clip.py
    (the app's CV entry point) as well as the CLI."""
    box = _load(os.path.join(_ROOT, "phase2", "out", f"{clip}_box_score.json"))
    loc = _load(os.path.join(_ROOT, "spikes", "out", f"{clip}_shot_locations.json"))
    att = _load(os.path.join(_ROOT, "spikes", "out", f"{clip}_shot_attempts.json"))

    out = build_measured_stats(clip, box, loc, att, court_length_for(clip))
    out_path = os.path.join(_ROOT, "spikes", "out", f"{clip}_measured_stats.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    d = out["shot_distribution"]
    print(f"[measured_stats] {clip}: {len(out['box_score'])} players, "
          f"{len(out['shots'])} shot(s) located ({out['shots_unlocated']} not located)")
    if d["n"]:
        print(f"  shot distribution: {d['pct_three']}% three / {d['pct_two']}% inside "
              f"(counts {d['counts']})")
    print(f"  wrote {out_path}")
    return out


def main():
    generate(sys.argv[1] if len(sys.argv) > 1 else "HARD")


if __name__ == "__main__":
    main()
