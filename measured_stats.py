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

# Pure logic (no cv2/numpy), so importing it keeps this module dependency-light
# the same way the geometry constants below do.
sys.path.insert(0, os.path.join(_ROOT, "phase2"))
import team_possessions  # noqa: E402

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


def build_touches(touch_doc, court_len):
    """BALL TOUCHES for the web contract -- who had the ball, for how long, and
    WHERE on the floor. A touch is ONE PLAYER holding the ball until she gives
    it up; it is NOT a possession (that is the team-level idea
    phase2/possessions.py owns). See spikes/ball_touch.py.

    Two honesty rules carried through verbatim, because the app is where a
    caveat gets lost:
      1. observed vs inferred seconds stay SEPARATE. Part of every touch is the
         ball seen in her hands and part is bridged across a detector dropout
         (DJ's "until proven otherwise" rule, 15s ceiling). A UI that shows only
         the total would present an assumption as a measurement.
      2. identity_status rides along. Most touches are review_item, not
         attributed -- the same confirmed-only discipline the box score uses.
    """
    out = []
    for t in touch_doc.get("touches", []):
        ft = t.get("court_feet_start")
        zone, dist = (classify_zone(ft[0], ft[1], court_len) if ft else (None, None))
        ident = t.get("identity") or {}
        out.append({
            "start_frame": t["start_frame"], "end_frame": t["end_frame"],
            "track_id": t["track_id"],
            "jersey_number": ident.get("jersey_number"),
            "identity_status": ident.get("status"),
            "observed_seconds": t.get("observed_seconds"),
            "inferred_seconds": t.get("inferred_seconds"),
            "total_seconds": t.get("total_seconds"),
            "court_x": ft[0] if ft else None, "court_y": ft[1] if ft else None,
            "zone": zone, "dist_ft": dist, "on_court": t.get("on_court"),
        })
    obs = sum(t["observed_seconds"] or 0.0 for t in out)
    inf = sum(t["inferred_seconds"] or 0.0 for t in out)
    named = [t for t in out if t["jersey_number"] is not None]
    return out, {
        "n_touches": len(out),
        "n_nameable": len(named),
        "observed_seconds": round(obs, 1),
        "inferred_seconds": round(inf, 1),
        "pct_inferred": round(100.0 * inf / (obs + inf), 0) if (obs + inf) else None,
        "zone_counts": {z: sum(1 for t in out if t["zone"] == z) for z in _ZONES},
    }


# How stale a touch may be and still speak to a shot. Basketball reason, not a
# fitted number: a player who last held the ball two seconds ago has had time to
# pass it, so the memory stops being evidence. Same value spikes/shooter_compare
# declared before its run -- kept identical so the app shows what that comparison
# scored, not a second, differently-tuned answer.
SHOOTER_MAX_BACK_FRAMES = 60


def attribute_shooter(touches, arc_start_frame):
    """WHO TOOK THIS SHOT -- the last player actually SEEN holding the ball
    before the arc began (spikes/ball_touch.shooter_from_touches, the method DJ
    proposed 2026-07-27).

    NOT VERIFIED, AND THE CONTRACT SAYS SO. The project's ground truth records
    WHICH arcs are real shots; it has never recorded WHO TOOK THEM
    (spikes/shooter_compare.py exists precisely to surface the disagreements for
    DJ to settle). So every answer here rides with shooter_verified=False, and
    the method ABSTAINS -- returns None -- rather than guessing whenever no
    touch was recorded in time. A shot with no attributable shooter stays
    unattributed; it is never assigned to the nearest body to fill the gap.
    """
    if not touches:
        return None
    best = None
    for t in touches:
        if t["start_frame"] > arc_start_frame:
            continue                       # began after the shot: cannot be it
        if arc_start_frame - t["end_frame"] > SHOOTER_MAX_BACK_FRAMES:
            continue                       # too stale to speak to this shot
        if best is None or t["end_frame"] > best["end_frame"]:
            best = t
    return best


def build_possessions(poss_doc):
    """TEAM POSSESSIONS for the web contract -- whose ball, from when to when.

    A POSSESSION is a team concept and is NOT a touch: several touches (pass,
    pass, drive, shoot) happen inside one possession. The two are kept as
    separate keys in the contract precisely so the app cannot conflate them.
    See phase2/team_possessions.py for the end rules.

    Passed straight through rather than reshaped -- the fields the film room
    needs (team, frame span, seconds, why it ended) are already the fields the
    layer produces.
    """
    if not poss_doc:
        return [], None
    poss = poss_doc.get("possessions", [])
    by_team = {}
    for p in poss:
        by_team[p["team"]] = by_team.get(p["team"], 0) + 1
    return poss, {
        "n_possessions": len(poss),
        "by_team": by_team,
        "seconds_by_team": {
            t: round(sum(p["seconds"] for p in poss if p["team"] == t), 1)
            for t in by_team},
        "teams": poss_doc.get("teams", []),
    }


def build_measured_stats(clip, box_doc, loc_doc, att_doc, court_len=COURT_LEN,
                         touch_doc=None, make_miss_results=None,
                         poss_doc=None):
    """Assemble the web-ready contract from the loaded docs."""
    # attempt lookup by frame span -> shot_type / hoop for a located shot
    att_by_span = {(a["start_frame"], a["end_frame"]): a
                   for a in att_doc.get("attempts", [])
                   if a.get("verdict") == "shot_attempt"}

    # make/miss lookup by frame span
    make_miss_by_span = {}
    if make_miss_results:
        for mm in make_miss_results:
            key = (mm["start_frame"], mm["end_frame"])
            make_miss_by_span[key] = mm

    touches, touch_summary = (build_touches(touch_doc, court_len)
                              if touch_doc else ([], None))

    shots = []
    unlocated = 0
    for loc in loc_doc.get("locations", []):
        if loc.get("status") != "located":
            unlocated += 1
            continue
        cx, cy = loc["court_feet"]
        zone, dist = classify_zone(cx, cy, court_len)
        att = att_by_span.get((loc["start_frame"], loc["end_frame"]), {})
        who = attribute_shooter(touches, loc["start_frame"])
        shot = {
            "start_frame": loc["start_frame"], "end_frame": loc["end_frame"],
            "court_x": cx, "court_y": cy, "zone": zone, "dist_ft": dist,
            "shot_type": att.get("shot_type"), "hoop": att.get("hoop"),
            "shooter_status": loc.get("shooter_status"),
            # --- who took it. INFERRED, never verified. See attribute_shooter.
            "shooter_number": who["jersey_number"] if who else None,
            "shooter_track_id": who["track_id"] if who else None,
            "shooter_method": "last_seen_holding_ball" if who else None,
            "shooter_verified": False,
        }

        # Add make/miss if available
        mm = make_miss_by_span.get((loc["start_frame"], loc["end_frame"]))
        if mm:
            shot["make_miss_outcome"] = mm["outcome"]
            shot["make_miss_score_from"] = mm["score_from"]
            shot["make_miss_score_to"] = mm["score_to"]
            shot["make_miss_score_change_frame"] = mm["score_change_frame"]
            shot["make_miss_score_change_time_sec"] = mm["score_change_time_sec"]

        shots.append(shot)

    box_score = [dict({k: p.get(k) for k in _BOX_FIELDS},
                      ambiguous=is_ambiguous(p.get("team")))
                 for p in box_doc.get("players", [])]

    # Stamp each shot with the possession it happened in, so the film room can
    # jump from a shot to the whole sequence that produced it. A shot that
    # lands outside every possession keeps possession_index None rather than
    # being snapped to the nearest one.
    possessions, poss_summary = build_possessions(poss_doc)
    if possessions:
        team_possessions.tag_shots(shots, possessions)
    else:
        for s in shots:
            s["possession_index"] = None

    n_attributed = sum(1 for s in shots if s["shooter_number"] is not None)

    return {
        "clip": clip,
        "meta": {
            "make_miss_available": bool(make_miss_results),
            "box_score_note": box_doc.get("note", ""),
            # Touches are CANDIDATES for review, never confirmed stats -- most
            # carry identity_status "review_item". And every touch is part seen,
            # part bridged; a UI that shows only total_seconds would present an
            # assumption as a measurement. Both facts ride in the contract so
            # the app cannot overpromise, the same way make_miss_available does.
            "touches_available": bool(touch_doc),
            # Possessions are a SEPARATE layer from touches and may be absent
            # on their own: a clip can have perfectly good touches and still be
            # unable to say whose ball it is, because the two jersey colours
            # could not be told apart on that footage. False means "we did not
            # answer", never "there were no possessions".
            "possessions_available": bool(possessions),
            "possession_note": (
                "A possession is a TEAM having the ball until they lose it -- "
                "several touches happen inside one. It ends when the other team "
                "gets the ball or the ball goes out of bounds (which restarts it "
                "even for the same team -- a film-room cut, not the stat-sheet "
                "rule). Teams come from the jersey colours entered at setup."
                if possessions else
                "No possessions for this clip: either the ball layer has not run, "
                "or the two jersey colours could not be told apart on this "
                "footage. The system abstained rather than guess which team had "
                "the ball."),
            # --- PER-PLAYER SHOTS. The player tab's heat map reads these.
            # Available does NOT mean verified: no ground truth for WHO took a
            # shot has ever existed in this project (only which arcs are real
            # shots). The UI must label this, exactly as it labels
            # make_miss_available and the seen-vs-inferred touch seconds.
            "shooter_attribution_available": bool(touch_doc) and n_attributed > 0,
            "shooter_attribution_verified": False,
            "shots_attributed": n_attributed,
            "shooter_note": (
                "Who took each shot is INFERRED from who was last SEEN holding "
                "the ball (within 2s). It has never been checked against ground "
                "truth -- this project has confirmed WHICH arcs are shots, never "
                "WHO took them. Shots with no recorded touch in time are left "
                "unattributed rather than guessed. Label as unverified."),
            "touch_note": ("Touches are review candidates, not confirmed stats. "
                           "Each one is part SEEN (ball visible in her hands) and "
                           "part FILLED IN (bridged across a dropout, 15s ceiling) "
                           "-- show both, never just the total."
                           if touch_doc else
                           "No ball-touch data for this clip (the ball layer has "
                           "not been run on it)."),
            "court": {"length_ft": court_len, "width_ft": COURT_WID,
                      "hoop_dx_ft": HOOP_DX, "three_radius_ft": THREE_RADIUS_FT},
        },
        "tracking": tracking_coverage(box_doc),
        "box_score": box_score,
        "shots": shots,
        "shots_unlocated": unlocated,
        "shot_distribution": shot_distribution([s["zone"] for s in shots]),
        "touches": touches,
        "touch_summary": touch_summary,
        "possessions": possessions,
        "possession_summary": poss_summary,
    }


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_optional(path, default):
    """A layer that was skipped is not a failure -- see generate()."""
    if not os.path.exists(path):
        return default
    return _load(path)


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
    # OPTIONAL, for the same reason touches are: a clip whose ball layer has
    # never run (ball_span_len = 0, which is every game set up in the browser
    # today -- nobody has marked its rims) writes no shot artifacts at all.
    # Crashing there would throw away a perfectly good box score over a layer
    # that was deliberately skipped. Absent shots are reported as zero shots,
    # never as a failure and never as an empty chart pretending to be complete.
    loc = _load_optional(os.path.join(_ROOT, "spikes", "out", f"{clip}_shot_locations.json"),
                         {"locations": []})
    att = _load_optional(os.path.join(_ROOT, "spikes", "out", f"{clip}_shot_attempts.json"),
                         {"attempts": []})
    # OPTIONAL: a clip whose ball layer has never run has no touches. The
    # contract then says so (meta.touches_available = false) rather than
    # shipping an empty list the UI might read as "she never had the ball".
    tp = os.path.join(_ROOT, "spikes", "out", f"{clip}_ball_touches.json")
    touch = _load(tp) if os.path.exists(tp) else None
    # OPTIONAL and separately so: the possession layer abstains on its own
    # whenever the two jersey colours cannot be told apart on this footage, so
    # a clip can have touches but no possessions. Absent means "not answered".
    pp = os.path.join(_ROOT, "spikes", "out", f"{clip}_team_possessions.json")
    poss = _load(pp) if os.path.exists(pp) else None

    # Try to load make/miss results from scoreboard analysis
    make_miss_results = None
    try:
        # Try Gemma (vision-based) first - works on all scoreboard styles
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            from spikes import gemma_make_miss_fast
            shots_path = os.path.join(_ROOT, "spikes", "out", f"{clip}_shot_attempts.json")
            try:
                make_miss_results = gemma_make_miss_fast.detect_makes_by_gemma_fast(
                    clip, shots_path, api_key)
                print(f"[measured_stats] using Gemma vision reader", flush=True)
            except Exception as e:
                print(f"[measured_stats] Gemma failed, falling back to OCR: {e}", flush=True)

        # Fallback to OCR if Gemma not available or fails
        if not make_miss_results:
            from spikes import scoreboard_make_miss
            shots_path = os.path.join(_ROOT, "spikes", "out", f"{clip}_shot_attempts.json")
            sb_path = os.path.join(_ROOT, "spikes", "out", f"{clip}_scoreboard_ocr.json")
            if os.path.exists(shots_path) and os.path.exists(sb_path):
                make_miss_results = scoreboard_make_miss.detect_makes_by_scoreboard(
                    clip, shots_path, sb_path)
                print(f"[measured_stats] using OCR reader", flush=True)
    except Exception as e:
        print(f"[measured_stats] warning: could not load make/miss data: {e}")

    out = build_measured_stats(clip, box, loc, att, court_length_for(clip), touch,
                              make_miss_results, poss)
    # Says WHY there are no shots: a skipped layer reads very differently from
    # a game where the ball was watched and nobody shot.
    out["meta"]["shot_layer_available"] = os.path.exists(
        os.path.join(_ROOT, "spikes", "out", f"{clip}_shot_locations.json"))
    out_path = os.path.join(_ROOT, "spikes", "out", f"{clip}_measured_stats.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    d = out["shot_distribution"]
    print(f"[measured_stats] {clip}: {len(out['box_score'])} players, "
          f"{len(out['shots'])} shot(s) located ({out['shots_unlocated']} not located)")
    if d["n"]:
        print(f"  shot distribution: {d['pct_three']}% three / {d['pct_two']}% inside "
              f"(counts {d['counts']})")
    ts = out["touch_summary"]
    if ts:
        print(f"  ball touches: {ts['n_touches']} ({ts['n_nameable']} nameable)  "
              f"{ts['observed_seconds']}s SEEN + {ts['inferred_seconds']}s FILLED IN "
              f"({ts['pct_inferred']:.0f}% inferred)  zones {ts['zone_counts']}")
    else:
        print(f"  ball touches: none for this clip (ball layer not run)")
    ps = out["possession_summary"]
    if ps:
        print(f"  possessions: {ps['n_possessions']}  "
              f"by team {ps['by_team']}  seconds {ps['seconds_by_team']}")
    else:
        print(f"  possessions: none for this clip (see meta.possession_note)")
    print(f"  wrote {out_path}")
    return out


def main():
    generate(sys.argv[1] if len(sys.argv) > 1 else "HARD")


if __name__ == "__main__":
    main()
