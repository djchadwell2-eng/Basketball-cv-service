"""TEAM POSSESSIONS -- one team has the ball until they lose it.

THIS MODULE OWNS THE WORD "POSSESSION" in this codebase. Nothing else may claim
it. phase2/window_boundaries.py used to (as possessions.py) by asking which half
of the floor the bodies stood on; that was deleted as a possession idea on
2026-08-02 -- DJ: "that's not how possessions work from a half court" -- and
demoted to what it always really was, cut points for the identity layer.

WHAT A POSSESSION IS HERE. A run of consecutive TOUCHES (spikes/ball_touch.py:
one player holding the ball) belonging to the SAME team. Pass, pass, drive,
shoot is four touches and ONE possession. It ends when the other team gets the
ball, or when the ball goes out of bounds.

THE TWO END RULES, both DJ's, both from basketball rather than from a fit:

  1. THE OTHER TEAM TOUCHES IT. Steal, defensive rebound, turnover -- the jersey
     colour on the ball changes, so the possession changes. This is the whole
     rule; there is no timer. A team keeping the ball off its OWN missed shot
     (offensive rebound) is the SAME possession, which falls out for free
     because the colour never changed (DJ confirmed, 2026-08-02).

  2. THE BALL GOES OUT OF BOUNDS -> RESTART, no matter who gets it back. DJ's
     call, 2026-08-02: "just restart the possession, I feel like." Note this is
     NOT the official stat-sheet definition -- by that definition a ball knocked
     out by the defence leaves the offence's possession running. This is
     deliberately a FILM ROOM definition: after a stoppage, play restarts from a
     set position, and a coach reviewing it wants that as its own clip. Recorded
     here so nobody later "fixes" it back to the stat-sheet rule by accident.

HOW OUT OF BOUNDS IS DETECTED -- and the way that was REJECTED. It is read off a
signal ball_touch.py already computes for free: a touch whose holder is standing
OFF THE COURT is an inbounds pass, and its own comment says so ("a real inbounds
pass is thrown from behind the baseline"). If someone is inbounding, the ball was
out. What is deliberately NOT used is the ball's own court position: the court
maths assumes whatever it maps is ON THE FLOOR, so a ball ten feet in the air
projects well past where it really is, and a high pass near the sideline would
read as out of bounds when it never left. That would invent turnovers that never
happened, which is worse than missing real ones.

ABSTENTION. A touch with no team (touch_teams.py could not tell) does NOT end a
possession -- it is skipped. This is the flicker guard DJ asked for: a dropout
in the colour signal is not a turnover, and the price of guessing is a fake
change of possession in the middle of a real one. Same shape as ball_touch's own
"the girl last seen with the ball has the ball until proven otherwise".

Pure functions over already-loaded documents. Nothing here writes into
team_events or any Phase 1/2 artifact (ROADMAP Principle 4).
"""

from __future__ import annotations


def build(touches, fps=30.0):
    """Touches (each with "team", in frame order) -> possessions.

    Returns [{"possession_index", "team", "start_frame", "end_frame",
              "start_time_s", "end_time_s", "seconds",
              "n_touches", "track_ids", "ended_by", "teamless_touches"}].

    ended_by is always recorded, because "why did this possession stop" is the
    first thing a human checks when a boundary looks wrong:
        "other_team"    the other team got the ball
        "out_of_bounds" an inbounds pass was seen (rule 2 above)
        "end_of_clip"   the footage ran out -- NOT a real basketball ending
    """
    ordered = sorted(touches, key=lambda t: (t["start_frame"], t["end_frame"]))

    poss, cur = [], None
    for t in ordered:
        team = t.get("team")

        # OUT OF BOUNDS. An off-court holder is someone inbounding, so the ball
        # was out: whatever was running has ended, and the restart is its own
        # possession -- even if the same team keeps the ball.
        if t.get("on_court") is False:
            if cur is not None:
                cur["ended_by"] = "out_of_bounds"
                poss.append(cur)
                cur = None
            # The inbounds pass itself opens the new possession when we know
            # whose it is; if we don't, it is skipped like any teamless touch.
            if team is not None:
                cur = _open(team, t)
            continue

        if team is None:
            # ABSTAINED: not evidence of anything. It neither ends the current
            # possession nor starts one. Counted so the JSON can show how much
            # of the clip the colour signal was silent for.
            if cur is not None:
                cur["teamless_touches"] += 1
            continue

        if cur is None:
            cur = _open(team, t)
        elif team == cur["team"]:
            _extend(cur, t)                  # same team: still their ball
        else:
            cur["ended_by"] = "other_team"   # the ball changed hands
            poss.append(cur)
            cur = _open(team, t)

    if cur is not None:
        cur["ended_by"] = "end_of_clip"
        poss.append(cur)

    for i, p in enumerate(poss):
        p["possession_index"] = i
        p["start_time_s"] = round(p["start_frame"] / fps, 2)
        p["end_time_s"] = round(p["end_frame"] / fps, 2)
        p["seconds"] = round((p["end_frame"] - p["start_frame"] + 1) / fps, 2)
    return poss


def _open(team, touch):
    return {"team": team,
            "start_frame": touch["start_frame"],
            "end_frame": touch["end_frame"],
            "n_touches": 1,
            "track_ids": [touch["track_id"]],
            "teamless_touches": 0,
            "ended_by": None}


def _extend(poss, touch):
    poss["end_frame"] = max(poss["end_frame"], touch["end_frame"])
    poss["n_touches"] += 1
    if touch["track_id"] not in poss["track_ids"]:
        poss["track_ids"].append(touch["track_id"])


# How long after a possession ends a shot may still belong to it. This is the
# SAME 2s ceiling measured_stats.SHOOTER_MAX_BACK_FRAMES already uses to decide
# how stale a touch may be and still speak to a shot -- deliberately reused
# rather than a second, separately-tuned number for the same physical fact.
SHOT_MAX_BACK_FRAMES = 60


def possession_of_frame(possessions, frame, max_back_frames=SHOT_MAX_BACK_FRAMES):
    """Which possession does this frame belong to -> index, or None.

    A frame INSIDE a possession is easy. The interesting case is the one just
    after one ends, and it is why max_back_frames exists:

    A POSSESSION ENDS WHEN THE BALL LEAVES HER HANDS, BUT THE SHOT HAPPENS
    AFTER THAT. A possession is built from touches, and a touch stops the
    moment she releases -- so a shot's arc starts a few frames PAST the end of
    the possession it came out of. Measured on HARD (2026-08-02): possession 3
    ran to frame 1179 and its own shot started at 1187, eight frames later, and
    the shot was reported as belonging to no possession at all.

    That is wrong on the basketball too: a ball in the air is still the
    shooting team's possession until somebody rebounds it. So a frame shortly
    after a possession ends is attributed back to it, bounded by
    max_back_frames so a shot two possessions later can never be dragged back.

    None is still a real answer for anything outside that: a shot in a stretch
    where nobody was confidently seen with the ball genuinely cannot be
    attributed, and is left unattributed rather than snapped to a neighbour.
    """
    for p in possessions:
        if p["start_frame"] <= frame <= p["end_frame"]:
            return p["possession_index"]
    # Not inside one -- fall back to the most recent possession that ended
    # within the ceiling, i.e. the one the ball was most likely still in flight
    # from. Never a possession that starts LATER: a shot cannot belong to one
    # that had not begun.
    best = None
    for p in possessions:
        gap = frame - p["end_frame"]
        if 0 < gap <= max_back_frames:
            if best is None or p["end_frame"] > best["end_frame"]:
                best = p
    return best["possession_index"] if best else None


def tag_shots(shots, possessions, max_back_frames=SHOT_MAX_BACK_FRAMES):
    """Stamp each shot with the possession it came out of, so the film room can
    jump from a shot to the sequence that produced it. Uses the shot's own start
    frame (the release). A shot too far from every possession keeps None rather
    than being snapped to the nearest one."""
    for s in shots:
        frame = s.get("start_frame")
        s["possession_index"] = (
            possession_of_frame(possessions, frame, max_back_frames)
            if frame is not None else None)
    return shots


def summary_lines(possessions, clip_name):
    """The raw table, printed before any interpretation."""
    if not possessions:
        return [f"TEAM POSSESSIONS ({clip_name}): none detected"]
    by_team = {}
    for p in possessions:
        by_team.setdefault(p["team"], []).append(p)
    ends = {}
    for p in possessions:
        ends[p["ended_by"]] = ends.get(p["ended_by"], 0) + 1

    out = [f"TEAM POSSESSIONS ({clip_name}) -- MEASURED, pending review",
           f"  {len(possessions)} possessions"]
    for team, ps in sorted(by_team.items()):
        secs = sum(p["seconds"] for p in ps)
        out.append(f"    {team:<32} {len(ps):>3} poss  {secs:>6.1f}s")
    out.append(f"  ended by: {ends}")
    skipped = sum(p["teamless_touches"] for p in possessions)
    if skipped:
        out.append(f"  {skipped} touch(es) had no team and were SKIPPED "
                   f"(they did not end a possession -- a dropout is not a "
                   f"turnover)")
    out.append("  every possession below is a CANDIDATE for review:")
    for p in possessions:
        out.append(f"    poss {p['possession_index']:>3}: {p['team']:<28} "
                   f"f{p['start_frame']:>6}..{p['end_frame']:<6} "
                   f"t={p['start_time_s']:>6.1f}s..{p['end_time_s']:<6.1f}s "
                   f"({p['seconds']:>5.1f}s, {p['n_touches']} touch) "
                   f"ended: {p['ended_by']}")
    return out
