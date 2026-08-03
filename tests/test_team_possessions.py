"""TEAM POSSESSIONS -- the rules DJ set on 2026-08-02, pinned down.

A possession is a run of touches by the same team. It ends when the OTHER team
gets the ball, or when the ball goes out of bounds (which restarts it no matter
who gets it back -- DJ's film-room rule, not the stat-sheet rule). A touch with
no team is SKIPPED, never treated as a change of possession: a dropout in the
colour signal is not a turnover.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "phase2"))

import team_possessions as tp  # noqa: E402

FPS = 30.0
HOME, AWAY = "Home", "Away"


def touch(track_id, start, end, team, on_court=True):
    return {"track_id": track_id, "start_frame": start, "end_frame": end,
            "team": team, "on_court": on_court}


# --------------------------------------------------------- the basic rule ---

def test_consecutive_same_team_touches_are_one_possession():
    """Pass, pass, shoot = three touches, ONE possession."""
    poss = tp.build([touch(1, 0, 30, HOME),
                     touch(2, 35, 60, HOME),
                     touch(3, 65, 90, HOME)], FPS)
    assert len(poss) == 1
    assert poss[0]["team"] == HOME
    assert poss[0]["n_touches"] == 3
    assert poss[0]["start_frame"] == 0 and poss[0]["end_frame"] == 90
    assert poss[0]["track_ids"] == [1, 2, 3]


def test_other_team_touch_ends_the_possession():
    poss = tp.build([touch(1, 0, 30, HOME),
                     touch(2, 40, 70, AWAY)], FPS)
    assert [p["team"] for p in poss] == [HOME, AWAY]
    assert poss[0]["ended_by"] == "other_team"
    assert poss[1]["ended_by"] == "end_of_clip"


def test_offensive_rebound_stays_the_same_possession():
    """DJ, 2026-08-02: a team rebounding its OWN miss is still one possession.
    Falls out for free -- the jersey colour never changed."""
    poss = tp.build([touch(1, 0, 30, HOME),      # shot
                     touch(2, 50, 80, HOME)], FPS)  # own rebound, put-back
    assert len(poss) == 1
    assert poss[0]["n_touches"] == 2


# ------------------------------------------------------------ out of bounds --

def test_out_of_bounds_restarts_even_for_the_same_team():
    """DJ's film-room rule: out of bounds restarts the possession no matter who
    gets it back. NOT the stat-sheet definition -- deliberate."""
    poss = tp.build([touch(1, 0, 30, HOME),
                     touch(2, 40, 55, HOME, on_court=False),   # inbounding
                     touch(3, 60, 90, HOME)], FPS)
    assert len(poss) == 2, "the same team inbounding still starts a new possession"
    assert poss[0]["ended_by"] == "out_of_bounds"
    assert all(p["team"] == HOME for p in poss)


def test_out_of_bounds_then_other_team_inbounds():
    poss = tp.build([touch(1, 0, 30, HOME),
                     touch(2, 40, 55, AWAY, on_court=False),
                     touch(3, 60, 90, AWAY)], FPS)
    assert [p["team"] for p in poss] == [HOME, AWAY]
    assert poss[0]["ended_by"] == "out_of_bounds"
    assert poss[1]["n_touches"] == 2, "the inbounds pass belongs to the new poss"


def test_teamless_inbounder_does_not_open_a_possession():
    """We saw a stoppage but cannot say whose ball it is -- end the old one,
    start nothing on a guess."""
    poss = tp.build([touch(1, 0, 30, HOME),
                     touch(2, 40, 55, None, on_court=False)], FPS)
    assert len(poss) == 1
    assert poss[0]["ended_by"] == "out_of_bounds"


# ------------------------------------------------------------- abstention ----

def test_teamless_touch_never_ends_a_possession():
    """THE FLICKER GUARD. A touch whose colour could not be read is skipped --
    a dropout is not a turnover."""
    poss = tp.build([touch(1, 0, 30, HOME),
                     touch(2, 40, 60, None),     # colour unreadable
                     touch(3, 70, 90, HOME)], FPS)
    assert len(poss) == 1, "an unreadable touch must not split a possession"
    assert poss[0]["teamless_touches"] == 1
    assert poss[0]["n_touches"] == 2, "the skipped touch is not counted as hers"


def test_teamless_touches_only_are_no_possessions():
    poss = tp.build([touch(1, 0, 30, None), touch(2, 40, 60, None)], FPS)
    assert poss == []


def test_no_touches_is_no_possessions():
    assert tp.build([], FPS) == []


# ------------------------------------------------------------- bookkeeping ---

def test_touches_are_sorted_before_chaining():
    poss = tp.build([touch(3, 65, 90, HOME),
                     touch(1, 0, 30, HOME),
                     touch(2, 35, 60, HOME)], FPS)
    assert len(poss) == 1
    assert poss[0]["start_frame"] == 0 and poss[0]["end_frame"] == 90


def test_times_and_indices_are_filled_in():
    poss = tp.build([touch(1, 0, 29, HOME), touch(2, 60, 89, AWAY)], FPS)
    assert [p["possession_index"] for p in poss] == [0, 1]
    assert poss[0]["start_time_s"] == 0.0
    assert poss[0]["end_time_s"] == round(29 / FPS, 2)
    assert poss[0]["seconds"] == 1.0


# ------------------------------------------------------------ shot tagging ---

def test_shots_are_tagged_with_their_possession():
    poss = tp.build([touch(1, 0, 100, HOME), touch(2, 200, 300, AWAY)], FPS)
    shots = tp.tag_shots([{"start_frame": 50}, {"start_frame": 250}], poss)
    assert [s["possession_index"] for s in shots] == [0, 1]


def test_a_shot_outside_every_possession_is_not_forced_into_one():
    poss = tp.build([touch(1, 0, 100, HOME)], FPS)
    shots = tp.tag_shots([{"start_frame": 5000}], poss)
    assert shots[0]["possession_index"] is None


def test_a_shot_just_after_a_possession_belongs_to_it():
    """REGRESSION, HARD 2026-08-02. A possession is built from touches, and a
    touch ends the moment the ball leaves her hands -- so the shot arc starts a
    few frames AFTER the possession that produced it. HARD's possession 3 ran
    to frame 1179 and its shot began at 1187, and the shot was being reported
    as belonging to no possession at all. A ball in the air is still the
    shooting team's possession."""
    poss = tp.build([touch(1, 1017, 1179, HOME)], FPS)
    shots = tp.tag_shots([{"start_frame": 1187}], poss)
    assert shots[0]["possession_index"] == 0


def test_a_shot_long_after_a_possession_is_still_not_attributed():
    """The look-back is bounded -- a shot much later is a different sequence."""
    poss = tp.build([touch(1, 0, 100, HOME)], FPS)
    shots = tp.tag_shots([{"start_frame": 100 + tp.SHOT_MAX_BACK_FRAMES + 1}], poss)
    assert shots[0]["possession_index"] is None


def test_the_look_back_never_reaches_a_possession_that_had_not_started():
    """A shot cannot come out of a possession that begins after it."""
    poss = tp.build([touch(1, 500, 600, HOME)], FPS)
    shots = tp.tag_shots([{"start_frame": 480}], poss)
    assert shots[0]["possession_index"] is None


def test_the_look_back_picks_the_most_recent_possession():
    poss = tp.build([touch(1, 0, 100, HOME), touch(2, 110, 200, AWAY)], FPS)
    shots = tp.tag_shots([{"start_frame": 210}], poss)
    assert shots[0]["possession_index"] == 1, "the one just ended, not the first"


def test_a_shot_with_no_frame_is_not_guessed():
    poss = tp.build([touch(1, 0, 100, HOME)], FPS)
    assert tp.tag_shots([{}], poss)[0]["possession_index"] is None


def test_summary_lines_do_not_crash_on_empty():
    assert tp.summary_lines([], "CLIP")
