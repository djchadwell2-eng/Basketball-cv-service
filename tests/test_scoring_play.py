"""is_scoring_play -- could basketball have produced this score change?

THE BUG BEING PINNED (TEST1, 2026-08-02). The Gemma make/miss reader treated
ANY difference between two scoreboard readings as a made basket. A real run
produced "MAKE [0,0]->[1,0]" (a one-point field goal) and "MAKE [1,0]->[0,0]"
-- a score going DOWN, confirmed as a basket. Scores do not decrease; that was
the vision model misreading a small low-contrast crop.

The make/miss layer's whole promise is that the scoreboard CONFIRMS makes
(DJ's rule -- it can never prove a miss). A confirmation built on a misread is
worse than abstaining.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "spikes"))

from gemma_make_miss_fast import is_scoring_play  # noqa: E402


# ------------------------------------------------------------- real baskets --

def test_a_two_point_basket_counts():
    assert is_scoring_play(10, 8, 12, 8) is True


def test_a_three_pointer_counts():
    assert is_scoring_play(10, 8, 13, 8) is True


def test_a_free_throw_counts():
    assert is_scoring_play(10, 8, 11, 8) is True


def test_the_away_team_scoring_counts_too():
    assert is_scoring_play(10, 8, 10, 10) is True


def test_scoring_from_nil():
    assert is_scoring_play(0, 0, 2, 0) is True


# ------------------------------------------------- what basketball cannot do --

def test_a_score_going_down_is_never_a_basket():
    """THE HEADLINE BUG: 'MAKE [1,0]->[0,0]' was really produced on TEST1."""
    assert is_scoring_play(1, 0, 0, 0) is False


def test_the_away_score_going_down_is_never_a_basket():
    assert is_scoring_play(0, 5, 0, 3) is False


def test_both_teams_changing_at_once_is_a_misread():
    """Two teams cannot score on the same possession."""
    assert is_scoring_play(10, 8, 12, 10) is False


def test_a_jump_of_four_or_more_is_a_misread():
    """No single basket is worth 4. A jump like this is a digit misread."""
    assert is_scoring_play(10, 8, 14, 8) is False
    assert is_scoring_play(0, 0, 8, 0) is False


def test_no_change_is_not_a_scoring_play():
    """Handled by the caller as 'still the same score', never as a make."""
    assert is_scoring_play(10, 8, 10, 8) is False


def test_a_wild_misread_is_refused():
    assert is_scoring_play(2, 0, 20, 0) is False
    assert is_scoring_play(12, 10, 1, 2) is False


# --------------------------------------------------------- the honest limit --

def test_a_misread_that_looks_legal_still_passes():
    """ON THE RECORD: this check removes the IMPOSSIBLE moves, it does not make
    the reader correct. A wrong reading that happens to look like a legal +2
    still gets through. It is a floor, not a guarantee -- do not read a passing
    result here as proof the board was read right."""
    assert is_scoring_play(10, 8, 12, 8) is True   # indistinguishable from real
