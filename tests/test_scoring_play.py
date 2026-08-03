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

import numpy as np  # noqa: E402

from gemma_make_miss_fast import (  # noqa: E402
    is_scoring_play, points_allowed, scoreboard_crop, scoreboard_region,
)


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


# ------------------------------------ FIX 3: the shot's own value as evidence --

def test_a_layup_cannot_be_worth_one():
    """THE TEST1 FAILURE. An occluded '2' reads reproducibly as '1', so the
    re-read agrees with itself and both values are legal scores. What breaks the
    tie is the shot chart: that shot was taken at (6.7, 27.5) ft -- at the rim.
    A layup is worth 2, never 1."""
    assert is_scoring_play(0, 0, 1, 0, zone="paint") is False


def test_a_layup_of_two_is_fine():
    assert is_scoring_play(0, 0, 2, 0, zone="paint") is True


def test_a_layup_cannot_be_worth_three():
    assert is_scoring_play(0, 0, 3, 0, zone="paint") is False


def test_a_three_must_be_worth_three():
    assert is_scoring_play(0, 0, 3, 0, zone="three") is True
    assert is_scoring_play(0, 0, 2, 0, zone="three") is False


def test_midrange_allows_a_free_throw_or_a_jumper():
    """A free throw sits at roughly the same distance as a midrange jumper, so
    both 1 and 2 stay legal there -- refusing one would delete real makes."""
    assert is_scoring_play(0, 0, 1, 0, zone="midrange") is True
    assert is_scoring_play(0, 0, 2, 0, zone="midrange") is True
    assert is_scoring_play(0, 0, 3, 0, zone="midrange") is False


def test_no_zone_means_no_opinion_not_rejection():
    """An unlocated shot is MISSING evidence, not evidence of a problem. It must
    behave exactly as it did before this check existed."""
    assert is_scoring_play(0, 0, 1, 0, zone=None) is True
    assert is_scoring_play(0, 0, 2, 0) is True


def test_an_unknown_zone_name_is_also_no_opinion():
    assert points_allowed("somewhere_new") is None
    assert is_scoring_play(0, 0, 1, 0, zone="somewhere_new") is True


def test_the_zone_check_never_rescues_an_impossible_move():
    """Zone is an EXTRA filter, never a bypass -- a decrease stays refused."""
    assert is_scoring_play(1, 0, 0, 0, zone="paint") is False
    assert is_scoring_play(10, 8, 12, 10, zone="paint") is False


# ------------------------------------------- FIX 1: crop the marked scorebug --

def test_the_marked_scorebug_region_is_used():
    """Every clip already records this box for SIFT masking -- reusing it costs
    no new human input."""
    cfg = {"exclude_regions": [(0.0, 810.0, 415.0, 1080.0)]}
    x1, y1, x2, y2 = scoreboard_region(cfg, 1920, 1080)
    assert 780 <= y1 <= 815 and y2 == 1080
    assert x1 == 0 and 410 <= x2 <= 430


def test_a_clip_with_no_marked_region_still_gets_a_crop():
    """Falls back rather than raising -- an unmarked clip loses accuracy, not
    its answer."""
    x1, y1, x2, y2 = scoreboard_region({}, 1920, 1080)
    assert (x2 - x1) > 0 and (y2 - y1) > 0


def test_the_biggest_marked_region_wins():
    cfg = {"exclude_regions": [(0, 0, 10, 10), (0, 800, 500, 1080)]}
    x1, y1, x2, y2 = scoreboard_region(cfg, 1920, 1080)
    assert y1 > 700, "the big scorebug box, not the small one"


def test_the_region_is_clamped_to_the_frame():
    """The pad must never push the crop off the edge of the image."""
    cfg = {"exclude_regions": [(0.0, 810.0, 1920.0, 1080.0)]}
    x1, y1, x2, y2 = scoreboard_region(cfg, 1920, 1080)
    assert x1 >= 0 and y1 >= 0 and x2 <= 1920 and y2 <= 1080


def test_test2_board_is_no_longer_cut_off():
    """REGRESSION: the old hardcoded crop took the left 22% of the frame = 422px,
    but TEST2's scorebug is 580px wide, so the away score was being cropped
    away entirely."""
    cfg = {"exclude_regions": [(0.0, 810.0, 580.0, 1080.0)]}
    _x1, _y1, x2, _y2 = scoreboard_region(cfg, 1920, 1080)
    assert x2 >= 580, "must cover the whole marked board"


def test_the_crop_is_enlarged_for_the_model():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cfg = {"exclude_regions": [(0.0, 810.0, 415.0, 1080.0)]}
    crop = scoreboard_crop(frame, cfg, upscale=3)
    assert crop is not None
    assert crop.shape[0] > 270 * 2, "upscaled, so small digits are readable"


def test_a_degenerate_region_yields_no_crop_rather_than_a_crash():
    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert scoreboard_crop(frame, {"exclude_regions": [(5, 5, 5, 5)]}) is None


# ----------------------------- the clip does not start at 0-0 (baseline bug) --

def test_a_midgame_opening_score_is_not_a_scoring_play():
    """FOUND 2026-08-03. The reader seeded the running score at (0, 0), which is
    only true of a clip opening at tip-off. HARD really starts 15-12 and TEST2
    starts 2-2, so the FIRST reading looked like an enormous score change and
    shipped as "MAKE [0,0]->[15,12]".

    The guard below is what surfaced it -- a 15-point jump is not a basket. The
    real fix is upstream (the baseline is now read from before the first shot,
    and a first reading is never a make), but this pins the arithmetic: nothing
    about a mid-game opening score may ever look like a made basket."""
    assert is_scoring_play(0, 0, 15, 12) is False
    assert is_scoring_play(0, 0, 2, 2) is False


def test_a_real_basket_from_a_midgame_baseline_still_counts():
    """The fix must not make mid-game clips unjudgeable -- once the baseline is
    right, an ordinary basket on top of it reads normally."""
    assert is_scoring_play(15, 12, 17, 12, zone="paint") is True
    assert is_scoring_play(15, 12, 15, 15, zone="three") is True
