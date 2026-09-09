"""FACING_CROP_PICK -- picking the most face-on crop in each time-slice
instead of always the biggest one.

WHY THIS EXISTS [MEASURED 2026-09-08, spikes/pose_facing_test.py]. The
reader's read rate is 3.4% per crop; the recorded cause is ANGLE, not size.
23 confident reads vs 189 other picked-but-unread crops of the SAME players
separated at 0.684 by apparent shoulder width (0.5 = no signal), consistent
in direction across all 4 clips checked, not explained by box size
(correlation 0.068). NOT a complete fix -- one player's plainly-legible
number went unread across 4 squared-on frames in a row.

These tests pin two things: the facing-based picker only reaches into the
FACING_SHORTLIST biggest boxes (bounded cost, matches what was measured),
and -- the one that matters most -- the picker is BYTE-IDENTICAL to the old
size-only picker when FACING_CROP_PICK is off, so every existing clip keeps
behaving exactly as it did before this file was touched.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "phase2"))
sys.path.insert(0, os.path.join(_ROOT, "spikes"))
sys.path.insert(0, _ROOT)

import stage6_ocr_confirm as s6  # noqa: E402


def _old_pick(frs, max_attempts=10):
    """The v3 algorithm this file replaced, kept here ONLY as a reference
    for the byte-identical test below -- never call this from real code."""
    if not frs:
        return []
    span_lo, span_hi = frs[0][0], frs[-1][0]
    width = max(1, span_hi - span_lo + 1)
    best_in_slice = {}
    for (f, bb) in frs:
        s = min(max_attempts - 1, (f - span_lo) * max_attempts // width)
        cur = best_in_slice.get(s)
        if cur is None or (bb[3] - bb[1]) > (cur[1][3] - cur[1][1]):
            best_in_slice[s] = (f, bb)
    return sorted(best_in_slice.values(),
                 key=lambda fb: -(fb[1][3] - fb[1][1]))[:max_attempts]


def _new_pick_size_only(frs):
    """The refactored path with facing_of=None -- what every candidate
    actually runs when FACING_CROP_PICK is off (the default)."""
    groups = s6._group_by_slice(frs)
    if not groups:
        return []
    winners = [s6._biggest(g) for g in groups.values()]
    return sorted(winners, key=lambda fb: -(fb[1][3] - fb[1][1]))[:10]


# --------------------------------------------------- flag-off = unchanged ----

def test_flag_off_reproduces_the_old_picker_exactly():
    """The one that matters most: no candidate's attempts may change just
    because this file was refactored."""
    frs = [(100, (0, 0, 50, 90)), (101, (0, 0, 50, 95)), (102, (0, 0, 50, 80)),
          (140, (0, 0, 50, 110)), (141, (0, 0, 50, 110)),   # exact tie: earliest wins
          (200, (0, 0, 50, 60)), (260, (0, 0, 50, 200))]
    assert _new_pick_size_only(frs) == _old_pick(frs)


def test_flag_off_empty_track_picks_nothing():
    assert _new_pick_size_only([]) == _old_pick([]) == []


def test_flag_off_single_frame_track():
    frs = [(5, (0, 0, 30, 100))]
    assert _new_pick_size_only(frs) == _old_pick(frs) == [(5, (0, 0, 30, 100))]


# ------------------------------------------------- the facing override -------

def test_facing_beats_size_within_the_shortlist():
    g = [(10, (0, 0, 100, 100)), (11, (0, 0, 100, 90)), (12, (0, 0, 100, 80))]
    facing = {(10, (0, 0, 100, 100)): 0.05,
             (11, (0, 0, 100, 90)): 0.90,     # smaller box, best facing
             (12, (0, 0, 100, 80)): 0.10}
    winner = s6._best_by_facing(g, lambda fb: facing[(fb[0], fb[1])])
    assert winner == (11, (0, 0, 100, 90))


def test_facing_never_reaches_past_the_shortlist():
    """A great facing score OUTSIDE the FACING_SHORTLIST biggest boxes must
    not win -- bounded cost is the whole point (spikes/pose_facing_test.py
    only ever validated facing among already reasonably-sized crops)."""
    g = [(1, (0, 0, 100, 100)), (2, (0, 0, 100, 90)), (3, (0, 0, 100, 80)),
        (4, (0, 0, 100, 10))]                # tiny box, best facing, 4th biggest
    facing = {(1, (0, 0, 100, 100)): 0.1, (2, (0, 0, 100, 90)): 0.1,
             (3, (0, 0, 100, 80)): 0.1, (4, (0, 0, 100, 10)): 0.99}
    winner = s6._best_by_facing(g, lambda fb: facing[(fb[0], fb[1])], shortlist=3)
    assert winner != (4, (0, 0, 100, 10))


def test_facing_falls_back_to_biggest_when_pose_has_no_answer():
    """Same abstain-to-today's-behaviour rule as everywhere else in this
    file: no usable pose anywhere in the shortlist means size decides,
    exactly like FACING_CROP_PICK was never turned on."""
    g = [(1, (0, 0, 100, 100)), (2, (0, 0, 100, 90))]
    winner = s6._best_by_facing(g, lambda fb: None)
    assert winner == s6._biggest(g) == (1, (0, 0, 100, 100))


def test_facing_score_needs_both_shoulders_confident():
    people = [{"box": (0, 0, 100, 100), "kp": [(0, 0)] * 17,
              "kpc": [1.0] * 17}]
    people[0]["kp"][s6.L_SHO] = (10, 10)
    people[0]["kp"][s6.R_SHO] = (90, 10)
    people[0]["kpc"][s6.R_SHO] = 0.1          # below KP_CONF
    assert s6._facing_score((0, 0, 100, 100), people) is None


def test_facing_score_needs_an_iou_match():
    """A sliver of overlap (~0.004 IoU) is still below IOU_MATCH -- must not
    be treated as a match just because it's nonzero."""
    people = [{"box": (90, 90, 200, 200), "kp": [(0, 0)] * 17, "kpc": [1.0] * 17}]
    assert s6._facing_score((0, 0, 100, 100), people) is None
