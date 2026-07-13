"""Shot outcome (make/miss) -- pure geometry tests (synthetic detections,
no video). ROADMAP Phase 5 step 5 + GATE 4: outputs are CANDIDATE labels
only (never a bare made/missed stat), and conflicting or absent evidence
must resolve to 'unknown', never a guess.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "spikes"))

from shot_outcome import (  # noqa: E402
    below_rim_fall_evidence, classify_outcome, deflection_evidence,
)


def const_hoop(hx, hy):
    return lambda f: (hx, hy)


def det(x1, y1, x2, y2):
    return {"bbox": [x1, y1, x2, y2]}


# --------------------------------------------------------- fall-through

def test_ball_falling_straight_through_the_corridor_is_make_evidence():
    hoop = const_hoop(500, 200)
    raw = {211: [det(490, 210, 510, 230)],   # cy=220
           212: [det(490, 230, 510, 250)],   # cy=240
           213: [det(490, 255, 510, 275)]}   # cy=265, still increasing
    out = below_rim_fall_evidence(raw, hoop, after_frame=210, window=10)
    assert out is not None
    assert out["n_points"] >= 3


def test_no_detections_near_hoop_is_no_make_evidence():
    hoop = const_hoop(500, 200)
    raw = {211: [det(1490, 1210, 1510, 1230)]}   # far from the hoop entirely
    out = below_rim_fall_evidence(raw, hoop, after_frame=210, window=10)
    assert out is None


def test_detections_above_the_hoop_dont_count_as_falling_through():
    hoop = const_hoop(500, 200)
    raw = {211: [det(490, 150, 510, 170)]}   # cy=160, ABOVE hoop (smaller y)
    out = below_rim_fall_evidence(raw, hoop, after_frame=210, window=10)
    assert out is None


def test_y_decreasing_run_does_not_count_as_falling():
    hoop = const_hoop(500, 200)
    raw = {211: [det(490, 260, 510, 280)],   # cy=270
           212: [det(490, 230, 510, 250)],   # cy=240 -- moving UP, not falling
           213: [det(490, 210, 510, 230)]}   # cy=220
    out = below_rim_fall_evidence(raw, hoop, after_frame=210, window=10)
    assert out is None


# --------------------------------------------------------- deflection

def test_chain_escaping_the_hoop_region_is_miss_evidence():
    hoop = const_hoop(500, 200)
    chain = {"points": [[211, 510, 210, 0.1], [212, 560, 195, 0.1],
                        [213, 650, 250, 0.1], [214, 750, 400, 0.1]]}
    out = deflection_evidence([chain], hoop, after_frame=210, window=10)
    assert out is not None
    assert out["chain_start_frame"] == 211


def test_chain_starting_far_from_the_hoop_is_not_deflection_evidence():
    hoop = const_hoop(500, 200)
    chain = {"points": [[211, 1500, 1200, 0.1], [212, 1600, 1300, 0.1]]}
    out = deflection_evidence([chain], hoop, after_frame=210, window=10)
    assert out is None


def test_chain_that_stays_near_the_hoop_is_not_deflection_evidence():
    """A rattle-in-the-rim that never clearly escapes must not count."""
    hoop = const_hoop(500, 200)
    chain = {"points": [[211, 510, 205, 0.1], [212, 495, 210, 0.1],
                        [213, 505, 195, 0.1]]}
    out = deflection_evidence([chain], hoop, after_frame=210, window=10)
    assert out is None


def test_chain_outside_the_search_window_is_ignored():
    hoop = const_hoop(500, 200)
    chain = {"points": [[300, 510, 210, 0.1], [301, 700, 400, 0.1]]}
    out = deflection_evidence([chain], hoop, after_frame=210, window=10)
    assert out is None


# --------------------------------------------------------- classification

def test_make_only_is_candidate_make():
    out = classify_outcome({"n_points": 3}, None)
    assert out["outcome"] == "candidate_make"


def test_miss_only_is_candidate_miss():
    out = classify_outcome(None, {"chain_start_frame": 1})
    assert out["outcome"] == "candidate_miss"


def test_both_signals_present_is_unknown_conflict_not_a_guess():
    out = classify_outcome({"n_points": 3}, {"chain_start_frame": 1})
    assert out["outcome"] == "unknown"
    assert "conflict" in out["reason"].lower()


def test_neither_signal_present_is_unknown_no_evidence():
    out = classify_outcome(None, None)
    assert out["outcome"] == "unknown"
    assert "no evidence" in out["reason"].lower()
