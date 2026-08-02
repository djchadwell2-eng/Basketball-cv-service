"""Player-signal check (TEST 16/19, wired into ball_stages 2026-07-31): does
the ball stay in a HAND through the window after a claimed arrival, or does
it reach the RIM? Locks in the UNANIMOUS rule chosen after the real-pipeline
experiment (spikes/player_signal_experiment.py) found plain majority-vote
over-rejects a real shot a nearby player rebounds quickly (TEST1's verified
layup 3, 571-589) -- unanimous keeps both HARD false-positive rejections
(403-415 rebound/dish, 1352-1375 cross-court pass) while recovering it.

Only the pure decision rules are tested here (no video/pose-model I/O) --
same split as ball_trajectory's physics gates vs ball_stages' glue.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "spikes"))

import pose_shot_check as psc  # noqa: E402


def test_window_unanimous_requires_every_vote_to_be_hand():
    assert psc.window_unanimous(["HAND", "HAND", "HAND"]) == "HAND"
    assert psc.window_unanimous(["rim", "HAND", "HAND"]) == "rim"
    assert psc.window_unanimous(["rim", "rim", "rim"]) == "rim"


def test_window_unanimous_no_votes_abstains():
    assert psc.window_unanimous([]) is None


def test_window_majority_still_available_but_not_the_live_rule():
    # window_majority is kept (used by the read-only TEST 16 script for
    # comparison) but is NOT what window_verdict calls anymore.
    assert psc.window_majority(["HAND", "HAND", "rim"]) == "HAND"
    assert psc.window_majority(["rim", "HAND", "rim"]) == "rim"


def test_hard_false_positives_stay_unanimous_hand_real_measured_votes():
    """Regression-locks the ACTUAL vote sequences measured 2026-07-31
    (spikes/out/player_signal_experiment.json) for the two known HARD false
    positives -- every sampled frame said HAND for both, which is exactly
    why unanimous (not just majority) correctly rejects them."""
    rebound_dish_votes = ["HAND"] * 6         # step-3 sample of a 31-frame all-HAND run
    cross_court_pass_votes = ["HAND"] * 6
    assert psc.window_unanimous(rebound_dish_votes) == "HAND"
    assert psc.window_unanimous(cross_court_pass_votes) == "HAND"


def test_layup_3_recovered_by_unanimous_not_majority():
    """TEST1's verified layup 3 (571-589): measured votes step-3 sampled were
    [rim, HAND, HAND, HAND, HAND, HAND] -- majority says HAND (wrongly
    rejects a real shot), unanimous says rim (correctly keeps it), because
    a rebounder grabbing the ball a few frames late must not overturn a
    shot that DID touch the rim first."""
    votes = ["rim", "HAND", "HAND", "HAND", "HAND", "HAND"]
    assert psc.window_majority(votes) == "HAND"     # the rule we moved away from
    assert psc.window_unanimous(votes) == "rim"     # the live rule
