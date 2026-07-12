"""Possession detection v1 -- side-of-court segmentation with hysteresis.

The stand-in 2.0s windows chopped play into arbitrary slices; possessions
follow the ball's side of the court. Honesty rules: a side switch must HOLD
before it counts (no midcourt flicker), blips shorter than a real possession
merge away, and a degenerate signal returns None so callers fall back to
fixed windows LOUDLY instead of trusting nonsense."""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "phase2"))

from possessions import detect  # noqa: E402

FPS = 30.0
LEN = 84.0


def xs(*segments):
    """segments: (n_frames, mean_x) -> dense [(frame, x)] starting at frame 0."""
    out, f = [], 0
    for (n, x) in segments:
        for _ in range(n):
            out.append((f, x))
            f += 1
    return out


def test_clean_switch_yields_two_possessions():
    sig = xs((300, 20.0), (300, 64.0))            # 10s left, 10s right
    segs = detect(sig, FPS, LEN)
    assert [s["side"] for s in segs] == ["L", "R"]
    assert segs[0]["start"] == 0
    assert segs[1]["start"] == 300
    assert segs[1]["end"] == 599


def test_midcourt_flicker_is_absorbed_by_hysteresis():
    # left possession with a 0.5s wobble past midcourt (a drive-and-kick)
    sig = xs((200, 20.0), (15, 64.0), (200, 20.0))
    segs = detect(sig, FPS, LEN)
    assert [s["side"] for s in segs] == ["L"], "a 0.5s wobble is not a possession"


def test_short_blip_merges_into_neighbor():
    # a 2s right excursion (steal + immediate turnover back) < min possession
    sig = xs((300, 20.0), (60, 64.0), (300, 20.0))
    segs = detect(sig, FPS, LEN, min_poss_s=4.0)
    assert [s["side"] for s in segs] == ["L"]


def test_neutral_midcourt_frames_carry_no_side():
    # dead-zone frames (transition) attach to the following possession
    sig = xs((300, 20.0), (45, 42.0), (300, 64.0))
    segs = detect(sig, FPS, LEN)
    assert [s["side"] for s in segs] == ["L", "R"]
    assert segs[1]["end"] == 644


def test_truncated_trailing_possession_survives_the_min_length_merge():
    """USER-CAUGHT BUG (2026-07-12, both clips): a possession CUT OFF by the
    span end can't prove its length; erasing it into the previous side was
    confidently-wrong. Edge segments are kept and flagged partial."""
    sig = xs((400, 20.0), (90, 64.0))            # 3s right tail, then span ends
    segs = detect(sig, FPS, LEN, min_poss_s=4.0)
    assert [s["side"] for s in segs] == ["L", "R"]
    assert segs[-1]["partial_end"] is True


def test_truncated_leading_possession_survives_too():
    sig = xs((60, 64.0), (400, 20.0))            # 2s right head (span starts mid-R)
    segs = detect(sig, FPS, LEN, min_poss_s=4.0)
    assert [s["side"] for s in segs] == ["R", "L"]
    assert segs[0]["partial_start"] is True


def test_interior_blip_still_merges():
    sig = xs((300, 20.0), (60, 64.0), (300, 20.0))
    segs = detect(sig, FPS, LEN, min_poss_s=4.0)
    assert [s["side"] for s in segs] == ["L"], "interior blips still merge away"


def test_degenerate_signal_returns_none_for_loud_fallback():
    assert detect([], FPS, LEN) is None
    assert detect([(f, None) for f in range(100)], FPS, LEN) is None
    assert detect(xs((20, 42.0)), FPS, LEN) is None   # all neutral


def test_segments_carry_confidence():
    sig = xs((290, 20.0), (10, 64.0), (300, 64.0))
    segs = detect(sig, FPS, LEN)
    assert all(0.5 < s["side_agreement"] <= 1.0 for s in segs)
