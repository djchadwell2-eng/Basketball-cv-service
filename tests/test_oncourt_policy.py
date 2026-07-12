"""Tests for the on-court MAJORITY policy (phase2/oncourt.on_court_by_window).

Pure-function tests over synthetic classification docs -- the policy that
decides who gets seeded/OCR'd must stay conservative: strict majority ON,
ties OFF, anchor-failed frames vote for nobody.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "phase2"))

from oncourt import on_court_by_window  # noqa: E402


def _doc(frames):
    """frames: [(frame_index, {tid: on_bool})] -> a minimal oncourt doc."""
    return {"frames": [
        {"frame_index": f,
         "tracks": {str(tid): {"on": on, "court_feet": [0.0, 0.0]}
                    for tid, on in tr.items()}}
        for (f, tr) in frames]}


def test_strict_majority_on():
    doc = _doc([(0, {1: True}), (1, {1: True}), (2, {1: False})])   # 2 on / 1 off
    assert on_court_by_window(doc, span_start=0, win_frames=10) == {0: {1}}


def test_tie_counts_off():
    doc = _doc([(0, {1: True}), (1, {1: False})])                   # 1 on / 1 off
    assert on_court_by_window(doc, span_start=0, win_frames=10) == {}


def test_off_majority_excluded():
    doc = _doc([(0, {1: False}), (1, {1: False}), (2, {1: True})])
    assert on_court_by_window(doc, span_start=0, win_frames=10) == {}


def test_votes_are_per_window():
    """On-court in window 0, off-court in window 1 -> only window 0 trusts it."""
    doc = _doc([(0, {1: True}), (1, {1: True}),          # window 0 (frames 0..4)
                (5, {1: False}), (6, {1: False})])       # window 1 (frames 5..9)
    out = on_court_by_window(doc, span_start=0, win_frames=5)
    assert out == {0: {1}}


def test_span_start_offsets_windows():
    doc = _doc([(300, {7: True}), (301, {7: True}),
                (305, {8: True}), (306, {8: True})])
    out = on_court_by_window(doc, span_start=300, win_frames=5)
    assert out == {0: {7}, 1: {8}}


def test_anchor_failed_frames_vote_for_nobody():
    doc = _doc([(0, {1: True})])
    doc["frames"].append({"frame_index": 1, "anchor": None, "tracks": {}})
    assert on_court_by_window(doc, span_start=0, win_frames=10) == {0: {1}}
