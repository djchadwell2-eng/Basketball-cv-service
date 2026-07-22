"""iter_frames() -- Phase 6's streaming replacement for extract_frames()'s
whole-span dict. Same single-pass frame-accurate semantics, verified
against extract_frames on a tiny synthetic video (no real footage needed).
"""

import os
import sys

import cv2
import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "spikes"))

from stage2_multikeyframe import extract_frames, iter_frames  # noqa: E402


@pytest.fixture
def synthetic_video(tmp_path):
    """20 frames, each a solid color encoding its own index (frame i is
    filled with pixel value i*10) so a wrong frame is trivially detectable."""
    path = str(tmp_path / "synthetic.mp4")
    w, h = 16, 16
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    for i in range(20):
        frame = np.full((h, w, 3), i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


def test_iter_frames_yields_same_frames_as_extract_frames(synthetic_video):
    indices = [2, 5, 5, 9, 0, 15]      # unsorted + duplicate, like a real caller might pass
    expected = extract_frames(synthetic_video, indices)
    got = dict(iter_frames(synthetic_video, indices))
    assert set(got) == set(expected)
    for idx in expected:
        # exact array equality against extract_frames (the property under
        # test), not against an intended constant -- mp4v compression drifts
        # pixel values slightly, so a hardcoded idx*10 check would be flaky.
        assert np.array_equal(got[idx], expected[idx])


def test_iter_frames_yields_in_increasing_order(synthetic_video):
    order = [f for f, _frame in iter_frames(synthetic_video, [7, 1, 4, 12])]
    assert order == [1, 4, 7, 12]


def test_iter_frames_is_a_true_generator_not_materialized_upfront(synthetic_video):
    gen = iter_frames(synthetic_video, [0, 5, 10])
    first_idx, _first_frame = next(gen)
    assert first_idx == 0
    # the rest only gets read on demand -- draining it now should still work
    rest = list(gen)
    assert [f for f, _ in rest] == [5, 10]


def test_iter_frames_raises_on_out_of_range_index(synthetic_video):
    with pytest.raises(RuntimeError, match=r"Could not read frames \[100\]"):
        list(iter_frames(synthetic_video, [3, 100]))


def test_iter_frames_raises_on_bad_path():
    with pytest.raises(RuntimeError, match="Could not open video"):
        list(iter_frames("does_not_exist.mp4", [0]))
