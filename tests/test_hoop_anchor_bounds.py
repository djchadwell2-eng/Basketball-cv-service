"""hoop_anchor.in_plausible_bounds -- the sanity gate that catches a
numerically-degenerate homography extrapolation (near-zero perspective
divide) before it's trusted as a hoop position. Real example that
motivated this (DECISIONS 18): a full-clip harvest run produced
hoop_px=(42625, 1191) in a 1920x1080 frame -- geometrically absurd, from
a SIFT match that otherwise cleared MIN_INLIERS.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "spikes"))

from hoop_anchor import in_plausible_bounds  # noqa: E402

W, H = 1920, 1080


def test_point_well_inside_frame_is_plausible():
    assert in_plausible_bounds((960, 540), W, H)


def test_point_moderately_outside_frame_is_still_plausible():
    # 50% margin means up to 960px beyond the right edge is allowed
    assert in_plausible_bounds((W + 500, 540), W, H)


def test_the_actual_bad_result_from_the_full_clip_harvest_is_rejected():
    assert not in_plausible_bounds((42625.0, 1191.3), W, H)


def test_moderately_negative_coordinates_are_still_plausible():
    assert in_plausible_bounds((-500, -200), W, H)


def test_wildly_negative_coordinates_are_rejected():
    assert not in_plausible_bounds((-50000, 500), W, H)
