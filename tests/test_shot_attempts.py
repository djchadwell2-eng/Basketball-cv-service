"""Shot-attempt classification -- pure geometry tests (synthetic arcs +
synthetic hoop positions, no video). A claimed arc (already physics-gated
by ball_trajectory.py) becomes a SHOT ATTEMPT only if, at or after its
apex (the descending half -- a rising ball hasn't reached the rim yet),
it passes within HOOP_RADIUS_PX of the hoop position AT THAT SAME FRAME.
Floor-level flight (dribbles) can never satisfy this by geometry alone,
not by a special case.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "spikes"))

from shot_attempts import (  # noqa: E402
    HOOP_RADIUS_PX, apex_index, classify_shot, nearest_track_feet,
)


def arc_points(frames, x0, vx, y0, vy, ay):
    """Same ballistic shape helper as test_ball_trajectory.py."""
    t0 = frames[0]
    return [(f, x0 + vx * (f - t0), y0 + vy * (f - t0) + 0.5 * ay * (f - t0) ** 2)
            for f in frames]


def const_hoop(hx, hy):
    return lambda frame: (hx, hy)


# ---------------------------------------------------------------- apex

def test_apex_is_the_min_y_point_since_image_y_grows_downward():
    pts = arc_points(range(0, 10), x0=0, vx=10, y0=300, vy=-20, ay=4)
    i = apex_index(pts)
    ys = [p[2] for p in pts]
    assert pts[i][2] == min(ys)


# ------------------------------------------------------------ classification

def test_arc_passing_through_hoop_region_on_descent_is_a_shot():
    pts = arc_points(range(0, 20), x0=100, vx=8, y0=300, vy=-9, ay=0.5)
    # apex ~ t=9 (y min); pick a hoop position matching a late point closely
    hoop_frame, hx, hy = 18, pts[18][1], pts[18][2]
    out = classify_shot(pts, const_hoop(hx, hy))
    assert out["verdict"] == "shot_attempt"
    assert out["min_dist"] <= HOOP_RADIUS_PX


def test_same_arc_hoop_moved_far_away_is_not_a_shot():
    pts = arc_points(range(0, 20), x0=100, vx=8, y0=300, vy=-9, ay=0.5)
    out = classify_shot(pts, const_hoop(5000, 5000))
    assert out["verdict"] == "not_shot"


def test_floor_bounce_arc_never_reaches_a_rim_high_in_the_frame():
    """A dribble bounce: apex near the floor (y~600), hoop near the top of
    the frame (y~200). Geometry alone rules it out -- no special case."""
    pts = arc_points(range(0, 12), x0=400, vx=3, y0=700, vy=-15, ay=3.0)
    out = classify_shot(pts, const_hoop(400, 200))
    assert out["verdict"] == "not_shot"


def test_proximity_before_apex_does_not_count_as_a_shot():
    """The ball passing near the hoop's (x,y) coordinates on the way UP
    (before apex) must not count -- only descending/at-apex does."""
    pts = arc_points(range(0, 20), x0=100, vx=8, y0=300, vy=-9, ay=0.5)
    i_apex = apex_index(pts)
    early_frame, ex, ey = pts[2]
    assert 2 < i_apex        # confirm this point IS before the apex
    out = classify_shot(pts, const_hoop(ex, ey))
    assert out["verdict"] == "not_shot"


def test_hoop_lookup_returning_none_for_a_frame_is_skipped_not_a_crash():
    pts = arc_points(range(0, 20), x0=100, vx=8, y0=300, vy=-9, ay=0.5)
    def sparse_hoop(f):
        return None if f < 15 else (pts[-1][1], pts[-1][2])
    out = classify_shot(pts, sparse_hoop)
    assert out["verdict"] == "shot_attempt"


def test_no_hoop_data_anywhere_is_honestly_not_shot_not_a_crash():
    pts = arc_points(range(0, 20), x0=100, vx=8, y0=300, vy=-9, ay=0.5)
    out = classify_shot(pts, lambda f: None)
    assert out["verdict"] == "not_shot"
    assert out["min_dist"] is None


# ------------------------------------------------------------ shooter join

def test_nearest_track_feet_picks_the_closest_body():
    tracks = [{"track_id": 1, "bbox": [0, 0, 20, 100]},      # feet (10,100)
              {"track_id": 2, "bbox": [190, 0, 210, 100]}]   # feet (200,100)
    tid, dist = nearest_track_feet(tracks, 15, 100)
    assert tid == 1
    assert dist < 10


def test_nearest_track_feet_on_empty_frame_is_none():
    assert nearest_track_feet([], 15, 100) is None
