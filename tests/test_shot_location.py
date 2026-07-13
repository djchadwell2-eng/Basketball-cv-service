"""Shot location -- pure lookup/geometry tests (synthetic data, no video).
Location = the hinted shooter's court_feet position at the ESTIMATED
RELEASE FRAME, read from the oncourt cache (trusted, already-computed
per-track court position) -- never a re-projection of the ball's own
(elevated) pixel position, which would be the wrong point on the floor.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "spikes"))

from shot_location import (  # noqa: E402
    COURT_WID, feet_to_px, find_shot_location, render_shot_chart,
)


def attempt(status, track_id=None, release_frame=None):
    rec = {"verdict": "shot_attempt", "start_frame": 100, "end_frame": 120,
           "shooter": {"status": status}}
    if track_id is not None:
        rec["shooter"]["release"] = {"track_id": track_id, "release_frame": release_frame}
        rec["shooter"]["track_id"] = track_id
    return rec


def oncourt(entries):
    """entries: {(frame, track_id): (on, feet_x, feet_y)}"""
    by_frame = {}
    for (f, tid), (on, x, y) in entries.items():
        by_frame.setdefault(f, {})[str(tid)] = {"on": on, "court_feet": [x, y]}
    return {"frames": [{"frame_index": f, "tracks": tr} for f, tr in by_frame.items()]}


def test_review_item_with_on_court_data_gets_a_location():
    a = attempt("review_item", track_id=5, release_frame=1178)
    oc = oncourt({(1178, 5): (True, 68.7, 42.3)})
    out = find_shot_location(a, oc)
    assert out["status"] == "located"
    assert out["court_feet"] == [68.7, 42.3]


def test_no_confident_shooter_is_location_unknown_not_a_crash():
    a = attempt("no_confident_shooter")
    out = find_shot_location(a, oncourt({}))
    assert out["status"] == "location_unknown"


def test_shooter_marked_off_court_is_location_unknown():
    """Trust the oncourt classifier's own abstention -- never override it."""
    a = attempt("review_item", track_id=5, release_frame=1178)
    oc = oncourt({(1178, 5): (False, 68.7, 42.3)})
    out = find_shot_location(a, oc)
    assert out["status"] == "location_unknown"


def test_missing_oncourt_entry_is_location_unknown():
    a = attempt("review_item", track_id=5, release_frame=1178)
    out = find_shot_location(a, oncourt({(1178, 9): (True, 1.0, 1.0)}))
    assert out["status"] == "location_unknown"


def test_feet_to_px_places_court_center_at_the_image_center():
    scale = 10
    w_ft, h_ft = 84.0, 50.0
    x, y = feet_to_px(w_ft / 2, h_ft / 2, scale)
    assert (x, y) == (int(w_ft / 2 * scale), int(h_ft / 2 * scale))


def test_feet_to_px_scales_linearly():
    assert feet_to_px(10, 5, 10) == (100, 50)
    assert feet_to_px(10, 5, 20) == (200, 100)


def test_render_orientation_matches_the_established_heatmap_convention(tmp_path):
    """REGRESSION: the very bug the user caught -- phase1/stage3_heatmap.py
    (already-validated) draws with matplotlib origin='lower' (near-sideline
    y=0 at the BOTTOM, far-sideline y=W at the TOP). A near-sideline shot
    (small y) must render in the BOTTOM half of the image, not the top."""
    scale = 5
    out = str(tmp_path / "chart.png")
    render_shot_chart("TEST", [{"status": "review_item", "court_feet": [42.0, 2.0]}],
                      out, scale=scale)
    import cv2
    img = cv2.imread(out)
    h = img.shape[0]
    # find the drawn marker (non-background, non-court-line, non-text pixel)
    # cheaply: the dot color (0,140,255 BGR orange) should appear only in
    # the BOTTOM half of the image for a near-sideline (y=2) shot.
    orange = (img[:, :, 0].astype(int) == 0) & (img[:, :, 2].astype(int) == 255)
    ys = orange.nonzero()[0]
    assert len(ys) > 0
    assert ys.mean() > h / 2   # bottom half
