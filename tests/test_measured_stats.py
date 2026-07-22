"""measured_stats.py -- the web-facing MEASURED-stats contract (Phase 7
slice A1). Pure functions only here: shot-zone classification (3pt-arc
geometry), the distribution summary math, and assembling the contract
from synthetic box-score / shot-location / shot-attempt docs. No files,
no video.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import measured_stats as ms  # noqa: E402


# ----------------------------------------------------------- zone geometry --

def test_shot_beyond_the_arc_is_a_three():
    # HARD's one real located shot: (68.7, 42.3), ~20ft from the right
    # basket (78.75, 25) -- just beyond the 19.75ft arc (DECISIONS 16:
    # "right at the 3pt line"). Must classify as three.
    zone, dist = ms.classify_zone(68.7, 42.3)
    assert zone == "three"
    assert 19.75 < dist < 21.0


def test_shot_at_the_rim_is_paint():
    zone, dist = ms.classify_zone(6.0, 25.0)   # ~0.75ft from left basket
    assert zone == "paint"
    assert dist < ms.PAINT_RADIUS_FT


def test_shot_between_paint_and_arc_is_midrange():
    zone, dist = ms.classify_zone(18.0, 25.0)  # ~12.75ft from left basket
    assert zone == "midrange"
    assert ms.PAINT_RADIUS_FT < dist < ms.THREE_RADIUS_FT


def test_zone_uses_nearest_basket():
    # a point near the RIGHT basket must be measured against the right
    # basket, not the left -- otherwise every right-side shot reads as a
    # 70ft "three".
    zone, dist = ms.classify_zone(76.0, 25.0)  # ~2.75ft from right basket
    assert zone == "paint"
    assert dist < 4.0


# ------------------------------------------------------------ distribution --

def test_distribution_percentages_sum_to_100():
    dist = ms.shot_distribution(["three", "three", "midrange", "paint"])
    assert dist["n"] == 4
    assert dist["counts"] == {"three": 2, "midrange": 1, "paint": 1}
    assert dist["pct"]["three"] == 50.0
    assert dist["pct"]["midrange"] == 25.0
    assert dist["pct"]["paint"] == 25.0
    # the headline split the goal cares about: behind vs inside the arc
    assert dist["pct_three"] == 50.0
    assert dist["pct_two"] == 50.0


def test_distribution_empty_is_safe():
    dist = ms.shot_distribution([])
    assert dist["n"] == 0
    assert dist["pct"] == {"three": 0.0, "midrange": 0.0, "paint": 0.0}
    assert dist["pct_three"] == 0.0


# --------------------------------------------------------- contract shape --

def _box_doc():
    return {"clip": "X", "note": "presence-seconds over this clip's tracked span",
            "players": [
                {"number": 24, "team": "Home", "seconds_total": 15.1,
                 "seconds_live": 15.1, "seconds_retro": 0.0, "windows_present": 2,
                 "zone_seconds": {"PERIMETER": 7.4, "PAINT": 0.6}, "top_zone": "PERIMETER",
                 "disputed_seconds": 1.1, "unpositioned_frames": 0},
            ]}


def _loc_doc():
    return {"clip": "X", "locations": [
        {"start_frame": 351, "end_frame": 375, "shooter_status": "no_confident_shooter",
         "status": "location_unknown", "reason": "no release estimate"},
        {"start_frame": 1188, "end_frame": 1213, "shooter_status": "review_item",
         "status": "located", "court_feet": [68.7, 42.3], "release_frame": 1178,
         "track_id": 1502},
    ]}


def _att_doc():
    return {"clip": "X", "attempts": [
        {"start_frame": 1188, "end_frame": 1213, "hoop": "far", "verdict": "shot_attempt",
         "shot_type": "jumpshot", "min_dist": 17.7, "at_frame": 1200},
        {"start_frame": 351, "end_frame": 375, "hoop": "near", "verdict": "shot_attempt",
         "shot_type": "jumpshot", "min_dist": 88.5, "at_frame": 360},
    ]}


def test_build_measured_stats_shape():
    out = ms.build_measured_stats("X", _box_doc(), _loc_doc(), _att_doc())
    assert out["clip"] == "X"
    assert out["meta"]["make_miss_available"] is False    # honest: no % yet
    assert "presence-seconds" in out["meta"]["box_score_note"]
    # box score passthrough
    assert len(out["box_score"]) == 1
    assert out["box_score"][0]["number"] == 24
    assert out["box_score"][0]["zone_seconds"]["PERIMETER"] == 7.4
    # exactly ONE located shot charted; the other is counted, not charted
    assert len(out["shots"]) == 1
    s = out["shots"][0]
    assert s["court_x"] == 68.7 and s["court_y"] == 42.3
    assert s["zone"] == "three"
    assert s["shot_type"] == "jumpshot"          # joined from the attempt by frame span
    assert s["shooter_status"] == "review_item"
    assert out["shots_unlocated"] == 1
    # distribution reflects only the located shot
    assert out["shot_distribution"]["n"] == 1
    assert out["shot_distribution"]["pct_three"] == 100.0


def test_build_measured_stats_no_located_shots_is_safe():
    """TEST1's real situation: every shot location_unknown -> empty chart,
    but the box score + a truthful '0 shots located' still render."""
    loc = {"clip": "X", "locations": [
        {"start_frame": 58, "end_frame": 70, "status": "location_unknown",
         "shooter_status": "no_confident_shooter", "reason": "no release estimate"}]}
    out = ms.build_measured_stats("X", _box_doc(), loc, _att_doc())
    assert out["shots"] == []
    assert out["shots_unlocated"] == 1
    assert out["shot_distribution"]["n"] == 0
    assert len(out["box_score"]) == 1            # box score still present
