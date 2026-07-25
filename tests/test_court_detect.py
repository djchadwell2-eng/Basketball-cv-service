"""Which court is this? -- tests for spikes/court_detect.py.

The bug these exist to prevent: TEST2 (Fairfield) is a 94-ft floor and its
config said 84 ft, copied from TEST1. Nothing noticed, because a wrong court
still produces a plausible-looking error number (0.94 ft), so two sessions were
spent blaming the clicks, the glare and the lens instead. These lock the three
real clips to the court they were actually filmed on, and -- just as important
-- lock the two cases where the detector must REFUSE rather than guess.
"""

import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "spikes"))

import clips_config                                    # noqa: E402
import court_detect                                    # noqa: E402

LEFT_KEY_ONLY = ("LB_side_far", "L_lane_base_near", "L_lane_base_far",
                 "L_FT_near", "L_FT_far")


def marks(clip):
    return clips_config.CLIPS[clip]["landmarks"]


# --- the three real clips ---------------------------------------------------
@pytest.mark.parametrize("clip,length", [
    ("TEST1", 84.0),    # Milford -- an ordinary high-school floor
    ("TEST2", 94.0),    # Fairfield -- full-size floor, high-school markings
    ("HARD", 94.0),     # Winton Woods -- also full-size (found 2026-07-25)
])
def test_real_clips_identify_their_actual_floor(clip, length):
    r = court_detect.identify(marks(clip))
    assert r["identified"], r["reason"]
    assert r["dims"]["length"] == length
    # every gym here plays high-school markings on whatever floor it has
    assert r["dims"]["lane_y0"] == 19.0 and r["dims"]["lane_y1"] == 31.0


@pytest.mark.parametrize("clip", ["TEST1", "TEST2", "HARD"])
def test_the_call_is_decisive_not_a_coin_flip(clip):
    """The right court must beat the next one by a clear margin, otherwise
    'identified' would just be noise picking a winner."""
    r = court_detect.identify(marks(clip))
    assert r["margin"] >= court_detect.DECISIVE_RATIO
    assert r["error_ft"] <= 0.35


def test_the_84ft_court_is_specifically_rejected_for_the_94ft_gyms():
    """The exact mistake that was live in the repo: TEST2 on dict(HS_COURT)."""
    hs84 = dict(court_detect.KNOWN_COURTS[0][1])
    assert hs84["length"] == 84.0
    for clip in ("TEST2", "HARD"):
        wrong, _, _ = court_detect.score(marks(clip), hs84)
        right, _, _ = court_detect.score(marks(clip), dict(hs84, length=94.0))
        assert right < wrong / 2.0, f"{clip}: 84 ft should fit far worse"


# --- when it must refuse ----------------------------------------------------
def test_refuses_when_the_marks_cannot_tell_the_floors_apart():
    """One key, marked perfectly, still cannot reveal the floor length: every
    landmark round a left-hand key sits at the same court feet on an 84 and a
    94 ft floor. The honest answer is 'not enough', not a 50/50 guess."""
    only_key = {240: [m for m in marks("TEST2")[240] if m[0] in LEFT_KEY_ONLY]}
    r = court_detect.identify(only_key)
    assert not r["identified"]
    assert "not tell these courts apart" in r["reason"]


def test_refuses_when_no_known_court_fits():
    """Two marks given each other's labels -- a real mis-click. No court can
    explain that, and inventing one would bury the mistake."""
    swap = {"R_FT_near": "R_FT_far", "R_FT_far": "R_FT_near"}
    mislabelled = {400: [(swap.get(t, t), x, y) for (t, x, y) in marks("TEST2")[400]]}
    r = court_detect.identify(mislabelled)
    assert not r["identified"]
    assert "no known court fits" in r["reason"]


def test_frames_with_four_marks_carry_no_evidence_and_are_skipped():
    """A homography has 8 degrees of freedom, so 4 marks fit ANY court exactly
    -- scoring them would dilute the frames that actually prove something."""
    four = {400: marks("TEST2")[400][:4]}
    err, used, frames = court_detect.score(four, court_detect.KNOWN_COURTS[0][1])
    assert (used, frames) == (0, 0)
    r = court_detect.identify(four)
    assert not r["identified"] and "marks" in r["reason"]


# --- the geometry the whole engine reads ------------------------------------
def test_court_model_places_the_far_end_by_the_floor_length():
    """The 10 ft that broke TEST2: on a 94 ft floor the right basket, right
    key and centre line all move, and the 3pt apex with them."""
    m84 = court_detect.court_model(court_detect.KNOWN_COURTS[0][1])
    m94 = court_detect.court_model(court_detect.KNOWN_COURTS[1][1])
    assert m84["center_logo"] == (42.0, 25.0)
    assert m94["center_logo"] == (47.0, 25.0)
    assert m84["R_lane_base_near"][0] == 84.0
    assert m94["R_lane_base_near"][0] == 94.0
    # arc apex stays 25 ft out from ITS OWN baseline on both floors
    assert m84["L_arc_top"] == m94["L_arc_top"] == (25.0, 25.0)
    assert m84["R_arc_top"][0] == 59.0 and m94["R_arc_top"][0] == 69.0


def test_stage4_builds_its_court_model_from_this_module():
    """One source of truth -- the engine's model and the detector's must not
    drift apart, or the detector would be checking a different court."""
    clips_config.ACTIVE = "TEST1"
    import stage4_courtmap
    assert stage4_courtmap.COURT_MODEL == court_detect.court_model(
        clips_config.CLIPS["TEST1"]["court"])


# --- the config path --------------------------------------------------------
def test_auto_court_resolves_test2_to_its_measured_94ft_floor():
    clips_config.ACTIVE = "TEST2"
    assert clips_config.CLIPS["TEST2"]["court"] == "auto"     # not hard-coded
    assert clips_config.active()["court"]["length"] == 94.0


def test_auto_court_stops_the_run_rather_than_guessing():
    """An unidentifiable court must raise. Quietly carrying on with a guessed
    court is exactly how the 84-ft model survived two sessions."""
    clips_config.CLIPS["_UNIDENTIFIABLE"] = {
        "court": "auto",
        "landmarks": {1: [m for m in marks("TEST2")[240] if m[0] in LEFT_KEY_ONLY]},
    }
    clips_config.ACTIVE = "_UNIDENTIFIABLE"
    try:
        with pytest.raises(ValueError, match="cannot tell which court"):
            clips_config.active()
    finally:
        del clips_config.CLIPS["_UNIDENTIFIABLE"]
        clips_config._RESOLVED.pop("_UNIDENTIFIABLE", None)
        clips_config.ACTIVE = "TEST1"
