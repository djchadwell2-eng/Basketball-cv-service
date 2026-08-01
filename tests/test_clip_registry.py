"""clip_registry -- the ONE config the web app writes and both Python config
systems read. These lock in the properties the setup flow depends on."""

import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import clip_registry as cr                                        # noqa: E402


@pytest.fixture
def clips_dir(tmp_path, monkeypatch):
    d = tmp_path / "clips"
    d.mkdir()
    monkeypatch.setattr(cr, "CLIPS_DIR", str(d))
    return d


def test_rejects_names_that_could_escape_the_directory():
    # A clip name becomes a file path and an artifact prefix; a traversal here
    # would write outside the project.
    for bad in ("../evil", "a/b", "with space", "", "sneaky.json"):
        assert not cr.valid_name(bad)
        with pytest.raises(ValueError):
            cr.path_for(bad)
    assert cr.valid_name("GAME_abc-123")


def test_update_merges_and_never_blanks_earlier_stages(clips_dir):
    """The setup flow writes a clip in stages -- roster first, landmarks later.
    Saving stage two must not wipe stage one."""
    cr.update("G1", video_path="v.mp4",
              teams=[{"name": "A", "jersey_color": "white", "numbers": [3, 4]}])
    cr.update("G1", keyframes=[10, 20])
    doc = cr.load("G1")
    assert doc["teams"][0]["numbers"] == [3, 4]      # survived
    assert doc["keyframes"] == [10, 20]
    assert doc["name"] == "G1"


def test_load_all_skips_malformed_without_losing_the_others(clips_dir):
    cr.save("GOOD", {"video_path": "v.mp4"})
    (clips_dir / "BAD.json").write_text("{not json", encoding="utf-8")
    got = cr.load_all()
    assert "GOOD" in got and "BAD" not in got


def test_calibration_adapter_restores_int_keys_and_tuples(clips_dir):
    """JSON has no int keys and no tuples; the engine needs both."""
    cr.save("G2", {
        "video_path": "v.mp4",
        "keyframes": [600, 900],
        "exclude_regions": [[0, 830, 330, 1080]],
        "landmarks": {"600": [["center_logo", 100.5, 200.25]]},
    })
    entry = cr.to_calibration_entry(cr.load("G2"))
    assert 600 in entry["landmarks"]                     # int, not "600"
    assert entry["landmarks"][600] == [("center_logo", 100.5, 200.25)]
    assert entry["exclude_regions"] == [(0, 830, 330, 1080)]
    assert entry["court"] == "auto"     # uploaded gyms are never assumed


def test_readiness_flags_are_honest_about_partial_clips(clips_dir):
    """Half a config must not look like a whole one."""
    cr.save("P", {"video_path": "v.mp4", "keyframes": [1], "landmarks": {}})
    doc = cr.load("P")
    assert not cr.has_calibration(doc)      # keyframes but no marks
    assert not cr.has_roster(doc)

    cr.update("P", landmarks={"1": [["center_logo", 1, 2]]},
              teams=[{"name": "A", "numbers": []}])
    doc = cr.load("P")
    assert cr.has_calibration(doc)
    assert not cr.has_roster(doc)           # a team with no numbers is not a roster

    cr.update("P", teams=[{"name": "A", "numbers": [5]}])
    assert cr.has_roster(cr.load("P"))


def test_missing_clip_and_missing_directory_are_not_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(cr, "CLIPS_DIR", str(tmp_path / "nope"))
    assert cr.load("anything") is None
    assert cr.load_all() == {}


def test_handwritten_clips_win_over_registry(clips_dir):
    """A stray uploaded clip named HARD must never redefine the validated
    baseline the calibration engine was proven against."""
    sys.path.insert(0, os.path.join(_ROOT, "spikes"))
    import clips_config
    before = clips_config.CLIPS["HARD"]["video_path"]
    cr.save("HARD", {"video_path": "HIJACKED.mp4", "keyframes": [1],
                     "landmarks": {"1": [["center_logo", 1, 2]]}})
    clips_config._merge_registry_clips()
    assert clips_config.CLIPS["HARD"]["video_path"] == before
