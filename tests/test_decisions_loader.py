"""Click-seeding decisions loader -- a human label must be on-roster or refused."""

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "phase2"))

from roster import load_decisions  # noqa: E402

ROSTER = {5, 13, 24}


def _write(tmp_path, track_labels):
    p = tmp_path / "decisions.json"
    p.write_text(json.dumps({"clip": "T", "track_labels": track_labels}),
                 encoding="utf-8")
    return str(p)


def test_valid_labels_load_with_int_keys(tmp_path):
    path = _write(tmp_path, {"17": 13, "6": 5})
    assert load_decisions(path, ROSTER) == {17: 13, 6: 5}


def test_off_roster_and_non_numeric_labels_refused(tmp_path, capsys):
    path = _write(tmp_path, {"17": 99, "6": 5, "9": "13", "3": None, "4": "ref"})
    out = load_decisions(path, ROSTER)
    assert out == {6: 5}, "only the on-roster int label survives"
    printed = capsys.readouterr().out
    assert "OFF-ROSTER" in printed and "99" in printed


def test_missing_file_is_empty_not_an_error(tmp_path):
    assert load_decisions(str(tmp_path / "nope.json"), ROSTER) == {}


# --- purity quarantine: a spliced track's label must be refused ---------------

def test_spliced_track_label_refused():
    from roster import resolve_label
    n, reason = resolve_label(49, {49: 44}, {}, spliced={49})
    assert n is None and reason == "refused_spliced"


def test_clean_track_label_resolves_decisions_over_config():
    from roster import resolve_label
    assert resolve_label(7, {7: 23}, {7: 24}, spliced=set()) == (23, "labeled")
    assert resolve_label(9, {}, {9: 5}, spliced=set()) == (5, "labeled")
    assert resolve_label(3, {}, {}, spliced=set()) == (None, "unlabeled")
