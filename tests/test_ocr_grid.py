"""Tests for the SHEET reader -- many crops in one call.

The reason this file exists: reading twelve girls in one request is the only
change that makes the vision reader affordable on a whole game (~150,000 crops
measured, ~698,000 calls one at a time), and it introduces a failure the
one-at-a-time reader never had -- an answer landing on the WRONG CELL, which is
one girl's number on another girl's floor time. Every test here is about
refusing that rather than being fast.

No network: the engine is faked, so each reply is chosen by the test.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "phase2"))
import ocr_reader  # noqa: E402

ROSTER = {3, 13, 24, 44}


def _crops(n):
    return [np.full((80, 60, 3), (37 * i) % 255, np.uint8) for i in range(n)]


def _fake_engine(monkeypatch, replies):
    """An engine that returns the given replies in order, counting calls."""
    box = {"calls": 0}
    it = iter(replies)

    class _Models:
        @staticmethod
        def generate_content(model=None, contents=None):
            box["calls"] += 1

            class _R:
                text = next(it)
            return _R()

    class _Client:
        models = _Models()

    client = _Client()
    monkeypatch.setattr(ocr_reader, "_get_engine", lambda: client)
    return box


def _sheet(mapping, labels):
    inner = ",".join(f'"{c}":{mapping.get(c, chr(34) + "NONE" + chr(34))}' for c in labels)
    return "{" + inner + "}"


def test_a_label_can_never_be_mistaken_for_a_jersey():
    """Cells are labelled with LETTERS. A digit painted beside a jersey is an
    invitation to read the label as the number."""
    _img, labels = ocr_reader._grid_image(_crops(ocr_reader.GRID_CELLS))
    assert labels and all(not c.isdigit() for c in labels)


def test_the_sheet_keeps_every_crop_full_size():
    """These numbers are already at the edge of legibility; a sheet that shrinks
    them defeats the whole point."""
    crops = _crops(ocr_reader.GRID_CELLS)
    img, _labels = ocr_reader._grid_image(crops)
    assert img.shape[0] >= crops[0].shape[0] and img.shape[1] >= crops[0].shape[1]
    assert max(img.shape[:2]) <= 1536, "past where vision APIs downscale"


def test_answers_that_do_not_line_up_with_the_cells_are_refused(monkeypatch):
    """A reply naming fewer cells than the sheet has cannot be aligned. Believing
    part of it is exactly how a number lands on the wrong girl."""
    labels = list(ocr_reader._CELL_LABELS[:ocr_reader.GRID_CELLS])
    _fake_engine(monkeypatch, ['{"A": 24}'])
    got = ocr_reader._gemma_grid_once(ocr_reader._get_engine(), "x", labels, ROSTER)
    assert got is None


def test_a_reply_that_is_not_json_is_refused(monkeypatch):
    labels = list(ocr_reader._CELL_LABELS[:3])
    _fake_engine(monkeypatch, ["sorry, I can't tell"])
    assert ocr_reader._gemma_grid_once(ocr_reader._get_engine(), "x", labels, ROSTER) is None


def test_an_off_roster_number_is_filtered_not_returned(monkeypatch):
    labels = list(ocr_reader._CELL_LABELS[:2])
    _fake_engine(monkeypatch, ['{"A": 99, "B": 24}'])
    got = ocr_reader._gemma_grid_once(ocr_reader._get_engine(), "x", labels, ROSTER)
    assert got == {"A": None, "B": 24}


def test_a_bad_sheet_falls_back_to_one_at_a_time(monkeypatch):
    """Refusing must never mean losing the crops -- they go the slow way."""
    crops = _crops(4)
    box = _fake_engine(monkeypatch, ["not json"] + ["NONE"] * 40)
    out = ocr_reader.read_jersey_batch(crops, ROSTER)
    assert len(out) == len(crops)
    assert box["calls"] > 1, "should have retried the crops individually"


def test_unanimous_across_sheets_confirms(monkeypatch):
    labels = list(ocr_reader._CELL_LABELS[:ocr_reader.GRID_CELLS])
    first = _sheet({"A": "24"}, labels)
    box = _fake_engine(monkeypatch, [first, '{"A":24}', '{"A":24}'])
    out = ocr_reader.read_jersey_batch(_crops(ocr_reader.GRID_CELLS), ROSTER)
    assert out[0] and out[0][0][0] == 24
    assert out[0][0][1] >= ocr_reader.OCR_CONFIRM_THRESHOLD
    assert box["calls"] == ocr_reader.GEMMA_READS


def test_disagreement_across_sheets_does_not_confirm(monkeypatch):
    labels = list(ocr_reader._CELL_LABELS[:ocr_reader.GRID_CELLS])
    first = _sheet({"A": "24"}, labels)
    _fake_engine(monkeypatch, [first, '{"A":13}', '{"A":13}'])
    out = ocr_reader.read_jersey_batch(_crops(ocr_reader.GRID_CELLS), ROSTER)
    assert not (out[0] and out[0][0][1] >= ocr_reader.OCR_CONFIRM_THRESHOLD)


def test_a_cell_that_reads_nothing_costs_one_sheet_not_three(monkeypatch):
    """The whole saving: a cell that cannot be read can never be unanimous, so
    the later sheets carry only the cells that named somebody."""
    labels = list(ocr_reader._CELL_LABELS[:ocr_reader.GRID_CELLS])
    box = _fake_engine(monkeypatch, [_sheet({}, labels)])
    out = ocr_reader.read_jersey_batch(_crops(ocr_reader.GRID_CELLS), ROSTER)
    assert box["calls"] == 1
    assert all(o == [] for o in out)


def test_never_returns_more_answers_than_crops(monkeypatch):
    labels = list(ocr_reader._CELL_LABELS[:5])
    _fake_engine(monkeypatch, [_sheet({"A": "24"}, labels), '{"A":24}', '{"A":24}'])
    out = ocr_reader.read_jersey_batch(_crops(5), ROSTER)
    assert len(out) == 5


def test_confidence_is_agreement_over_the_full_read_budget(monkeypatch):
    """A cell dropped after one read has one vote of three -- 0.33 -- the same
    answer the one-at-a-time path gives for the same evidence."""
    labels = list(ocr_reader._CELL_LABELS[:2])
    _fake_engine(monkeypatch, ['{"A":24,"B":"NONE"}', '{"A":"NONE"}'])
    out = ocr_reader.read_jersey_batch(_crops(2), ROSTER)
    assert out[0] and out[0][0][0] == 24
    assert out[0][0][1] == pytest.approx(1.0 / ocr_reader.GEMMA_READS)
