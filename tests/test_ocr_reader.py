"""The jersey reader: agreement across repeated reads IS the confidence score.

From the 2026-08-03 head-to-head (tasks/todo.md): on identical crops the vision
reader named 12 correctly where EasyOCR named 1, and unanimity is what stopped a
REFEREE being named "10" (his crop read [13, 10, 10] -- majority would have
taken it).
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "phase2"))

import numpy as np  # noqa: E402

import ocr_reader  # noqa: E402


# ----------------------------------------- agreement IS the confidence score --

def test_unanimous_reads_clear_the_confirm_threshold():
    """3 of 3 = 1.00, comfortably over OCR_CONFIRM_THRESHOLD."""
    conf = 3 / float(ocr_reader.GEMMA_READS)
    assert conf >= ocr_reader.OCR_CONFIRM_THRESHOLD


def test_a_split_vote_does_NOT_clear_the_threshold():
    """THE REFEREE CASE. A referee crop read [13, 10, 10]; majority-of-3 would
    have named him '10'. 2 of 3 = 0.67, which the existing 0.85 dial rejects on
    its own -- no second threshold needed."""
    conf = 2 / float(ocr_reader.GEMMA_READS)
    assert conf < ocr_reader.OCR_CONFIRM_THRESHOLD


def test_raising_the_read_count_keeps_that_property():
    """4 of 5 = 0.8, still under the bar. Unanimity survives a bigger sample."""
    assert 4 / 5.0 < ocr_reader.OCR_CONFIRM_THRESHOLD
    assert 5 / 5.0 >= ocr_reader.OCR_CONFIRM_THRESHOLD


def test_a_refusal_counts_against_agreement(monkeypatch):
    """Two reads of 24 and one 'cannot tell' is 0.67, not 1.0 -- the crop was
    not legible three times running, so it is not confirmed."""
    calls = iter([24, 24, None])
    monkeypatch.setattr(ocr_reader, "_get_engine", lambda: object())
    monkeypatch.setattr(ocr_reader, "_gemma_once",
                        lambda *_a, **_k: next(calls))
    up = np.zeros((80, 60, 3), dtype=np.uint8)
    got = ocr_reader._read_gemma(up, {24, 13})
    assert got and got[0][0] == 24
    assert got[0][1] < ocr_reader.OCR_CONFIRM_THRESHOLD


def test_all_three_agreeing_is_confirmable(monkeypatch):
    monkeypatch.setattr(ocr_reader, "_get_engine", lambda: object())
    monkeypatch.setattr(ocr_reader, "_gemma_once", lambda *_a, **_k: 24)
    up = np.zeros((80, 60, 3), dtype=np.uint8)
    got = ocr_reader._read_gemma(up, {24, 13})
    assert got == [(24, 1.0)]


def test_three_refusals_return_nothing(monkeypatch):
    monkeypatch.setattr(ocr_reader, "_get_engine", lambda: object())
    monkeypatch.setattr(ocr_reader, "_gemma_once", lambda *_a, **_k: None)
    up = np.zeros((80, 60, 3), dtype=np.uint8)
    assert ocr_reader._read_gemma(up, {24, 13}) == []


def test_an_api_error_is_not_a_vote(monkeypatch):
    """_gemma_once swallows failures into None. A call that errored is not
    evidence, and must never be counted as agreement."""
    monkeypatch.setattr(ocr_reader, "_get_engine", lambda: object())
    seq = iter([24, None, None])          # one good read, two failed calls
    monkeypatch.setattr(ocr_reader, "_gemma_once", lambda *_a, **_k: next(seq))
    up = np.zeros((80, 60, 3), dtype=np.uint8)
    got = ocr_reader._read_gemma(up, {24, 13})
    assert got[0][1] < ocr_reader.OCR_CONFIRM_THRESHOLD


# ------------------------------------------------------------- crop guards --

def test_a_crop_too_short_to_carry_a_number_is_not_attempted():
    tiny = np.zeros((ocr_reader.MIN_CROP_HEIGHT_PX - 1, 40, 3), dtype=np.uint8)
    assert ocr_reader.read_jersey(tiny, {24}) == []


def test_an_empty_crop_is_not_attempted():
    assert ocr_reader.read_jersey(None, {24}) == []
    assert ocr_reader.read_jersey(np.zeros((0, 0, 3), dtype=np.uint8), {24}) == []
