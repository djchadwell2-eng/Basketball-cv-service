"""Track-purity classification -- two confident DIFFERENT numbers on one
track means the tracker jumped players (spliced); quarantine, never label."""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "phase2"))

from purity import classify_reads  # noqa: E402

TH = 0.85


def test_two_confident_numbers_is_spliced_with_interval():
    reads = [(100, 44, 0.99), (140, 44, 0.95), (300, 13, 0.98)]
    v = classify_reads(reads, TH)
    assert v["verdict"] == "spliced"
    assert set(v["numbers"]) == {44, 13}
    assert v["splice_interval"] == [140, 300]      # last #44 .. first #13
    assert v["interleaved"] is False


def test_interleaved_numbers_still_spliced():
    reads = [(100, 44, 0.99), (200, 13, 0.98), (250, 44, 0.97)]
    v = classify_reads(reads, TH)
    assert v["verdict"] == "spliced"
    assert v["interleaved"] is True


def test_one_number_is_consistent():
    reads = [(100, 24, 0.99), (300, 24, 0.91)]
    v = classify_reads(reads, TH)
    assert v["verdict"] == "consistent"
    assert v["numbers"][24]["reads"] == 2
    assert v["numbers"][24]["max_conf"] == 0.99


def test_low_confidence_reads_carry_no_verdict_weight():
    reads = [(100, 44, 0.99), (300, 13, 0.5)]     # the 13 is below the bar
    v = classify_reads(reads, TH)
    assert v["verdict"] == "consistent", "a weak read must not convict a track"


def test_no_reads_is_no_evidence_not_purity():
    v = classify_reads([], TH)
    assert v["verdict"] == "no_evidence", \
        "absence of reads is absence of evidence, never a clean bill"
