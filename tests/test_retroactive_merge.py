"""Retroactive stat merge -- safety-property tests (written BEFORE the merge).

The merge re-credits a candidate's provisional span to a player ONLY when the
OCR second signal AGREED (gate-emitted authorization). The failure this suite
exists to prevent: silently re-crediting the wrong player's span -- so:

  * merge consumes ONLY agree records; position-plausible candidates without
    an agree record are NEVER touched (never-on-reappearance, by construction);
  * only candidate-stamped events restamp (LOST gaps are never invented;
    unknown events never restamp);
  * a span overlapping frames already credited to the SAME number in the same
    window is a CONTRADICTION: no merge, loud flag;
  * live 'confirmed' and 'confirmed_retroactive' stay distinguishable forever;
  * the input document is never mutated.
"""

import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "phase2"))

from stage7_merge import merge_events  # noqa: E402


def ev(window, ident, frame, state):
    return {"window": window, "frame": frame, "identity_id": ident,
            "track_id": ident + 100, "event": "on_floor_presence",
            "identity_state": state}


def events_doc(rows):
    return {"clip": "T", "player_events": rows}


def ocr_doc(agrees=(), identities=()):
    return {"clip": "T",
            "outcomes": {"agree": list(agrees), "disagree": [],
                         "no_confident_read": [], "no_position_hypothesis": []},
            "identities": list(identities)}


def agree(window, ident, number, conf=0.99, frame=10):
    return {"window": window, "identity_id": ident, "read_number": number,
            "read_confidence": conf, "read_frame": frame}


def ident_row(window, ident, number, state="candidate"):
    return {"window": window, "identity_id": ident, "track_id": ident + 100,
            "roster_number": number, "final_state": state}


# --- the happy path -----------------------------------------------------------

def test_agree_restamps_candidate_span_only():
    rows = [ev(0, 5, f, "confirmed") for f in range(100, 105)]          # pre-break
    rows += [ev(0, 5, f, "candidate") for f in range(110, 115)]         # post-break
    doc = merge_events(events_doc(rows),
                       ocr_doc([agree(0, 5, 13)], [ident_row(0, 5, 13)]))
    out = doc["player_events"]
    assert all(e["identity_state"] == "confirmed"
               for e in out if e["frame"] < 105), "live-confirmed must stay 'confirmed'"
    retro = [e for e in out if e["identity_state"] == "confirmed_retroactive"]
    assert {e["frame"] for e in retro} == set(range(110, 115))
    assert all(e["merge"]["number"] == 13 for e in retro)
    assert len(doc["merges"]) == 1 and doc["merges"][0]["frames_restamped"] == 5
    assert doc["contradictions"] == []


def test_lost_gap_is_never_invented():
    rows = [ev(0, 5, f, "confirmed") for f in range(100, 105)]
    rows += [ev(0, 5, f, "candidate") for f in range(110, 115)]         # gap 105..109
    doc = merge_events(events_doc(rows),
                       ocr_doc([agree(0, 5, 13)], [ident_row(0, 5, 13)]))
    frames = {e["frame"] for e in doc["player_events"] if e["identity_id"] == 5}
    assert frames & set(range(105, 110)) == set(), \
        "LOST frames attributed nothing before the merge and must stay that way"


# --- the property the whole layer exists for ----------------------------------

def test_candidate_without_agree_is_never_touched():
    """A position-plausible candidate with NO agree record must stay candidate.
    The merge takes no position input at all -- there is nothing to 'match'."""
    rows = [ev(0, 7, f, "candidate") for f in range(200, 210)]
    doc = merge_events(events_doc(rows), ocr_doc([], [ident_row(0, 7, 24)]))
    assert all(e["identity_state"] == "candidate" for e in doc["player_events"])
    assert doc["merges"] == []


def test_unknown_events_never_restamp():
    rows = [ev(0, 9, f, "unknown") for f in range(300, 305)]
    rows += [ev(0, 5, f, "candidate") for f in range(300, 305)]
    doc = merge_events(events_doc(rows),
                       ocr_doc([agree(0, 5, 13)], [ident_row(0, 5, 13),
                                                   ident_row(0, 9, None, "unknown")]))
    assert all(e["identity_state"] == "unknown"
               for e in doc["player_events"] if e["identity_id"] == 9)


# --- contradictions ------------------------------------------------------------

def test_contradiction_blocks_merge_and_flags():
    """#13 already has live-confirmed presence on the same frames the candidate
    claims -- one player cannot be in two places; flag, do not merge."""
    rows = [ev(0, 1, f, "confirmed") for f in range(110, 120)]          # Y: #13 live
    rows += [ev(0, 5, f, "candidate") for f in range(112, 118)]         # X overlaps
    doc = merge_events(events_doc(rows),
                       ocr_doc([agree(0, 5, 13)],
                               [ident_row(0, 1, 13, "confirmed"),
                                ident_row(0, 5, 13)]))
    assert doc["merges"] == []
    assert len(doc["contradictions"]) == 1
    c = doc["contradictions"][0]
    assert c["number"] == 13 and c["n_overlap"] == 6
    assert all(e["identity_state"] == "candidate"
               for e in doc["player_events"] if e["identity_id"] == 5), \
        "a contradicted span must stay candidate (review), never merged"


def test_disjoint_same_number_merges_fine():
    rows = [ev(0, 1, f, "confirmed") for f in range(100, 105)]          # Y: #13 early
    rows += [ev(0, 5, f, "candidate") for f in range(110, 115)]         # X later
    doc = merge_events(events_doc(rows),
                       ocr_doc([agree(0, 5, 13)],
                               [ident_row(0, 1, 13, "confirmed"),
                                ident_row(0, 5, 13)]))
    assert len(doc["merges"]) == 1 and doc["contradictions"] == []


def test_second_agree_colliding_with_first_retro_span_contradicts():
    rows = [ev(0, 5, f, "candidate") for f in range(110, 115)]
    rows += [ev(0, 8, f, "candidate") for f in range(112, 118)]         # overlaps 5's
    doc = merge_events(events_doc(rows),
                       ocr_doc([agree(0, 5, 13), agree(0, 8, 13)],
                               [ident_row(0, 5, 13), ident_row(0, 8, 13)]))
    assert len(doc["merges"]) == 1, "first agree merges"
    assert len(doc["contradictions"]) == 1, "second collides with the retro span"


# --- hygiene --------------------------------------------------------------------

def test_windows_are_isolated():
    rows = [ev(0, 1, f, "confirmed") for f in range(110, 120)]          # #13 in w0
    rows += [ev(1, 5, f, "candidate") for f in range(112, 118)]         # w1 overlap ok
    doc = merge_events(events_doc(rows),
                       ocr_doc([agree(1, 5, 13)],
                               [ident_row(0, 1, 13, "confirmed"),
                                ident_row(1, 5, 13)]))
    assert len(doc["merges"]) == 1 and doc["contradictions"] == []


def test_original_document_is_not_mutated():
    rows = [ev(0, 5, f, "candidate") for f in range(110, 115)]
    src = events_doc(rows)
    merge_events(src, ocr_doc([agree(0, 5, 13)], [ident_row(0, 5, 13)]))
    assert all(e["identity_state"] == "candidate" for e in src["player_events"])
    assert "merges" not in src


def test_agree_for_unknown_identity_fails_loud():
    doc = events_doc([ev(0, 1, 100, "confirmed")])
    with pytest.raises(SystemExit):
        merge_events(doc, ocr_doc([agree(0, 99, 13)], [ident_row(0, 99, 13)]))


# --- queue-resolution v2: HUMAN resolutions ride the same machinery -----------

def hres(window, ident, number):
    return {"window": window, "identity_id": ident, "number": number}


def test_human_resolution_restamps_candidate_and_unknown_span():
    rows = [ev(0, 7, f, "candidate") for f in range(200, 210)]
    rows += [ev(0, 7, f, "unknown") for f in range(190, 195)]   # its own pre-span
    doc = merge_events(events_doc(rows), ocr_doc([], [ident_row(0, 7, None)]),
                       human_resolutions=[hres(0, 7, 44)])
    out = [e for e in doc["player_events"] if e["identity_id"] == 7]
    assert all(e["identity_state"] == "confirmed_retroactive" for e in out), \
        "the human vouched for the identity's whole span"
    assert all(e["merge"]["source"] == "human" for e in out)
    assert doc["merges"][0]["source"] == "human"


def test_human_resolution_hits_the_same_contradiction_check():
    rows = [ev(0, 1, f, "confirmed") for f in range(100, 120)]  # #44 live
    rows += [ev(0, 7, f, "candidate") for f in range(110, 118)]  # overlaps
    doc = merge_events(events_doc(rows),
                       ocr_doc([], [ident_row(0, 1, 44, "confirmed"),
                                    ident_row(0, 7, None)]),
                       human_resolutions=[hres(0, 7, 44)])
    assert doc["merges"] == [] and len(doc["contradictions"]) == 1, \
        "a human can be wrong; the ledger defends itself the same way"


def test_ocr_and_human_conflict_is_flagged_never_resolved():
    rows = [ev(0, 5, f, "candidate") for f in range(100, 120)]
    doc = merge_events(events_doc(rows),
                       ocr_doc([agree(0, 5, 13)], [ident_row(0, 5, 13)]),
                       human_resolutions=[hres(0, 5, 44)])
    assert len(doc["merges"]) == 1 and doc["merges"][0]["number"] == 13
    assert len(doc["conflicts"]) == 1, "OCR said 13, human said 44 -> flag"
    assert doc["conflicts"][0]["ocr_number"] == 13
    assert doc["conflicts"][0]["human_number"] == 44


def test_human_resolution_for_missing_identity_fails_loud():
    with pytest.raises(SystemExit):
        merge_events(events_doc([ev(0, 1, 100, "confirmed")]),
                     ocr_doc([], [ident_row(0, 1, 5, "confirmed")]),
                     human_resolutions=[hres(0, 99, 13)])


def test_reject_is_recorded_and_never_credited():
    rows = [ev(0, 7, f, "candidate") for f in range(200, 210)]
    doc = merge_events(events_doc(rows), ocr_doc([], [ident_row(0, 7, None)]),
                       human_resolutions=[hres(0, 7, "reject")])
    assert all(e["identity_state"] == "candidate" for e in doc["player_events"])
    assert doc["rejected_by_review"] == [{"window": 0, "identity_id": 7}]


# --- the gate emits the authorization records (identity.py side) ---------------

def test_gate_emits_confirmation_records():
    import identity as idmod
    from types import SimpleNamespace
    m = idmod.IdentityStateMachine()
    m.update(0, [SimpleNamespace(track_id=1, bbox=(0, 0, 20, 40))])
    m.seed(1, roster_number=13)
    assert m.confirmations == [{"identity_id": 0, "track_id": 1,
                                "provenance": "seed", "roster_number": 13}]
    ident = m.active()[0]
    with pytest.raises(ValueError):
        m.set_confirmed(ident, provenance="position_continuity")
    assert len(m.confirmations) == 1, "a refused confirm must emit nothing"
