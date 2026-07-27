"""SAFETY-PROPERTY regression tests for the identity state machine.

These are the IRON-RULE proofs from phase2/identity.py, runnable in seconds on
every change (this suite replaces the broken demo proof in phase2/stage1_states.py).
No video, no YOLO, no OCR engine -- synthetic tracks drive the real machine.

The property under test, stated once: the system must NEVER silently attribute
identity to the wrong player. Concretely:
  * CONFIRMED is reachable ONLY via set_confirmed(provenance in {seed, second_signal}).
  * Position/motion continuity can produce at most CANDIDATE.
  * LOST attributes nothing; ambiguous reappearances abstain (UNKNOWN).
  * A window boundary is a hard wall: no relink crosses it.
  * OCR promotes only on AGREE (position hypothesis + confident on-roster read);
    DISAGREE flags and demotes; a lone confident read confirms nothing.

If you edit phase2/identity.py or phase2/windows.py and any test here fails,
you have broken the safety property, not the test.

Run:  .venv/Scripts/python -m pytest tests/ -v
"""

import json
import os
import sys
from types import SimpleNamespace

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "phase2"))

import identity as idmod                      # noqa: E402
import windows as winmod                      # noqa: E402
from identity import IdentityState            # noqa: E402
from ocr_reader import OCR_CONFIRM_THRESHOLD  # noqa: E402

CONF_OK = OCR_CONFIRM_THRESHOLD               # >= threshold counts as confident
CONF_LOW = OCR_CONFIRM_THRESHOLD - 0.2


def T(tid, cx, cy, w=20.0, h=40.0):
    """A synthetic track: something with .track_id and .bbox centered at (cx, cy)."""
    return SimpleNamespace(track_id=tid,
                           bbox=(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))


def walk(machine, frames):
    """Drive the machine over [(frame_index, [tracks]), ...]."""
    for fidx, tracks in frames:
        machine.update(fidx, tracks)
    return machine


def make_candidate(roster_number=None):
    """The canonical break: a (optionally seeded) track is lost, then reappears
    cleanly under the SAME track id -> the same Identity relinked as CANDIDATE.

    Same-id is the only continuity relink that still exists. Relinking onto a
    DIFFERENT track id was removed 2026-07-25 after measurement: it overruled
    ByteTrack on ~150px of proximity alone and was wrong about 70% of the time
    (HARD 10/10, TEST1 2/2 wrong), merging up to three different players into
    one identity. Same-id relinks measured 52/52 correct. The safety property is
    unchanged and in fact stricter -- a different-id reappearance now abstains
    to UNKNOWN, which attributes even less than CANDIDATE did. Locked by
    test_relink_onto_a_different_track_id_is_refused below."""
    m = idmod.IdentityStateMachine()
    m.update(0, [T(1, 100, 100)])
    if roster_number is not None:
        m.seed(1, roster_number=roster_number)
    # steady motion so the relink has a real velocity prediction
    walk(m, [(f, [T(1, 100 + 10 * f, 100)]) for f in range(1, 5)])
    walk(m, [(5, []), (6, [])])                      # occlusion -> LOST
    m.update(7, [T(1, 170, 100)])                    # reappear, SAME track id
    ident = m.active()[0]
    return m, ident


# ---------------------------------------------------------------------------
# 1. Abstention baseline + the confirmation lock itself
# ---------------------------------------------------------------------------

def test_new_tracks_are_unknown():
    m = idmod.IdentityStateMachine()
    m.update(0, [T(1, 100, 100), T(2, 300, 200)])
    states = {i.state for i in m.active()}
    assert states == {IdentityState.UNKNOWN}, "before anyone is identified, claim nothing"


@pytest.mark.parametrize("provenance", [
    "position_continuity", "continuity", "motion", "relink", "", "SEED"])
def test_continuity_provenance_refused(provenance):
    """THE lock: set_confirmed rejects everything outside {seed, second_signal}."""
    m = idmod.IdentityStateMachine()
    m.update(0, [T(1, 100, 100)])
    ident = m.active()[0]
    with pytest.raises(ValueError):
        m.set_confirmed(ident, provenance=provenance)
    assert ident.state is IdentityState.UNKNOWN, "a refused confirm must change nothing"


def test_seed_confirms_with_seed_provenance():
    m = idmod.IdentityStateMachine()
    m.update(0, [T(1, 100, 100)])
    assert m.seed(1, roster_number=13, label="w0_t1") is True
    ident = m.active()[0]
    assert ident.state is IdentityState.CONFIRMED
    assert ident.roster_number == 13
    assert "seed" in ident.evidence["reason"]


def test_seed_unknown_track_id_is_loud_and_confirms_nothing(capsys):
    """A typo'd/stale seed label must warn, not vanish silently."""
    m = idmod.IdentityStateMachine()
    m.update(0, [T(1, 100, 100)])
    assert m.seed(999, roster_number=13) is False
    assert "WARNING" in capsys.readouterr().out
    assert all(i.state is not IdentityState.CONFIRMED for i in m.all_identities())


# ---------------------------------------------------------------------------
# 2. Honest loss and recovery: LOST attributes nothing; continuity's ceiling
#    is CANDIDATE; ambiguity abstains
# ---------------------------------------------------------------------------

def test_occlusion_goes_lost_and_attributes_nothing():
    m = idmod.IdentityStateMachine()
    walk(m, [(f, [T(1, 100, 100)]) for f in range(5)])
    m.update(5, [])                                   # the body vanishes
    assert m.active() == [], "a LOST identity must not be attributable"
    lost = m.lost()
    assert len(lost) == 1 and lost[0].state is IdentityState.LOST
    assert lost[0].evidence["reason"] == "occlusion"


def test_relink_is_candidate_never_confirmed():
    """A clean reappearance exactly where motion predicts -- the strongest
    possible continuity evidence -- may still only reach CANDIDATE."""
    m, ident = make_candidate()
    assert ident.state is IdentityState.CANDIDATE, (
        "continuity's ceiling is CANDIDATE; CONFIRMED here = the silent-swap bug")
    assert ident.track_id == 1                        # same track, re-acquired
    assert ident.evidence["gap_frames"] == 3
    assert "distance_px" in ident.evidence
    assert m.breaks and m.breaks[0]["result"] == "candidate"


def test_candidate_stays_candidate_under_continuity():
    m, ident = make_candidate()
    walk(m, [(f, [T(1, 170 + 10 * (f - 7), 100)]) for f in range(8, 14)])
    assert ident.state is IdentityState.CANDIDATE, "continuity NEVER promotes"


def test_ambiguous_reappearance_stays_unknown():
    """Two players vanish and one body reappears between them -> abstain, and
    claim NEITHER lost identity. This is the exact scenario that used to invent
    a merge, so the abstention outcome is the property that matters.

    (The old build reached this outcome via an 'ambiguous, >=2 contenders' path.
    A body reappearing under a new track id can no longer contend for anyone's
    identity at all, so it is simply a fresh UNKNOWN -- the same abstention,
    reached earlier and more cheaply.)"""
    m = idmod.IdentityStateMachine()
    walk(m, [(f, [T(1, 100, 100), T(2, 160, 100)]) for f in range(3)])
    m.update(3, [])                                   # both vanish
    m.update(4, [T(9, 130, 100)])                     # dead center between them
    ident = m.active()[0]
    assert ident.state is IdentityState.UNKNOWN
    assert ident.roster_number is None, "must not inherit anyone's number"
    assert len(m.lost()) == 2, "neither lost identity may be claimed"


def test_relink_onto_a_different_track_id_is_refused():
    """A body appearing under a NEW track id -- however close, however clean the
    motion prediction -- may not be relinked onto a lost identity.

    Removed 2026-07-25 on measurement: these guesses overruled ByteTrack, which
    had already declined the association, on proximity alone. Judged against the
    human's own track labels they were wrong ~70% of the time (HARD 10/10,
    TEST1 2/2), because a body is lost precisely WHEN it is in a crowd, so the
    body that reappears nearby is usually the player who crowded it. One HARD
    identity walked through tracks the human labelled #20, #23 and #44 while
    carrying #20 throughout -- 47.2s credited to the wrong player, now 0.0s."""
    m = idmod.IdentityStateMachine()
    m.update(0, [T(1, 100, 100)])
    m.seed(1, roster_number=23)
    walk(m, [(f, [T(1, 100 + 10 * f, 100)]) for f in range(1, 5)])
    walk(m, [(5, []), (6, [])])                       # lost
    m.update(7, [T(2, 170, 100)])                     # PERFECT prediction, new id

    fresh = m.active()[0]
    assert fresh.state is IdentityState.UNKNOWN, "a new track id is a new person"
    assert fresh.roster_number is None, (
        "inheriting #23 here is the silent-swap bug: it credits one player's "
        "stats to whoever happened to reappear nearby")
    assert m.breaks == [], "no relink may be recorded across track ids"
    assert len(m.lost()) == 1, "the real #23 stays honestly LOST"


def test_refusing_a_bad_relink_lets_the_human_label_land():
    """Why refusing raises coverage instead of lowering it. A relinked CANDIDATE
    can never be late-seeded (windows.seed_labeled_newcomers refuses inherited
    history), so a wrong relink also BLOCKED the coach's own click. Left as a
    fresh UNKNOWN, the label applies -- which is why 'your clicks used' rose
    from 46.9% to 55.2% on HARD while wrong-player time fell to zero."""
    m = idmod.IdentityStateMachine()
    m.update(0, [T(1, 100, 100)])
    walk(m, [(f, [T(1, 100 + 10 * f, 100)]) for f in range(1, 5)])
    walk(m, [(5, []), (6, [])])
    m.update(7, [T(2, 170, 100)])                     # new body, new id

    seeded = winmod.seed_labeled_newcomers(
        m, [T(2, 170, 100)], seen=set(), on_set={2}, label_fn={2: 44}.get)
    assert seeded == [2], "a fresh UNKNOWN must accept the human's label"
    assert m._by_track[2].state is IdentityState.CONFIRMED
    assert m._by_track[2].roster_number == 44


def test_ever_unresolved_survives_the_identity_dying():
    """The review queue and OCR ask 'was this EVER unresolved', not 'is it
    unresolved now'. An identity that spent its whole life as CANDIDATE and then
    vanished is LOST at the end -- under the old final-state test it was offered
    to neither the coach nor OCR, hiding ~40% of HARD's player time."""
    m, ident = make_candidate()
    assert ident.state is IdentityState.CANDIDATE and ident.ever_unresolved
    m.update(20, [])                                  # dies without ever resolving
    assert ident.state is IdentityState.LOST
    assert ident.ever_unresolved, "dying is not evidence that nobody should look"


def test_gap_too_long_is_a_fresh_unknown():
    m = idmod.IdentityStateMachine()
    walk(m, [(f, [T(1, 100, 100)]) for f in range(3)])
    m.update(3, [])
    m.update(3 + idmod.MAX_GAP_FRAMES + 10, [T(1, 100, 100)])   # way past the window
    ident = m.active()[0]
    assert ident.state is IdentityState.UNKNOWN
    assert m.breaks == [], "no relink may be recorded for an expired gap"
    assert len(m.lost()) == 1, "the old identity stays honestly LOST"


# ---------------------------------------------------------------------------
# 3. The OCR second signal: three outcomes + the no-lone-signal rule
# ---------------------------------------------------------------------------

def test_ocr_no_confident_read_stays_candidate():
    m, ident = make_candidate(roster_number=13)
    assert m.promote_via_second_signal(ident, None, None) == "no_confident_read"
    assert ident.state is IdentityState.CANDIDATE
    assert m.promote_via_second_signal(ident, 13, CONF_LOW) == "no_confident_read"
    assert ident.state is IdentityState.CANDIDATE, "a weak read must not confirm"


def test_ocr_confident_read_alone_does_not_confirm():
    """No position hypothesis -> even a perfect read abstains (two signals rule)."""
    m, ident = make_candidate(roster_number=None)
    assert ident.roster_number is None
    res = m.promote_via_second_signal(ident, 13, 0.99)
    assert res == "no_position_hypothesis"
    assert ident.state is IdentityState.CANDIDATE


def test_ocr_agree_confirms_via_second_signal():
    """The flagship designed flow: seed -> CONFIRMED -> occlusion -> CANDIDATE
    (roster hypothesis SURVIVES the break) -> OCR agrees -> CONFIRMED again."""
    m, ident = make_candidate(roster_number=13)
    assert ident.state is IdentityState.CANDIDATE
    assert ident.roster_number == 13, "the position hypothesis must survive the break"
    res = m.promote_via_second_signal(ident, 13, CONF_OK)
    assert res == "agree"
    assert ident.state is IdentityState.CONFIRMED
    assert ident.evidence["jersey"] == 13


def test_ocr_disagree_flags_and_never_reattributes():
    m, ident = make_candidate(roster_number=13)
    res = m.promote_via_second_signal(ident, 5, CONF_OK)
    assert res == "disagree"
    assert ident.state is IdentityState.UNKNOWN, "disagreement = swap flag, not a pick"
    assert ident.evidence["position_says"] == 13
    assert ident.evidence["ocr_read"] == 5


# ---------------------------------------------------------------------------
# 4. Window containment: a boundary is a wall for relinks
# ---------------------------------------------------------------------------

def test_window_boundary_blocks_cross_window_relink():
    """The same break that relinks inside one machine must NOT relink when it
    spans a window boundary (fresh machine, empty lost pool)."""
    frames = [(f, [T(1, 100, 100)]) for f in range(9)]      # window 0: frames 0..9
    frames += [(9, []), (10, []), (11, [])]                 # lost near the boundary
    frames += [(12, [T(1, 100, 100)])]                      # reappears in window 1

    # Control: ONE machine, no boundaries -> this relinks (gap 4, distance 0).
    single = walk(idmod.IdentityStateMachine(), frames)
    assert [b["result"] for b in single.breaks] == ["candidate"], (
        "control failed: this scenario should relink without a boundary")

    # With the boundary at frame 10: window 1 starts fresh -> no relink possible.
    wid = winmod.WindowedIdentity(span_start=0, window_frames=10)
    for fidx, tracks in frames:
        wid.update(fidx, tracks)
    win1 = wid.machines()[1]
    ident = win1.active()[0]
    assert ident.state is IdentityState.UNKNOWN, (
        "cross-window relink detected: the re-seed boundary must contain breaks")
    assert all(b.get("result") != "candidate" for b in win1.breaks)
    assert len(win1.lost()) == 0, "window 1 must start with an empty lost pool"
    assert len(wid.machines()[0].lost()) == 1, "window 0 keeps its own LOST honestly"


# ---------------------------------------------------------------------------
# 4b. Late seeding: a label seeds a FRESH track any time; NEVER a relink
# ---------------------------------------------------------------------------

def test_labeled_newcomer_is_seeded_mid_window():
    m = idmod.IdentityStateMachine()
    m.update(0, [T(1, 100, 100)])
    m.update(5, [T(1, 100, 100), T(38, 400, 200)])       # t38 appears fresh
    seeded = winmod.seed_labeled_newcomers(
        m, [T(1, 100, 100), T(38, 400, 200)], seen={1},
        on_set={1, 38}, label_fn={38: 44}.get)
    assert seeded == [38]
    ident = m._by_track[38]
    assert ident.state is IdentityState.CONFIRMED and ident.roster_number == 44


def test_relinked_candidate_is_never_late_seeded():
    """The label vouches for the track; a relinked CANDIDATE carries inherited
    continuity history the human never saw -- OCR/queue must resolve it."""
    m, ident = make_candidate()                          # relinked, same track id
    seeded = winmod.seed_labeled_newcomers(
        m, [T(1, 170, 100)], seen=set(), on_set={1}, label_fn={1: 44}.get)
    assert seeded == []
    assert ident.state is IdentityState.CANDIDATE, "continuity history is not vouched"


def test_unlabeled_and_off_court_newcomers_stay_unknown():
    m = idmod.IdentityStateMachine()
    m.update(0, [T(9, 100, 100), T(10, 300, 100)])
    seeded = winmod.seed_labeled_newcomers(
        m, [T(9, 100, 100), T(10, 300, 100)], seen=set(),
        on_set={9}, label_fn={10: 5}.get)                # 9 unlabeled; 10 off-court
    assert seeded == []
    assert all(i.state is IdentityState.UNKNOWN for i in m.active())


# ---------------------------------------------------------------------------
# 5. Vocabulary guard: every identity always carries one of the four states
# ---------------------------------------------------------------------------

def test_states_are_always_the_four_enum_members():
    m, _ = make_candidate(roster_number=13)
    for ident in m.all_identities():
        assert ident.state in (IdentityState.CONFIRMED, IdentityState.LOST,
                               IdentityState.CANDIDATE, IdentityState.UNKNOWN)


# ---------------------------------------------------------------------------
# 6. The COACH-DECLARED splice (2026-07-27). A track that follows two different
#    girls can never carry a number: whichever one you pick, the other's play is
#    credited to her. purity.py is supposed to catch this and does not on real
#    footage -- TEST2 t8 and t137 both walk from a black jersey to a white one
#    and purity scores them "consistent" -- so the coach's eye is a third
#    signal, with the same consequence as an automatic splice verdict.
# ---------------------------------------------------------------------------
def test_a_spliced_track_can_never_carry_a_number(tmp_path, monkeypatch):
    import roster
    label_file = tmp_path / "X_decisions.json"
    label_file.write_text(json.dumps({
        "track_labels": {"7": "spliced", "9": 23, "5": "ref", "6": "bench"}}),
        encoding="utf-8")
    monkeypatch.setattr(roster, "DECISIONS_JSON", str(label_file))

    # a non-number label never becomes a jersey hypothesis
    got = roster.load_decisions(str(label_file), {23, 44})
    assert got == {9: 23}, "only the real number survives as a hypothesis"

    # and the coach's splice is quarantined exactly like an automatic one
    assert roster.human_spliced_tracks() == {7}
    assert roster.resolve_label(7, {7: 23}, {}, spliced={7}) == (None, "refused_spliced")
    # an ordinary track is unaffected
    assert roster.resolve_label(9, {9: 23}, {}, spliced={7}) == (23, "labeled")


def test_ref_and_bench_are_excluded_but_splices_are_not_non_players(tmp_path, monkeypatch):
    """Refs/bench are NOT players, so they leave the readable denominator.
    A spliced track IS real player time we simply cannot attribute -- it must
    stay countable, or the coverage number flatters itself by deleting the
    footage it failed on."""
    import roster
    f = tmp_path / "X_decisions.json"
    f.write_text(json.dumps({"track_labels": {"5": "ref", "6": "bench",
                                              "7": "spliced"}}), encoding="utf-8")
    assert roster.load_ref_tracks(str(f)) == {5, 6}, "spliced is not a non-player"
