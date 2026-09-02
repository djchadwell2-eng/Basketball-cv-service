"""establish_via_reads -- letting the JERSEY name a girl nobody clicked.

WHY THIS PATH EXISTS [MEASURED 2026-08-31]. On a game with no human clicks every
identity carries roster_number = None, so promote_via_second_signal returns
"no_position_hypothesis" for every read however good. Full_Game's reader had
read #3, #13, #24 and #33 at confidence 1.00, two of them corroborated on a
second crop, and ALL of it was discarded. A fresh game could not be named at
all, by construction.

These tests exist to pin the REFUSALS, not the successes. The path adds a third
way to reach CONFIRMED, and the whole value of this codebase's identity layer is
that CONFIRMED is hard to reach.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "phase2"))

import identity as idmod  # noqa: E402
from identity import IdentityState  # noqa: E402
from tracking import Track  # noqa: E402

CONF_OK = 0.99
CONF_LOW = 0.50


def _fresh_identity(machine, track_id=1, frame=0):
    machine.update(frame, [Track(track_id, (10, 10, 60, 160))])
    return machine._by_track[track_id]


# ------------------------------------------------------------ it can work ----

def test_two_agreeing_reads_from_different_crops_establish_a_name():
    m = idmod.IdentityStateMachine()
    ident = _fresh_identity(m)
    res = m.establish_via_reads(ident, 24, CONF_OK, corroborating_frames=[100, 260])
    assert res == "established"
    assert ident.state is IdentityState.CONFIRMED
    assert ident.roster_number == 24


def test_it_goes_through_the_confirmation_gate():
    """The gate is the single choke point; this path must not bypass it."""
    m = idmod.IdentityStateMachine()
    ident = _fresh_identity(m)
    m.establish_via_reads(ident, 24, CONF_OK, corroborating_frames=[100, 260])
    assert m.confirmations, "set_confirmed must have emitted an authorization"
    rec = m.confirmations[-1]
    assert rec["provenance"] == "read_established"
    assert rec["roster_number"] == 24, "the number must ride on the gate record"


def test_the_provenance_is_distinguishable_forever():
    """A name that came from a jersey must never be mistaken later for a name a
    human clicked -- that distinction is what makes an audit possible."""
    assert "read_established" in idmod._CONFIRMING_PROVENANCES
    assert "seed" in idmod._CONFIRMING_PROVENANCES
    assert "read_established" != "seed"


# --------------------------------------------------------- the refusals ------

def test_one_crop_is_never_enough():
    """THE MEASURED FAILURE MODE. The reader's two known wrong answers both came
    back UNANIMOUS at 1.00 off a single clipped crop (44 read as 14, 10 read as
    13). Repeating a read on one picture cannot catch that; a second picture
    can."""
    m = idmod.IdentityStateMachine()
    ident = _fresh_identity(m)
    assert m.establish_via_reads(ident, 14, CONF_OK,
                                 corroborating_frames=[100]) == "needs_second_crop"
    assert ident.state is not IdentityState.CONFIRMED


def test_the_same_frame_twice_is_still_one_crop():
    m = idmod.IdentityStateMachine()
    ident = _fresh_identity(m)
    assert m.establish_via_reads(ident, 14, CONF_OK,
                                 corroborating_frames=[100, 100]) == "needs_second_crop"


def test_no_corroboration_at_all_is_refused():
    m = idmod.IdentityStateMachine()
    ident = _fresh_identity(m)
    assert m.establish_via_reads(ident, 14, CONF_OK,
                                 corroborating_frames=None) == "needs_second_crop"


def test_a_low_confidence_read_is_refused():
    m = idmod.IdentityStateMachine()
    ident = _fresh_identity(m)
    assert m.establish_via_reads(ident, 24, CONF_LOW,
                                 corroborating_frames=[1, 2]) == "no_confident_read"
    assert ident.state is not IdentityState.CONFIRMED


def test_it_never_overwrites_a_human_click():
    """A read that disagrees with a click must raise a SWAP FLAG through
    promote_via_second_signal, never quietly win here."""
    m = idmod.IdentityStateMachine()
    ident = _fresh_identity(m)
    ident.roster_number = 13                       # a human said 13
    res = m.establish_via_reads(ident, 24, CONF_OK, corroborating_frames=[1, 90])
    assert res == "already_has_hypothesis"
    assert ident.roster_number == 13, "the human's number must survive untouched"
    assert ident.state is not IdentityState.CONFIRMED


def test_a_relinked_candidate_is_never_established():
    """A CANDIDATE carries frames inherited through continuity from a body
    nobody verified. Naming it would retroactively vouch for history no read
    ever saw -- the same reason seed_labeled_newcomers refuses one."""
    m = idmod.IdentityStateMachine()
    ident = _fresh_identity(m)
    ident.state = IdentityState.CANDIDATE
    assert m.establish_via_reads(ident, 24, CONF_OK,
                                 corroborating_frames=[1, 90]) == "not_fresh"
    assert ident.state is IdentityState.CANDIDATE


# ------------------------------------------- the lock is still a lock --------

def test_continuity_still_has_no_way_to_confirm():
    """The whole safety property. Adding a third provenance must not open a
    fourth."""
    m = idmod.IdentityStateMachine()
    ident = _fresh_identity(m)
    for bad in ("continuity", "relink", "position", "guess", ""):
        try:
            m.set_confirmed(ident, provenance=bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} was allowed to confirm")


def test_a_single_read_still_cannot_confirm_through_the_old_path():
    """promote_via_second_signal is unchanged: with no hypothesis it still
    abstains rather than establishing anything."""
    m = idmod.IdentityStateMachine()
    ident = _fresh_identity(m)
    assert m.promote_via_second_signal(ident, 24, CONF_OK) == "no_position_hypothesis"
    assert ident.state is not IdentityState.CONFIRMED
