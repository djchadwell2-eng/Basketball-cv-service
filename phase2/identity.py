"""Abstention-first identity STATE MACHINE (the safety-critical core).

SAFETY PROPERTY (the whole point): this system must NEVER silently attribute a
stat to the wrong player. A spike proved position-continuity ALONE is not safe --
one normal single occlusion produced a silent swap. So here, silent promotion to
a confident identity is made STRUCTURALLY IMPOSSIBLE until a SECOND SIGNAL (jersey
OCR, a later step) exists.

The four states every track carries every frame:
  CONFIRMED : identity certain. Reachable ONLY via a human seed, or later a second
              signal. NEVER via position/motion continuity.
  LOST      : occlusion in progress. NO stat is attributed during this span.
  CANDIDATE : reappeared and position/motion says probably-the-same, but NOT yet
              second-signal-verified. The strongest that continuity alone can reach.
  UNKNOWN   : mass occlusion, ambiguous, or simply not-yet-identified. Abstain;
              flag for coach review.

IRON RULE, enforced in code (see set_confirmed): the ONLY ways into CONFIRMED are
`seed()` and `promote_via_second_signal()`. Continuity code cannot reach CONFIRMED
-- it can produce at most CANDIDATE. promote_via_second_signal() is an unimplemented
seam for the next step; it is never called now, so no candidate can become confirmed
in this build. That is deliberate, not incomplete.

This module holds NO detection and NO court math -- just the state logic, so the
safety-critical part is small and auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class IdentityState(str, Enum):
    CONFIRMED = "confirmed"
    LOST = "lost"
    CANDIDATE = "candidate"
    UNKNOWN = "unknown"


# The only provenances allowed to reach CONFIRMED. Continuity is deliberately absent.
_CONFIRMING_PROVENANCES = frozenset({"seed", "second_signal"})


@dataclass
class Identity:
    """A logical identity following one (or, across relinks, several) track_ids."""
    identity_id: int
    state: IdentityState = IdentityState.UNKNOWN
    track_id: int | None = None            # the ByteTrack id it currently follows
    evidence: dict = field(default_factory=dict)   # why it is in its current state
    last_bbox: tuple | None = None
    last_seen_frame: int | None = None


class IdentityStateMachine:
    """Owns the identities and the ONLY transitions allowed between states."""

    def __init__(self):
        self._by_track: dict[int, Identity] = {}   # active track_id -> Identity
        self._all: list[Identity] = []
        self._next_id = 0

    # -- creation -------------------------------------------------------------
    def _new_identity(self, track_id: int, bbox, frame: int) -> Identity:
        ident = Identity(identity_id=self._next_id, track_id=track_id,
                         state=IdentityState.UNKNOWN,          # abstain until seeded
                         evidence={"reason": "new track, not yet identified"},
                         last_bbox=tuple(bbox), last_seen_frame=frame)
        self._next_id += 1
        self._by_track[track_id] = ident
        self._all.append(ident)
        return ident

    # -- the guarded transition INTO confirmed (IRON RULE lives here) ---------
    def set_confirmed(self, ident: Identity, *, provenance: str) -> None:
        """Move an identity to CONFIRMED. Rejected unless provenance is a real
        second signal (seed / OCR). This is the single choke point that makes
        silent promotion impossible: continuity code has no valid provenance."""
        if provenance not in _CONFIRMING_PROVENANCES:
            raise ValueError(
                f"REFUSED to confirm identity {ident.identity_id} via "
                f"'{provenance}'. CONFIRMED is reachable only by {sorted(_CONFIRMING_PROVENANCES)}. "
                "Position/motion continuity must never confirm (silent-swap risk).")
        ident.state = IdentityState.CONFIRMED
        ident.evidence = {"reason": f"confirmed via {provenance}"}

    def seed(self, track_id: int, *, label: str | None = None) -> None:
        """Human seed in post-processing (Stage 4). A real second signal."""
        ident = self._by_track.get(track_id)
        if ident is not None:
            self.set_confirmed(ident, provenance="seed")
            if label is not None:
                ident.evidence["label"] = label

    def promote_via_second_signal(self, ident: "Identity") -> None:
        """SEAM for the next step (jersey OCR / re-ID). When a second signal agrees
        with a CANDIDATE, it will confirm here via set_confirmed(provenance=
        'second_signal'). UNIMPLEMENTED and never called in this build -- its
        absence is exactly why no candidate can become confirmed yet."""
        raise NotImplementedError(
            "Second signal (jersey OCR) not built yet -- see Phase 2 next step. "
            "Until it exists, candidates stay candidates by design.")

    # -- per-frame update (Stage 1: base states only) ------------------------
    def update(self, frame_index: int, tracks) -> None:
        """Advance one frame. Stage 1 assigns the abstaining base state (UNKNOWN)
        to every active track and keeps it. Loss/candidate transitions arrive in
        Stage 2; confirmation arrives via seed()/second signal only."""
        for t in tracks:
            ident = self._by_track.get(t.track_id)
            if ident is None:
                ident = self._new_identity(t.track_id, t.bbox, frame_index)
            ident.last_bbox = tuple(t.bbox)
            ident.last_seen_frame = frame_index

    # -- read -----------------------------------------------------------------
    def active(self) -> list[Identity]:
        return list(self._by_track.values())

    def all_identities(self) -> list[Identity]:
        return list(self._all)
