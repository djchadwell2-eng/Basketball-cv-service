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

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class IdentityState(str, Enum):
    CONFIRMED = "confirmed"
    LOST = "lost"
    CANDIDATE = "candidate"
    UNKNOWN = "unknown"


# The only provenances allowed to reach CONFIRMED. Continuity is deliberately absent.
_CONFIRMING_PROVENANCES = frozenset({"seed", "second_signal"})

# Continuity-relink thresholds (produce CANDIDATE only, never CONFIRMED).
MAX_GAP_FRAMES = 30        # a lost track can be relinked within ~1s (30fps)
MAX_MATCH_DIST_PX = 150    # reappearance must be near the motion prediction
AMBIG_RATIO = 1.5          # best match must be this much closer than the 2nd-best
HISTORY = 5                # frames of centre history used for the velocity estimate


def _center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass
class Identity:
    """A logical identity following one (or, across relinks, several) track_ids."""
    identity_id: int
    state: IdentityState = IdentityState.UNKNOWN
    track_id: int | None = None            # the ByteTrack id it currently follows
    evidence: dict = field(default_factory=dict)   # why it is in its current state
    roster_number: int | None = None       # jersey number from the seed (position hypothesis)
    last_bbox: tuple | None = None
    last_seen_frame: int | None = None
    # centre history (frame, cx, cy) for the motion model used in relinking.
    _hist: deque = field(default_factory=lambda: deque(maxlen=HISTORY))

    def push(self, bbox, frame):
        c = _center(bbox)
        self._hist.append((frame, c[0], c[1]))
        self.last_bbox = tuple(bbox)
        self.last_seen_frame = frame

    def last_center(self):
        return (self._hist[-1][1], self._hist[-1][2]) if self._hist else _center(self.last_bbox)

    def velocity(self):
        """Per-frame centre velocity from recent history (0,0 if too short)."""
        if len(self._hist) < 2:
            return (0.0, 0.0)
        f0, x0, y0 = self._hist[0]
        f1, x1, y1 = self._hist[-1]
        dt = max(1, f1 - f0)
        return ((x1 - x0) / dt, (y1 - y0) / dt)

    def predicted_center(self, gap):
        cx, cy = self.last_center()
        vx, vy = self.velocity()
        return (cx + vx * gap, cy + vy * gap)


class IdentityStateMachine:
    """Owns the identities and the ONLY transitions allowed between states."""

    def __init__(self):
        self._by_track: dict[int, Identity] = {}   # active track_id -> Identity
        self._lost: list[Identity] = []            # occluded, awaiting reappearance
        self._all: list[Identity] = []
        self.breaks: list[dict] = []               # log of occlusion/relink events
        # Every CONFIRMED transition, emitted by the gate itself. This is the
        # ONLY authorization source for retroactive stat merges: a merge may
        # consume these records and nothing else (never position scans).
        self.confirmations: list[dict] = []
        self._next_id = 0

    # -- creation -------------------------------------------------------------
    def _new_identity(self, track_id: int, bbox, frame: int, reason: str) -> Identity:
        ident = Identity(identity_id=self._next_id, track_id=track_id,
                         state=IdentityState.UNKNOWN,          # abstain until seeded
                         evidence={"reason": reason})
        ident.push(bbox, frame)
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
        self.confirmations.append({                # gate-emitted merge authorization
            "identity_id": ident.identity_id, "track_id": ident.track_id,
            "provenance": provenance, "roster_number": ident.roster_number})

    def seed(self, track_id: int, *, roster_number: int | None = None,
             label: str | None = None) -> bool:
        """Human seed in post-processing. A real (first) signal. May carry the
        player's jersey number, which position-continuity then keeps through breaks
        so a later OCR read can AGREE/DISAGREE with it.

        Returns True if seeded. A track_id with no active track is a LOUD no-op
        (returns False) -- a typo'd or stale hand-verified seed label must never
        disappear silently."""
        ident = self._by_track.get(track_id)
        if ident is None:
            print(f"  WARNING: seed for track_id {track_id} ignored -- no active "
                  f"track with that id at the seed frame (typo or stale label?)")
            return False
        ident.roster_number = roster_number        # before confirming, so the
        self.set_confirmed(ident, provenance="seed")   # gate record carries it
        if label is not None:
            ident.evidence["label"] = label
        return True

    def promote_via_second_signal(self, ident: Identity, number, confidence) -> str:
        """SANCTIONED second-signal path (jersey OCR). Applies the three outcomes to
        the accumulated best on-roster read (`number`, `confidence`). AGREE is the
        only NEW path to CONFIRMED; it goes through set_confirmed so the lock holds.
        Returns: 'agree' | 'disagree' | 'no_confident_read' | 'no_position_hypothesis'.
        """
        from ocr_reader import OCR_CONFIRM_THRESHOLD          # the single autonomy dial
        # Outcome 3: no confident on-roster read -> stay CANDIDATE (honest uncertainty).
        if number is None or confidence is None or confidence < OCR_CONFIRM_THRESHOLD:
            return "no_confident_read"
        expected = ident.roster_number
        if expected is None:
            return "no_position_hypothesis"     # nothing to agree with -> stay CANDIDATE
        if number == expected:                  # Outcome 1: AGREE -> confirm
            self.set_confirmed(ident, provenance="second_signal")
            ident.roster_number = number
            ident.evidence = {"reason": "OCR agrees with position",
                              "jersey": number, "confidence": round(confidence, 2)}
            return "agree"
        # Outcome 2: DISAGREE -> swap detector. NEVER silently resolve.
        ident.state = IdentityState.UNKNOWN
        ident.evidence = {"reason": "position/jersey disagreement -- possible swap",
                          "position_says": expected, "ocr_read": number,
                          "confidence": round(confidence, 2)}
        return "disagree"

    # -- per-frame update (Stage 2: honest loss + candidate recovery) --------
    def update(self, frame_index: int, tracks) -> None:
        """Advance one frame:
          - a track_id that vanished -> its identity goes LOST (attribute nothing);
          - a continuing track_id keeps its state (a CANDIDATE stays CANDIDATE --
            continuity NEVER promotes);
          - a brand-new track_id is either a continuity RELINK of a recently-lost
            identity -> CANDIDATE (with evidence), or, if none/ambiguous, a fresh
            UNKNOWN. No path here reaches CONFIRMED.
        """
        current = {t.track_id for t in tracks}

        # 1. disappearances -> LOST (no attribution during the gap)
        for tid in list(self._by_track):
            if tid not in current:
                ident = self._by_track.pop(tid)
                ident.state = IdentityState.LOST
                ident.evidence = {"reason": "occlusion",
                                  "lost_after_frame": ident.last_seen_frame,
                                  "lost_track_id": tid}
                self._lost.append(ident)

        # 2. continuing + new tracks
        for t in tracks:
            ident = self._by_track.get(t.track_id)
            if ident is not None:
                ident.push(t.bbox, frame_index)         # state unchanged (no promo)
                continue
            self._handle_reappearance(t, frame_index)

    def _handle_reappearance(self, t, frame_index):
        matches = self._match_lost(t, frame_index)      # [(ident, evidence), ...]
        if matches and self._is_confident(matches):
            ident, ev = matches[0]
            self._lost.remove(ident)
            old_track = ident.track_id
            ident.track_id = t.track_id
            ident.state = IdentityState.CANDIDATE       # the CEILING for continuity
            ev["reappeared_frame"] = frame_index
            ident.evidence = ev
            ident.push(t.bbox, frame_index)
            self._by_track[t.track_id] = ident
            self.breaks.append({
                "identity_id": ident.identity_id, "lost_track_id": old_track,
                "reappeared_track_id": t.track_id,
                "lost_after_frame": ev["lost_after_frame"],
                "reappeared_frame": frame_index, "gap_frames": ev["gap_frames"],
                "distance_px": ev["distance_px"],
                "reappeared_at": ev["reappeared_center"], "result": "candidate"})
        else:
            reason = ("ambiguous reappearance (>=2 lost tracks contend)"
                      if matches else "new track, not yet identified")
            ident = self._new_identity(t.track_id, t.bbox, frame_index, reason)
            if matches:                                 # ambiguous -> stays UNKNOWN
                ident.evidence["contenders"] = len(matches)
                self.breaks.append({
                    "identity_id": ident.identity_id, "reappeared_track_id": t.track_id,
                    "reappeared_frame": frame_index, "result": "unknown (ambiguous)",
                    "contenders": len(matches)})

    def _match_lost(self, t, frame_index):
        """Lost identities that plausibly relink to this reappearance, by motion."""
        cx, cy = _center(t.bbox)
        out = []
        for ident in self._lost:
            gap = frame_index - ident.last_seen_frame
            if gap <= 0 or gap > MAX_GAP_FRAMES:
                continue
            px, py = ident.predicted_center(gap)
            dist = math.hypot(cx - px, cy - py)
            dist_last = math.hypot(cx - ident.last_center()[0],
                                   cy - ident.last_center()[1])
            if min(dist, dist_last) > MAX_MATCH_DIST_PX:
                continue
            out.append((ident, {
                "reason": "position/motion continuity relink",
                "gap_frames": gap, "distance_px": round(dist, 1),
                "distance_from_last_px": round(dist_last, 1),
                "predicted_center": [round(px, 1), round(py, 1)],
                "reappeared_center": [round(cx, 1), round(cy, 1)],
                "lost_after_frame": ident.last_seen_frame}))
        out.sort(key=lambda m: m[1]["distance_px"])
        return out

    @staticmethod
    def _is_confident(matches):
        """One clear match, or a best clearly closer than the runner-up. A confident
        continuity match is still only CANDIDATE -- this gates candidate vs unknown."""
        if len(matches) == 1:
            return True
        return matches[0][1]["distance_px"] * AMBIG_RATIO <= matches[1][1]["distance_px"]

    # -- read -----------------------------------------------------------------
    def active(self) -> list[Identity]:
        return list(self._by_track.values())

    def lost(self) -> list[Identity]:
        return list(self._lost)

    def all_identities(self) -> list[Identity]:
        return list(self._all)
