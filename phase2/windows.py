"""Per-window identity containment -- the re-seed boundary.

A break (lost / candidate / unknown) inside one window must NOT be able to corrupt
the next window. We enforce that structurally: at every window boundary we RESET
identity -- a fresh IdentityStateMachine, with an empty lost pool -- so nothing
relinks across a boundary and every window is (re-)seeded independently.

WINDOW = ~15s is a STAND-IN for real possession boundaries (possession detection is
a later step); it is explicitly NOT final possession logic. The confirmation lock
(identity.set_confirmed) is untouched: this stage is about containment, not
confirmation, so there is still no path to CONFIRMED without a seed/second signal.
"""

from __future__ import annotations

from bisect import bisect_right

import identity as idmod


class WindowedIdentity:
    """Runs a separate IdentityStateMachine per window. Windows are either
    fixed-length (span_start + window_frames, the original stand-in) or a
    sorted list of start-frame `boundaries` (detected possessions)."""

    def __init__(self, span_start: int | None = None,
                 window_frames: int | None = None,
                 boundaries: list | None = None):
        self.boundaries = sorted(boundaries) if boundaries is not None else None
        self.span_start = span_start
        self.window_frames = window_frames
        self._machines: dict[int, idmod.IdentityStateMachine] = {}
        self._cur_win: int | None = None
        self._machine: idmod.IdentityStateMachine | None = None

    def window_of(self, frame_index: int) -> int:
        if self.boundaries is not None:
            return max(0, bisect_right(self.boundaries, frame_index) - 1)
        return (frame_index - self.span_start) // self.window_frames

    def update(self, frame_index: int, tracks) -> int:
        win = self.window_of(frame_index)
        if win != self._cur_win:                 # boundary -> RESET (fresh machine)
            self._machine = idmod.IdentityStateMachine()
            self._machines[win] = self._machine
            self._cur_win = win
        self._machine.update(frame_index, tracks)
        return win

    def current_machine(self) -> idmod.IdentityStateMachine:
        return self._machine

    def machines(self) -> dict[int, idmod.IdentityStateMachine]:
        return dict(self._machines)


def seed_labeled_newcomers(machine, tracks, seen: set, on_set: set, label_fn):
    """Mid-window seeding for LABELED tracks at their FIRST appearance.

    A human label vouches for the TRACK itself (the body ByteTrack follows
    under that id), so it is a legitimate first signal at any time -- not only
    at a window boundary. Guard: only a FRESH identity (state UNKNOWN, no
    relink history) may be late-seeded. A relinked CANDIDATE carries frames
    inherited from a lost identity via continuity; confirming it on a track
    label would retroactively vouch for history the human never saw -- it must
    earn confirmation via OCR or the review queue. Same seed() gate as always.

    Returns the list of track_ids seeded; caller maintains `seen` per window."""
    out = []
    for t in tracks:
        if t.track_id in seen or t.track_id not in on_set:
            continue
        n = label_fn(t.track_id)
        if n is None:
            continue
        ident = machine._by_track.get(t.track_id)
        if ident is None or ident.state is not idmod.IdentityState.UNKNOWN:
            continue                      # relinked CANDIDATE: never late-seeded
        machine.seed(t.track_id, roster_number=n, label=f"late_t{t.track_id}")
        out.append(t.track_id)
    return out
