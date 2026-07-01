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

import identity as idmod


class WindowedIdentity:
    """Runs a separate IdentityStateMachine per fixed-length window."""

    def __init__(self, span_start: int, window_frames: int):
        self.span_start = span_start
        self.window_frames = window_frames
        self._machines: dict[int, idmod.IdentityStateMachine] = {}
        self._cur_win: int | None = None
        self._machine: idmod.IdentityStateMachine | None = None

    def window_of(self, frame_index: int) -> int:
        return (frame_index - self.span_start) // self.window_frames

    def update(self, frame_index: int, tracks) -> int:
        win = self.window_of(frame_index)
        if win != self._cur_win:                 # boundary -> RESET (fresh machine)
            self._machine = idmod.IdentityStateMachine()
            self._machines[win] = self._machine
            self._cur_win = win
        self._machine.update(frame_index, tracks)
        return win

    def machines(self) -> dict[int, idmod.IdentityStateMachine]:
        return dict(self._machines)
