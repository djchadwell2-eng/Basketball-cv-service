"""RUN THE REAL CALIBRATION MATH on the FULL_GAME_CHAIN clip (DJ's 2026-07-30
clicks on the verified 4-frame chain: 600 / 127200 / 151200 / 171000).

This is phase1/refit_keyframes.main() -- the same script whose numbers are
quoted everywhere in TEST_LOG.md (TEST 36's "15.45 ft mean / 50.52 max" came
from exactly this). Nothing new is built here; this just points the existing
tool at the new clip entry (clips_config.ACTIVE = "FULL_GAME_CHAIN") instead of
FULL_GAME.

Bar: <= 0.3 ft is "glued" (TEST1's benchmark). 0.94 ft is what DJ called broken
by eye. Above that, do not trust it, no matter how the number is framed.

Usage:  .venv/Scripts/python.exe spikes/run_chain_calibration.py [CLIP_NAME]
        (default FULL_GAME_CHAIN; the second gym is FULL_GAME2)
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "phase1"))

import clips_config                                               # noqa: E402
clips_config.ACTIVE = sys.argv[1] if len(sys.argv) > 1 else "FULL_GAME_CHAIN"

import refit_keyframes                                            # noqa: E402

if __name__ == "__main__":
    refit_keyframes.main()
