"""Roster access -- the CLOSED SET jersey OCR matches against.

The roster values themselves now live in the per-clip ClipConfig (clip_config.py):
per-team numbers + jersey colour, and the hand-verified seed labels (a first signal
independent of the automated OCR reader). This module is a thin accessor over the
ACTIVE_CLIP so the OCR stage keeps its existing calls.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from clip_config import ACTIVE_CLIP

# Closed set of jersey numbers to match against (union of both teams' numbers).
ROSTER_NUMBERS = ACTIVE_CLIP.roster_numbers()


def seed_number_for(clip: str, track_id: int):
    """track_id -> jersey number for the active clip's hand-verified seeds."""
    return ACTIVE_CLIP.seed_labels.get(track_id)
