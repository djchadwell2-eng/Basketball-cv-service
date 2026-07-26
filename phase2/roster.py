"""Roster access -- the CLOSED SET jersey OCR matches against.

The roster values themselves now live in the per-clip ClipConfig (clip_config.py):
per-team numbers + jersey colour, and the hand-verified seed labels (a first signal
independent of the automated OCR reader). This module is a thin accessor over the
ACTIVE_CLIP so the OCR stage keeps its existing calls.

CLICK-SEEDING (review bundle): if phase2/out/{clip}_decisions.json exists (the
file the review-bundle HTML downloads after a human labels tracks), its track
labels are merged in -- human decisions OVERRIDE legacy config seed_labels, and
off-roster labels are REFUSED loudly (a typo must never become a hypothesis).
A human click is a seed: it flows through the SAME set_confirmed gate.
"""

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))                          # repo root
from clip_config import ACTIVE_CLIP

# Closed set of jersey numbers to match against (union of both teams' numbers).
ROSTER_NUMBERS = ACTIVE_CLIP.roster_numbers()

DECISIONS_JSON = os.path.join(_HERE, "out", f"{ACTIVE_CLIP.name}_decisions.json")

# Labels that mean "this body is not a player in this footage". Both are
# excluded from the stats; kept apart because they diagnose different things
# (see load_ref_tracks).
NON_PLAYER_LABELS = ("ref", "bench")


def load_decisions(path: str, roster_numbers: set) -> dict:
    """{track_id: number} from a decisions file. Missing file -> {}. Labels
    that are not ints on the roster are REFUSED with a loud warning (never a
    silent bad hypothesis). Non-number labels ('ref', null) are skipped."""
    if not os.path.exists(path):
        return {}
    doc = json.load(open(path, encoding="utf-8"))
    out = {}
    for tid_s, n in doc.get("track_labels", {}).items():
        if n is None or n in NON_PLAYER_LABELS:
            continue
        if not isinstance(n, int) or n not in roster_numbers:
            print(f"  WARNING: decisions label t{tid_s} -> {n!r} is OFF-ROSTER "
                  f"or not a number -- REFUSED (fix the label or the roster)")
            continue
        out[int(tid_s)] = n
    return out


def load_ref_tracks(path: str, kinds=NON_PLAYER_LABELS) -> set:
    """Track ids the human marked as NOT a player in this footage.

    Two distinct kinds, both excluded from the stats but recorded separately
    because they mean different things:
      'ref'   -- an official. Correctly classified as on-court; simply not a
                 player. Nothing to fix upstream.
      'bench' -- a substitute sitting at the sideline. Being on-court AT ALL is
                 an on-court-classifier miss: MARGIN_FT is 1.5 ft of slack for
                 players who step on the line, and a seated sub sits right at
                 it. TEST2 t18/t21 were credited 8.7s and 7.9s of "floor time"
                 from the bench. So the count of these is a bug signal, not
                 just a labelling chore.

    These labels were being read and thrown away, which wasted the one signal
    that can keep officials out of the player stats. stage4 seeds EVERY
    on-court track (a ref is on-court by the Phase-1 rule), so referees become
    CONFIRMED identities with real floor time: measured 41.5s on HARD and
    40.2s on TEST1 -- and it is why TEST1 reported MORE confirmed player-time
    than there are player slots. A ref stands in the paint all possession, so
    one counted as a player invents exactly the positional tendency the
    product sells. Excluding them ADDS information; it never lowers a
    confidence bar.
    """
    if not os.path.exists(path):
        return set()
    doc = json.load(open(path, encoding="utf-8"))
    return {int(tid) for tid, n in doc.get("track_labels", {}).items()
            if n in kinds}


_decisions_cache = None
_spliced_cache = None
_ref_cache = None
_splice_warned: set = set()


def ref_tracks() -> set:
    """Track ids the human marked 'ref', cached. Empty set if never labelled."""
    global _ref_cache
    if _ref_cache is None:
        _ref_cache = load_ref_tracks(DECISIONS_JSON)
        if _ref_cache:
            print(f"  non-players: {len(_ref_cache)} track(s) the human marked "
                  f"ref/bench -- excluded from seeding, so they cannot become "
                  f"players")
    return _ref_cache


def _decisions() -> dict:
    global _decisions_cache
    if _decisions_cache is None:
        _decisions_cache = load_decisions(DECISIONS_JSON, ROSTER_NUMBERS)
        if _decisions_cache:
            print(f"  click-seeding: loaded {len(_decisions_cache)} human track "
                  f"labels from {os.path.basename(DECISIONS_JSON)}")
    return _decisions_cache


def _spliced() -> set:
    global _spliced_cache
    if _spliced_cache is None:
        import purity
        _spliced_cache = purity.spliced_tracks(ACTIVE_CLIP)
        if _spliced_cache:
            print(f"  purity: {len(_spliced_cache)} SPLICED track(s) -- their "
                  f"labels will be refused (wrong-player risk)")
    return _spliced_cache


def resolve_label(track_id: int, decisions: dict, config_labels: dict,
                  spliced: set):
    """(number | None, reason). A spliced track's label is REFUSED: the track
    followed two different players, so any single number would credit the
    wrong one for part of the span."""
    label = decisions.get(track_id, config_labels.get(track_id))
    if label is None:
        return None, "unlabeled"
    if track_id in spliced:
        return None, "refused_spliced"
    return label, "labeled"


def seed_number_for(clip: str, track_id: int):
    """track_id -> jersey number. Human decisions first, then legacy config
    seed labels; labels on SPLICED tracks (purity check) are refused loudly."""
    n, reason = resolve_label(track_id, _decisions(), ACTIVE_CLIP.seed_labels,
                              _spliced())
    if reason == "refused_spliced" and track_id not in _splice_warned:
        print(f"  PURITY: t{track_id} is SPLICED (two different numbers read "
              f"on one track) -- label REFUSED; its time stays unnamed")
        _splice_warned.add(track_id)
    return n
