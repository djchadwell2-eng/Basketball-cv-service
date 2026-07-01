"""STAGE 3b -- team stats as deterministic reads over the team_event JSON.

ONLY what ~16 sampled frames can honestly support:
  - per-zone occupancy: how many on-court body-positions fell in each zone, over
    the TRUSTED frames
  - mean on-court count per (trusted) frame

NO possessions, NO pace, NO shooting % -- those need dense frames or shot
detection we don't have. NO identity. This module only reads positions.

FRAME TRUST filter: skip any frame whose homography_confidence.state ==
"low_confidence". (The former explicit frame-450 skip was removed: the
keyframe-consistency re-fit + direct anchoring made 450's homography correct, so
it is a normal trusted frame now -- see phase1/DECISIONS.md.)
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))                          # repo root (clip_config)

import team_event_schema as schema
import zones as Z
from clip_config import ACTIVE_CLIP as CLIP

JSON_PATH = os.path.join(_HERE, "out", f"{CLIP.name}_team_events.json")

def load_trusted(path: str):
    """Return (trusted_events, skip_log). skip_log = list of (frame_index, reason)."""
    events, meta = schema.read_events(path)
    trusted, skipped = [], []
    for ev in events:
        if ev.homography_confidence.state == "low_confidence":
            skipped.append((ev.frame_index, "low_confidence homography"))
        else:
            trusted.append(ev)
    return trusted, skipped, meta


def per_zone_occupancy(events) -> dict:
    """Count on-court body-positions per zone across the given frames."""
    counts = {z: 0 for z in Z.ZONES}
    for ev in events:
        for det in ev.detections:
            x, y = det.court_feet
            counts[Z.zone_of(x, y)] += 1
    return counts


def mean_on_court(events) -> float:
    if not events:
        return 0.0
    return sum(len(ev.detections) for ev in events) / len(events)


def main():
    trusted, skipped, meta = load_trusted(JSON_PATH)

    print(f"clip={meta['clip']}  fps={meta['fps']}")
    print(f"\nframes skipped ({len(skipped)}):")
    for (fi, reason) in skipped:
        print(f"  frame {fi}: {reason}")
    print(f"frames used (trusted): {len(trusted)}  "
          f"{[ev.frame_index for ev in trusted]}")

    occ = per_zone_occupancy(trusted)
    total = sum(occ.values())
    print(f"\nper-zone occupancy over {len(trusted)} trusted frames "
          f"({total} body-positions):")
    print(f"  {'zone':<14} {'count':>6} {'share':>7}")
    for z in sorted(occ, key=lambda k: -occ[k]):
        share = (occ[z] / total) if total else 0.0
        print(f"  {z:<14} {occ[z]:>6} {share:>6.1%}")

    print(f"\nmean on-court count per trusted frame: {mean_on_court(trusted):.1f}")
    print("(NOTE: 16 sampled frames -> render/zone proof, not real basketball stats.)")


if __name__ == "__main__":
    main()
