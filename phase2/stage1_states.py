"""STAGE 1 -- drive the identity state machine over cached tracks; print states.

Reads the raw tracks (phase2/run_tracking.py output), runs the state machine, and
prints the identity STATE of every track at every frame for a sample span. In
Stage 1 nothing is seeded and no occlusion logic runs yet, so every track is
UNKNOWN -- that is the correct abstention-first baseline: before we identify
anyone, we claim nothing. (Loss/candidate arrive in Stage 2; confirmed only via
seeding in Stage 4.)

Also prints a proof that the IRON RULE holds: confirming via continuity is refused.
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "spikes"))

import clips_config as cfg
import identity as idmod
from tracking import Track

TRACKS_JSON = os.path.join(_HERE, "out", f"{cfg.ACTIVE}_tracks_raw.json")
SAMPLE_SPAN = 12          # how many frames of per-track state to print


def load_tracks(path):
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    frames = []
    for fr in doc["frames"]:
        tracks = [Track(t["track_id"], tuple(t["bbox"])) for t in fr["tracks"]]
        frames.append((fr["frame_index"], tracks))
    return frames, doc


def main():
    frames, doc = load_tracks(TRACKS_JSON)
    print(f"clip={doc['clip']} span={doc['span_start']}..+{doc['span_len']} "
          f"frames={len(frames)}")

    machine = idmod.IdentityStateMachine()
    per_frame_states = []
    for (fidx, tracks) in frames:
        machine.update(fidx, tracks)
        states = {t.track_id: machine._by_track[t.track_id].state.value for t in tracks}
        per_frame_states.append((fidx, states))

    # --- print a sample span: track states per frame ---
    print(f"\nper-frame identity states (first {SAMPLE_SPAN} frames of the span):")
    print("  (every track UNKNOWN = abstain; nobody identified yet -- correct)")
    for (fidx, states) in per_frame_states[:SAMPLE_SPAN]:
        cells = "  ".join(f"t{tid}:{st}" for tid, st in sorted(states.items()))
        print(f"  f={fidx}  n={len(states):>2}  {cells}")

    # --- state tally over the whole span ---
    from collections import Counter
    tally = Counter(st for (_f, states) in per_frame_states for st in states.values())
    n_ids = len(machine.all_identities())
    print(f"\ntrack-frames by state over the span: {dict(tally)}")
    print(f"distinct identities seen: {n_ids}  (all UNKNOWN -- abstention baseline)")

    # --- IRON RULE proof: continuity cannot confirm ---
    print("\nIRON RULE check (silent promotion must be impossible):")
    ident = machine.all_identities()[0]
    try:
        machine.set_confirmed(ident, provenance="position_continuity")
        print("  *** FAIL: continuity was allowed to confirm ***")
    except ValueError as e:
        print(f"  set_confirmed(provenance='position_continuity') -> REFUSED ({str(e)[:60]}...)")
    try:
        machine.promote_via_second_signal(ident)
        print("  *** FAIL: second-signal promotion is implemented but must not be ***")
    except NotImplementedError:
        print("  promote_via_second_signal() -> NotImplementedError (seam, unbuilt) OK")
    print("  => no code path promotes candidate->confirmed in this build.")


if __name__ == "__main__":
    main()
