"""STAGE 3 -- per-window (re-seed boundary) containment.

Shows that a lost/candidate/unknown in window N cannot carry into window N+1:
identities reset at each boundary and the lost pool is cleared, so no relink
spans a boundary. Prints, per window, the contained state; and the concrete
containment win -- cross-window relinks that the boundary PREVENTS (found by
comparing to a single-machine run with no boundaries).
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))                          # repo root (clip_config)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "spikes"))

from clip_config import ACTIVE_CLIP as CLIP
import identity as idmod
import windows as winmod
from tracking import Track

TRACKS_JSON = CLIP.tracks_cache_path

# ~15s is the STAND-IN for a real possession boundary (NOT final possession logic).
STANDIN_WINDOW_SECONDS = 15.0
# This validation span is only ~4s, so the DEMO window is shortened to fit >=2
# boundaries and actually exercise the reset. Clearly a demo scaling, not the real value.
DEMO_WINDOW_SECONDS = CLIP.accumulation_window_seconds


def load(path):
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    frames = [(fr["frame_index"],
               [Track(t["track_id"], tuple(t["bbox"])) for t in fr["tracks"]])
              for fr in doc["frames"]]
    return frames, doc


def main():
    frames, doc = load(TRACKS_JSON)
    fps = doc["fps"]
    span_start = doc["span_start"]
    win_frames = int(round(DEMO_WINDOW_SECONDS * fps))

    print(f"clip={doc['clip']} span={span_start}..+{doc['span_len']} fps={fps:.0f}")
    print(f"STAND-IN possession window = {STANDIN_WINDOW_SECONDS:.0f}s "
          f"({int(STANDIN_WINDOW_SECONDS*fps)} frames) -- NOT final possession logic.")
    print(f"DEMO window (this {doc['span_len']/fps:.0f}s span) = {DEMO_WINDOW_SECONDS:.0f}s "
          f"({win_frames} frames) to show >=2 boundaries.")

    def window_of(f):
        return (f - span_start) // win_frames

    # --- (A) single machine, NO boundaries: find relinks that cross a boundary ---
    single = idmod.IdentityStateMachine()
    for (fidx, tracks) in frames:
        single.update(fidx, tracks)
    cross = [b for b in single.breaks if b.get("result") == "candidate"
             and window_of(b["lost_after_frame"]) != window_of(b["reappeared_frame"])]
    print(f"\n[no boundaries] candidate relinks that CROSS a window boundary: {len(cross)}")
    for b in cross:
        print(f"  t{b['lost_track_id']} lost f={b['lost_after_frame']} (win "
              f"{window_of(b['lost_after_frame'])}) -> reappeared f={b['reappeared_frame']} "
              f"(win {window_of(b['reappeared_frame'])}) as t{b['reappeared_track_id']} "
              f"-- would leak identity across the boundary")

    # --- (B) windowed: reset at each boundary ---
    wid = winmod.WindowedIdentity(span_start, win_frames)
    for (fidx, tracks) in frames:
        wid.update(fidx, tracks)

    print("\n[with boundaries] per-window contained state:")
    all_cross_windowed = 0
    for w, machine in sorted(wid.machines().items()):
        tally = Counter(i.state.value for i in machine.all_identities())
        relinks = [b for b in machine.breaks if b.get("result") == "candidate"]
        ambig = [b for b in machine.breaks if str(b.get("result", "")).startswith("unknown")]
        # any relink in this window that references a frame outside it? (must be none)
        leaks = [b for b in relinks
                 if window_of(b["lost_after_frame"]) != w or window_of(b["reappeared_frame"]) != w]
        all_cross_windowed += len(leaks)
        f0 = span_start + w * win_frames
        print(f"  window {w}  frames {f0}..{f0+win_frames-1}:  identities={len(machine.all_identities())} "
              f"states={dict(tally)}  relinks={len(relinks)} ambiguous={len(ambig)} "
              f"confirmed={tally.get('confirmed',0)}  cross-boundary_relinks={len(leaks)}")

    print(f"\nCONTAINMENT: cross-window relinks  without boundaries = {len(cross)}  "
          f"-> with boundaries = {all_cross_windowed}")
    print("Each window's identities are FRESH (own machine, empty lost pool); a "
          "lost/candidate/unknown in window N has no successor in window N+1.")
    total_confirmed = sum(Counter(i.state.value for i in m.all_identities()).get("confirmed", 0)
                          for m in wid.machines().values())
    print(f"CONFIRMED across all windows: {total_confirmed} "
          f"(lock intact -- no seed/second signal yet)")


if __name__ == "__main__":
    main()
