"""STAGE 5 -- identity-stamped player_events + confirmed-only box score.

Every individual attribution (here: a player's on-floor PRESENCE at a frame) is
stamped with that identity's state AT THAT MOMENT. Uncertainty propagates: while an
identity is CONFIRMED its presence is a confirmed event; after a break, its
recovered presence is a CANDIDATE event; during a LOST gap it is not active, so
nothing is attributed at all. A shaky identity cannot launder into a confident stat.

Box score trusts CONFIRMED only. Candidate/unknown events are SURFACED for review,
never silently counted. The set_confirmed lock is intact; second_signal unbuilt, so
no candidate becomes confirmed here.

(player_events sit on the tracks/identity layer, ON TOP of the identity-free
team_event spine -- team_events are not touched.)
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))                          # repo root (clip_config)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "spikes"))

from clip_config import ACTIVE_CLIP as CLIP
import oncourt
import window_boundaries
import roster
import windows as winmod
from tracking import Track

TRACKS_JSON = CLIP.tracks_cache_path
OUT_JSON = os.path.join(_HERE, "out", f"{CLIP.name}_player_events.json")


def load(path):
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    return [(fr["frame_index"],
             [Track(t["track_id"], tuple(t["bbox"])) for t in fr["tracks"]])
            for fr in doc["frames"]], doc


def spans(frame_states):
    """Compress sorted [(frame, state), ...] into runs of (state, start, end)."""
    out, prev = [], None
    for (f, st) in frame_states:
        if prev and st == prev[0] and f == prev[2] + 1:
            prev[2] = f
        else:
            if prev:
                out.append(tuple(prev))
            prev = [st, f, f]
    if prev:
        out.append(tuple(prev))
    return out       # list of (state, start_frame, end_frame)


def main():
    frames, doc = load(TRACKS_JSON)
    span_start, fps = doc["span_start"], doc["fps"]

    # WINDOWS = court-side cut points, NOT possessions (fixed-window
    # fallback is loud inside). Real possessions: team_possessions.py.
    boundaries, wlabel = window_boundaries.load_windows(CLIP)

    # ROI-MASK SEEDING: only on-court tracks are auto-trusted. Off-court bodies
    # stay UNKNOWN (their events are still stamped honestly; the box score
    # trusts CONFIRMED only, so it now contains on-court players exclusively).
    onc = oncourt.on_court_by_window(oncourt.load_checked(CLIP),
                                     boundaries=boundaries)

    wid = winmod.WindowedIdentity(boundaries=boundaries)
    events = []                         # the identity-stamped player_events
    seen = set()
    prev_win = None
    for (fidx, tracks) in frames:
        win = wid.update(fidx, tracks)
        machine = wid.current_machine()
        if win != prev_win:             # re-seed point -> CONFIRMED (provenance=seed)
            seen = set()
            on = onc.get(win, set())
            n_off = sum(1 for t in tracks if t.track_id not in on)
            refs = roster.ref_tracks()      # the human says these are officials
            refs = refs | roster._spliced()   # two-player tracks: never credited
            for t in tracks:
                if t.track_id in on and t.track_id not in refs:
                    machine.seed(t.track_id, label=f"w{win}_t{t.track_id}")
            print(f"  window {win}: seeded {len(tracks) - n_off} on-court at f={fidx}, "
                  f"skipped {n_off} off-court (ROI mask)")
            prev_win = win
        else:                           # LABELED on-court newcomers seed on arrival
            winmod.seed_labeled_newcomers(
                machine, tracks, seen, onc.get(win, set()),
                lambda tid: roster.seed_number_for(CLIP.name, tid))
        seen |= {t.track_id for t in tracks}
        for ident in machine.active():  # LOST identities are NOT active -> no attribution
            events.append({"window": win, "frame": fidx,
                           "identity_id": ident.identity_id, "track_id": ident.track_id,
                           "event": "on_floor_presence", "identity_state": ident.state.value})

    # --- aggregate per logical identity (window, identity_id) ---
    per = defaultdict(Counter)
    frame_states = defaultdict(list)
    for e in events:
        key = (e["window"], e["identity_id"])
        per[key][e["identity_state"]] += 1
        frame_states[key].append((e["frame"], e["identity_state"]))

    # --- box score: CONFIRMED only; candidate/unknown -> review ---
    box, review = [], []
    for key, c in sorted(per.items()):
        if c["confirmed"] > 0:
            box.append((key, c["confirmed"]))
        if c["candidate"] or c["unknown"]:
            review.append((key, dict(c)))

    counted = sum(c["confirmed"] for c in per.values())
    not_counted = sum(c["candidate"] + c["unknown"] for c in per.values())

    print(f"clip={doc['clip']} span={span_start}..+{doc['span_len']}  "
          f"player_events={len(events)}")
    print(f"\nattributions by state (each event stamped with identity_state):")
    print(f"  {dict(Counter(e['identity_state'] for e in events))}")

    # --- SAMPLE: an identity whose stamp CHANGES (confirmed -> break -> candidate) ---
    changed = [k for k, c in per.items() if c["confirmed"] > 0 and c["candidate"] > 0]
    print(f"\nsample propagation (identity confirmed BEFORE a break, candidate AFTER):")
    if changed:
        key = changed[0]
        print(f"  identity (window {key[0]}, id {key[1]}):")
        prev_end = None
        for (st, a, b) in spans(sorted(frame_states[key])):
            if prev_end is not None and a > prev_end + 1:      # a LOST gap sits here
                print(f"    {'LOST':<9} f={prev_end+1}..{a-1} ({a-1-prev_end:>2}f)   "
                      f"NOT ACTIVE, nothing attributed")
            verdict = "COUNTED in box score" if st == "confirmed" else "-> REVIEW (not counted)"
            print(f"    {st:<9} f={a}..{b} ({b - a + 1:>2}f)   {verdict}")
            prev_end = b
        print("  => the SAME identity's post-break presence is CANDIDATE; it cannot be")
        print("     counted as a confirmed stat. Uncertainty propagated onto the events.")
    else:
        print("  (no confirmed->candidate identity in this span)")

    # --- box score + review summary ---
    print(f"\nBOX SCORE (trusts CONFIRMED only) -- confirmed presence-frames per identity:")
    for (key, n) in box[:12]:
        print(f"  window {key[0]} identity {key[1]}: {n} confirmed frames")
    if len(box) > 12:
        print(f"  ... and {len(box) - 12} more confirmed identities")
    print(f"\ncounted (confirmed) attributions:      {counted}")
    print(f"surfaced for REVIEW (candidate+unknown): {not_counted}  "
          f"({len(review)} identities)")
    print("candidate/unknown events are NEVER silently counted -- a shaky identity "
          "cannot launder into a confident stat.")

    total_conf = sum(1 for k, c in per.items() if c["confirmed"] > 0)
    print(f"\nCONFIRMED identities all originate from seeds (provenance='seed'); "
          f"second_signal unbuilt -> 0 continuity confirmations.")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"clip": doc["clip"], "player_events": events,
                   "box_score_confirmed_frames": [
                       {"window": k[0], "identity_id": k[1], "confirmed_frames": n}
                       for (k, n) in box],
                   "review_items": [
                       {"window": k[0], "identity_id": k[1], "states": c}
                       for (k, c) in review]}, f, indent=2)
    print(f"\nsaved player_events + box score + review -> {OUT_JSON}")


if __name__ == "__main__":
    main()
