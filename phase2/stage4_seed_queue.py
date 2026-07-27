"""STAGE 4 -- seeding (post-process, re-seed-on-loss) + coach review queue.

Seeding happens AFTER upload, not live. Model = re-seed at each window start:
establish identity for the tracks present (a stand-in for the coach clicking each
on-court player), track until a break, recover what continuity confidently gives
(CANDIDATE), and surface the rest. Seeding is the FIRST legitimate path to
CONFIRMED (provenance='seed') -- the first green, and ONLY from an explicit seed,
never from continuity.

Because NO second signal (jersey OCR) exists yet, every CANDIDATE and UNKNOWN goes
to the coach review queue as a one-click item. The queue is LONG by design: OCR
(the next step) is what shrinks it by safely auto-promoting candidates. We do NOT
shrink it by loosening the candidate bar or auto-confirming continuity.

SEED POOL (ROI mask): only tracks standing ON the court at the window start are
seeded -- per-window majority over the cached on/off classification
(phase2/oncourt.py, built once per clip by cache_oncourt.py, reusing the
validated Phase-1 court rules). Off-court bodies (crowd/bench) stay UNKNOWN and
are excluded from the coach queue but always counted + recorded. Refs are
on-court by the Phase-1 rule, so they are seeded (no roster number -> they end
up in review, honestly). A coach click UI remains the eventual real seeder.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))                          # repo root (clip_config)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "spikes"))

from clip_config import ACTIVE_CLIP as CLIP
import oncourt
import possessions
import roster
import windows as winmod
from identity import IdentityState
from tracking import Track

TRACKS_JSON = CLIP.tracks_cache_path
OUT_DIR = os.path.join(_HERE, "out")
QUEUE_JSON = os.path.join(OUT_DIR, f"{CLIP.name}_review_queue.json")


def load(path):
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    frames = [(fr["frame_index"],
               [Track(t["track_id"], tuple(t["bbox"])) for t in fr["tracks"]])
              for fr in doc["frames"]]
    return frames, doc


def flagged_frame(ident):
    if ident.state == IdentityState.CANDIDATE:
        return ident.evidence.get("reappeared_frame", ident.last_seen_frame)
    return ident._hist[0][0] if ident._hist else ident.last_seen_frame


def why(ident):
    ev = ident.evidence
    if ident.state == IdentityState.CANDIDATE:
        return (f"candidate: reappeared after {ev.get('gap_frames','?')}f occlusion, "
                f"{ev.get('distance_px','?')}px from motion prediction -- NOT "
                f"second-signal-verified")
    return f"unknown: {ev.get('reason','unidentified')}"


def main():
    frames, doc = load(TRACKS_JSON)
    span_start, fps = doc["span_start"], doc["fps"]

    # WINDOWS = detected possessions (fixed-window fallback is loud inside).
    boundaries, wlabel = possessions.load_windows(CLIP)

    # ROI-MASK SEEDING: only tracks standing ON the court (per-window majority
    # over the cached classification) are auto-trusted with a seed. Off-court
    # bodies (crowd/bench) stay UNKNOWN -- counted below, never silently used.
    onc = oncourt.on_court_by_window(oncourt.load_checked(CLIP),
                                     boundaries=boundaries)

    wid = winmod.WindowedIdentity(boundaries=boundaries)
    seed_frames = {}                 # window -> (frame, [seeded], [skipped off-court])
    late_seeded = defaultdict(list)  # window -> [labeled tracks seeded on first appearance]
    seen = set()
    prev_win = None
    for (fidx, tracks) in frames:
        win = wid.update(fidx, tracks)
        machine = wid.current_machine()
        if win != prev_win:          # window start = the (re-)seed point
            seen = set()
            on = onc.get(win, set())
            seeded, skipped = [], []
            refs = roster.ref_tracks()
            refs = refs | roster._spliced()   # two-player tracks: never credited
            for t in tracks:
                if t.track_id not in on:          # off-court -> NOT auto-trusted
                    skipped.append(t.track_id)
                    continue
                if t.track_id in refs:            # the human says this is an official
                    skipped.append(t.track_id)    # -- never a player, never seeded
                    continue
                machine.seed(t.track_id, label=f"w{win}_t{t.track_id}")   # -> CONFIRMED
                seeded.append(t.track_id)
            seed_frames[win] = (fidx, seeded, skipped)
            prev_win = win
        else:                        # LABELED on-court newcomers seed on arrival
            late_seeded[win] += winmod.seed_labeled_newcomers(
                machine, tracks, seen, onc.get(win, set()),
                lambda tid: roster.seed_number_for(CLIP.name, tid))
        seen |= {t.track_id for t in tracks}

    # --- per-window state + build the review queue (on-court only) ---
    queue = []
    off_court_excluded = 0
    print(f"clip={doc['clip']} span={span_start}..+{doc['span_len']}  "
          f"windows: {wlabel}")
    for w, machine in sorted(wid.machines().items()):
        tally = Counter(i.state.value for i in machine.all_identities())
        sf, seeded, skipped = seed_frames[w]
        print(f"\nwindow {w}: seeded {len(seeded)} ON-COURT tracks at f={sf} -> CONFIRMED "
              f"(provenance='seed'); skipped {len(skipped)} off-court (ROI mask); "
              f"+{len(late_seeded[w])} LABELED newcomers seeded on arrival")
        print(f"  final states: {dict(tally)}")
        on = onc.get(w, set())
        for ident in machine.all_identities():
            # EVER unresolved, not just unresolved at the END of the run: an
            # identity that was CANDIDATE for hundreds of frames and then died
            # is LOST by now, and so was never put in front of the coach --
            # 2290 track-frames, ~40% of HARD's player pool, silently skipped.
            if ident.ever_unresolved and ident.state is not IdentityState.CONFIRMED:
                if ident.track_id not in on:      # crowd/bench: keep OUT of the
                    off_court_excluded += 1       # coach's queue, but COUNT it
                    continue
                queue.append({"window": w, "frame": flagged_frame(ident),
                              "track": ident.track_id, "state": ident.state.value,
                              "why": why(ident)})

    queue.sort(key=lambda q: (q["window"], q["frame"] or 0, q["track"] or 0))

    # --- confirmed came ONLY from seeds (safety) ---
    total_conf = sum(Counter(i.state.value for i in m.all_identities()).get("confirmed", 0)
                     for m in wid.machines().values())
    total_seeded = sum(len(s) for (_f, s, _sk) in seed_frames.values())
    print(f"\nCONFIRMED total = {total_conf} (all via provenance='seed'; seeds issued "
          f"= {total_seeded}). No continuity confirmation. second_signal still unbuilt.")

    # --- the review queue ---
    print(f"\n================ COACH REVIEW QUEUE ================")
    print(f"items needing one click: {len(queue)}")
    print("(window, frame, track, state, why)")
    for q in queue[:20]:
        print(f"  w{q['window']} f={q['frame']} t{q['track']} [{q['state']}]  {q['why']}")
    if len(queue) > 20:
        print(f"  ... and {len(queue)-20} more")
    n_cand = sum(1 for q in queue if q["state"] == "candidate")
    n_unk = sum(1 for q in queue if q["state"] == "unknown")
    print(f"\nqueue = {n_cand} candidates + {n_unk} unknowns (ON-COURT bodies only).")
    print(f"off-court candidates/unknowns EXCLUDED from the coach queue: "
          f"{off_court_excluded}  (crowd/bench -- counted, recorded, never seeded)")
    print("The on-court queue shrinks via OCR auto-promotes only -- never by")
    print("loosening the candidate bar.")

    with open(QUEUE_JSON, "w", encoding="utf-8") as f:
        json.dump({"clip": doc["clip"], "windows": wlabel,
                   "window_boundaries": boundaries,
                   "seed_policy": "ROI-mask: on-court majority per (window, track)",
                   "off_court_excluded": off_court_excluded,
                   "items": queue}, f, indent=2)
    print(f"\nsaved review queue -> {QUEUE_JSON}")

    # --- a still per window start: seeded GREEN, off-court skipped RED ---
    # (the ROI-mask eyeball artifact: crowd/bench should be red, players green)
    cap = cv2.VideoCapture(CLIP.video_path)
    for w, (sf, seeded, skipped) in seed_frames.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for _ in range(sf):
            cap.grab()
        ok, frame = cap.read()
        if not ok:
            continue
        _fi, seed_tracks = frames[sf - span_start]
        seeded_set = set(seeded)
        for t in seed_tracks:
            x1, y1, x2, y2 = [int(v) for v in t.bbox]
            if t.track_id in seeded_set:
                col, lbl = (0, 255, 0), f"t{t.track_id}:confirmed(seed)"
            else:
                col, lbl = (0, 0, 255), f"t{t.track_id}:OFF-COURT (not seeded)"
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            cv2.putText(frame, lbl, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        cv2.putText(frame, f"WINDOW {w} SEED f={sf}: {len(seeded)} on-court seeded "
                           f"(green), {len(skipped)} off-court skipped (red)",
                    (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite(os.path.join(OUT_DIR, f"{CLIP.name}_stage4_seed_w{w}_f{sf}.jpg"),
                    cv2.resize(frame, (1280, 720)))
    cap.release()
    print(f"saved seed stills (green=seeded on-court, red=skipped off-court) in {OUT_DIR}")


if __name__ == "__main__":
    main()
