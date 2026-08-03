"""OCR SECOND SIGNAL -- temporal accumulation + strict auto-confirm.

For each CANDIDATE identity, attempt jersey reads across MULTIPLE frames of its
possession window (most frames NO-READ -- expected), accumulate the best on-roster
read, and apply the three outcomes to it:
  AGREE (read == position's number, >= OCR_CONFIRM_THRESHOLD) -> CONFIRMED (2nd signal)
  DISAGREE (confident, DIFFERENT on-roster number)            -> flag swap (UNKNOWN)
  NO CONFIDENT READ across the whole window                   -> stay CANDIDATE (review)

Measures per-FRAME vs per-POSSESSION readability (the gap is the argument for
accumulating across the window). set_confirmed lock intact: confirmed only via
{seed, second_signal}, never continuity.
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))                          # repo root (clip_config)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "spikes"))

from clip_config import ACTIVE_CLIP as CLIP
import ocr_reader
import oncourt
import window_boundaries
import roster
import stage2_multikeyframe as s2mk    # Phase 6: iter_frames (targeted streaming)
import windows as winmod
from identity import IdentityState
from tracking import Track

TRACKS_JSON = CLIP.tracks_cache_path
OUT_DIR = os.path.join(_HERE, "out")
OUT_JSON = os.path.join(OUT_DIR, f"{CLIP.name}_ocr_confirms.json")   # persisted outcomes
MIN_OCR_HEIGHT = 90      # only attempt OCR on player boxes >= this tall (else unreadable)
OCR_STRIDE = 2           # subsample the window's frames (CPU OCR is slow)
MAX_ATTEMPTS = 10        # cap reads per candidate


def load(path):
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    return [(fr["frame_index"],
             [Track(t["track_id"], tuple(t["bbox"])) for t in fr["tracks"]])
            for fr in doc["frames"]], doc


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    frames, doc = load(TRACKS_JSON)
    span_start, fps, clip = doc["span_start"], doc["fps"], doc["clip"]
    # Phase 6: NO whole-span image load here. Which frames are actually
    # needed depends only on track bboxes (computed below from active_log),
    # not on the images themselves -- so frame selection happens FIRST, and
    # only the frames actually picked for an OCR attempt get read. This
    # dict scales with (candidates x MAX_ATTEMPTS), never with clip length.

    # WINDOWS = court-side cut points, NOT possessions (fixed-window
    # fallback is loud inside). Real possessions: team_possessions.py.
    boundaries, wlabel = window_boundaries.load_windows(CLIP)

    # ROI-MASK: on-court majority per (window, track). Seeds, OCR attempts, and
    # the review queue are scoped to ON-COURT bodies; off-court exclusions are
    # counted and persisted below (never silent).
    onc = oncourt.on_court_by_window(oncourt.load_checked(CLIP),
                                     boundaries=boundaries)

    # --- windowed run + seed (roster labels give some identities a position number) ---
    wid = winmod.WindowedIdentity(boundaries=boundaries)
    active_log = defaultdict(list)          # (win, id) -> [(frame, bbox)]
    ident_of = {}                           # (win, id) -> Identity
    seen = set()
    prev_win = None
    for (fidx, tracks) in frames:
        win = wid.update(fidx, tracks)
        m = wid.current_machine()
        if win != prev_win:
            seen = set()
            on = onc.get(win, set())
            refs = roster.ref_tracks()  # officials never become players
            refs = refs | roster._spliced()   # two-player tracks: never credited
            for t in tracks:
                if t.track_id in on and t.track_id not in refs:
                    m.seed(t.track_id, roster_number=roster.seed_number_for(clip, t.track_id))
            prev_win = win
        else:                               # LABELED on-court newcomers seed on arrival
            winmod.seed_labeled_newcomers(
                m, tracks, seen, onc.get(win, set()),
                lambda tid: roster.seed_number_for(clip, tid))
        seen |= {t.track_id for t in tracks}
        for ident in m.active():
            active_log[(win, ident.identity_id)].append((fidx, ident.last_bbox))
            ident_of[(win, ident.identity_id)] = ident

    machines = wid.machines()

    def _on_court(key):
        return ident_of[key].track_id in onc.get(key[0], set())

    # EVER unresolved, not just unresolved AT THE END. An identity that spent
    # hundreds of frames as CANDIDATE and then died reads as LOST here, and was
    # therefore never offered to OCR at all -- 83% of HARD's 'one read away'
    # mass. Dying is not evidence that the number is unreadable.
    all_cands = [k for k, i in ident_of.items()
                 if i.ever_unresolved and i.state is not IdentityState.CONFIRMED]
    candidates = [k for k in all_cands if _on_court(k)]           # the OCR pool
    off_court_candidates = len(all_cands) - len(candidates)
    before_review = [k for k, i in ident_of.items()
                     if i.ever_unresolved and i.state is not IdentityState.CONFIRMED
                     and _on_court(k)]

    # --- PICK frames per candidate first (bbox data only, no images yet) ---
    picked_by_key = {}
    attempted_cands = set()
    for key in candidates:
        frs = [(f, bb) for (f, bb) in active_log[key]
               if bb and (bb[3] - bb[1]) >= MIN_OCR_HEIGHT]
        # BEST-CROPS-FIRST (attempt policy v2): legibility tracks box size
        # (montage diagnosis, DECISIONS 4b), so spend the attempt budget on the
        # candidate's LARGEST boxes, keeping picks >= OCR_STRIDE frames apart
        # so attempts aren't near-duplicate frames. Budget/threshold unchanged.
        frs.sort(key=lambda fb: -(fb[1][3] - fb[1][1]))
        picked = []
        for (f, bb) in frs:
            if all(abs(f - g) >= OCR_STRIDE for (g, _b) in picked):
                picked.append((f, bb))
            if len(picked) >= MAX_ATTEMPTS:
                break
        picked_by_key[key] = picked
        if picked:
            attempted_cands.add(key)

    # --- NOW load only the frames actually picked (targeted, single pass) ---
    needed_frames = sorted({f for picked in picked_by_key.values() for (f, _bb) in picked})
    imgs = dict(s2mk.iter_frames(CLIP.video_path, needed_frames))
    print(f"[stage6] loaded {len(imgs)} frame(s) for OCR attempts "
          f"(span holds {doc['span_len']} -- targeted read, not the whole span)")

    # --- temporal OCR accumulation per candidate, IN ROUNDS -----------------
    # Round N attempts every still-unread candidate's Nth-best crop, all in
    # parallel, then drops the ones that got a confident read before round N+1.
    #
    # WHY (measured 2026-08-03). EasyOCR runs locally and a plain nested loop was
    # fine. The vision reader is a network call taking ~7.8 s on its own, and it
    # makes GEMMA_READS of them per crop -- TEST1's 43 candidates x 10 crops x 3
    # reads is over an hour of waiting, and the first attempt at this timed out.
    # Two changes fix it without touching the attempt BUDGET or the threshold:
    #   PARALLEL: the reads are independent, so they overlap. 6 workers measured
    #     stable end to end; 16 was faster per call but the API rate-limited.
    #   BEST-CROP-FIRST + EARLY EXIT: picks are already sorted biggest-first, so
    #     a candidate whose clearest crop reads confidently needs none of her
    #     remaining nine. On real clips most candidates either read immediately
    #     or never, so this removes the bulk of the calls.
    # Neither changes WHICH crops are eligible, only how many get spent.
    OCR_WORKERS = 6
    attempts = crops_any = crops_conf = 0
    best = {}                               # (win,id) -> (number, conf, frame, bbox)

    def _attempt(job):
        key, f, bb = job
        crop = ocr_reader.jersey_crop(imgs[f], bb)
        return key, f, bb, ocr_reader.read_jersey(crop, roster.ROSTER_NUMBERS)

    max_round = max((len(v) for v in picked_by_key.values()), default=0)
    for rnd in range(max_round):
        jobs = [(key, *picked_by_key[key][rnd])
                for key in candidates
                if key not in best and len(picked_by_key[key]) > rnd]
        if not jobs:
            break
        with ThreadPoolExecutor(max_workers=OCR_WORKERS) as ex:
            results = list(ex.map(_attempt, jobs))
        for key, f, bb, reads in results:
            attempts += 1
            if not reads:
                continue
            crops_any += 1
            n, c = max(reads, key=lambda r: r[1])
            if c >= ocr_reader.OCR_CONFIRM_THRESHOLD:
                crops_conf += 1
                if key not in best or c > best[key][1]:
                    best[key] = (n, c, f, bb)
        print(f"[stage6] attempt round {rnd + 1}: {len(jobs)} crop(s), "
              f"{len(best)} candidate(s) now read", flush=True)

    # --- CORROBORATION: the same number, off a DIFFERENT picture of her -----
    #
    # WHY (measured on TEST1, 2026-08-03). The vision reader's confidence is how
    # many of its repeated reads agreed, and that catches a wobbly crop -- but
    # its two real mistakes were UNANIMOUS and still wrong: a jersey plainly
    # reading 44 came back 14 three times running, and one reading 10 came back
    # 13. Asking the same picture again cannot fix a picture that is clipped or
    # half-occluded; it just gets the same wrong answer with full marks.
    #
    # So a confident read now has to survive a SECOND, DIFFERENT crop of the
    # same candidate. Two pictures disagreeing is exactly the evidence that one
    # of them is unreadable, and it is evidence three reads of one picture can
    # never produce.
    #
    # Cost is small and bounded: only candidates that ALREADY read get a second
    # look (9-17 per clip here), not the 30-odd who never read at all. Early
    # exit still applies to everyone else.
    corroboration = {}                       # (win,id) -> "corroborated" | ...
    verify_jobs = []
    for key, (_n, _c, f, _bb) in best.items():
        alt = next(((g, gb) for (g, gb) in picked_by_key[key] if g != f), None)
        if alt is None:
            corroboration[key] = "single_crop_only"   # she has no other picture
        else:
            verify_jobs.append((key, *alt))
    if verify_jobs:
        with ThreadPoolExecutor(max_workers=OCR_WORKERS) as ex:
            vres = list(ex.map(_attempt, verify_jobs))
        conflicts = []
        for key, f, bb, reads in vres:
            first = best[key][0]
            conf_reads = [(n, c) for (n, c) in reads
                          if c >= ocr_reader.OCR_CONFIRM_THRESHOLD]
            if not conf_reads:
                # The second picture was not legible. That is NOT a conflict --
                # most crops are unreadable, which is the whole reason this
                # pipeline accumulates across a window. Kept, but marked so a
                # human can see it rested on one picture.
                corroboration[key] = "single_crop_only"
                continue
            second = max(conf_reads, key=lambda r: r[1])[0]
            if second == first:
                corroboration[key] = "corroborated"
            else:
                # TWO legible pictures of one girl disagree. One of them is a
                # misread and we cannot tell which, so she is NOT auto-confirmed
                # -- she goes to the human queue, which is the point.
                corroboration[key] = f"conflict_{first}_vs_{second}"
                conflicts.append((key, first, second, f))
        for (key, a, b_, f) in conflicts:
            del best[key]
        if conflicts:
            print(f"[stage6] CORROBORATION rejected {len(conflicts)} read(s) -- a "
                  f"second crop of the same girl read a DIFFERENT number:")
            for (key, a, b_, f) in conflicts:
                print(f"    w{key[0]} id{key[1]}: first said #{a}, second crop "
                      f"(f{f}) said #{b_} -> sent to review, not confirmed")
    n_corr = sum(1 for v in corroboration.values() if v == "corroborated")
    n_single = sum(1 for v in corroboration.values() if v == "single_crop_only")
    print(f"[stage6] confidence: {n_corr} read(s) corroborated by a second crop, "
          f"{n_single} rest on a single crop (kept, but flagged for review)")

    # --- apply the three outcomes to the accumulated best read ---
    outcomes = {"agree": [], "disagree": [], "no_confident_read": [],
                "no_position_hypothesis": []}
    for key in candidates:
        ident = ident_of[key]
        b = best.get(key)
        num, conf, frame = (b[0], b[1], b[2]) if b else (None, None, None)
        res = machines[key[0]].promote_via_second_signal(ident, num, conf)
        outcomes[res].append((key, b))

    # --- readability measurement ---
    print(f"clip={clip} span={span_start}..+{doc['span_len']}  "
          f"candidates={len(candidates)} (ON-COURT; ROI mask excluded "
          f"{off_court_candidates} off-court candidates from the OCR pool)")
    pf_any = crops_any / attempts if attempts else 0
    pf_conf = crops_conf / attempts if attempts else 0
    pp_conf = len(best) / len(attempted_cands) if attempted_cands else 0
    print(f"\nREADABILITY (why we accumulate across the window):")
    print(f"  per-FRAME:      {attempts} crops attempted -> any on-roster read "
          f"{crops_any} ({pf_any:.0%}), confident {crops_conf} ({pf_conf:.0%})")
    print(f"  per-POSSESSION: {len(attempted_cands)} candidates attempted -> "
          f">=1 confident read {len(best)} ({pp_conf:.0%})")
    print(f"  -> per-possession ({pp_conf:.0%}) >> per-frame ({pf_conf:.0%}): one lucky "
          f"frame per window is enough.")

    # --- auto-confirms (eyeball correctness against the visible number) ---
    print(f"\nAUTO-CONFIRMS via OCR (AGREE): {len(outcomes['agree'])}")
    for (key, b) in outcomes["agree"]:
        ident = ident_of[key]
        print(f"  window {key[0]} identity {key[1]}: position said #{ident.roster_number}, "
              f"OCR read #{b[0]} conf {b[1]:.2f} at f={b[2]}  -> CONFIRMED (second_signal)")
    print(f"\nDISAGREEMENTS (swap flags, NEVER silently resolved): {len(outcomes['disagree'])}")
    for (key, b) in outcomes["disagree"]:
        ev = ident_of[key].evidence
        print(f"  window {key[0]} id {key[1]}: position #{ev.get('position_says')} vs "
              f"OCR #{ev.get('ocr_read')} conf {ev.get('confidence')} -> FLAG possible swap")
    stayed = len(outcomes["no_confident_read"]) + len(outcomes["no_position_hypothesis"])
    print(f"\nSTAYED CANDIDATE (no confident on-roster read all window): {stayed}  "
          f"(expected -- number never faced the camera; NOT a failure)")

    # --- queue before vs after (on-court only; matches stage4's queue policy) ---
    after_review = sum(1 for k, i in ident_of.items()
                       if i.ever_unresolved and i.state is not IdentityState.CONFIRMED
                       and _on_court(k))
    print(f"\nREVIEW QUEUE (on-court):  before OCR = {len(before_review)}  ->  "
          f"after OCR = {after_review}")
    print(f"  auto-confirmed (removed from queue): {len(outcomes['agree'])}")
    print(f"  disagreement flags (added, highest value): {len(outcomes['disagree'])}")

    # --- confirmed-only + lock check ---
    total_conf = sum(1 for i in ident_of.values() if i.state == IdentityState.CONFIRMED)
    via_2nd = len(outcomes["agree"])
    print(f"\nCONFIRMED identities total = {total_conf} "
          f"(via seed + {via_2nd} via second_signal). Continuity confirmations = 0.")

    # --- stills: OCR-confirmed players (green + jersey number) ---
    for (key, b) in outcomes["agree"]:
        n, c, f, bb = b
        img = imgs[f].copy()
        x1, y1, x2, y2 = [int(v) for v in bb]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img, f"#{n} CONFIRMED via OCR ({c:.2f})", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(OUT_DIR, f"{clip}_ocr_confirm_w{key[0]}_id{key[1]}_f{f}.jpg"),
                    cv2.resize(img, (1280, 720)))
    print(f"\nsaved OCR-confirm stills in {OUT_DIR}")

    # --- PERSIST the outcomes (they are the pipeline's most valuable signal;
    # until now they lived only in stdout + stills). This JSON is also the
    # input contract for the future retroactive stat merge. -------------------
    def _row(key, b):
        ident = ident_of[key]
        row = {"window": key[0], "identity_id": key[1], "track_id": ident.track_id,
               "position_hypothesis": ident.roster_number,
               "final_state": ident.state.value, "evidence": dict(ident.evidence)}
        if b is not None:
            row.update({"read_number": b[0], "read_confidence": round(b[1], 3),
                        "read_frame": b[2], "read_bbox": [round(v, 1) for v in b[3]]})
        # HOW SURE ARE WE, in a form a human can sort by. read_confidence alone
        # is not enough: the reader's worst mistakes came back unanimous (1.00)
        # off a clipped crop. This says whether a SECOND picture of her agreed.
        #   corroborated      two different crops, same number -- strongest
        #   single_crop_only  only one crop was ever legible -- worth an eyeball
        #   conflict_A_vs_B   two crops disagreed -- NOT confirmed, needs a human
        row["corroboration"] = corroboration.get(key)
        return row

    out_doc = {
        "clip": clip, "span_start": span_start, "span_len": doc["span_len"],
        "windows": wlabel, "window_boundaries": boundaries,
        "ocr_confirm_threshold": ocr_reader.OCR_CONFIRM_THRESHOLD,
        "seed_policy": "ROI-mask: on-court majority per (window, track)",
        "attempt_policy": "best_crops_first_v2",
        "off_court_candidates_excluded": off_court_candidates,
        "readability": {
            "crops_attempted": attempts, "crops_any_read": crops_any,
            "crops_confident_read": crops_conf,
            "candidates_attempted": len(attempted_cands),
            "candidates_with_confident_read": len(best)},
        "review_queue_before": len(before_review),
        "review_queue_after": after_review,
        "outcomes": {name: [_row(k, b) for (k, b) in rows]
                     for name, rows in outcomes.items()},
        # Per-identity registry (ALL identities, not just candidates): the
        # merge stage needs number + final state to run its contradiction check.
        "identities": [
            {"window": k[0], "identity_id": k[1], "track_id": i.track_id,
             "roster_number": i.roster_number, "final_state": i.state.value}
            for k, i in sorted(ident_of.items())],
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, indent=2)
    print(f"saved OCR outcomes -> {OUT_JSON}")


if __name__ == "__main__":
    main()
