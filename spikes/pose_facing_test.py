"""POSE TEST: does shoulder-facing predict a confident jersey read?

WHY THIS TEST EXISTS. The jersey reader is accurate when it gets a look -- 8
of 8 confident reads matched a human label (HANDOFF_NAMING_2026_09_07 S2) --
but it only gets a confident look 3.4% of the time. Crops are currently picked
by BOX SIZE (close to camera), not by which way the player is facing. A pose
model's shoulder keypoints say which way she is facing: shoulders far apart
IN THE PICTURE = squared up to the camera (front or back -- either can carry
a number); shoulders close together = turned sideways, where no number faces
the lens at all.

THE CHEAP TEST, on cached data, no new footage (HANDOFF_NAMING_2026_09_07 S6):
  WIN  = the exact (frame, bbox) that produced every confident read on disk,
         already recorded in {clip}_ocr_confirms.json outcomes.
  LOSS = the OTHER frames phase2/stage6_ocr_confirm.py picked for that same
         candidate (same identity, same window, same size-based time-sliced
         selection -- rebuilt here identically, see _picked_frames) that were
         NOT the winner.
Both piles get the same pose model; the only question is whether facing
separates them.

REBUILDING "the same candidate" IS NOT A PLAIN track_id LOOKUP. Tried that
first -- 28 of 58 candidates' read_frame landed on a bbox nowhere near the
current track_id's real position that frame (one was off by 400+ px: a
different person). identity.Identity is explicit about why: "a logical
identity following one (or, across relinks, several) track_ids" -- ident.
track_id is only the one it CURRENTLY follows. The only faithful source is
the actual identity replay stage6 runs (_replay_active_log below, copied
from its main()), which is why this script drives itself with `--clip NAME`
as a fresh subprocess per clip: phase2/roster.py binds ACTIVE_CLIP at IMPORT
time, so a second clip in the same interpreter would silently score against
the first clip's roster/decisions (the project's own "one clip per process"
rule, HANDOFF_NAMING_2026_09_07 S10).

KILL NUMBER (written down before running): if WIN and LOSS facing scores come
out about the same (separation score near 0.5), crop selection stays
size-based. Pose only earns a place in the pipeline if it visibly separates
the two piles.

Usage:
    .venv/Scripts/python.exe spikes/pose_facing_test.py
    (delete spikes/out/pose_facing_test.json to force a re-run of the pose
    model; delete spikes/out/pose_facing_pool_<CLIP>.json to force a re-replay
    of that clip's identity machine. Re-running this script otherwise just
    re-scores whatever is already cached.)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "phase2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                  # noqa: E402

import stage2_multikeyframe as s2mk                                 # noqa: E402
import clip_config                                                  # noqa: E402

CLIPS = ("TEST1", "TEST1_REG", "HARD", "TEST2", "Full_Game_9eb8bf2a")
MIN_OCR_HEIGHT = 90     # phase2/stage6_ocr_confirm.py -- same eligibility filter
MAX_ATTEMPTS = 10       # phase2/stage6_ocr_confirm.py -- same slice count
POSE_WEIGHTS = "yolo11x-pose.pt"   # same weights spikes/pose_shot_check.py proved out
IMGSZ = 1280                       # DECISIONS 20 proven optimum
POSE_CONF = 0.25
KP_CONF = 0.3            # spikes/pose_shot_check.py -- below this a keypoint is a guess
IOU_MATCH = 0.3          # minimum overlap to say "this pose is that tracker box"
L_SHO, R_SHO = 5, 6      # COCO keypoint order

RESULTS_JSON = os.path.join(_HERE, "out", "pose_facing_test.json")


def _pool_json(clip):
    return os.path.join(_HERE, "out", f"pose_facing_pool_{clip}.json")


def _picked_frames(frs):
    """IDENTICAL selection to phase2/stage6_ocr_confirm.py main() (the
    "best crop in each slice of her time" block): cut a candidate's eligible
    frames into MAX_ATTEMPTS time-slices, keep the biggest box in each slice,
    sort biggest-first. frs: [(frame, bbox), ...] frame-ordered.
    -> [(frame, bbox), ...] picked, biggest-first."""
    if not frs:
        return []
    span_lo, span_hi = frs[0][0], frs[-1][0]
    width = max(1, span_hi - span_lo + 1)
    best_in_slice = {}
    for (f, bb) in frs:
        s = min(MAX_ATTEMPTS - 1, (f - span_lo) * MAX_ATTEMPTS // width)
        cur = best_in_slice.get(s)
        if cur is None or (bb[3] - bb[1]) > (cur[1][3] - cur[1][1]):
            best_in_slice[s] = (f, bb)
    return sorted(best_in_slice.values(), key=lambda fb: -(fb[1][3] - fb[1][1]))


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-6, area_a + area_b - inter)


def _load_tracks(path):
    """Same as phase2/stage6_ocr_confirm.py's own load(): tracks_raw.json ->
    [(frame_index, [Track, ...]), ...], frame-ordered."""
    import tracking
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    return [(fr["frame_index"],
             [tracking.Track(t["track_id"], tuple(t["bbox"])) for t in fr["tracks"]])
            for fr in doc["frames"]]


def _replay_active_log(clip):
    """Faithful replay of phase2/stage6_ocr_confirm.py main(), stopping right
    after active_log is built (no OCR calls -- this never reads a crop).
    MUST be the first clip-aware code in this interpreter: run_clip._sync_and_
    guard sets clip_config.ACTIVE_CLIP before roster/oncourt/windows are
    imported, because roster.py binds ACTIVE_CLIP at import time and caches
    decisions/ref-tracks/spliced-tracks in globals that never invalidate.

    -> ([dict(clip, window, identity_id, track_id, frame, bbox, is_positive),
    ...], mismatches). mismatches counts candidates whose read_frame could not
    be found in the rebuilt picked-set even with the real replay -- kept as an
    honest count rather than silently dropped or forced in."""
    import run_clip
    import clip_config
    cfg = clip_config.get_clip(clip)
    run_clip._sync_and_guard(cfg)          # sets ACTIVE_CLIP BEFORE the imports below

    import oncourt
    import roster
    import windows as winmod

    ocr_path = os.path.join(_ROOT, "phase2", "out", f"{clip}_ocr_confirms.json")
    ocr = json.load(open(ocr_path, encoding="utf-8"))
    boundaries = ocr["window_boundaries"]

    frames = _load_tracks(cfg.tracks_cache_path)
    onc = oncourt.on_court_by_window(oncourt.load_checked(cfg), boundaries=boundaries)

    wid = winmod.WindowedIdentity(boundaries=boundaries)
    active_log = defaultdict(list)          # (win, id) -> [(frame, bbox)]
    seen = set()
    prev_win = None
    for (fidx, tracks) in frames:
        win = wid.update(fidx, tracks)
        m = wid.current_machine()
        if win != prev_win:
            seen = set()
            on = onc.get(win, set())
            refs = roster.ref_tracks() | roster._spliced()
            for t in tracks:
                if t.track_id in on and t.track_id not in refs:
                    m.seed(t.track_id, roster_number=roster.seed_number_for(clip, t.track_id))
            prev_win = win
        else:
            winmod.seed_labeled_newcomers(
                m, tracks, seen, onc.get(win, set()),
                lambda tid: roster.seed_number_for(clip, tid))
        seen |= {t.track_id for t in tracks}
        for ident in m.active():
            active_log[(win, ident.identity_id)].append((fidx, ident.last_bbox))

    positives = []   # (window, identity_id, track_id, read_frame, read_bbox)
    for rows in ocr["outcomes"].values():
        for row in rows:
            if row.get("read_frame") is not None:
                positives.append((row["window"], row["identity_id"], row["track_id"],
                                  row["read_frame"], row["read_bbox"]))

    # A MATCH NEEDS THE SAME BOX, NOT JUST THE SAME FRAME NUMBER. First cut of
    # this rebuild only checked the frame index -- MEASURED on TEST1, that let
    # a false match through: same frame, a box 450px away (a different track
    # entirely). read_bbox is rounded to 1dp in ocr_confirms.json, so exact
    # equality is too strict; IoU >= BBOX_MATCH_IOU allows for that rounding
    # while still rejecting an unrelated box at the same frame.
    BBOX_MATCH_IOU = 0.9
    out_rows, mismatches = [], 0
    for (w, ident_id, track_id, read_frame, read_bbox) in positives:
        frs = [(f, bb) for (f, bb) in active_log.get((w, ident_id), [])
               if bb and (bb[3] - bb[1]) >= MIN_OCR_HEIGHT]
        picked = _picked_frames(frs)
        # frames are unique within `picked` (each source frame belongs to
        # exactly one time-slice), so a frame-number match is also the only
        # candidate is_positive can mean below.
        hit = any(f == read_frame and _iou(bb, read_bbox) >= BBOX_MATCH_IOU
                 for (f, bb) in picked)
        if not hit:
            mismatches += 1
            continue
        for (f, bb) in picked:
            out_rows.append(dict(clip=clip, window=w, identity_id=ident_id,
                                 track_id=track_id, frame=f, bbox=list(bb),
                                 is_positive=(f == read_frame)))
    return out_rows, mismatches


def collect_win_loss(clip):
    """-> (rows, mismatches) for one clip, via a FRESH subprocess (see
    _replay_active_log's docstring for why) that dumps its pool to
    POOL_JSON(clip). Reused across runs like every other cache in this repo."""
    path = _pool_json(clip)
    if not os.path.exists(path):
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--clip", clip],
                           cwd=_ROOT, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(path):
            print(r.stdout[-4000:])
            print(r.stderr[-4000:])
            raise RuntimeError(f"{clip}: identity replay subprocess failed "
                               f"(returncode {r.returncode})")
    doc = json.load(open(path, encoding="utf-8"))
    return doc["rows"], doc["mismatches"]


def _pose_people(model, img):
    res = model.predict(img, imgsz=IMGSZ, conf=POSE_CONF, verbose=False)[0]
    people = []
    if res.keypoints is None:
        return people
    kxy = res.keypoints.xy.cpu().numpy()
    kcf = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else None
    boxes = res.boxes.xyxy.cpu().numpy()
    for i in range(len(kxy)):
        conf = kcf[i] if kcf is not None else np.ones(len(kxy[i]))
        people.append({"kp": kxy[i], "kpc": conf, "box": boxes[i]})
    return people


def add_facing_scores(rows):
    """Runs pose once per (clip, frame) -- several rows can share a frame --
    matches each row's tracker bbox to a pose detection by IoU, and adds
    matched/iou/facing_score fields IN PLACE. facing_score = normalised
    apparent shoulder width (wide = squared to camera, narrow = side-on),
    same body-height normalisation spikes/pose_shot_check.py already uses."""
    from ultralytics import YOLO
    model = YOLO(POSE_WEIGHTS)

    by_clip_frame = defaultdict(list)
    for r in rows:
        by_clip_frame[(r["clip"], r["frame"])].append(r)

    clips_needed = sorted({c for (c, _f) in by_clip_frame})
    for clip in clips_needed:
        video = clip_config.get_clip(clip).video_path
        need = sorted({f for (c, f) in by_clip_frame if c == clip})
        done = 0
        for f, img in s2mk.iter_frames(video, need):
            people = _pose_people(model, img)
            for r in by_clip_frame[(clip, f)]:
                best_iou, best = 0.0, None
                for p in people:
                    iou = _iou(r["bbox"], p["box"])
                    if iou > best_iou:
                        best_iou, best = iou, p
                matched = best is not None and best_iou >= IOU_MATCH
                r["matched"] = bool(matched)
                r["iou"] = round(float(best_iou), 3)   # numpy float32 isn't JSON-serializable
                r["facing_score"] = None
                if matched:
                    lc, rc = best["kpc"][L_SHO], best["kpc"][R_SHO]
                    if lc >= KP_CONF and rc >= KP_CONF:
                        lx, rx = best["kp"][L_SHO][0], best["kp"][R_SHO][0]
                        h = float(best["box"][3] - best["box"][1])
                        r["facing_score"] = round(float(abs(lx - rx)) / max(h, 1e-6), 4)
            done += 1
            if done % 50 == 0 or done == len(need):
                print(f"  [{clip}] pose: {done}/{len(need)} frames", flush=True)
    return rows


def main():
    if os.path.exists(RESULTS_JSON):
        rows = json.load(open(RESULTS_JSON, encoding="utf-8"))
        print(f"reusing {len(rows)} cached crop(s) from "
              f"{os.path.basename(RESULTS_JSON)} (delete it to re-run pose)")
        return report(rows)

    all_rows = []
    skipped_clips = []
    for clip in CLIPS:
        try:
            rows, mismatches = collect_win_loss(clip)
        except RuntimeError as e:
            print(f"{clip}: SKIPPED -- {e}")
            skipped_clips.append(clip)
            continue
        n_pos = sum(1 for r in rows if r["is_positive"])
        n_neg = len(rows) - n_pos
        print(f"{clip}: {n_pos} WIN crop(s), {n_neg} LOSS crop(s) from the same "
              f"tracks ({mismatches} candidate(s) skipped -- rebuilt pool didn't "
              f"agree with the recorded read_bbox, a cache-consistency gap not a "
              f"facing question)")
        all_rows.extend(rows)
    if skipped_clips:
        print(f"\nSkipped entirely: {skipped_clips} -- their caches need attention "
              f"before they can be used here (see stderr above). Not fixed in this "
              f"run: that is separate from the facing question this test is asking.")

    n_frames = len({(r["clip"], r["frame"]) for r in all_rows})
    print(f"\n{len(all_rows)} total crops on {n_frames} unique frames -- running "
          f"pose ({POSE_WEIGHTS}, imgsz={IMGSZ}) once per frame...")
    add_facing_scores(all_rows)

    # REPORT FIRST, SAVE SECOND. The pose pass is the expensive part (~15 min
    # of CPU here) -- a save failure must never be able to eat that, so the
    # printed report is not allowed to depend on the save succeeding.
    report(all_rows)
    try:
        os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
        json.dump(all_rows, open(RESULTS_JSON, "w", encoding="utf-8"), indent=1)
        print(f"\nsaved -> {RESULTS_JSON}")
    except TypeError as e:
        print(f"\nWARNING: results were NOT saved to {RESULTS_JSON} ({e}). "
              f"The report above is still correct -- this only affects re-running "
              f"the report without redoing the pose pass.")


def report(rows):
    pos = [r["facing_score"] for r in rows if r["is_positive"] and r["facing_score"] is not None]
    neg = [r["facing_score"] for r in rows if not r["is_positive"] and r["facing_score"] is not None]
    pos_nopose = sum(1 for r in rows if r["is_positive"] and r["facing_score"] is None)
    neg_nopose = sum(1 for r in rows if not r["is_positive"] and r["facing_score"] is None)

    print(f"\n{'=' * 72}")
    print("POSE FACING TEST -- WIN (confident reads) vs LOSS (other picked crops)")
    print(f"{'=' * 72}")
    print(f"WIN : {len(pos)} with a usable pose ({pos_nopose} had no matched/confident shoulders)")
    print(f"LOSS: {len(neg)} with a usable pose ({neg_nopose} had no matched/confident shoulders)")

    if not pos or not neg:
        print("\nNot enough usable poses on either side to say anything. STOP.")
        return

    def stats(xs):
        xs = sorted(xs)
        n = len(xs)
        mean = sum(xs) / n
        median = xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2
        return mean, median

    pm, pmed = stats(pos)
    nm, nmed = stats(neg)
    print(f"\n{'':6s} {'n':>4s} {'mean':>8s} {'median':>8s}")
    print(f"{'WIN':6s} {len(pos):4d} {pm:8.3f} {pmed:8.3f}")
    print(f"{'LOSS':6s} {len(neg):4d} {nm:8.3f} {nmed:8.3f}")

    # SEPARATION SCORE: probability a random WIN scores higher than a random
    # LOSS (ties count half). 0.5 = no difference at all; 1.0 = WIN crops are
    # ALWAYS more face-on than LOSS crops. Same statistic as Mann-Whitney U /
    # (n1*n2) -- computed directly since both piles are small.
    wins = sum(1 for a in pos for b in neg if a > b)
    ties = sum(1 for a in pos for b in neg if a == b)
    total = len(pos) * len(neg)
    auc = (wins + 0.5 * ties) / total

    print(f"\nSEPARATION SCORE: {auc:.3f}  (0.5 = no difference at all, "
          f"1.0 = WIN crops are ALWAYS more face-on than LOSS crops)")
    print(f"{'-' * 72}")
    if auc >= 0.65:
        print("VERDICT: facing separates WIN from LOSS. Promising -- worth building "
              "pose-guided crop selection next.")
    elif auc <= 0.55:
        print("VERDICT: facing does NOT separate WIN from LOSS. DEAD -- crop "
              "selection stays size-based.")
    else:
        print("VERDICT: weak/unclear signal. Not a clean kill, not a clean win -- "
              "look at the numbers by hand before building anything.")

    print("\n  every WIN crop, for the record:")
    for r in sorted(rows, key=lambda r: (r["clip"], r["track_id"])):
        if r["is_positive"]:
            fs = f"{r['facing_score']:.3f}" if r["facing_score"] is not None else "no-pose"
            print(f"    {r['clip']:18s} t{r['track_id']:<5} f{r['frame']:<6} facing={fs}")


if __name__ == "__main__":
    if "--clip" in sys.argv:
        _clip = sys.argv[sys.argv.index("--clip") + 1]
        _rows, _mismatches = _replay_active_log(_clip)
        os.makedirs(os.path.dirname(_pool_json(_clip)), exist_ok=True)
        json.dump({"rows": _rows, "mismatches": _mismatches},
                  open(_pool_json(_clip), "w", encoding="utf-8"))
        _n_pos = sum(1 for r in _rows if r["is_positive"])
        print(f"{_clip}: {_n_pos} WIN, {len(_rows) - _n_pos} LOSS, "
              f"{_mismatches} mismatch (identity replay subprocess)")
    else:
        main()
