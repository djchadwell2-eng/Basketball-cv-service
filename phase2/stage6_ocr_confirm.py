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

# FACING-GUIDED CROP PICK (spikes/pose_facing_test.py, 2026-09-08). MEASURED:
# the reader's read rate is 3.4% per crop, and the recorded cause is ANGLE
# (backs, side-ons) -- box size (below) is a proxy for "close to camera", not
# "facing it". On 23 confident reads vs 189 other picked-but-unread crops of
# the SAME players, a pose model's apparent shoulder width separated the two
# piles at 0.684 (0.5 = no signal, 1.0 = perfect) -- consistent in the same
# direction across all 4 clips checked, not a proxy for box size (correlation
# 0.068), and confirmed by eye on real crops. NOT a complete fix: one player's
# plainly-legible number went unread across 4 squared-on frames in a row, so
# this removes ONE cause of missed reads, not the only one.
# Default OFF -- every existing clip and all tests behave exactly as before;
# turn on deliberately to test it on a real clip.
FACING_CROP_PICK = bool(int(os.environ.get("CV_FACING_CROP_PICK", "0") or 0))
FACING_SHORTLIST = 3     # only re-rank the N biggest boxes in a slice by facing
                         # -- bounds the added pose cost, and matches what was
                         # actually measured (reasonably-sized crops only)
POSE_WEIGHTS = "yolo11x-pose.pt"   # same weights the measurement used
POSE_IMGSZ = 1280                  # DECISIONS 20 proven optimum
POSE_CONF = 0.25
KP_CONF = 0.3            # below this a keypoint is a guess, not a reading
IOU_MATCH = 0.3          # minimum overlap to say "this pose is that tracker box"
L_SHO, R_SHO = 5, 6      # COCO keypoint order


def load(path):
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    frames = [(fr["frame_index"],
               [Track(t["track_id"], tuple(t["bbox"])) for t in fr["tracks"]])
              for fr in doc["frames"]]
    # THE PARSED JSON IS DROPPED ONCE THE TRACKS ARE BUILT. Holding both meant
    # every body in every frame existed twice over -- once as the dict json
    # parsed, once as the Track built from it -- and nothing downstream ever
    # reads doc["frames"] again; only the header fields (clip/fps/span). On a
    # 15-second clip that duplication is invisible. MEASURED at full-game
    # scale it was ~0.88 GB per slice, heading for ~9 GB on a whole game
    # against the 3.85 GB of worker memory this project has ever proven.
    doc.pop("frames", None)
    return frames, doc


# --------------------------------------------------------------------------
# FACING-GUIDED CROP PICK -- pure selection logic first (no images, testable
# by hand), then the pose-model I/O that feeds it. See FACING_CROP_PICK above
# for why this exists and what it does and doesn't fix.
# --------------------------------------------------------------------------

def _group_by_slice(frs):
    """frs: [(frame, bbox), ...] frame-ordered, already size-eligible.
    -> {slice_idx: [(frame, bbox), ...]}, each list in frame order, slice
    indices inserted in ascending order -- SAME grouping the plain size-only
    picker always used, just not reduced to one entry per slice yet."""
    if not frs:
        return {}
    span_lo, span_hi = frs[0][0], frs[-1][0]
    width = max(1, span_hi - span_lo + 1)
    groups = defaultdict(list)
    for (f, bb) in frs:
        s = min(MAX_ATTEMPTS - 1, (f - span_lo) * MAX_ATTEMPTS // width)
        groups[s].append((f, bb))
    return groups


def _biggest(frs_in_slice):
    """today's rule: the tallest box in the slice, earliest frame on a tie
    (max() keeps the first-seen maximum, and frs_in_slice is frame-ordered)."""
    return max(frs_in_slice, key=lambda fb: fb[1][3] - fb[1][1])


def _best_by_facing(frs_in_slice, facing_of, shortlist=FACING_SHORTLIST):
    """Among the `shortlist` biggest boxes in this slice, keep whichever has
    the best facing score. Falls back to _biggest (today's rule) if nothing
    in the shortlist has a usable score -- same abstain-to-today's-behaviour
    this file uses everywhere else. facing_of: (frame, bbox) -> float | None.
    """
    by_size = sorted(frs_in_slice, key=lambda fb: -(fb[1][3] - fb[1][1]))
    top = by_size[:shortlist]
    scored = [(fb, facing_of(fb)) for fb in top]
    scored = [(fb, s) for (fb, s) in scored if s is not None]
    if not scored:
        return by_size[0]
    return max(scored, key=lambda fbs: fbs[1])[0]


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


def _pose_people(model, img):
    res = model.predict(img, imgsz=POSE_IMGSZ, conf=POSE_CONF, verbose=False)[0]
    people = []
    if res.keypoints is None:
        return people
    kxy = res.keypoints.xy.cpu().numpy()
    kcf = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else None
    boxes = res.boxes.xyxy.cpu().numpy()
    for i in range(len(kxy)):
        people.append({"kp": kxy[i], "box": boxes[i],
                       "kpc": kcf[i] if kcf is not None else [1.0] * len(kxy[i])})
    return people


def _facing_score(bbox, people):
    """Best IoU match in `people`, then normalised apparent shoulder width
    (wide = squared to the camera, narrow = side-on), or None if nothing
    matches closely enough or the shoulders aren't confidently seen. Same
    method measured in spikes/pose_facing_test.py (separation 0.684)."""
    best_iou, best = 0.0, None
    for p in people:
        iou = _iou(bbox, p["box"])
        if iou > best_iou:
            best_iou, best = iou, p
    if best is None or best_iou < IOU_MATCH:
        return None
    lc, rc = best["kpc"][L_SHO], best["kpc"][R_SHO]
    if lc < KP_CONF or rc < KP_CONF:
        return None
    lx, rx = best["kp"][L_SHO][0], best["kp"][R_SHO][0]
    h = float(best["box"][3] - best["box"][1])
    return float(abs(lx - rx)) / max(h, 1e-6)


def _facing_scores_for_shortlists(slices_by_key, video_path, shortlist=FACING_SHORTLIST):
    """Runs the pose model once per unique video frame any candidate's
    size-shortlist actually needs, and returns facing_of: (frame, bbox) ->
    float | None. A slice with only one eligible frame needs no scoring (it
    has no alternative to pick over) so those are skipped, same as the
    budget/attempt logic elsewhere in this file: bounded cost, never the
    whole candidate pool."""
    need = defaultdict(list)             # frame -> [bbox, ...]
    for groups in slices_by_key.values():
        for g in groups.values():
            if len(g) < 2:
                continue
            top = sorted(g, key=lambda fb: -(fb[1][3] - fb[1][1]))[:shortlist]
            for (f, bb) in top:
                need[f].append(bb)
    if not need:
        return lambda fb: None

    from ultralytics import YOLO
    model = YOLO(POSE_WEIGHTS)
    scores = {}                          # (frame, bbox-as-tuple) -> float | None
    for f, im in s2mk.iter_frames(video_path, sorted(need)):
        people = _pose_people(model, im)
        for bb in need[f]:
            scores[(f, tuple(bb))] = _facing_score(bb, people)
    print(f"[stage6] FACING_CROP_PICK: scored {len(scores)} shortlisted crop(s) "
          f"on {len(need)} frame(s)", flush=True)
    return lambda fb: scores.get((fb[0], tuple(fb[1])))


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

    # THE BUDGET: spend the reader where it buys the most floor time.
    #
    # A 95-minute game offers 23,288 candidates for a roster of about 24 girls
    # (MEASURED). Those are TRACKING FRAGMENTS, not people -- every time a
    # player is occluded and re-acquired she becomes another candidate -- and
    # most of them are worth almost nothing: a fragment lasting five frames
    # credits a sixth of a second of floor time, while one lasting half a minute
    # is a real line on the box score. Reading them all is 698,000 vision calls;
    # reading the biggest ones first buys nearly all the value for a fraction of
    # that, and it turns cost from something you discover into a number you set.
    #
    # WHAT IT DOES NOT DO: lose anybody. A candidate that is not read stays
    # exactly what it is today -- CANDIDATE, in the coach's review queue, never
    # confirmed and never guessed. The skipped count is printed and persisted,
    # so an unread pool can never look like an unreadable one.
    #
    # Default 0 = no budget, so every existing clip and all 395 tests behave
    # exactly as before; a whole game sets it deliberately.
    budget = int(os.environ.get("CV_OCR_MAX_CANDIDATES", "0") or 0)
    skipped_for_budget = 0
    if budget and len(candidates) > budget:
        floor_time = {k: len(active_log[k]) for k in candidates}
        # frames of presence, then the key itself so ties break the same way
        # every run -- a budget that shuffles under you is not a measurement.
        candidates = sorted(candidates, key=lambda k: (-floor_time[k], k))[:budget]
        skipped_for_budget = len(floor_time) - len(candidates)
        kept = sum(floor_time[k] for k in candidates)
        total = sum(floor_time.values())
        print(f"[stage6] BUDGET {budget}: reading {len(candidates)} of "
              f"{len(floor_time)} candidates, the ones carrying the most floor "
              f"time -- {kept / max(1, total):.1%} of all tracked presence. The "
              f"other {skipped_for_budget} stay CANDIDATE in the review queue "
              f"(not read, NOT unreadable).", flush=True)
    before_review = [k for k, i in ident_of.items()
                     if i.ever_unresolved and i.state is not IdentityState.CONFIRMED
                     and _on_court(k)]

    # --- PICK frames per candidate first (bbox data only, no images yet) ---
    # BEST CROP IN EACH SLICE OF HER TIME (attempt policy v3).
    #
    # v2 sorted by box height and required picks to be OCR_STRIDE=2 frames
    # apart. Size is the right legibility proxy (montage diagnosis,
    # DECISIONS 4b) but two frames is a fifteenth of a second, so "spread"
    # was never enforced in any meaningful sense: the ten biggest boxes are
    # the ten frames where she was nearest the camera, which are usually
    # consecutive.
    #
    # MEASURED on real on-court tracks from DJ's game: the ten attempts span
    # 1.6 s at the median against a 4.3 s tracked life, and 30% of players
    # get ALL TEN inside a single second. Ten pictures of one instant, at one
    # angle. If her back is turned for that second, every attempt fails and
    # she is recorded as unreadable -- when she was only ever shown once.
    #
    # A jersey number is unreadable because of ANGLE far more often than
    # because of size: the crop montage is full of backs, side-ons, arms and
    # a referee, next to a perfectly crisp 23. Attempts are therefore spread
    # over her whole time on screen -- her frames are cut into MAX_ATTEMPTS
    # slices, and (FACING_CROP_PICK off) the LARGEST box in each is taken, so
    # every attempt is a different moment and each is still the best look
    # available then. Measured effect: 2.2x wider window, same number of
    # attempts, same cost.
    #
    # FACING_CROP_PICK on: within each slice, the biggest box no longer wins
    # automatically -- the FACING_SHORTLIST biggest are re-ranked by pose-
    # measured facing, on the MEASURED finding that size predicts "close to
    # camera", not "facing it" (see the constant's comment above). Falls back
    # to the plain biggest wherever pose has nothing to say, same as v3.
    #
    # Still ordered biggest-first afterwards, so the early rounds (and the
    # early exit) still spend on her clearest look.
    slices_by_key = {}
    for key in candidates:
        frs = [(f, bb) for (f, bb) in active_log[key]
               if bb and (bb[3] - bb[1]) >= MIN_OCR_HEIGHT]
        slices_by_key[key] = _group_by_slice(frs)

    facing_of = None
    if FACING_CROP_PICK:
        facing_of = _facing_scores_for_shortlists(slices_by_key, CLIP.video_path)

    picked_by_key = {}
    attempted_cands = set()
    overridden_by_facing = 0
    for key in candidates:
        groups = slices_by_key[key]
        if not groups:                  # never big enough to try: no attempts
            picked_by_key[key] = []
            continue
        winners = []
        for g in groups.values():
            default = _biggest(g)
            w = _best_by_facing(g, facing_of) if facing_of is not None else default
            if w != default:
                overridden_by_facing += 1
            winners.append(w)
        picked = sorted(winners, key=lambda fb: -(fb[1][3] - fb[1][1]))[:MAX_ATTEMPTS]
        picked_by_key[key] = picked
        if picked:
            attempted_cands.add(key)
    if facing_of is not None:
        print(f"[stage6] FACING_CROP_PICK: swapped the biggest crop for a more "
              f"face-on one in {overridden_by_facing} slice(s)", flush=True)

    # --- NOW cut only the crops actually picked (targeted, single pass) -----
    # KEEP THE CROP, NOT THE FRAME. This used to hold every picked frame in one
    # dict, at a MEASURED 6.38 MB each. That is fine for a 15-second clip whose
    # whole span is 461 frames, and fatal for a game: the pool is one frame per
    # OCR attempt, so a few thousand candidates is tens of gigabytes, against
    # the only worker memory ever proven (>=3.85 GB). A jersey crop is a few
    # kilobytes, and it is the only part of the frame this stage ever looks at.
    #
    # .copy() is not optional: jersey_crop returns a numpy VIEW, which would
    # keep the whole 6.22 MB frame alive and undo the entire fix.
    by_frame = defaultdict(list)
    for key, picked in picked_by_key.items():
        for (f, bb) in picked:
            by_frame[f].append((key, bb))
    crops = {}                              # (key, frame) -> jersey crop
    for f, im in s2mk.iter_frames(CLIP.video_path, sorted(by_frame)):
        for (key, bb) in by_frame[f]:
            crops[(key, f)] = ocr_reader.jersey_crop(im, bb).copy()
    print(f"[stage6] cut {len(crops)} crop(s) from {len(by_frame)} frame(s) for OCR "
          f"attempts (span holds {doc['span_len']} -- targeted read, one frame at a time)")

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
    # 32, not 6. MEASURED on the worker against real crops from DJ's own film:
    #   1 thread  0.0581 s/crop     6 threads 0.0357     32 threads 0.0250
    #   96 threads 0.0250 -- no better, the interpreter's lock is the ceiling
    # Over a whole game (~150,000 crops, measured) that is 89 minutes at 6 and
    # 62 at 32, inside a job RunPod kills at 180. Nothing about WHICH crops are
    # read, how many attempts a candidate gets, or the confirm threshold moves;
    # the rounds are still lock-step and ex.map still returns in order, so the
    # accumulated best read is the same one.
    # The 6 was measured against the GEMMA reader, where the limit was the API
    # rate, not the CPU -- keep that in mind before raising it for that engine.
    OCR_WORKERS = 32 if ocr_reader._get_engine() is None else 6
    attempts = crops_any = crops_conf = 0
    best = {}                               # (win,id) -> (number, conf, frame, bbox)

    def _attempt(job):
        key, f, bb = job
        crop = crops[(key, f)]          # cut above, while its frame was in hand
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
    # TRY MORE THAN ONE OTHER PICTURE. The first version tried exactly one
    # alternate crop and, if that crop happened to be illegible, wrote the
    # candidate off as "single_crop_only". Measured across every run on disk,
    # only 3 of 143 candidates ever reached "corroborated" (2.1%) -- and most of
    # those failures are not "she has no second readable moment", they are "we
    # looked once". Since the picks are already spread across her whole time on
    # court, several alternates are usually available for free.
    # HOW MANY alternate crops to try. Stays at 3 -- raising it was TESTED AND
    # REFUTED on 2026-09-09, and the refutation is worth more than the setting.
    #
    # The prediction: if each crop reads independently at rate p, corroboration
    # succeeds at 1-(1-p)^TRIES, so 3 -> 12 tries should lift TEST2 from ~18%
    # to ~55%. The observed 2-of-10 at TRIES=3 fit that model exactly (predicted
    # 17-22%), which is what made it credible.
    # The test: TRIES=12 on TEST2. Verified deterministically that the cap was
    # really binding -- crops actually tried per candidate went 3.00 -> 9.25
    # (HARD 2.65 -> 6.90), so the run genuinely spent 3x the reads.
    # THE RESULT: corroborated stayed at **2**. Zero extra corroborations for
    # three times the calls.
    #
    # WHAT THAT MEANS, and it closes a whole family of ideas: crops of one
    # player are NOT independent samples. Whether her number can be read is a
    # property of the PLAYER-STRETCH -- angle, occlusion, distance, motion --
    # not of which frame you happen to sample from it. If she is readable an
    # early crop reads; if she is not, neither re-ranking the crops
    # (CORROB_PICK_BY_SIZE, null) nor sampling more of them (this, null) nor
    # choosing them by facing (2026-09-09, null) recovers her.
    # Three independent interventions on WHICH/HOW MANY crops all measured
    # null. Do not propose a fourth. The remaining levers change what a crop
    # CONTAINS (torso framing -- the fixed 15-50% box is measurably slicing
    # numbers in half), or the reader, or the confirmation bar.
    CORROBORATION_TRIES = int(os.environ.get("CV_CORROBORATION_TRIES", "3") or 3)
    corroboration = {}                       # (win,id) -> "corroborated" | ...
    corrob_frames = {}                       # (win,id) -> [frames that agreed]
    verify_jobs = []
    # A SECOND OPINION MUST COME FROM A DIFFERENT MOMENT, and it cannot come
    # from picked_by_key.
    #
    # picked_by_key is BEST-CROPS-FIRST (sorted by box height) and spaced by
    # OCR_STRIDE = 2 frames, so all ten picks are the frames where she is
    # nearest the camera -- one moment. MEASURED on TEST1: the whole pick set
    # spans a median of 0.8 s, the four crops corroboration used spanned 0.3 s,
    # and 70 of 86 candidates had them under a second apart. Asking that set
    # again is asking "is this the same pose?", which is why it corroborated
    # 0 of 21 confident reads.
    # The reader's measured failure mode is ANGLE (backs, side-ons), not size,
    # so corroboration draws from her WHOLE track life instead, taking the
    # usable crops farthest in time from the read being checked.
    # Cost stays bounded: only candidates that ALREADY read get this, a handful
    # per clip, at CORROBORATION_TRIES frames each.
    #
    # RANK BY LEGIBILITY, NOT BY DISTANCE IN TIME (CORROB_PICK_BY_SIZE).
    # The two `>= 15` guards below ALREADY guarantee a different moment: one
    # keeps every pick 15+ frames from the read being checked, the other keeps
    # the picks 15+ frames from each other. Temporal diversity is therefore a
    # property of the CONSTRAINTS, and the sort adds none of it -- all the sort
    # decides is WHICH of the eligible crops get spent.
    # Sorting by -abs(g - f) spends them on the two ends of her track, which is
    # exactly where a player tends to be furthest from the camera and smallest.
    # Size is this project's proven legibility proxy (DECISIONS 4b) and the main
    # picker already uses it; this step was the one place that ignored it
    # completely.
    #
    # MEASURED 2026-09-09, deterministically, on HARD/TEST2/TEST1_REG (36 of 36
    # candidates, no reader calls so no noise): the new rule picks crops
    # **+19-24% taller**, 36 better / 0 worse -- AND it *increases* temporal
    # spread (34 -> 85 frames mean). The old comment claimed it drew "from her
    # WHOLE track life"; in practice sorting by distance takes the farthest
    # crop and then its immediate neighbours, CLUSTERING all three picks at one
    # end of the track ~1 second apart. So the old rule was not doing what it
    # said, and the new rule is strictly better on BOTH axes.
    #
    # ⚠ BUT IT DOES NOT MOVE THE OUTCOME. Corroboration succeeded 2 of 10-13 in
    # every TEST2 run on 2026-09-09 regardless of this flag. Sampling MORE crops
    # was then tried as the follow-up and also came back null (see
    # CORROBORATION_TRIES above) -- which together say the crops of one player
    # are not independent samples, so no re-pick of them recovers a stretch the
    # reader cannot read.
    # Kept ON anyway: free, strictly better on both measured axes, and it costs
    # nothing to leave a better crop in place. It is NOT a fix, and must not be
    # cited as one.
    # NOT facing: facing was A/B'd on 2026-09-09 and changed nothing, because
    # buying facing there meant paying in box size. Here size is not being
    # traded away -- it is introduced where there was no legibility signal at
    # all, and spread improves too.
    CORROB_PICK_BY_SIZE = bool(int(os.environ.get("CV_CORROB_BY_SIZE", "1") or 0))
    corrob_picks = {}
    for key, (_n, _c, f, _bb) in best.items():
        usable = [(g, bb) for (g, bb) in active_log[key]
                  if bb and (bb[3] - bb[1]) >= MIN_OCR_HEIGHT and abs(g - f) >= 15]
        if CORROB_PICK_BY_SIZE:
            usable.sort(key=lambda gb: -(gb[1][3] - gb[1][1]))   # most legible first
        else:
            usable.sort(key=lambda gb: -abs(gb[0] - f))          # farthest in time first
        spread = []
        for (g, bb) in usable:
            if all(abs(g - h) >= 15 for (h, _b) in spread):   # not near each other
                spread.append((g, bb))
            if len(spread) >= CORROBORATION_TRIES:
                break
        if not spread:
            corroboration[key] = "single_crop_only"   # no other moment exists
        corrob_picks[key] = spread

    need_frames = sorted({g for v in corrob_picks.values() for (g, _b) in v})
    corrob_crops = {}
    for g, im in s2mk.iter_frames(CLIP.video_path, need_frames):
        for key, v in corrob_picks.items():
            for (gg, bb) in v:
                if gg == g:
                    corrob_crops[(key, g)] = ocr_reader.jersey_crop(im, bb).copy()
    verify_jobs = [(key, g, bb) for key, v in corrob_picks.items() for (g, bb) in v]
    if verify_jobs:
        def _verify(job):
            key, g, bb = job
            crop = corrob_crops.get((key, g))
            reads = (ocr_reader.read_jersey(crop, roster.ROSTER_NUMBERS)
                     if crop is not None else [])
            return key, g, bb, reads
        with ThreadPoolExecutor(max_workers=OCR_WORKERS) as ex:
            vres = list(ex.map(_verify, verify_jobs))
        by_key = defaultdict(list)
        for key, f, bb, reads in vres:
            by_key[key].append((f, reads))
        conflicts = []
        for key, tries in by_key.items():
            first_num, _c, first_frame, _bb = best[key]
            agreed_on, disagreed_with, disagree_frame = None, None, None
            for f, reads in tries:
                conf_reads = [(n, c) for (n, c) in reads
                              if c >= ocr_reader.OCR_CONFIRM_THRESHOLD]
                if not conf_reads:
                    continue                  # illegible crop: no evidence either way
                second = max(conf_reads, key=lambda r: r[1])[0]
                if second == first_num:
                    agreed_on = f
                    break                     # one agreeing picture is enough
                disagreed_with, disagree_frame = second, f
            if agreed_on is not None:
                corroboration[key] = "corroborated"
                corrob_frames[key] = [first_frame, agreed_on]
            elif disagreed_with is not None:
                # TWO legible pictures of one girl disagree. One is a misread and
                # we cannot tell which, so she is NOT auto-confirmed -- she goes
                # to the human queue, which is the point.
                corroboration[key] = f"conflict_{first_num}_vs_{disagreed_with}"
                conflicts.append((key, first_num, disagreed_with, disagree_frame))
            else:
                # Every other picture was illegible. NOT a conflict -- most crops
                # are unreadable, which is the whole reason this stage
                # accumulates across a window.
                corroboration[key] = "single_crop_only"
        for (key, _a, _b, _f) in conflicts:
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
                "no_position_hypothesis": [], "established": []}

    # NUMBERS THAT ARE ON BOTH ROSTERS CANNOT ESTABLISH ANYTHING. HARD lists #3
    # and #23 on both teams, TEST2 lists #1, #4 and #13. A read of one of those
    # names two different girls at once, so it may CONFIRM a click (which
    # already carries a team) but must never CREATE an identity. color_tiebreak
    # is the tool that would resolve it; until it is wired in here, abstain.
    dual = set()
    for i, t1 in enumerate(CLIP.teams):
        for t2 in CLIP.teams[i + 1:]:
            dual |= set(t1.numbers) & set(t2.numbers)
    if dual:
        print(f"[stage6] numbers on BOTH rosters, never used to establish an "
              f"identity: {sorted(dual)}")

    for key in candidates:
        ident = ident_of[key]
        b = best.get(key)
        num, conf, frame = (b[0], b[1], b[2]) if b else (None, None, None)
        res = machines[key[0]].promote_via_second_signal(ident, num, conf)
        # A CONFIDENT READ WITH NOTHING TO AGREE WITH USED TO BE BINNED HERE.
        # On a game nobody has clicked that is EVERY read: all 21 of Full_Game's
        # candidates carried roster_number = None, and four reads at confidence
        # 1.00 (two corroborated) were discarded [MEASURED 2026-08-31]. The
        # jersey may now name her itself -- but only on two agreeing crops from
        # different moments, and never on a dual-roster number.
        if (res == "no_position_hypothesis"
                and corroboration.get(key) == "corroborated"
                and num not in dual):
            res2 = machines[key[0]].establish_via_reads(
                ident, num, conf, corroborating_frames=corrob_frames.get(key))
            if res2 == "established":
                res = "established"
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
    # Re-read just these frames, one at a time. They are only the players who
    # were actually named -- a handful, not the whole attempt pool -- so this
    # costs one seek each and holds one frame, instead of keeping every OCR
    # frame alive to the end of the stage for the sake of a few pictures.
    confirmed_by_frame = defaultdict(list)
    for (key, b) in outcomes["agree"]:
        confirmed_by_frame[b[2]].append((key, b))
    for f, img in s2mk.iter_frames(CLIP.video_path, sorted(confirmed_by_frame)):
        for (key, b) in confirmed_by_frame[f]:
            n, c, _f, bb = b
            still = img.copy()          # one frame, possibly two players on it
            x1, y1, x2, y2 = [int(v) for v in bb]
            cv2.rectangle(still, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(still, f"#{n} CONFIRMED via OCR ({c:.2f})", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imwrite(
                os.path.join(OUT_DIR, f"{clip}_ocr_confirm_w{key[0]}_id{key[1]}_f{f}.jpg"),
                cv2.resize(still, (1280, 720)))
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
        # NOT READ is not the same as UNREADABLE. Persisted so a budgeted run
        # can never be mistaken for a run where the reader tried and failed.
        "candidate_budget": budget or None,
        "candidates_skipped_for_budget": skipped_for_budget,
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
