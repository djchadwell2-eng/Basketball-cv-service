"""CAN A MODEL PIECE A NUMBER TOGETHER ACROSS FRAMES that no single frame
shows clearly?

WHY THIS IS NOT ALREADY CLOSED. INDIVIDUAL_TRACKER S4b closed three
interventions -- pick face-on crops, pick bigger crops, pick more crops -- all
null, with the shared cause "readability is a property of the player-stretch,
not of the frame you sample from it." **That conclusion assumed every frame is
judged ON ITS OWN**, which is what the pipeline does today: pick one crop, ask,
throw the rest away. If a model can INTEGRATE across frames, the information
available is the UNION of what is visible, not the best single frame -- frame 1
showing "2_" behind an arm and frame 5 showing "_3" is unreadable twice and
readable together. No experiment here has tested that.

NOT the same as ocr_reader's existing sheet path, which packs TWELVE DIFFERENT
players into one call for COST and takes twelve separate answers out. This is
N frames of ONE player -> ONE answer.

THE DANGER, and the reason this measures what it measures: asking a model to
combine partial glimpses is an invitation to confabulate. A wrong name is worse
than no name here (a wrong name is silently attributed to a real girl for a
whole stretch of the game). So the number that decides this is NOT how many
extra girls get named -- it is **how many get named WRONG**. A method that
finds 5 more and invents 2 is a regression, not a win.

DISCIPLINE: the population that matters is candidates the CURRENT single-frame
reader FAILED on. A method that only works where the pipeline already succeeds
adds nothing.

Usage:
    .venv/Scripts/python.exe spikes/multiframe_read_test.py <CLIP>
    (votes cache to spikes/out/multiframe_<CLIP>.json -- rescoring is free)
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
from collections import Counter, defaultdict

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "phase2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CLIP = sys.argv[1] if len(sys.argv) > 1 else "TEST1"
MIN_OCR_HEIGHT = 90
N_FRAMES = 8          # frames of one player shown in a single call
MIN_GAP = 8           # frames apart, so they are genuinely different moments
READS = 3             # same unanimous-of-3 bar the single-frame reader uses
CACHE = os.path.join(_HERE, "out", f"multiframe_{CLIP}.json")

PROMPT = (
    "These {n} photos are the SAME basketball player, from different moments "
    "of one play. Her jersey number may be partly hidden, blurred or turned "
    "away in ANY single photo -- use ALL of them together to work out the "
    "number.\n"
    "It must be one of exactly these numbers: {roster}.\n"
    "Answer with ONLY the number. If the photos together still do not let you "
    "read it, answer exactly: NONE"
)


def _b64(img):
    ok, jpg = cv2.imencode(".jpg", img)
    return base64.standard_b64encode(jpg.tobytes()).decode() if ok else None


def ask(client, model, crops, roster_numbers, retries=3):
    """N crops of ONE player -> one on-roster number, or None."""
    parts = [PROMPT.format(n=len(crops), roster=sorted(roster_numbers))]
    for c in crops:
        b = _b64(c)
        if b is None:
            continue
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": b}})
    for attempt in range(retries):
        try:
            r = client.models.generate_content(model=model, contents=parts)
        except Exception:
            time.sleep(1.5 * (attempt + 1))
            continue
        txt = (r.text or "").strip().upper()
        digits = "".join(ch for ch in txt if ch.isdigit())
        if not digits or "NONE" in txt:
            return None
        n = int(digits)
        return n if n in roster_numbers else None      # closed-set filter
    return None


def main():
    import clip_config
    import run_clip
    cfg = clip_config.get_clip(CLIP)
    run_clip._sync_and_guard(cfg)

    import oncourt
    import roster
    import ocr_reader
    import windows as winmod
    import stage2_multikeyframe as s2mk
    from tracking import Track

    ocr = json.load(open(os.path.join(_ROOT, "phase2", "out",
                                      f"{CLIP}_ocr_confirms.json"), encoding="utf-8"))
    boundaries = ocr["window_boundaries"]

    doc = json.load(open(cfg.tracks_cache_path, encoding="utf-8"))
    frames = [(fr["frame_index"],
               [Track(t["track_id"], tuple(t["bbox"])) for t in fr["tracks"]])
              for fr in doc["frames"]]

    onc = oncourt.on_court_by_window(oncourt.load_checked(cfg), boundaries=boundaries)
    wid = winmod.WindowedIdentity(boundaries=boundaries)
    active_log, ident_of = defaultdict(list), {}
    seen, prev_win = set(), None
    for (fidx, tracks) in frames:
        win = wid.update(fidx, tracks)
        m = wid.current_machine()
        if win != prev_win:
            seen = set()
            on = onc.get(win, set())
            refs = roster.ref_tracks() | roster._spliced()
            for t in tracks:
                if t.track_id in on and t.track_id not in refs:
                    m.seed(t.track_id, roster_number=roster.seed_number_for(CLIP, t.track_id))
            prev_win = win
        else:
            winmod.seed_labeled_newcomers(m, tracks, seen, onc.get(win, set()),
                                          lambda tid: roster.seed_number_for(CLIP, tid))
        seen |= {t.track_id for t in tracks}
        for ident in m.active():
            active_log[(win, ident.identity_id)].append((fidx, ident.last_bbox))
            ident_of[(win, ident.identity_id)] = ident

    # WHAT THE SINGLE-FRAME READER MANAGED, so the two can be compared on the
    # population that actually matters (the ones it failed).
    single = {}
    for rows in ocr["outcomes"].values():
        for r in rows:
            single[(r["window"], r["identity_id"])] = r.get("read_number")

    # GROUND TRUTH: the human's own track labels. Known to contain errors
    # (a referee recorded as #30) -- disagreements get eyeballed, not trusted.
    truth = {}
    for key, ident in ident_of.items():
        n = roster.seed_number_for(CLIP, ident.track_id)
        if n is not None:
            truth[key] = n

    jobs = []
    for key, log in active_log.items():
        if key not in truth:
            continue                    # nothing to score against
        usable = [(f, bb) for (f, bb) in log
                  if bb and (bb[3] - bb[1]) >= MIN_OCR_HEIGHT]
        picks = []
        for (f, bb) in sorted(usable, key=lambda fb: -(fb[1][3] - fb[1][1])):
            if all(abs(f - g) >= MIN_GAP for (g, _b) in picks):
                picks.append((f, bb))
            if len(picks) >= N_FRAMES:
                break
        if len(picks) >= 2:             # one frame is not a multi-frame test
            jobs.append((key, sorted(picks)))

    print(f"{CLIP}: {len(jobs)} labelled candidate(s) with >=2 usable frames")
    if not jobs:
        return

    if os.path.exists(CACHE):
        results = {tuple(json.loads(k)): v for k, v in
                   json.load(open(CACHE, encoding="utf-8")).items()}
        print(f"reusing {len(results)} cached answer(s) (delete {CACHE} to re-ask)")
    else:
        import env_local
        env_local.load()
        import google.genai
        client = google.genai.Client(api_key=os.environ["GEMINI_API_KEY"])

        need = sorted({f for (_k, picks) in jobs for (f, _b) in picks})
        imgs = dict(s2mk.iter_frames(cfg.video_path, need))
        results = {}
        for i, (key, picks) in enumerate(jobs, 1):
            crops = []
            for (f, bb) in picks:
                im = imgs.get(f)
                if im is None:
                    continue
                c = ocr_reader.jersey_crop(im, bb)
                if c is not None and c.size:
                    crops.append(cv2.resize(c, None, fx=2, fy=2,
                                            interpolation=cv2.INTER_CUBIC))
            if len(crops) < 2:
                continue
            votes = [ask(client, ocr_reader.GEMMA_MODEL, crops, roster.ROSTER_NUMBERS)
                     for _ in range(READS)]
            named = [v for v in votes if v is not None]
            if named:
                num, cnt = Counter(named).most_common(1)[0]
                results[key] = [num, cnt / float(READS), len(crops)]
            else:
                results[key] = [None, 0.0, len(crops)]
            print(f"  [{i}/{len(jobs)}] w{key[0]} id{key[1]}: {results[key]}", flush=True)
            # SAVE AFTER EVERY CANDIDATE. An 8-image call takes ~90 s (MEASURED
            # -- roughly 3x a single-image read), so a whole clip runs long
            # enough to hit any wall clock a caller sets. The first attempt
            # timed out at candidate 10 of 13 and lost all ten. Answers are the
            # expensive part; never hold them in memory to the end.
            json.dump({json.dumps(list(k)): v for k, v in results.items()},
                      open(CACHE, "w", encoding="utf-8"), indent=1)
        print(f"saved -> {CACHE}")

    report(results, single, truth, ocr)


def report(results, single, truth, ocr):
    THRESH = ocr["ocr_confirm_threshold"]
    buckets = {"rescued": [], "wrong_new": [], "still_none": [],
               "agreed": [], "contradicted": [], "lost": []}
    for key, (num, conf, ncrops) in results.items():
        t = truth.get(key)
        s = single.get(key)
        confident = num is not None and conf >= THRESH
        if s is None:                                  # single-frame FAILED
            if not confident:
                buckets["still_none"].append((key, num, conf))
            elif num == t:
                buckets["rescued"].append((key, num, conf))
            else:
                buckets["wrong_new"].append((key, num, conf, t))
        else:                                          # single-frame succeeded
            if not confident:
                buckets["lost"].append((key, s))
            elif num == s:
                buckets["agreed"].append((key, num))
            else:
                buckets["contradicted"].append((key, num, s, t))

    print(f"\n{'=' * 70}\nMULTI-FRAME READ  ({len(results)} labelled candidates)\n{'=' * 70}")
    print("\nON CANDIDATES THE SINGLE-FRAME READER FAILED  <- the point of this")
    print(f"   RESCUED (new, and CORRECT) : {len(buckets['rescued'])}")
    print(f"   WRONG   (new, and WRONG)   : {len(buckets['wrong_new'])}   <-- the number that decides it")
    print(f"   still abstained            : {len(buckets['still_none'])}")
    print("\nON CANDIDATES IT ALREADY READ  <- checking it does no harm")
    print(f"   agreed                     : {len(buckets['agreed'])}")
    print(f"   CONTRADICTED               : {len(buckets['contradicted'])}")
    print(f"   went quiet (lost a read)   : {len(buckets['lost'])}")

    for (key, num, conf) in buckets["rescued"]:
        print(f"\n   RESCUED w{key[0]} id{key[1]}: read #{num} @ {conf:.2f} (label agrees)")
    for (key, num, conf, t) in buckets["wrong_new"]:
        print(f"\n   ⚠ WRONG w{key[0]} id{key[1]}: said #{num} @ {conf:.2f}, label says #{t}")
    for (key, num, s, t) in buckets["contradicted"]:
        print(f"\n   ⚠ CONTRADICTS w{key[0]} id{key[1]}: multi #{num}, single #{s}, label #{t}")

    n_new = len(buckets["rescued"]) + len(buckets["wrong_new"])
    if n_new:
        print(f"\n   precision on NEW names: {len(buckets['rescued'])}/{n_new} "
              f"= {100.0 * len(buckets['rescued']) / n_new:.0f}%")
    print("\nREMINDER: every disagreement with a label must be EYEBALLED before it "
          "is called an error --\n{clip}_decisions.json is known to contain wrong "
          "labels (a referee recorded as #30).")


if __name__ == "__main__":
    main()
