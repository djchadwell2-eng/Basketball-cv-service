"""Pluggable jersey-number reader (closed-set) + the ONE autonomy dial.

read_jersey() is the swappable engine seam: today it wraps EasyOCR; a future
fine-grained reader drops in behind the same signature. Reads are CLOSED-SET --
only numbers on the roster count, so open-world OCR noise is filtered out.

OCR_CONFIRM_THRESHOLD is THE autonomy dial (kept here, one place, not scattered):
only reads at least this confident are allowed to auto-confirm. Lower it LATER,
one notch at a time, while watching for swaps -- never to shrink the review queue.

easyocr is imported lazily (inside the reader) so importing this module for the
threshold does not drag the heavy OCR stack into the other phase-2 stages.
"""

from __future__ import annotations

import os
import sys
import threading

import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# --- THE AUTONOMY DIAL (single source of truth) ------------------------------
OCR_CONFIRM_THRESHOLD = 0.85     # strict first pass. Clear jerseys read ~0.95-0.98.

MIN_CROP_HEIGHT_PX = 24          # below this the number is unresolvable -> no attempt
_UPSCALE = 4                     # both engines do better on upscaled small crops

# --- THE ENGINE (the seam this module was always built around) ---------------
# How many independent reads the vision engine must agree on. THIS IS THE
# CONFIDENCE MECHANISM: a VLM returns a number with no calibrated score, so
# agreement across repeated reads stands in for one. Measured 2026-08-03 -- it
# refused a real "23" that read [3, 30, 3], and refused a REFEREE that read
# [13, 10, 10]. Majority-of-3 would have named that referee "10".
GEMMA_READS = 3
GEMMA_MODEL = "gemma-4-26b-a4b-it"
# Milliseconds. A read that has not come back in this long is not coming back
# usefully: a legible crop answers in ~8 s (MEASURED), so 90 s is generous, and
# the alternative is a thread that waits forever (see _init_engine).
GEMMA_TIMEOUT_MS = 90_000

# The reported confidence is the AGREEMENT FRACTION, not a probability, and that
# is deliberate: with 3 reads, unanimous = 1.00 and 2-of-3 = 0.67, so the
# existing OCR_CONFIRM_THRESHOLD of 0.85 enforces unanimity on its own. One
# dial, still meaning "how sure are we", no second threshold to drift apart.
# Raising GEMMA_READS keeps that property (4-of-5 = 0.8, still below the bar).

_reader = None
_engine = None
_engine_lock = threading.Lock()


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr                              # lazy: heavy import on first use only
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def _get_engine():
    """The Gemma client, or None to fall back to EasyOCR.

    Says out loud which reader is in use. A silent fallback is what hid the
    scoreboard key bug for a whole session (see env_local): a run that names
    almost nobody must not look identical to a run that is working.
    Set JERSEY_ENGINE=easyocr to force the old reader.

    THREAD-SAFE ON PURPOSE. stage6 reads crops from a thread pool, and the
    unlocked version raced: several workers ran this at once, some printed
    "no GEMINI_API_KEY" and fell back to EasyOCR while others built the Gemma
    client, so the same run silently used two different readers on different
    crops. The lock makes the choice happen once for the whole process.
    """
    global _engine
    with _engine_lock:
        return _init_engine()


def _init_engine():
    global _engine
    if _engine is not None:
        return _engine or None
    if os.environ.get("JERSEY_ENGINE", "").lower() == "easyocr":
        print("[ocr_reader] JERSEY_ENGINE=easyocr -- using EasyOCR")
        _engine = False
        return None
    try:
        import env_local
        env_local.load()
    except Exception:
        pass
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        print("[ocr_reader] no GEMINI_API_KEY -- falling back to EasyOCR, which "
              "reads far fewer numbers on this footage (see read_jersey)")
        _engine = False
        return None
    try:
        import google.genai
        # A CALL THAT NEVER RETURNS IS WORSE THAN A CALL THAT FAILS. There was
        # no timeout here at all, and a single hung request blocks its thread
        # forever: measured 2026-08-24, a sheet request stopped responding and
        # sat for 28 minutes with no error and no progress. In a stage that
        # makes tens of thousands of these, one hang burns the whole job until
        # RunPod's 180-minute cap kills it -- after the slices have been paid
        # for. _gemma_once already treats a failure as "not evidence", so a
        # timeout costs one crop's read and nothing else.
        try:
            _engine = google.genai.Client(
                api_key=key, http_options={"timeout": GEMMA_TIMEOUT_MS})
        except TypeError:              # older SDK without http_options
            _engine = google.genai.Client(api_key=key)
            print("[ocr_reader] WARNING: this google-genai cannot take a "
                  "timeout -- a hung call will block until the job is killed")
        print(f"[ocr_reader] jersey reader: Gemma {GEMMA_MODEL}, "
              f"unanimous-of-{GEMMA_READS}")
    except Exception as e:
        print(f"[ocr_reader] Gemma unavailable ({e}) -- falling back to EasyOCR")
        _engine = False
    return _engine or None


# THE SCOREBUG GUARD THAT DID NOT WORK -- recorded so nobody rebuilds it.
#
# The one wrong read in the 2026-08-03 head-to-head was TEST1 track 467: the
# burned-in scoreboard, tracked as a player, read as a roster number three times
# running. The obvious fix looked free -- every clip already marks a rectangle in
# clips_config exclude_regions, so skip any body box sitting inside it.
#
# MEASURED BEFORE SHIPPING, and it is WORSE than the problem. Those rectangles
# are SIFT masks, not scoreboards: they are drawn generously to cover the whole
# burned-in corner. TEST2's covers the bench, and the guard would have deleted
# tracks 16 and 22 -- two real players whose "24" and "13" are the most legible
# numbers in that entire clip. Rendered and eyeballed to confirm.
#
# Motion does not separate them either: the static graphic moves 0.67 px/frame
# and a real player standing on the bench moves 1.95, with plenty of overlap in
# between. There is no clean automatic line here.
#
# WHAT TO DO INSTEAD: the pipeline already has a designed path for "this track
# is not a player" -- roster.load_ref_tracks, the human ref/bench labels that
# ball_touch also uses to stop crediting referees with the ball. Labelling the
# scoreboard track once, the same way a referee is labelled, costs one click and
# cannot delete anybody by accident.

def jersey_crop(frame_bgr, bbox):
    """Upper-central torso patch of a player box (where the number sits)."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    w, h = x2 - x1, y2 - y1
    jx1, jx2 = x1 + int(0.15 * w), x1 + int(0.85 * w)
    jy1, jy2 = y1 + int(0.15 * h), y1 + int(0.50 * h)
    H, W = frame_bgr.shape[:2]
    jx1, jy1 = max(0, jx1), max(0, jy1)
    jx2, jy2 = min(W, jx2), min(H, jy2)
    return frame_bgr[jy1:jy2, jx1:jx2]


def _read_easyocr(up, roster_numbers):
    out = []
    for (_box, txt, conf) in _get_reader().readtext(up, allowlist="0123456789"):
        if txt.isdigit() and int(txt) in roster_numbers:      # closed-set filter
            out.append((int(txt), float(conf)))
    return out


def _gemma_once(client, jpg_b64, roster_numbers):
    """One read -> an on-roster number, or None. Any failure is None (a call
    that errored is NOT evidence of anything, and must not count as a vote)."""
    prompt = (
        "This is a cropped photo of a basketball player's torso. What number is "
        "printed on her jersey?\n"
        f"It must be one of exactly these numbers: {sorted(roster_numbers)}.\n"
        "Answer with ONLY the number. If you cannot clearly read a number, or "
        "the picture is not a player in a jersey, answer exactly: NONE")
    try:
        r = client.models.generate_content(
            model=GEMMA_MODEL,
            contents=[prompt, {"inline_data": {"mime_type": "image/jpeg",
                                               "data": jpg_b64}}])
    except Exception:
        return None
    txt = (r.text or "").strip().upper()
    digits = "".join(c for c in txt if c.isdigit())
    if not digits or "NONE" in txt:
        return None
    n = int(digits)
    return n if n in roster_numbers else None                 # closed-set filter


def _read_gemma(up, roster_numbers):
    """Vision-model reader. Returns at most ONE candidate, with the agreement
    fraction as its confidence (see GEMMA_READS)."""
    import base64
    from collections import Counter
    client = _get_engine()
    ok, jpg = cv2.imencode(".jpg", up)
    if not ok:
        return []
    b64 = base64.standard_b64encode(jpg.tobytes()).decode("utf-8")

    # STOP THE MOMENT THE BAR IS OUT OF REACH.
    #
    # Confidence is agreement over GEMMA_READS, and OCR_CONFIRM_THRESHOLD is
    # 0.85, so with three reads only a unanimous 3/3 = 1.00 ever clears it --
    # 2-of-3 is 0.67 and already a reject. The answer is therefore DECIDED the
    # first time a read disagrees, and every call after that buys nothing.
    #
    # It buys nothing on the common case in particular: most crops are simply
    # unreadable and come back NONE on the first call. Measured on DJ's game,
    # a whole game offers ~23,288 candidates and up to ten crops each, which is
    # ~698,000 calls at three per crop. Gating on the first read makes that
    # ~256,000 -- the same confirmations, for a third of the waiting.
    #
    # It BREAKS rather than returning: the reads already made were paid for and
    # are still evidence, so the confidence below is computed over the votes we
    # actually have, out of GEMMA_READS exactly as before. Two of this module's
    # own tests caught the earlier version throwing that away -- [24, None,
    # None] must still report 24 at 0.33, not nothing.
    #
    # WHAT DOES CHANGE, said out loud rather than hidden: when the FIRST read
    # fails, later reads that might have named somebody are never made, so that
    # crop reports no read where before it could have reported a sub-threshold
    # one. No confirmation moves -- a sub-threshold read is a reject either way,
    # and nothing below the bar ever reaches the state machine -- but stage6's
    # "any on-roster read" tally is a readability diagnostic, and it now counts
    # slightly fewer.
    votes = []
    for i in range(GEMMA_READS):
        votes.append(_gemma_once(client, b64, roster_numbers))
        named_so_far = [v for v in votes if v is not None]
        best = Counter(named_so_far).most_common(1)[0][1] if named_so_far else 0
        if (best + GEMMA_READS - (i + 1)) / float(GEMMA_READS) < OCR_CONFIRM_THRESHOLD:
            break                      # the bar is out of reach; the rest cannot change it
    named = [v for v in votes if v is not None]
    if not named:
        return []
    number, count = Counter(named).most_common(1)[0]
    # Denominator is EVERY read, so a refusal counts against agreement. Two
    # reads saying 24 and one saying "cannot tell" is 0.67, not 1.0 -- the
    # crop was not legible enough three times running.
    return [(number, count / float(GEMMA_READS))]


# --- MANY CROPS, ONE CALL ------------------------------------------------
# A whole game offers ~150,000 crops (MEASURED). One call each is the single
# largest cost in the pipeline, and a vision model can be shown a sheet of
# crops and asked about all of them at once.
#
# WHY TWELVE, and not forty-eight. MEASURED on real crops from DJ's film: a
# cell must be 372 x 356 to hold 90% of them WITHOUT SHRINKING, and shrinking
# is not available -- these numbers are already at the edge of what anyone can
# read. So cells can only be bought with picture size:
#     6 cells  1116 x 712        12 cells  1488 x 1068
#    24 cells  2232 x 1424       48 cells  2976 x 2136
# Vision APIs downscale above roughly 1536 on a side, which would shrink every
# jersey and quietly undo the whole exercise. Twelve is about the largest sheet
# that keeps every crop at full size. It is also where the returns are: 1 -> 12
# is twelvefold, 12 -> 24 only another double, while the chance of the model
# mapping a number to the wrong cell grows the whole way.
GRID_CELLS = 12
GRID_COLS = 4
_CELL_W, _CELL_H = 372, 356
# Cells are labelled with LETTERS, never digits. A digit painted next to a
# jersey is an invitation to read the label as the number.
_CELL_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _cell_labels(n):
    """n distinct letter labels: A..Z, then AA, AB, ... Still never a digit.

    A fixed twelve-letter alphabet silently capped how big a sheet could ever
    be -- raising GRID_CELLS past it crashed rather than saying so, which is a
    poor way for a limit to announce itself.
    """
    out = []
    i = 0
    while len(out) < n:
        q, r = divmod(i, 26)
        out.append((_CELL_LABELS[q - 1] if q else "") + _CELL_LABELS[r])
        i += 1
    return out


def _grid_image(crops):
    """Lay crops out as a labelled sheet. Returns (image, [labels])."""
    import numpy as np
    n = len(crops)
    rows = (n + GRID_COLS - 1) // GRID_COLS
    canvas = np.zeros((rows * _CELL_H, GRID_COLS * _CELL_W, 3), np.uint8)
    labs = _cell_labels(n)
    labels = []
    for i, c in enumerate(crops[:GRID_CELLS]):
        r, col = divmod(i, GRID_COLS)
        y0, x0 = r * _CELL_H, col * _CELL_W
        h, w = min(c.shape[0], _CELL_H), min(c.shape[1], _CELL_W)
        canvas[y0:y0 + h, x0:x0 + w] = c[:h, :w]
        cv2.rectangle(canvas, (x0, y0), (x0 + _CELL_W - 2, y0 + _CELL_H - 2),
                      (0, 255, 255), 2)
        lab = labs[i]
        cv2.rectangle(canvas, (x0 + 2, y0 + 2),
                      (x0 + 14 + 20 * len(lab), y0 + 30), (0, 0, 0), -1)
        cv2.putText(canvas, lab, (x0 + 8, y0 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        labels.append(lab)
    return canvas, labels


def _gemma_grid_once(client, jpg_b64, labels, roster_numbers):
    """One read of a sheet -> {label: number|None}, or None if the reply is not
    trustworthy. A malformed or short answer is REFUSED rather than partially
    believed: a sheet whose cells cannot be lined up with its answers is exactly
    how one girl's number lands on another girl."""
    import json as _json
    prompt = (
        "This sheet shows " + str(len(labels)) + " separate cropped photos of "
        "basketball players' torsos, in cells labelled "
        + ", ".join(labels) + ".\n"
        "For EACH cell, what number is printed on that player's jersey?\n"
        f"Each number must be one of exactly these: {sorted(roster_numbers)}.\n"
        "The letter in the corner of a cell is a label, NOT a jersey number.\n"
        'Answer with ONLY a JSON object mapping every cell label to its number, '
        'or to "NONE" if you cannot clearly read that one. Example: '
        '{"A": 24, "B": "NONE"}')
    try:
        r = client.models.generate_content(
            model=GEMMA_MODEL,
            contents=[prompt, {"inline_data": {"mime_type": "image/jpeg",
                                               "data": jpg_b64}}])
        txt = (r.text or "").strip()
    except Exception:
        return None
    a, b = txt.find("{"), txt.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        raw = _json.loads(txt[a:b + 1])
    except ValueError:
        return None
    if not isinstance(raw, dict) or set(raw) != set(labels):
        return None                    # cells and answers do not line up: refuse
    out = {}
    for lab in labels:
        v = raw[lab]
        if isinstance(v, bool):
            return None
        try:
            n = int(v)
        except (TypeError, ValueError):
            out[lab] = None            # "NONE" and anything unparseable
            continue
        out[lab] = n if n in roster_numbers else None      # closed-set filter
    return out


def read_jersey_batch(crops, roster_numbers):
    """Read up to GRID_CELLS crops in ONE call each round. Returns a list the
    same length as `crops`, each entry shaped exactly like read_jersey's.

    The confidence rule is unchanged: a cell is only believed if every read
    agrees, so GEMMA_READS sheets are still needed for a confirmation. But the
    cheap half of lever one applies here too -- a cell that reads NONE on the
    first sheet can never be unanimous, so the later sheets carry ONLY the cells
    that named somebody. On real footage that is a handful, so sheets two and
    three are tiny.

    Repacking those survivors also moves them to different positions, which
    means a positional mix-up would have to happen twice, in two different
    layouts, to survive -- and stage6's corroboration rule then demands a third
    agreement from a DIFFERENT crop of the same player.

    Falls back to one-at-a-time whenever the batch reader is unavailable or a
    sheet comes back untrustworthy. Never guesses.
    """
    crops = list(crops)[:GRID_CELLS]
    if not crops:
        return []
    client = _get_engine()
    if client is None:
        return [read_jersey(c, roster_numbers) for c in crops]

    import base64
    from collections import Counter as _C
    votes = {i: [] for i in range(len(crops))}
    alive = list(range(len(crops)))
    for rnd in range(GEMMA_READS):
        if not alive:
            break
        sheet, labels = _grid_image([crops[i] for i in alive])
        ok, jpg = cv2.imencode(".jpg", sheet)
        if not ok:
            return [read_jersey(c, roster_numbers) for c in crops]
        b64 = base64.standard_b64encode(jpg.tobytes()).decode("utf-8")
        got = _gemma_grid_once(client, b64, labels, roster_numbers)
        if got is None:                       # untrustworthy sheet -> do it slowly
            return [read_jersey(c, roster_numbers) for c in crops]
        still = []
        for lab, idx in zip(labels, alive):
            votes[idx].append(got[lab])
            if got[lab] is not None:
                still.append(idx)
        alive = still                         # a NONE can never become unanimous

    out = []
    for i in range(len(crops)):
        named = [v for v in votes[i] if v is not None]
        if not named:
            out.append([])
            continue
        # Agreement over GEMMA_READS, exactly as the single-crop path computes
        # it. A cell dropped after one read has one vote out of three -- 0.33,
        # nowhere near the bar -- which is the same answer the slow path gives
        # for the same evidence.
        number, count = _C(named).most_common(1)[0]
        out.append([(number, count / float(GEMMA_READS))])
    return out


def read_jersey(crop_bgr, roster_numbers):
    """Return [(number:int, confidence:float), ...] for CLOSED-SET roster matches.

    Pluggable engine seam. Empty list = no confident on-roster read (the common
    case when the number is turned away/blurred -- expected, not a failure).

    WHICH ENGINE, and why it changed (measured 2026-08-03). On identical crops,
    with identical selection and the same closed-set filter:
        EasyOCR @0.85          1 correct,  0 wrong
        Gemma unanimous-of-3  12 correct,  1 wrong
    EasyOCR is not broken -- this footage is simply past its limit, exactly as
    DECISIONS 4b concluded (players are small and far; 22 confident reads out of
    232 crops on TEST1, 7 of ~395 on HARD). The vision model reads numbers a
    human can read and EasyOCR cannot.
    The single wrong read was the SCOREBOARD GRAPHIC, not a player, and is now
    refused upstream by is_burned_in_graphic().

    EasyOCR remains the fallback whenever no API key is configured, so the
    pipeline still runs offline -- it just names far fewer girls.
    """
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < MIN_CROP_HEIGHT_PX:
        return []
    up = cv2.resize(crop_bgr, (crop_bgr.shape[1] * _UPSCALE, crop_bgr.shape[0] * _UPSCALE),
                    interpolation=cv2.INTER_CUBIC)
    return (_read_gemma(up, roster_numbers) if _get_engine()
            else _read_easyocr(up, roster_numbers))


def best_on_roster_read(crop_bgr, roster_numbers):
    """Highest-confidence closed-set read for one crop, or None."""
    reads = read_jersey(crop_bgr, roster_numbers)
    return max(reads, key=lambda r: r[1]) if reads else None
