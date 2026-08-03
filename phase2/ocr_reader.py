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
        _engine = google.genai.Client(api_key=key)
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
    votes = [_gemma_once(client, b64, roster_numbers) for _ in range(GEMMA_READS)]
    named = [v for v in votes if v is not None]
    if not named:
        return []
    number, count = Counter(named).most_common(1)[0]
    # Denominator is EVERY read, so a refusal counts against agreement. Two
    # reads saying 24 and one saying "cannot tell" is 0.67, not 1.0 -- the
    # crop was not legible enough three times running.
    return [(number, count / float(GEMMA_READS))]


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
