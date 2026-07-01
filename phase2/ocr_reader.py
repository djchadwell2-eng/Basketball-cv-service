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

import cv2

# --- THE AUTONOMY DIAL (single source of truth) ------------------------------
OCR_CONFIRM_THRESHOLD = 0.85     # strict first pass. Clear jerseys read ~0.95-0.98.

MIN_CROP_HEIGHT_PX = 24          # below this the number is unresolvable -> no attempt
_UPSCALE = 4                     # EasyOCR does better on upscaled small crops

_reader = None


def _get_reader():
    global _reader
    if _reader is None:
        import easyocr                              # lazy: heavy import on first use only
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


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


def read_jersey(crop_bgr, roster_numbers):
    """Return [(number:int, confidence:float), ...] for CLOSED-SET roster matches.

    Pluggable engine seam. Empty list = no confident on-roster read (the common
    case when the number is turned away/blurred -- expected, not a failure).
    """
    if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < MIN_CROP_HEIGHT_PX:
        return []
    up = cv2.resize(crop_bgr, (crop_bgr.shape[1] * _UPSCALE, crop_bgr.shape[0] * _UPSCALE),
                    interpolation=cv2.INTER_CUBIC)
    out = []
    for (_box, txt, conf) in _get_reader().readtext(up, allowlist="0123456789"):
        if txt.isdigit() and int(txt) in roster_numbers:      # closed-set filter
            out.append((int(txt), float(conf)))
    return out


def best_on_roster_read(crop_bgr, roster_numbers):
    """Highest-confidence closed-set read for one crop, or None."""
    reads = read_jersey(crop_bgr, roster_numbers)
    return max(reads, key=lambda r: r[1]) if reads else None
