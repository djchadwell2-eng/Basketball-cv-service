"""IDENTITY WINDOW BOUNDARIES -- where to cut the clip so names can't leak.

WHAT THIS IS FOR, precisely: the identity/OCR layer resets itself at every
window boundary (phase2/windows.py) so a wrong name in one stretch cannot
corrupt the next. That job needs SOME cut points. It does not care what the
cuts mean -- only that they land where the play changed, so a reset doesn't
chop a girl's identity in half mid-drive.

WHAT THIS IS NOT -- read this before reusing it: these cuts are NOT
possessions. This module used to be called possessions.py and claimed they
were. They are "which half of the court is the centre of mass on", which is a
camera/court signal, not a ball signal -- both teams stand at the same end on
every made basket, and an offence can spend a whole possession in its own
backcourt. DJ (2026-08-02): "that's not how possessions work from a half
court. That's not what I want."

REAL possessions live in phase2/team_possessions.py, built on ball tracking
plus jersey colour. That module owns the word "possession". This one owns
nothing but cut points, and its output is named {clip}_id_windows.json so no
one mistakes it for basketball again.

The signal is FREE: the on-court cache already stores court_feet for every
track on every frame. No new perception, no new cache -- boundaries are
derived deterministically at load time, and an INSPECTION artifact is written
so a human can scrub the video at each boundary timestamp.

ABSTENTION: a degenerate signal (no frames confidently on either side) returns
None and callers fall back to the fixed windows LOUDLY. Bad boundaries are
worse than boring ones.
"""

from __future__ import annotations

import json
import os
import sys
from bisect import bisect_right

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "spikes"))

HOLD_S = 1.5          # a side switch must hold this long to count
MIN_SEG_S = 4.0       # shorter same-side runs merge into their neighbor
MARGIN_FT = 6.0       # dead zone around midcourt (transition = no side)


def detect(xs, fps, court_len, hold_s=HOLD_S, min_poss_s=MIN_SEG_S,
           margin_ft=MARGIN_FT):
    """xs: dense [(frame, mean_on_court_x | None)] -> court-side segments
    [{'side','start','end','side_agreement'}] or None if degenerate.

    A segment is a stretch where the action sat on one half of the floor. It
    is a CUT POINT for the identity layer, not a possession."""
    mid = court_len / 2.0
    pts = [(f, ("L" if x < mid - margin_ft else
                "R" if x > mid + margin_ft else "N"))
           for (f, x) in xs if x is not None]
    side_pts = [(f, s) for (f, s) in pts if s != "N"]
    if not side_pts:
        return None

    hold = max(1, int(round(hold_s * fps)))
    segs = []                                   # [side, start, end]
    cur_side, cur_start = side_pts[0][1], side_pts[0][0]
    pend_side, pend_start, pend_count = None, None, 0
    for (f, s) in side_pts:
        if s == cur_side:
            pend_side, pend_count = None, 0
            continue
        if pend_side != s:
            pend_side, pend_start, pend_count = s, f, 1
        else:
            pend_count += 1
        if pend_count >= hold:                  # the switch is real
            segs.append([cur_side, cur_start, pend_start - 1])
            cur_side, cur_start = pend_side, pend_start
            pend_side, pend_count = None, 0
    segs.append([cur_side, cur_start, side_pts[-1][0]])

    # merge INTERIOR blips shorter than a real segment, then coalesce same
    # sides. Segments touching the span edges are EXEMPT: a segment cut off
    # by the clip boundary cannot prove its length -- erasing it into the
    # neighboring side is confidently-wrong (user-caught on both clips,
    # 2026-07-12). They are kept and flagged partial instead.
    min_frames = int(round(min_poss_s * fps))
    span_a, span_b = side_pts[0][0], side_pts[-1][0]
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i, seg in enumerate(segs):
            if seg[1] <= span_a or seg[2] >= span_b:    # edge segment: keep
                continue
            if seg[2] - seg[1] + 1 < min_frames:
                segs[i - 1][2] = seg[2]                 # interior => i > 0
                segs.pop(i)
                changed = True
                break
        merged = [segs[0]]
        for seg in segs[1:]:
            if seg[0] == merged[-1][0]:
                merged[-1][2] = seg[2]
            else:
                merged.append(seg)
        segs = merged

    out = []
    for (side, a, b) in segs:
        inside = [(f, s) for (f, s) in pts if a <= f <= b and s != "N"]
        agree = (sum(1 for (_f, s) in inside if s == side) / len(inside)
                 if inside else 0.0)
        out.append({"side": side, "start": a, "end": b,
                    "side_agreement": round(agree, 3),
                    "partial_start": a <= span_a,
                    "partial_end": b >= span_b})
    return out


def load_windows(config, verbose=True):
    """-> (boundaries, label). Identity-window cut frames derived from the
    on-court cache; LOUD fixed-window fallback when the signal is degenerate.
    Also writes {clip}_id_windows.json for eyeball validation."""
    import oncourt
    import clips_config

    odoc = oncourt.load_checked(config)
    tdoc = json.load(open(config.tracks_cache_path, encoding="utf-8"))
    fps, span_start = tdoc["fps"], tdoc["span_start"]
    span_len = tdoc["span_len"]
    court_len = clips_config.active()["court"]["length"]

    xs = []
    for fr in odoc["frames"]:
        on_x = [i["court_feet"][0] for i in fr["tracks"].values() if i["on"]]
        xs.append((fr["frame_index"],
                   sum(on_x) / len(on_x) if on_x else None))
    segs = detect(xs, fps, court_len)

    if segs is None:
        wf = int(round(config.accumulation_window_seconds * fps))
        boundaries = list(range(span_start, span_start + span_len, wf))
        label = (f"FIXED {config.accumulation_window_seconds}s FALLBACK "
                 f"(court-side signal degenerate -- do not trust boundaries)")
        if verbose:
            print(f"  WINDOWS: {label}")
        return boundaries, label

    boundaries = [s["start"] for s in segs]
    boundaries[0] = span_start                  # window 0 covers the span head
    label = f"{len(segs)} court-side segments"
    if verbose:
        print(f"  WINDOWS: {label} (hold {HOLD_S}s, min {MIN_SEG_S}s, "
              f"dead zone +/-{MARGIN_FT:.0f}ft) -- CUT POINTS, not possessions:")
        for i, s in enumerate(segs):
            print(f"    seg {i}: {s['side']}  frames {s['start']}..{s['end']}"
                  f"  t={s['start'] / fps:.1f}s..{s['end'] / fps:.1f}s"
                  f"  ({(s['end'] - s['start'] + 1) / fps:.1f}s, "
                  f"side agreement {s['side_agreement']:.0%})")
    path = os.path.join(_HERE, "out", f"{config.name}_id_windows.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"clip": config.name, "fps": fps, "hold_s": HOLD_S,
                   "min_seg_s": MIN_SEG_S, "margin_ft": MARGIN_FT,
                   "note": "court-side cut points for the identity layer; "
                           "NOT possessions (see team_possessions.py)",
                   "segments": segs, "boundaries": boundaries}, f, indent=2)
    return boundaries, label


def window_of(boundaries, frame):
    return max(0, bisect_right(boundaries, frame) - 1)
