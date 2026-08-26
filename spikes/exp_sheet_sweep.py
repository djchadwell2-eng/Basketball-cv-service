"""exp_sheet_sweep.py -- how many crops can share one call before it breaks?

TWO UNKNOWNS THIS SETTLES, and they are the two the sheet reader currently
rests on:

  1. WHERE THE API DOWNSCALES. The sheet reader assumes a vision model shrinks
     pictures above roughly 1536 on a side. That was assumed, never measured,
     and everything hangs off it: shrink a jersey that is already at the edge of
     legibility and the reads collapse.
  2. WHETHER MORE CELLS COSTS ACCURACY. Twelve crops in one call is a twelvefold
     saving; thirty-six would be three times better again. But one answer landing
     on the wrong cell is one girl's number on another girl's floor time, and
     that failure does not exist when crops are read one at a time.

METHOD. The same real crops, read four ways: one at a time (the reader the
pipeline trusts today, used here as the yardstick), then as sheets of 12, 24 and
36. Compare the CONFIRMATIONS, not the raw text -- a read below the 0.85 bar is
a reject either way, and what matters is whether a girl gets named.

Runs on the laptop, deliberately: it is crop-cutting and network calls, so a
rented GPU would be paid to sit still.

    .venv/Scripts/python.exe spikes/exp_sheet_sweep.py [n_crops]
"""

from __future__ import annotations

import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "phase2"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def biggest_crops(clip_cache, video, n):
    """The n most legible crops in the cache -- biggest boxes first.

    Deliberately the big ones: a sweep over crops nobody could read would return
    NONE everywhere and prove nothing about cells. These are also the crops a
    budgeted run would actually spend the reader on.
    """
    import fast_frames
    import ocr_reader
    with open(clip_cache, encoding="utf-8") as fh:
        doc = json.load(fh)
    picks = []
    for fr in doc["frames"]:
        for t in fr["tracks"]:
            x1, y1, x2, y2 = t["bbox"]
            picks.append((y2 - y1, fr["frame_index"], t["bbox"]))
    picks.sort(key=lambda p: -p[0])
    by_frame = {}
    for _h, f, bb in picks[:n * 3]:
        by_frame.setdefault(f, []).append(bb)
    frames = fast_frames.read_frames(video, sorted(by_frame))
    crops = []
    for f in sorted(by_frame):
        img = frames.get(f)
        if img is None:
            continue
        for bb in by_frame[f]:
            c = ocr_reader.jersey_crop(img, bb)
            if c is None or c.size == 0 or c.shape[0] < ocr_reader.MIN_CROP_HEIGHT_PX:
                continue
            crops.append(cv2.resize(c, (c.shape[1] * 4, c.shape[0] * 4),
                                    interpolation=cv2.INTER_CUBIC))
            if len(crops) >= n:
                return crops
    return crops


def _confirmed(reads, threshold):
    """(number, confidence) if this read would name somebody, else None."""
    if not reads:
        return None
    n, c = max(reads, key=lambda r: r[1])
    return (n, round(c, 3)) if c >= threshold else None


def run(n_crops=36, cells=(12, 24, 36)):
    import ocr_reader
    import env_local
    env_local.load()
    if ocr_reader._get_engine() is None:
        return {"error": "no Gemma engine -- GEMINI_API_KEY not reaching the reader"}

    clip = "Full_Game_9eb8bf2a"
    cache = os.path.join(_ROOT, "phase2", "out", f"{clip}_tracks_raw.json")
    import clip_registry
    video = clip_registry.load(clip)["video_path"]
    crops = biggest_crops(cache, video, int(n_crops))
    if not crops:
        return {"error": "no crops"}
    roster = set()
    for t in clip_registry.load(clip)["teams"]:
        roster |= set(t["numbers"])

    hs = [c.shape[0] for c in crops]
    ws = [c.shape[1] for c in crops]
    out = {"crops": len(crops), "roster_size": len(roster),
           "crop_px": [int(np.median(ws)), int(np.median(hs))], "runs": {}}
    calls = {"n": 0}
    _orig = ocr_reader._gemma_once
    _orig_grid = ocr_reader._gemma_grid_once

    def counted(*a, **k):
        calls["n"] += 1
        return _orig(*a, **k)

    def counted_grid(*a, **k):
        calls["n"] += 1
        return _orig_grid(*a, **k)

    ocr_reader._gemma_once = counted
    ocr_reader._gemma_grid_once = counted_grid

    # --- the yardstick: one crop at a time, the reader in use today ----------
    calls["n"] = 0
    t0 = time.time()
    base = [_confirmed(ocr_reader.read_jersey(c, roster),
                       ocr_reader.OCR_CONFIRM_THRESHOLD) for c in crops]
    out["runs"]["one_at_a_time"] = {
        "calls": calls["n"], "seconds": round(time.time() - t0, 1),
        "named": sum(1 for b in base if b),
        "reads": [b[0] if b else None for b in base]}

    # --- sheets -------------------------------------------------------------
    for n_cells in cells:
        ocr_reader.GRID_CELLS = int(n_cells)
        ocr_reader.GRID_COLS = max(1, int(round(np.sqrt(n_cells * 1.4))))
        sheet, _labels = ocr_reader._grid_image(crops[:n_cells])
        calls["n"] = 0
        t0 = time.time()
        got = []
        for i in range(0, len(crops), n_cells):
            got.extend(ocr_reader.read_jersey_batch(crops[i:i + n_cells], roster))
        conf = [_confirmed(g, ocr_reader.OCR_CONFIRM_THRESHOLD) for g in got]
        agree = sum(1 for a, b in zip(base, conf)
                    if (a[0] if a else None) == (b[0] if b else None))
        out["runs"][f"sheet_{n_cells}"] = {
            "sheet_px": [int(sheet.shape[1]), int(sheet.shape[0])],
            "calls": calls["n"], "seconds": round(time.time() - t0, 1),
            "named": sum(1 for c in conf if c),
            "agrees_with_one_at_a_time": f"{agree}/{len(base)}",
            "reads": [c[0] if c else None for c in conf],
            "disagreements": [{"crop": i, "one": (a[0] if a else None),
                               "sheet": (b[0] if b else None)}
                              for i, (a, b) in enumerate(zip(base, conf))
                              if (a[0] if a else None) != (b[0] if b else None)][:12]}
        print(f"  sheet {n_cells:>3}: {out['runs'][f'sheet_{n_cells}']['sheet_px']} px, "
              f"{calls['n']} calls, named {out['runs'][f'sheet_{n_cells}']['named']}, "
              f"agrees {agree}/{len(base)}", flush=True)

    ocr_reader._gemma_once = _orig
    ocr_reader._gemma_grid_once = _orig_grid
    base_calls = out["runs"]["one_at_a_time"]["calls"] or 1
    for k, v in out["runs"].items():
        v["calls_vs_one_at_a_time"] = round(base_calls / max(1, v["calls"]), 2)
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 36
    res = run(n)
    print(json.dumps(res, indent=1)[:6000])
