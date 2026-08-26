"""exp_legibility.py -- how big does a girl have to be before the reader can
actually read her?

THE QUESTION, and why it is the one that matters. The pipeline offers the vision
reader every on-court body whose box is at least 90 pixels tall, which on a
whole game is ~150,000 crops. MEASURED 2026-08-24: reading the same 120 of them
twice named 13 girls, then 2, with exactly ONE crop getting the same name both
times. The reader was not disagreeing with a sheet or with EasyOCR -- it was
disagreeing with ITSELF, thirteen times out of fourteen.

That is not a reader that needs tuning. That is a reader being asked to read
something that is not there: at the median those crops are ~65x84 real pixels of
torso, so the number on the jersey is a handful of pixels tall. A vision model
asked an impossible question does not reliably answer "I cannot tell" -- it
varies, and unanimous-of-3 turns most of that variation into silence rather than
into a wrong name, which is the system working. But it means most of the cost is
spent on crops that could never have produced anything.

SO THIS MEASURES REPRODUCIBILITY AGAINST SIZE. Real on-court crops, binned by
how tall the player's box was, each read TWICE. A bin is only useful where the
reader gives the SAME answer both times -- a name that changes between runs is
not a name, it is a coin toss that happens to land on a roster number.

The output is one number the whole naming design hangs off: the box height at
which reading becomes repeatable. Above it, spend the good reader. Below it, do
not spend anything -- send her to the coach, who can see what the model cannot.

Laptop only. Crop-cutting and network calls.

    .venv/Scripts/python.exe spikes/exp_legibility.py [per_bin]
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "phase2"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

WORKERS = 6
# Box-height bins. The floor is the pipeline's current bar; the question is
# whether that bar is anywhere near where reading becomes possible.
BINS = [(90, 130), (130, 180), (180, 250), (250, 350), (350, 500), (500, 10000)]


def _caches(cache_dir):
    clip = "Full_Game_9eb8bf2a"
    tp = os.path.join(cache_dir, "slice0_tracks.json")
    op = os.path.join(cache_dir, "slice0_oncourt.json")
    with open(tp, encoding="utf-8") as fh:
        tdoc = json.load(fh)
    with open(op, encoding="utf-8") as fh:
        odoc = json.load(fh)
    return clip, tdoc, odoc


def crops_by_size(cache_dir, per_bin):
    """per_bin on-court crops in each height band, spread across the footage."""
    import fast_frames
    import ocr_reader
    import clip_registry
    clip, tdoc, odoc = _caches(cache_dir)
    on_by_frame = {fr["frame_index"]: {int(k) for k, v in fr["tracks"].items() if v.get("on")}
                   for fr in odoc["frames"]}
    banded = defaultdict(list)
    for fr in tdoc["frames"]:
        on = on_by_frame.get(fr["frame_index"], set())
        for t in fr["tracks"]:
            if t["track_id"] not in on:
                continue
            h = t["bbox"][3] - t["bbox"][1]
            for lo, hi in BINS:
                if lo <= h < hi:
                    banded[(lo, hi)].append((fr["frame_index"], t["bbox"], h))
                    break
    want = defaultdict(list)
    for band, rows in banded.items():
        step = max(1, len(rows) // (per_bin * 3))     # spread, not one moment
        want[band] = rows[::step][:per_bin * 3]
    need_frames = sorted({f for rows in want.values() for (f, _b, _h) in rows})
    frames = fast_frames.read_frames(clip_registry.load(clip)["video_path"], need_frames)
    out = {}
    for band, rows in want.items():
        got = []
        for (f, bb, h) in rows:
            img = frames.get(f)
            if img is None:
                continue
            c = ocr_reader.jersey_crop(img, bb)
            if c is None or c.size == 0 or c.shape[0] < ocr_reader.MIN_CROP_HEIGHT_PX:
                continue
            got.append((cv2.resize(c, (c.shape[1] * 4, c.shape[0] * 4),
                                   interpolation=cv2.INTER_CUBIC), h, c.shape[0]))
            if len(got) >= per_bin:
                break
        out[band] = got
    return out, {b: len(v) for b, v in banded.items()}


def _named(reads, th):
    if not reads:
        return None
    n, c = max(reads, key=lambda r: r[1])
    return n if c >= th else None


def run(per_bin=25):
    import ocr_reader
    import env_local
    env_local.load()
    if ocr_reader._get_engine() is None:
        return {"error": "GEMINI_API_KEY is not reaching the reader"}
    TH = ocr_reader.OCR_CONFIRM_THRESHOLD
    cache_dir = os.environ.get("SPDIR") or os.path.join(_ROOT, "phase2", "out")

    banded, pool = crops_by_size(cache_dir, int(per_bin))
    import clip_registry
    roster = set()
    for t in clip_registry.load("Full_Game_9eb8bf2a")["teams"]:
        roster |= set(t["numbers"])

    calls = {"n": 0}
    _one = ocr_reader._gemma_once
    ocr_reader._gemma_once = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1),
                                              _one(*a, **k))[1]

    print(f"roster {len(roster)} numbers; on-court pool per band: "
          f"{ {f'{lo}-{hi}': n for (lo, hi), n in sorted(pool.items())} }", flush=True)
    print(f"\n{'box px':>10} {'crop px':>8} {'n':>4} {'run1':>6} {'run2':>6} "
          f"{'SAME BOTH':>10} {'calls':>7}")
    rows = []
    for band in BINS:
        crops = banded.get(band, [])
        if not crops:
            continue
        imgs = [c for (c, _h, _ch) in crops]
        calls["n"] = 0
        t0 = time.time()

        def one_pass():
            with ThreadPoolExecutor(max_workers=WORKERS) as ex:
                return list(ex.map(
                    lambda c: _named(ocr_reader.read_jersey(c, roster), TH), imgs))

        a, b = one_pass(), one_pass()
        same = sum(1 for x, y in zip(a, b) if x is not None and x == y)
        n1 = sum(1 for x in a if x is not None)
        n2 = sum(1 for x in b if x is not None)
        med_box = int(np.median([h for (_c, h, _ch) in crops]))
        med_crop = int(np.median([ch for (_c, _h, ch) in crops]))
        rows.append({"band": f"{band[0]}-{band[1]}", "median_box_px": med_box,
                     "median_crop_px": med_crop, "n": len(imgs),
                     "named_run1": n1, "named_run2": n2, "same_both_runs": same,
                     "reproducible_rate": round(same / max(1, len(imgs)), 3),
                     "calls": calls["n"], "seconds": round(time.time() - t0, 1)})
        print(f"{band[0]:>5}-{band[1]:<4} {med_crop:>8} {len(imgs):>4} {n1:>6} {n2:>6} "
              f"{same:>10} {calls['n']:>7}", flush=True)

    ocr_reader._gemma_once = _one
    best = [r for r in rows if r["reproducible_rate"] >= 0.5]
    return {"threshold": TH, "current_bar_px": 90, "bands": rows,
            "reproducible_from_box_px": best[0]["band"] if best else None,
            "note": ("a name that changes between two runs is not a name; the bar "
                     "belongs where reading becomes repeatable")}


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 25
    print("\n" + json.dumps(run(n), indent=1)[:4000])
