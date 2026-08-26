"""exp_sheet_max.py -- how many crops can share ONE call before naming breaks?

WHY A SECOND ATTEMPT. The first sweep answered a question nobody asked. It
scored "agrees 36/36" twice -- once when both readers named three girls, and
once when both named nobody. Two readers that both say NONE agree perfectly, so
that number could never tell a working sheet from a sheet that reads nothing.
It also fed the reader the BIGGEST boxes in the frame, which in basketball
footage are the press table and the front row, not the players.

WHAT THIS MEASURES INSTEAD, on crops of bodies the pipeline says are ON THE
COURT:
    matched   the slow reader named her, the sheet named her the same
    lost      the slow reader named her, the sheet did not
    invented  the slow reader named nobody, the sheet named somebody
`invented` is the dangerous column: a number appearing on a girl who was never
read is a wrong name on real floor time, which is the failure this project
exists to refuse. `lost` only costs a name.

AND IT MEASURES THE READER'S OWN WOBBLE FIRST. The same crops read twice, one at
a time, gave 3 names and then 0. A sheet size cannot be blamed for a swing the
reader produces on its own, so the baseline is read twice and only crops it
names CONSISTENTLY are used as the yardstick.

Laptop only: crop-cutting and network calls, so a rented GPU would be paid to
sit still.

    .venv/Scripts/python.exe spikes/exp_sheet_max.py [n_crops] [sizes...]
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "phase2"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

WORKERS = 6          # measured stable; 16 was rate-limited


def on_court_crops(n, cache_dir=None):
    """The tallest ON-COURT bodies -- players, not the front row.

    Height is still the legibility proxy (DECISIONS 4b), but only among bodies
    the on-court cache places on the floor. Sorting the whole frame by height
    returns spectators, which is how the first attempt ended up asking a vision
    model to read numbers off a press table.
    """
    import fast_frames
    import ocr_reader
    import clip_registry
    clip = "Full_Game_9eb8bf2a"
    # NOT phase2/out: that is the pipeline's working directory and other runs
    # write there -- a cache fetched for this experiment was overwritten
    # mid-flight by an unrelated 150-frame run, which is how the first attempt
    # ended up intersecting two different stretches of the game and finding
    # nothing in common.
    out_dir = cache_dir or os.path.join(_ROOT, "phase2", "out")
    tp = os.path.join(out_dir, "slice0_tracks.json")
    op = os.path.join(out_dir, "slice0_oncourt.json")
    if not (os.path.exists(tp) and os.path.exists(op)):
        tp = os.path.join(out_dir, f"{clip}_tracks_raw.json")
        op = os.path.join(out_dir, f"{clip}_oncourt.json")
    with open(tp, encoding="utf-8") as fh:
        tdoc = json.load(fh)
    with open(op, encoding="utf-8") as fh:
        odoc = json.load(fh)
    span = (tdoc["frames"][0]["frame_index"], tdoc["frames"][-1]["frame_index"])
    ospan = (odoc["frames"][0]["frame_index"], odoc["frames"][-1]["frame_index"])
    print(f"  tracks {span[0]}..{span[1]}   oncourt {ospan[0]}..{ospan[1]}", flush=True)
    on_by_frame = {fr["frame_index"]: {int(k) for k, v in fr["tracks"].items() if v.get("on")}
                   for fr in odoc["frames"]}
    picks = []
    for fr in tdoc["frames"]:
        on = on_by_frame.get(fr["frame_index"], set())
        for t in fr["tracks"]:
            if t["track_id"] not in on:
                continue
            x1, y1, x2, y2 = t["bbox"]
            picks.append((y2 - y1, fr["frame_index"], t["bbox"]))
    picks.sort(key=lambda p: -p[0])
    by_frame = {}
    for _h, f, bb in picks[:n * 4]:
        by_frame.setdefault(f, []).append(bb)
    frames = fast_frames.read_frames(clip_registry.load(clip)["video_path"],
                                     sorted(by_frame))
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
                return crops, len(picks)
    return crops, len(picks)


def _named(reads, th):
    if not reads:
        return None
    n, c = max(reads, key=lambda r: r[1])
    return n if c >= th else None


def run(n_crops=120, sizes=(12, 24, 36, 48)):
    import ocr_reader
    import env_local
    env_local.load()
    if ocr_reader._get_engine() is None:
        return {"error": "GEMINI_API_KEY is not reaching the reader"}
    TH = ocr_reader.OCR_CONFIRM_THRESHOLD

    crops, pool = on_court_crops(int(n_crops), os.environ.get("SPDIR"))
    if not crops:
        return {"error": "no on-court crops"}
    import clip_registry
    roster = set()
    for t in clip_registry.load("Full_Game_9eb8bf2a")["teams"]:
        roster |= set(t["numbers"])
    hs = [c.shape[0] for c in crops]
    print(f"{len(crops)} ON-COURT crops from a pool of {pool:,}; "
          f"median {int(np.median([c.shape[1] for c in crops]))}x{int(np.median(hs))} px",
          flush=True)

    calls = {"n": 0}
    _one, _grid = ocr_reader._gemma_once, ocr_reader._gemma_grid_once
    ocr_reader._gemma_once = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1),
                                              _one(*a, **k))[1]
    ocr_reader._gemma_grid_once = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1),
                                                   _grid(*a, **k))[1]

    def baseline_pass():
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            return list(ex.map(lambda c: _named(ocr_reader.read_jersey(c, roster), TH), crops))

    out = {"crops": len(crops), "roster": len(roster), "threshold": TH, "runs": {}}

    # --- the reader's own wobble, before any sheet is blamed for it ---------
    calls["n"] = 0
    t0 = time.time()
    a = baseline_pass()
    b = baseline_pass()
    stable = [i for i in range(len(crops)) if a[i] is not None and a[i] == b[i]]
    flaky = [i for i in range(len(crops)) if (a[i] is not None or b[i] is not None)
             and a[i] != b[i]]
    out["baseline"] = {
        "calls": calls["n"], "seconds": round(time.time() - t0, 1),
        "named_run1": sum(1 for x in a if x is not None),
        "named_run2": sum(1 for x in b if x is not None),
        "named_BOTH_runs_same": len(stable),
        "unstable": len(flaky)}
    print(f"baseline: named {out['baseline']['named_run1']} then "
          f"{out['baseline']['named_run2']}; {len(stable)} the SAME both times, "
          f"{len(flaky)} unstable ({calls['n']} calls)", flush=True)
    if not stable:
        out["error"] = ("the one-at-a-time reader named nobody consistently, so "
                        "there is no yardstick to measure a sheet against")
        ocr_reader._gemma_once, ocr_reader._gemma_grid_once = _one, _grid
        return out
    truth = {i: a[i] for i in stable}

    for n_cells in sizes:
        ocr_reader.GRID_CELLS = int(n_cells)
        ocr_reader.GRID_COLS = max(1, int(round((n_cells * 1.4) ** 0.5)))
        sheet, _l = ocr_reader._grid_image(crops[:n_cells])
        calls["n"] = 0
        t0 = time.time()
        got = []
        for i in range(0, len(crops), n_cells):
            got.extend(ocr_reader.read_jersey_batch(crops[i:i + n_cells], roster))
        res = [_named(g, TH) for g in got]
        matched = sum(1 for i, v in truth.items() if res[i] == v)
        lost = sum(1 for i, v in truth.items() if res[i] is None)
        wrong = sum(1 for i, v in truth.items() if res[i] is not None and res[i] != v)
        invented = sum(1 for i in range(len(crops))
                       if i not in truth and a[i] is None and b[i] is None
                       and res[i] is not None)
        row = {"sheet_px": [int(sheet.shape[1]), int(sheet.shape[0])],
               "calls": calls["n"], "seconds": round(time.time() - t0, 1),
               "of_the_yardstick": len(truth), "matched": matched,
               "lost": lost, "WRONG_NAME": wrong, "INVENTED": invented,
               "named_total": sum(1 for r in res if r is not None)}
        out["runs"][str(n_cells)] = row
        print(f"  {n_cells:>3} cells {str(row['sheet_px']):>13}  calls {calls['n']:>4}  "
              f"matched {matched}/{len(truth)}  lost {lost}  wrong {wrong}  "
              f"invented {invented}", flush=True)

    ocr_reader._gemma_once, ocr_reader._gemma_grid_once = _one, _grid
    return out


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 120
    sizes = tuple(int(x) for x in sys.argv[2:]) or (12, 24, 36, 48)
    print(json.dumps(run(n, sizes), indent=1)[:5000])
