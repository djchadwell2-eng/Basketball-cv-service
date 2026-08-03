"""Track-PURITY check -- catches the tracker jumping between players.

A ByteTrack id is supposed to follow ONE body; during collisions/crossings it
can walk off with a different player (a "splice"). A human first caught this
on TEST1 t49 (crops showed #44, then #13). This module automates that catch:
run the jersey reader across each labelable track's whole lifespan; if it
confidently reads TWO DIFFERENT on-roster numbers on one track_id, the track
is SPLICED -- quarantined from labeling (review bundle) and its seed labels
are REFUSED (roster.py). Conservative by design: a quarantined track costs
some unnamed floor time; a trusted spliced track credits the wrong player.

LIMITATION (recorded): number-only evidence cannot see a splice between two
players wearing the SAME number (the dual-team #3/#23 cases) -- that needs
the jersey-color tiebreak.

Builder is heavy (OCR sweep, once per clip via root cache_purity.py) ->
phase2/out/{clip}_purity.json. classify_reads() and the loader are light.
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "phase1"))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "spikes"))

from clip_config import ACTIVE_CLIP as CLIP

OUT_JSON = os.path.join(_HERE, "out", f"{CLIP.name}_purity.json")

SEGMENTS = 6            # lifespan slices; best crop per slice gets OCR'd
MIN_BOX_H = 90          # same readability gate as stage6
MIN_SPACING = 3         # frames between attempts


def classify_reads(reads, threshold):
    """[(frame, number, confidence)] -> verdict dict. Only confident reads
    (>= threshold) carry weight; a weak read must never convict a track."""
    conf = [(f, n) for (f, n, c) in reads if c >= threshold]
    stats: dict = {}
    for (f, n, c) in reads:
        if c < threshold:
            continue
        s = stats.setdefault(n, {"reads": 0, "first_frame": f,
                                 "last_frame": f, "max_conf": 0.0})
        s["reads"] += 1
        s["first_frame"] = min(s["first_frame"], f)
        s["last_frame"] = max(s["last_frame"], f)
        s["max_conf"] = max(s["max_conf"], c)
    if not stats:
        return {"verdict": "no_evidence", "numbers": {}}
    if len(stats) == 1:
        return {"verdict": "consistent", "numbers": stats}
    ordered = sorted(stats, key=lambda n: (stats[n]["first_frame"]
                                           + stats[n]["last_frame"]) / 2.0)
    a, b = ordered[0], ordered[-1]
    interleaved = stats[a]["last_frame"] > stats[b]["first_frame"]
    interval = sorted([stats[a]["last_frame"], stats[b]["first_frame"]])
    return {"verdict": "spliced", "numbers": stats,
            "splice_interval": interval, "interleaved": interleaved}


def load(config) -> dict:
    """{track_id: verdict-dict} for the clip, or {} if the purity cache has
    not been built (callers treat missing as 'no evidence', never as clean)."""
    path = os.path.join(_HERE, "out", f"{config.name}_purity.json")
    if not os.path.exists(path):
        return {}
    doc = json.load(open(path, encoding="utf-8"))
    return {int(t): v for t, v in doc["tracks"].items()}


def spliced_tracks(config) -> set:
    return {t for t, v in load(config).items() if v["verdict"] == "spliced"}


# --------------------------------------------------------------------------
# builder (heavy: OCR sweep over the labelable tracks; run via cache_purity)
# --------------------------------------------------------------------------

def build():
    import cv2  # noqa: F401  (frame reading below)
    import oncourt
    import ocr_reader

    import window_boundaries
    tdoc = json.load(open(CLIP.tracks_cache_path, encoding="utf-8"))
    odoc = oncourt.load_checked(CLIP)
    boundaries, _wlabel = window_boundaries.load_windows(CLIP, verbose=False)
    tracks = oncourt.seed_frame_tracks(tdoc, odoc, boundaries=boundaries)
    roster = CLIP.roster_numbers()

    occ: dict = {tid: [] for tid in tracks}
    for fr in tdoc["frames"]:
        for t in fr["tracks"]:
            if t["track_id"] in occ:
                occ[t["track_id"]].append((fr["frame_index"], t["bbox"]))

    # best (tallest) box per lifespan segment, spaced apart
    picks: dict = {}
    need = set()
    for tid, rows in occ.items():
        big = [(f, bb) for (f, bb) in rows if bb[3] - bb[1] >= MIN_BOX_H]
        if not big:
            picks[tid] = []
            continue
        k = max(1, len(big) // SEGMENTS)
        sel = []
        for i in range(0, len(big), k):
            part = big[i:i + k]
            f, bb = max(part, key=lambda r: r[1][3] - r[1][1])
            if all(abs(f - g) >= MIN_SPACING for (g, _b) in sel):
                sel.append((f, bb))
        picks[tid] = sel[:SEGMENTS * 2]
        need |= {f for (f, _b) in sel[:SEGMENTS * 2]}

    print(f"[purity] {CLIP.name}: {len(tracks)} labelable tracks, "
          f"{sum(len(v) for v in picks.values())} OCR attempts "
          f"(CPU, once per clip)")
    import cv2
    frames = {}
    cap = cv2.VideoCapture(CLIP.video_path)
    it = iter(sorted(need))
    nxt, idx = next(it, None), 0
    while nxt is not None and cap.grab():
        if idx == nxt:
            ok, fr = cap.retrieve()
            if ok:
                frames[idx] = fr
            nxt = next(it, None)
        idx += 1
    cap.release()

    out_tracks, n_spliced = {}, 0
    for i, (tid, sel) in enumerate(sorted(picks.items())):
        reads = []
        for (f, bb) in sel:
            if f not in frames:
                continue
            crop = ocr_reader.jersey_crop(frames[f], bb)
            for (n, c) in ocr_reader.read_jersey(crop, roster):
                reads.append((f, n, c))
        v = classify_reads(reads, ocr_reader.OCR_CONFIRM_THRESHOLD)
        v["attempts"] = len(sel)
        out_tracks[str(tid)] = v
        if v["verdict"] == "spliced":
            n_spliced += 1
            nums = {n: s["max_conf"] for n, s in v["numbers"].items()}
            print(f"  *** SPLICED t{tid}: numbers {nums} "
                  f"splice interval ~{v['splice_interval']} ***")
        if i % 10 == 0:
            print(f"  ...{i}/{len(picks)}")

    doc = {"clip": CLIP.name, "span_start": tdoc["span_start"],
           "span_len": tdoc["span_len"],
           "threshold": __import__("ocr_reader").OCR_CONFIRM_THRESHOLD,
           "tracks": out_tracks}
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)
    tally = {}
    for v in out_tracks.values():
        tally[v["verdict"]] = tally.get(v["verdict"], 0) + 1
    print(f"[purity] verdicts: {tally}  -> {OUT_JSON}")
    if n_spliced:
        print(f"[purity] {n_spliced} SPLICED track(s): quarantined from "
              f"labeling; their seed labels will be refused.")


if __name__ == "__main__":
    build()
