"""THE UNBLOCKING RUN: is bytetrack_mt09.yaml safe to adopt?

WHAT IS SITTING UNCLAIMED. `phase2/bytetrack_mt09.yaml` loosens match_thresh
0.8 -> 0.9 and its own header records the measurement: **"the single biggest
fragmentation-reduction lever measured to date (122->93 distinct ids, +35% mean
lifespan on TEST1)"** -- a 24% cut in fragments. It says, in the same breath,
"NOT adopted -- pending the ID-switch eyeball check this file exists to
support." That check was believed to need a human labelling session, so it
never happened, and a measured 24% has been sitting on the shelf.

**The blocker no longer exists.** spikes/tracker_switch_metric.py answers
exactly that question with NO labels: if one candidate id absorbs two committed
tracks that were alive at the same time in different places, it glued two
people together. Nobody connected the tool to the config it unblocks.

WHY FRAGMENTATION IS THE RIGHT TARGET [MEASURED 2026-09-10]: window resets are
NOT what costs the clicks -- tracks appear in a mean of 1.21 windows, i.e. they
DIE BEFORE they reach a boundary, and letting a name ride a continuous track
across boundaries would save only 17% (HARD) / 0% (TEST2). What actually costs
is that one girl becomes many tracks. Every naming route we have -- clicking,
reading, exclusion -- is paid PER FRAGMENT, so fragmentation is the denominator
under all of them at once.

THE ASYMMETRY, which is the whole point of running this rather than just
adopting it: a fragment costs a CLICK. A merge puts one girl's floor time on
another and nothing downstream can tell. Fewer ids is only better if no two
girls were glued to get there.

Writes a CANDIDATE tracks file under a different name. It must never touch the
committed cache.

Usage:  .venv/Scripts/python.exe spikes/mt09_switch_check.py <CLIP>
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "phase2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

CLIP = sys.argv[1] if len(sys.argv) > 1 else "TEST1"
CANDIDATE_YAML = os.path.join(_ROOT, "phase2", "bytetrack_mt09.yaml")


def main():
    import clip_config
    cfg = clip_config.get_clip(CLIP)
    out_path = os.path.join(_HERE, "out", f"{CLIP}_tracks_mt09.json")

    # NEVER write the committed cache. That file is the pipeline's ground
    # truth for every downstream stage, and a probe overwriting it would be
    # indistinguishable from the real thing afterwards.
    assert os.path.abspath(out_path) != os.path.abspath(cfg.tracks_cache_path), \
        "probe must not write the committed tracks cache"

    # TAKE THE SPAN FROM THE COMMITTED CACHE, NOT FROM THE CONFIG. They are not
    # always the same file's idea of the same thing: TEST1's committed cache
    # covers frames 300-449 while its ClipConfig says 120..581, so tracking the
    # config's span produced a candidate covering 3x more film than the
    # reference and a meaningless "+58% ids". The only span that makes the two
    # comparable is the one the reference actually contains.
    _committed = json.load(open(cfg.tracks_cache_path, encoding="utf-8"))
    lo = _committed["span_start"]
    hi = lo + _committed["span_len"]
    if lo != cfg.tracking_span_start or hi != cfg.tracking_span_start + cfg.tracking_span_len:
        print(f"NOTE: committed cache covers {lo}..{hi}, but ClipConfig says "
              f"{cfg.tracking_span_start}..{cfg.tracking_span_start + cfg.tracking_span_len}. "
              f"Using the CACHE's span so the two are comparable.")

    if os.path.exists(out_path):
        print(f"reusing {out_path} (delete to re-track)")
    else:
        import tracking
        import stage2_multikeyframe as s2mk
        print(f"{CLIP}: tracking frames {lo}..{hi} with "
              f"{os.path.basename(CANDIDATE_YAML)} (match_thresh 0.9)", flush=True)

        frames_iter = (im for _f, im in s2mk.iter_frames(cfg.video_path, range(lo, hi)))
        out_frames = []
        for i, (_idx, _img, tracks) in enumerate(
                tracking.iter_tracks_over(frames_iter, tracker_config=CANDIDATE_YAML)):
            out_frames.append({"frame_index": lo + i,
                               "tracks": [{"track_id": t.track_id,
                                           "bbox": list(t.bbox)} for t in tracks]})
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{hi - lo} frames", flush=True)
        json.dump({"clip": CLIP, "fps": _committed["fps"], "span_start": lo,
                   "span_len": hi - lo, "tracker": "bytetrack_mt09.yaml",
                   "frames": out_frames},
                  open(out_path, "w", encoding="utf-8"))
        print(f"saved -> {out_path}")

    # --- fragmentation, both sides of the ledger ---------------------------
    cand = json.load(open(out_path, encoding="utf-8"))
    comm = json.load(open(cfg.tracks_cache_path, encoding="utf-8"))

    def stats(doc):
        life = {}
        for fr in doc["frames"]:
            for t in fr["tracks"]:
                life.setdefault(t["track_id"], []).append(fr["frame_index"])
        n = len(life)
        mean_life = sum(len(v) for v in life.values()) / max(n, 1)
        return n, mean_life

    cn, cl = stats(cand)
    mn, ml = stats(comm)
    print(f"\n{'':14}{'ids':>7}{'mean lifespan':>16}")
    print(f"{'committed':14}{mn:>7}{ml:>16.1f}")
    print(f"{'mt09':14}{cn:>7}{cl:>16.1f}")
    if mn:
        print(f"{'change':14}{100.0 * (cn - mn) / mn:>6.0f}%"
              f"{100.0 * (cl - ml) / max(ml, 1e-6):>15.0f}%")

    print("\n--- THE SAFETY HALF: did it glue two girls together? ---")
    import subprocess
    r = subprocess.run(
        [sys.executable, os.path.join(_HERE, "tracker_switch_metric.py"),
         cfg.tracks_cache_path, out_path],
        cwd=_ROOT, capture_output=True, text=True)
    print(r.stdout[-3000:] or r.stderr[-2000:])


if __name__ == "__main__":
    main()
