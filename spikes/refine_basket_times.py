"""Pin down WHEN each basket happened, so the expensive pipeline can be aimed.

THE PROBLEM THIS SOLVES. scoreboard_timeline.py samples every 10 seconds, so it
knows a basket happened somewhere inside a 10-second window -- not when. The
pipeline costs ~9.2 s of CPU per frame of film, so we can afford roughly 5
seconds of video per basket. Aiming a 5-second window at a 10-second
uncertainty misses the shot about half the time.

Refining the time is CHEAP because it needs the scoreboard and nothing else:
binary-search the moment the number changed. Four reads take a 10-second window
down to under a second, at a few model calls per basket -- against 9.2 s/frame
for the machinery that would otherwise have to cover the whole window.

WHY THE WINDOW SITS BEFORE THE CHANGE. The board updates AFTER the ball goes in
-- a human presses a button. So the shot is EARLIER than the moment the number
moves, never later. The emitted window runs from `pre_seconds` before the change
to `post_seconds` after it.

Usage:
    .venv/Scripts/python spikes/refine_basket_times.py FULL_GAME --max 20
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "phase2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2                                                          # noqa: E402

from gemma_make_miss_fast import (SCOREBOARD_PROMPT, _parse_response,  # noqa: E402
                                  scoreboard_crop)

READS = 3            # a probe must come back unanimous to move the search
MIN_AGREE = 2        # ...on at least this many answers (an API error is not a vote)
BISECT_STEPS = 4     # 10s -> ~0.6s. More steps buy precision we cannot use.
WORKERS = 6


def _read(client, b64, retries=3):
    for attempt in range(retries):
        try:
            r = client.models.generate_content(
                model="gemma-4-26b-a4b-it",
                contents=[SCOREBOARD_PROMPT,
                          {"inline_data": {"mime_type": "image/jpeg", "data": b64}}])
        except Exception:
            time.sleep(1.5 * (attempt + 1) + random.random())
            continue
        h, a = _parse_response((r.text or "").strip())
        return None if h is None or a is None else (h, a)
    return None


def _score_at(client, cap, calib, frame):
    """The score on one frame, or None if it will not read unanimously."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame)))
    ok, img = cap.read()
    if not ok:
        return None
    crop = scoreboard_crop(img, calib)
    if crop is None:
        return None
    ok, jpg = cv2.imencode(".jpg", crop)
    if not ok:
        return None
    b64 = base64.standard_b64encode(jpg.tobytes()).decode("utf-8")
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        votes = [v for v in ex.map(lambda _i: _read(client, b64), range(READS))
                 if v is not None]
    if len(votes) < MIN_AGREE:
        return None
    top, n = Counter(votes).most_common(1)[0]
    return top if n == len(votes) else None


def refine(client, cap, calib, lo, hi, before, after, steps=BISECT_STEPS):
    """Narrow (lo, hi] -- where the score went from `before` to `after` -- down
    to the frame the change landed on.

    An unreadable probe does NOT stop the search: it just cannot move the
    bracket, so we keep the wider bracket and stop early. Half an answer is
    still better than a 10-second guess.
    """
    for _ in range(steps):
        mid = (lo + hi) // 2
        if mid <= lo or mid >= hi:
            break
        s = _score_at(client, cap, calib, mid)
        if s is None:
            break                       # cannot narrow further, keep the bracket
        if list(s) == list(before):
            lo = mid                    # still the old score -> change is later
        elif list(s) == list(after):
            hi = mid                    # already the new score -> change is earlier
        else:
            break                       # a third value: the window hid more than
                                        # one basket, so stop and keep what we have
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--max", type=int, default=20, help="how many baskets to refine")
    ap.add_argument("--pre", type=float, default=4.0, help="seconds of film before the change")
    ap.add_argument("--post", type=float, default=1.0, help="seconds after")
    args = ap.parse_args()

    import clips_config
    import env_local
    env_local.load()
    import google.genai

    calib = clips_config.CLIPS[args.clip]
    tl_path = os.path.join(_HERE, "out", f"{args.clip}_scoreboard_timeline.json")
    tl = json.load(open(tl_path, encoding="utf-8"))
    fps = tl["fps"]
    changes = tl["confirmed_changes"]

    # PREFER THE CLEAN ONES. A window where both teams scored hid at least two
    # baskets, so the shot we would be aiming at is ambiguous -- those go last.
    single = [e for e in changes if not (e["points"][0] and e["points"][1])]
    single.sort(key=lambda e: e["frame"])
    # Spread the picks across the game rather than taking the first N, so the
    # sample is not all first-quarter footage.
    if len(single) > args.max:
        step = len(single) / float(args.max)
        single = [single[int(i * step)] for i in range(args.max)]

    client = google.genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    cap = cv2.VideoCapture(calib["video_path"])
    print(f"{args.clip}: refining {len(single)} of {len(changes)} confirmed baskets",
          flush=True)

    path = os.path.join(_HERE, "out", f"{args.clip}_basket_windows.json")

    def _save(rows):
        """Written after EVERY basket. The first version saved only at the end
        and a timeout killed it one basket short, throwing away 12 minutes of
        model calls for nothing."""
        json.dump({"clip": args.clip, "fps": fps, "pre_s": args.pre,
                   "post_s": args.post, "baskets": rows,
                   "total_frames_to_run": sum(b["run_frames"] for b in rows),
                   "note": ("Each window is aimed BEFORE the score change, "
                            "because the board is updated by hand after the ball "
                            "goes in. change_uncertainty_s is how tightly the "
                            "moment is pinned; anything much above pre_s may "
                            "still miss its shot.")},
                  open(path, "w", encoding="utf-8"), indent=2)

    out, t0 = [], time.time()
    for i, e in enumerate(single):
        lo, hi = refine(client, cap, calib, e["prev_frame"], e["frame"],
                        e["from"], e["to"])
        span = (hi - lo) / fps
        start = max(0, int(hi - args.pre * fps))
        out.append({
            "basket_index": i,
            "points": max(e["points"]),
            "team": "home" if e["points"][0] else "away",
            "score_from": e["from"], "score_to": e["to"],
            "change_bracket_frames": [lo, hi],
            "change_uncertainty_s": round(span, 2),
            "run_start_frame": start,
            "run_frames": int((args.pre + args.post) * fps),
            "run_seconds": [round(start / fps, 1),
                            round((start + (args.pre + args.post) * fps) / fps, 1)],
        })
        print(f"  basket {i:>2}: {e['from']}->{e['to']}  was a {span:.1f}s window "
              f"(from {(e['frame'] - e['prev_frame']) / fps:.0f}s)  "
              f"-> run f{start}..{start + int((args.pre + args.post) * fps)}", flush=True)
        _save(out)
    cap.release()

    _save(out)
    tot = sum(b["run_frames"] for b in out)
    unc = [b["change_uncertainty_s"] for b in out]
    print(f"\n  refined {len(out)} baskets in {time.time() - t0:.0f}s")
    print(f"  uncertainty: median {sorted(unc)[len(unc) // 2]:.1f}s, "
          f"worst {max(unc):.1f}s (was 10s for every one)")
    print(f"  frames to run: {tot}  = {tot / 30.0:.0f}s of film")
    print(f"  estimated CPU cost at 9.2 s/frame: {tot * 9.2 / 3600:.1f} hours")
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
