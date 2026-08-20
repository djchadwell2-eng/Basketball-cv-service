"""Read the scoreboard across a WHOLE game and build the scoring timeline.

WHY THIS EXISTS. As of 2026-08-19 almost nothing in this pipeline has a real
accuracy number, because every measurement sits on a 15-40 second clip: 8
detected shots in total, of which exactly ONE has a confirmed outcome. You
cannot compute a shooting percentage from that.

The blocker was always cost. But the expensive machinery -- the SIFT camera
anchor (3.6 s/frame), tracking (3.1) and ball detection (2.5) -- exists to find
WHERE things are on the floor. Ground truth about WHAT HAPPENED needs none of
it. The scoreboard is already in the picture, and reading it costs one model
call per sampled frame and nothing else.

So this walks the whole 95 minutes cheaply and produces the real scoring
timeline: every score change, roughly when it happened, and the final score.
That is ground truth we have never had, and it is what turns "the make/miss
detector seems to work" into a measured number.

HOW IT AVOIDS BELIEVING NONSENSE. A vision model always answers, so:
  - a score may never DECREASE (a scoreboard does not count down)
  - between two samples it may only rise by a plausible amount (see MAX_PER_MIN)
  - every candidate CHANGE is re-read READS times on BOTH sides and must come
    back unanimous; a change that will not reproduce is reported as unconfirmed
    rather than entered into the timeline
An unreadable frame (timeout, replay, camera elsewhere) reads as None and is
skipped -- that is missing evidence, not evidence of no score.

Two passes, because they do different jobs:
  COARSE  one read every --every seconds, to find WHERE the score moved
  REFINE  three reads either side of each move, to confirm it

Usage:
    .venv/Scripts/python spikes/scoreboard_timeline.py FULL_GAME
    .venv/Scripts/python spikes/scoreboard_timeline.py FULL_GAME --every 10 --limit-min 20
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "phase2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2                                                          # noqa: E402

# REUSED, not rewritten: the same crop and prompt the make/miss reader already
# uses, so the two can never disagree about what the board says.
from gemma_make_miss_fast import (SCOREBOARD_PROMPT, _parse_response,  # noqa: E402
                                  scoreboard_crop)

WORKERS = 6          # 16 was faster per call but the API rate-limited (measured)
READS = 3            # a change must reproduce this many times, unanimously
MIN_AGREE = 2        # ...and at least this many must actually come back
MAX_PER_MIN = 30     # points one team can plausibly add per minute of clock.
                     # Generous on purpose: this rejects only absurd misreads
                     # (a "12" read off a "2"), never real scoring.


def _read_once(client, jpg_b64, retries=3):
    """One read -> (home, away), or None if the BOARD could not be read.

    RETRIES ON API FAILURE, and that distinction is the whole point. The first
    version returned None on any exception, so a rate-limited call looked
    exactly like an illegible scoreboard. The confirmation pass then demanded
    three clean reads in a 150-call burst and threw away 24 of 29 real score
    changes -- frames the coarse pass had read perfectly moments earlier. A
    failed CALL is missing evidence about our network; an unreadable BOARD is
    evidence about the film. They must not collapse into the same answer.
    """
    for attempt in range(retries):
        try:
            r = client.models.generate_content(
                model="gemma-4-26b-a4b-it",
                contents=[SCOREBOARD_PROMPT,
                          {"inline_data": {"mime_type": "image/jpeg", "data": jpg_b64}}])
        except Exception:
            time.sleep(1.5 * (attempt + 1) + random.random())
            continue
        h, a = _parse_response((r.text or "").strip())
        return None if h is None or a is None else (h, a)
    return None


def _encode(frame, calib):
    crop = scoreboard_crop(frame, calib)
    if crop is None:
        return None
    ok, jpg = cv2.imencode(".jpg", crop)
    return base64.standard_b64encode(jpg.tobytes()).decode("utf-8") if ok else None


def _grab(video_path, frames):
    """Decode just the frames we want. Seeking is what makes a sparse walk over
    95 minutes affordable -- we touch a few hundred frames, not 171,000."""
    cap = cv2.VideoCapture(video_path)
    out = {}
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, img = cap.read()
        if ok:
            out[f] = img
    cap.release()
    return out


def _read_frames(client, calib, video_path, frames, reads=1):
    """{frame: (home, away)} for the frames that could be read. reads > 1 means
    the answer must be UNANIMOUS or the frame counts as unreadable."""
    imgs = _grab(video_path, frames)
    b64 = {f: _encode(im, calib) for f, im in imgs.items()}
    jobs = [(f, k) for f, b in b64.items() if b for k in range(reads)]
    if not jobs:
        return {}
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        res = list(ex.map(lambda fk: _read_once(client, b64[fk[0]]), jobs))
    votes = {}
    for (f, _k), v in zip(jobs, res):
        votes.setdefault(f, []).append(v)
    out = {}
    for f, vs in votes.items():
        named = [v for v in vs if v is not None]
        # UNANIMOUS AMONG THE READS THAT CAME BACK, with at least MIN_AGREE of
        # them. Requiring all `reads` to succeed makes a network hiccup
        # indistinguishable from an illegible board -- see _read_once.
        if len(named) < min(reads, MIN_AGREE):
            continue
        top, n = Counter(named).most_common(1)[0]
        if n == len(named):                # every read that answered, agreed
            out[f] = top
    return out


def plausible(prev, cur, seconds_apart):
    """Could the score have gone from prev to cur in this much time?"""
    dh, da = cur[0] - prev[0], cur[1] - prev[1]
    if dh < 0 or da < 0:
        return False                       # a scoreboard does not count down
    cap = max(3, int(MAX_PER_MIN * seconds_apart / 60.0) + 3)
    return dh <= cap and da <= cap


def build_timeline(coarse, fps):
    """Sampled reads -> (changes, rejected). Pure, so it is testable without
    touching a video or the API."""
    changes, rejected, prev_f = [], [], None
    for f in sorted(coarse):
        if prev_f is None:
            prev_f = f
            continue
        a, b = coarse[prev_f], coarse[f]
        if b == a:
            prev_f = f
            continue
        if not plausible(a, b, (f - prev_f) / fps):
            rejected.append({"frame": f, "from": list(a), "to": list(b),
                             "why": "impossible jump for the elapsed time"})
            continue                       # prev stays put: a is still trusted
        changes.append({"frame": f, "prev_frame": prev_f,
                        "from": list(a), "to": list(b)})
        prev_f = f
    return changes, rejected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--every", type=float, default=10.0, help="coarse sample seconds")
    ap.add_argument("--limit-min", type=float, default=None,
                    help="only scan the first N minutes (for a cheap trial)")
    args = ap.parse_args()

    import clips_config
    import env_local
    env_local.load()
    import google.genai

    calib = clips_config.CLIPS[args.clip]
    video = calib["video_path"]
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    if args.limit_min:
        total = min(total, int(args.limit_min * 60 * fps))
    step = max(1, int(args.every * fps))
    frames = list(range(0, total, step))

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise SystemExit("no GEMINI_API_KEY -- this tool IS the vision reader")
    client = google.genai.Client(api_key=key)

    print(f"{args.clip}: {total} frames ({total / fps / 60:.1f} min), "
          f"sampling every {args.every}s = {len(frames)} coarse reads", flush=True)
    t0 = time.time()

    coarse = _read_frames(client, calib, video, frames, reads=1)
    print(f"  coarse pass: {len(coarse)}/{len(frames)} frames readable "
          f"({time.time() - t0:.0f}s)", flush=True)

    changes, rejected = build_timeline(coarse, fps)
    print(f"  {len(changes)} candidate change(s), {len(rejected)} rejected as "
          f"misreads -- now confirming each", flush=True)

    need = sorted({e["frame"] for e in changes} | {e["prev_frame"] for e in changes})
    firm = _read_frames(client, calib, video, need, reads=READS)
    confirmed, unconfirmed = [], []
    for e in changes:
        a, b = firm.get(e["prev_frame"]), firm.get(e["frame"])
        if a is None or b is None or list(a) != e["from"] or list(b) != e["to"]:
            e["seen_on_recheck"] = {"prev": list(a) if a else None,
                                    "at": list(b) if b else None}
            unconfirmed.append(e)
        else:
            e["points"] = [b[0] - a[0], b[1] - a[1]]
            e["time_s"] = round(e["frame"] / fps, 1)
            e["window_s"] = [round(e["prev_frame"] / fps, 1),
                             round(e["frame"] / fps, 1)]
            confirmed.append(e)

    last = max(coarse) if coarse else None
    out = {
        "clip": args.clip, "fps": fps, "frames_scanned": len(frames),
        "sample_seconds": args.every, "coarse_readable": len(coarse),
        "confirmed_changes": confirmed,
        "unconfirmed_changes": unconfirmed,
        "rejected_misreads": rejected,
        "final_score": list(coarse[last]) if last is not None else None,
        "note": ("Ground truth from the scoreboard only. A change is listed once "
                 "it survives a unanimous re-read on BOTH sides. Timing is only "
                 "as precise as the sample interval -- the basket happened "
                 "somewhere inside window_s, not exactly at time_s."),
    }
    path = os.path.join(_HERE, "out", f"{args.clip}_scoreboard_timeline.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w", encoding="utf-8"), indent=2)

    print(f"\n{'=' * 66}\nSCORING TIMELINE -- {args.clip}\n{'=' * 66}")
    home = away = 0
    for e in confirmed:
        home += e["points"][0]
        away += e["points"][1]
        who = "HOME" if e["points"][0] else "AWAY"
        print(f"  t={e['time_s']:>7.1f}s  {who} +{max(e['points'])}"
              f"   {e['from']} -> {e['to']}")
    print(f"\n  confirmed scoring plays:  {len(confirmed)}")
    print(f"  points from those plays:  home {home}, away {away}")
    print(f"  final score seen:         {out['final_score']}")
    print(f"  unconfirmed (recheck disagreed): {len(unconfirmed)}")
    print(f"  rejected as misreads:            {len(rejected)}")
    print(f"  wall clock: {time.time() - t0:.0f}s")
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
