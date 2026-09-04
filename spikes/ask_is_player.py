"""Ask the vision model: is this a PLAYER, a REFEREE, or a COACH?

WHY THIS IS THE LAST ROUTE. Five-on-court exclusion needs an accurate count of
players per team. Measured 2026-09-03, the count reads 1 to 9 bodies per team
per frame, with SIX and SEVEN common -- impossible in basketball -- and exclusion
fired 0 of 9 at real relink moments as a direct result.

The cause was chased all the way down:
  - NOT the geometry margin: the median distance outside the painted lines is
    0.00 ft, and TEST2's known bench tracks sit 0.16-0.51 ft out, INSIDE the
    slack a real player needs for stepping on the line.
  - NOT duplicate detections: 99.8% of the excess bodies do not overlap an
    already-counted player (IoU < 0.5). They are separate people.
  - It IS that referees, coaches and bench bodies really are standing inside the
    court rectangle, and nothing tells them apart from players.

Both automatic routes to that are already measured dead:
  MOTION  (DJ, 2026-08-29) -- a coach walked 26 ft; the median player 9.6 ft.
  COLOUR  (2026-08-30) -- players' distance to the nearest team centroid runs
          4.7-63.7, non-players 10.1-63.4. The ranges overlap end to end, and
          every threshold that catches a referee also deletes a real player.

WHY THIS ONE MIGHT SURVIVE WHERE THOSE DIED. Both dead routes asked for a
MEASUREMENT of the crop -- a distance, a displacement -- and then a threshold.
This asks a SEMANTIC question, which is what a vision model is actually good at,
and a COARSE one: a striped shirt, a tracksuit, street clothes and a basketball
kit do not resemble each other at any distance. That is the opposite of reading
a 40px number, the task this same model measurably struggles with.

GROUND TRUTH IS FREE AND ALREADY ON DISK: the human track labels give 34 real
players and 19 real non-players across TEST1/HARD/TEST2.

THE NUMBER THAT DECIDES IT is not how many referees were caught. It is HOW MANY
REAL PLAYERS GET CALLED NON-PLAYERS. A referee wrongly kept leaves the count one
too high, which the exclusion gate already refuses on -- costly but safe. A
player wrongly dropped makes a five look like a four and can FORCE A WRONG NAME.

Usage:
    .venv/Scripts/python.exe spikes/ask_is_player.py
"""

from __future__ import annotations

import base64
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "phase2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2                                                          # noqa: E402

import stage2_multikeyframe as s2                                   # noqa: E402

CLIPS = ("TEST1", "HARD", "TEST2")
MIN_BOX_H = 90
CROPS_PER_TRACK = 3      # spread across her life; a pose is not a person
WORKERS = 6
NON_PLAYER = {"ref", "bench"}

PROMPT = (
    "This is a cropped photo of one person at a high-school basketball game.\n"
    "Which is it?\n"
    "  PLAYER - wearing a team uniform (sleeveless jersey and shorts)\n"
    "  REFEREE - wearing a black-and-white striped shirt\n"
    "  COACH - an adult in street clothes, a tracksuit or a suit\n"
    "  OTHER - a spectator, or you cannot tell\n"
    "Answer with ONE word: PLAYER, REFEREE, COACH or OTHER."
)


def ask_once(client, b64, retries=3):
    """One classification, or None. Retries a FAILED CALL -- an API error is
    missing evidence about our network, not evidence about the picture."""
    for attempt in range(retries):
        try:
            r = client.models.generate_content(
                model="gemma-4-26b-a4b-it",
                contents=[PROMPT, {"inline_data": {"mime_type": "image/jpeg",
                                                   "data": b64}}])
        except Exception:
            time.sleep(1.5 * (attempt + 1) + random.random())
            continue
        t = (r.text or "").strip().upper()
        for word in ("REFEREE", "PLAYER", "COACH", "OTHER"):
            if word in t:
                return word
        return None
    return None


def body_crop(frame, bbox):
    """The WHOLE body, not the jersey patch. A striped shirt, a tracksuit and a
    kit are told apart by the whole silhouette; a torso rectangle throws away
    the shorts, the legs and the stance that carry most of that signal."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    H, W = frame.shape[:2]
    px, py = int(0.1 * (x2 - x1)), int(0.05 * (y2 - y1))
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(W, x2 + px), min(H, y2 + py)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


VOTES_JSON = os.path.join(_HERE, "out", "is_player_votes.json")


def main():
    # REUSE THE VOTES. The model calls are the only expensive part, and the
    # SCORING RULE is the thing being iterated on -- rescoring must never cost
    # another API run. Delete the file to re-ask.
    if os.path.exists(VOTES_JSON):
        rows = [tuple(r) for r in json.load(open(VOTES_JSON, encoding="utf-8"))]
        print(f"reusing {len(rows)} cached votes from "
              f"{os.path.basename(VOTES_JSON)} (delete it to re-ask)")
        return report(rows)

    import clip_config
    import env_local
    env_local.load()
    import google.genai
    sys.argv = ["x"]                       # exclusion_precondition reads argv
    import exclusion_precondition as ep

    client = google.genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    rows = []
    for clip in CLIPS:
        labs = ep.labels_of(clip)
        if not labs:
            continue
        cfg = clip_config.get_clip(clip)
        trk = json.load(open(cfg.tracks_cache_path, encoding="utf-8"))
        boxes = defaultdict(list)
        for fr in trk["frames"]:
            for t in fr["tracks"]:
                if t["track_id"] in labs and (t["bbox"][3] - t["bbox"][1]) >= MIN_BOX_H:
                    boxes[t["track_id"]].append((fr["frame_index"], t["bbox"]))
        picks = {}
        for tid, bb in boxes.items():
            bb.sort()
            step = max(1, len(bb) // CROPS_PER_TRACK)
            picks[tid] = bb[::step][:CROPS_PER_TRACK]
        need = sorted({f for v in picks.values() for (f, _b) in v})
        if not need:
            continue
        imgs = dict(s2.iter_frames(cfg.video_path, need))

        jobs = []
        for tid, v in picks.items():
            for (f, bb) in v:
                im = imgs.get(f)
                if im is None:
                    continue
                crop = body_crop(im, bb)
                if crop is None or crop.size == 0:
                    continue
                big = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                ok, jpg = cv2.imencode(".jpg", big)
                if ok:
                    jobs.append((tid, base64.standard_b64encode(jpg.tobytes()).decode()))
        print(f"{clip}: asking about {len(jobs)} crops from {len(picks)} "
              f"labelled tracks", flush=True)
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            res = list(ex.map(lambda j: ask_once(client, j[1]), jobs))

        votes = defaultdict(list)
        for (tid, _b), v in zip(jobs, res):
            if v:
                votes[tid].append(v)
        for tid, vs in votes.items():
            lab = labs[tid]
            truth = ("non-player" if lab in NON_PLAYER
                     else "player" if lab.isdigit() else None)
            if truth is None:
                continue
            top, n = Counter(vs).most_common(1)[0]
            rows.append((clip, tid, truth, top, n, len(vs)))

    os.makedirs(os.path.dirname(VOTES_JSON), exist_ok=True)
    json.dump([list(r) for r in rows], open(VOTES_JSON, "w"), indent=1)
    print(f"saved votes -> {VOTES_JSON}")
    report(rows)


def report(rows):
    """Score the SAME votes under two rules.

    The asymmetry is the whole point. Keeping a referee leaves the count ONE TOO
    HIGH, and the exclusion gate already refuses on that (it needs exactly five)
    -- costly but safe. Dropping a real player makes a five look like a FOUR,
    and a four with one unnamed body FORCES A WRONG NAME. So a rule is judged on
    players deleted first, referees caught second.
    """
    def score(name, is_drop):
        tp = fn = tn = fp = 0
        for (clip, tid, truth, top, n, tot_v) in rows:
            drop = is_drop(top, n, tot_v)
            if truth == "player":
                fn += drop
                tp += (not drop)
            else:
                tn += drop
                fp += (not drop)
        total = tp + fn + tn + fp
        print(f"\n  {name}")
        print(f"     real PLAYERS     : {tp} kept, {fn} WRONGLY DROPPED")
        print(f"     real NON-PLAYERS : {tn} refused, {fp} kept (safe)")
        print(f"     overall correct  : {tp + tn}/{total} "
              f"({100.0 * (tp + tn) / max(total, 1):.0f}%)")
        if fn:
            for (clip, tid, truth, top, n, tot_v) in rows:
                if truth == "player" and is_drop(top, n, tot_v):
                    print(f"       deleted a real player: {clip} t{tid} "
                          f"called {top} ({n}/{tot_v})")

    print(f"\n{'=' * 68}\nIS THIS A PLAYER?  ({len(rows)} labelled tracks)\n{'=' * 68}")
    score("RULE A -- majority says anything but PLAYER",
          lambda top, n, tot_v: top != "PLAYER")
    score("RULE B -- UNANIMOUS REFEREE or COACH only "
          "(OTHER means 'cannot tell', and a split vote is not evidence)",
          lambda top, n, tot_v: top in ("REFEREE", "COACH") and n == tot_v)

    print("\n  every track, for the record:")
    for (clip, tid, truth, top, n, tot_v) in sorted(rows, key=lambda r: (r[0], r[2])):
        print(f"     {clip:<6} t{tid:<5} truth={truth:<11} model={top:<8} "
              f"{n}/{tot_v} votes")
    print("\n  COMPARE: colour (2026-08-30) had NO threshold that caught a single "
          "non-player\n  without also deleting a real player.")


if __name__ == "__main__":
    main()
