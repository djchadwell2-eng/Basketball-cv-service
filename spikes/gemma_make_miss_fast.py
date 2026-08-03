"""Fast make/miss detection using Gemma - optimized for speed.

Strategy: Read only 2-3 key frames per shot instead of many.
- Frame 1: ~0.5s after shot (catch late score update)
- Frame 2: ~1.5s after shot (if score not changed yet)
- Stop if score changes

This reduces API calls from 6-10 per shot down to 2-3.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from typing import TypedDict

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)


class ShotMakeResult(TypedDict):
    start_frame: int
    end_frame: int
    outcome: str
    score_from: list[int] | None
    score_to: list[int] | None
    score_change_frame: int | None
    score_change_time_sec: float | None


SCOREBOARD_PROMPT = (
    "Look at this basketball scoreboard. What is the home score (the LEFT "
    "number) and the away score (the RIGHT number)? Answer ONLY in this "
    "format, no other text: Home: X Away: X\n"
    "If you cannot read it, answer: Home: ? Away: ?"
)


# --- FIX 1: crop to the board the human already marked, and enlarge it -------
#
# The old crop was a hardcoded fraction of the frame (bottom 28%, left 22%). It
# was wrong in both directions: on TEST1 it swept in the court, a referee's legs
# and the sponsor banner below the board, and on TEST2 it was NARROWER than the
# board (580 px of scorebug, 422 px of crop) so it cut the away score off.
#
# Every clip ALREADY records its scorebug rectangle -- clips_config
# exclude_regions, marked by a human so SIFT ignores the burned-in graphic. That
# is exactly the box wanted here, it is per-clip accurate, and reusing it costs
# no new human input (the same argument touch_teams makes for jersey colours).
UPSCALE = 3          # small digits; VLMs read an enlarged crop far better
_PAD = 0.02          # a hair of margin so a slightly tight box keeps its edges


def scoreboard_region(clip_cfg, frame_w, frame_h):
    """(x1, y1, x2, y2) of the scorebug in native pixels.

    Falls back to the old fixed fraction when a clip has no marked region, so a
    clip that was never marked still gets an answer rather than an exception --
    but the marked box is always preferred because it is measured, not guessed.
    """
    regions = (clip_cfg or {}).get("exclude_regions") if isinstance(clip_cfg, dict) \
        else getattr(clip_cfg, "exclude_regions", None)
    if regions:
        # The scorebug is the region a human drew for it. If a clip ever marks
        # several, take the largest -- the board is the big one.
        x1, y1, x2, y2 = max(regions, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
        px, py = (x2 - x1) * _PAD, (y2 - y1) * _PAD
        x1, y1, x2, y2 = x1 - px, y1 - py, x2 + px, y2 + py
    else:
        x1, y1, x2, y2 = 0.0, frame_h * 0.72, frame_w * 0.22, float(frame_h)
    return (max(0, int(x1)), max(0, int(y1)),
            min(frame_w, int(round(x2))), min(frame_h, int(round(y2))))


def scoreboard_crop(frame, clip_cfg, upscale=UPSCALE):
    """The scoreboard, cropped to its marked box and enlarged. None if empty."""
    import cv2
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = scoreboard_region(clip_cfg, w, h)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    if upscale and upscale != 1:
        crop = cv2.resize(crop, None, fx=upscale, fy=upscale,
                          interpolation=cv2.INTER_CUBIC)
    return crop


# --- FIX 3: does this score change match the shot that supposedly caused it? --
#
# We already know WHERE every shot was taken (spikes/shot_locations.json, court
# feet) and therefore what it was worth. A layup cannot be worth 1. That is free
# evidence the reader was never using, and it is what catches the TEST1 failure:
# an occluded "2" read reproducibly as "1" survives every other guard here,
# because 1 and 2 are both legal scores -- but not both legal for a shot at the
# rim.
_POINTS_BY_ZONE = {
    "paint": (2,),          # layup/dunk. Never 1, never 3.
    "midrange": (1, 2),     # a jumper is 2; a free throw also lands here (~15ft)
    "three": (3,),          # behind the arc
}


def points_allowed(zone):
    """Which score jumps this shot could possibly have caused. Unknown zone ->
    None, meaning "no opinion", and the caller must not filter on it: an
    unlocated shot is missing evidence, not evidence of a problem."""
    return _POINTS_BY_ZONE.get(zone)


def _zones_by_span(clip_name):
    """{(start_frame, end_frame): zone} from the shot chart this clip already
    produced. Empty dict when the shot-location layer has not run -- the caller
    then simply has no zone opinion, which is treated as "no opinion", never as
    a reason to reject a make."""
    path = os.path.join(_HERE, "out", f"{clip_name}_shot_locations.json")
    if not os.path.exists(path):
        return {}
    try:
        import measured_stats
        doc = json.load(open(path, encoding="utf-8"))
        court_len = measured_stats.court_length_for(clip_name)
    except Exception:
        return {}
    out = {}
    for loc in doc.get("locations", []):
        ft = loc.get("court_feet")
        if loc.get("status") != "located" or not ft:
            continue
        zone, _dist = measured_stats.classify_zone(ft[0], ft[1], court_len)
        out[(loc["start_frame"], loc["end_frame"])] = zone
    return out


def _read_frame_score(client, cap, frame_num, calib):
    """The score on one frame of the video -> (home, away), or None."""
    import cv2
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_num)))
    ok, frame = cap.read()
    if not ok:
        return None
    crop = scoreboard_crop(frame, calib)
    if crop is None:
        return None
    _ok, jpg = cv2.imencode(".jpg", crop)
    return _reread(client, base64.standard_b64encode(jpg.tobytes()).decode("utf-8"))


def _reread(client, image_data):
    """Read the same crop again -> (home, away), or None if it cannot be read.

    Used only to confirm a detected score change (see the call site). Any
    failure returns None, which the caller treats as disagreement -- an
    unconfirmable change must not become a confirmed make."""
    try:
        r = client.models.generate_content(
            model="gemma-4-26b-a4b-it",
            contents=[SCOREBOARD_PROMPT,
                      {"inline_data": {"mime_type": "image/jpeg",
                                       "data": image_data}}])
        h, a = _parse_response(r.text.strip())
        return None if h is None or a is None else (h, a)
    except Exception:
        return None


def is_scoring_play(last_home, last_away, home, away, zone=None):
    """Could basketball have produced this change in the score?

    WHY THIS EXISTS (found 2026-08-02 on TEST1). The reader used to treat ANY
    difference from the previous reading as a made basket. On a real run that
    produced "MAKE [0,0]->[1,0]" (a one-point field goal) and, worse,
    "MAKE [1,0]->[0,0]" -- a score going DOWN being confirmed as a basket. A
    scoreboard number cannot decrease, so that was never a make; it was the
    vision model misreading a small, low-contrast crop.

    That failure mode is the one this project forbids everywhere else: the
    make/miss layer's entire promise is "the scoreboard CONFIRMS makes"
    (DJ's rule), and a confirmation built on a misread is worse than no answer.

    A real made basket moves exactly ONE team's score UP by 1, 2 or 3 (free
    throw, field goal, three). Anything else -- a decrease, both teams moving at
    once, or a jump of 4+ -- is the board being misread, and is refused.

    ZONE, when given, is the shot's own zone from its measured court position,
    and it is what catches the case the rest of this cannot. TEST1's board is
    partially occluded at the deciding frame and Gemma reads "2" as "1" four
    times out of four -- reproducibly, so re-reading agrees with itself, and
    both 1 and 2 are legal scores so the check above passes. But that shot was
    taken at (6.7, 27.5) ft: a LAYUP, and a layup is never worth 1. The shot
    chart already knew that and nobody was asking it.

    NOTE this still cannot catch every misread: a wrong reading that happens to
    look like a legal jump FOR THAT ZONE still passes. It removes the impossible
    ones, which is a floor, not a guarantee.
    """
    dh, da = home - last_home, away - last_away
    if dh and da:
        return False                     # both teams cannot score at once
    delta = dh or da
    if delta not in (1, 2, 3):           # up by a legal amount; 0 handled by caller
        return False
    allowed = points_allowed(zone)
    if allowed is not None and delta not in allowed:
        return False                     # not worth that many from where she shot
    return True


def detect_makes_by_gemma_fast(
    clip_name: str,
    shots_json_path: str,
    api_key: str,
    fps: float = 60.0,
) -> list[ShotMakeResult]:
    """Fast make/miss detection - only reads key frames."""
    import cv2
    import clip_config

    try:
        import google.genai
    except ImportError:
        print("[gemma_fast] ERROR: google.genai not installed")
        return []

    if not os.path.exists(shots_json_path):
        return []

    try:
        with open(shots_json_path) as f:
            shots_doc = json.load(f)
    except Exception as e:
        print(f"[gemma_fast] ERROR reading shots: {e}")
        return []

    shots = [
        (a["start_frame"], a["end_frame"], a.get("hoop", "far"))
        for a in shots_doc.get("attempts", [])
        if a.get("verdict") == "shot_attempt"
    ]
    if not shots:
        return []

    # WHERE each shot was taken -> what it could be worth (FIX 3). Optional on
    # purpose: a clip with no located shots still gets make/miss, it just loses
    # this one cross-check rather than failing.
    zones = _zones_by_span(clip_name)

    # Get the clip's video
    clip_config_map = {
        "TEST1": clip_config.TEST1_CLIP,
        "TEST2": clip_config.TEST2_CLIP,
        "HARD": clip_config.HARD_CLIP,
    }

    if clip_name not in clip_config_map:
        print(f"[gemma_fast] ERROR: Unknown clip {clip_name}")
        return []

    clip = clip_config_map[clip_name]
    cap = cv2.VideoCapture(clip.video_path)
    if not cap.isOpened():
        print(f"[gemma_fast] ERROR: Could not open {clip.video_path}")
        return []

    client = google.genai.Client(api_key=api_key)

    shots_sorted = sorted(shots, key=lambda s: s[0])
    # THE SCORE AT THE START OF THIS CLIP IS NOT ZERO. This used to be seeded
    # (0, 0), which is only true of a clip that opens at tip-off. HARD starts at
    # 15-12 and TEST2 at 2-2, so the very first reading looked like a colossal
    # score change and was reported as a made basket -- "MAKE [0,0]->[15,12]".
    # Found 2026-08-03, once the impossible-move guard started refusing it
    # instead of shipping it.
    # None means "we have not established a baseline yet". The first successful
    # reading SETS it and is never itself a make: seeing a score for the first
    # time is not evidence that it just changed.
    confirmed_home, confirmed_away = None, None
    results = []

    print(f"[gemma_fast] Detecting makes for {len(shots_sorted)} shots...")

    # The CALIBRATION config carries the marked scorebug box (FIX 1). Separate
    # from clip_config, which is the pipeline half -- see clip_registry's
    # docstring on why this project has two.
    try:
        import clips_config
        calib = clips_config.CLIPS.get(clip_name)
    except Exception:
        calib = None
    if not (calib or {}).get("exclude_regions"):
        print(f"[gemma_fast] no marked scorebug region for {clip_name} -- "
              f"falling back to the fixed frame fraction (less accurate)")

    # SEED THE BASELINE FROM BEFORE THE FIRST SHOT. Without this the first shot
    # spends its frames just learning what the score already was and can never
    # be judged. One extra call buys the whole clip an answer for shot 1.
    seed_frame = max(0, shots_sorted[0][0] - 30)
    seeded = _read_frame_score(client, cap, seed_frame, calib)
    if seeded:
        confirmed_home, confirmed_away = seeded
        print(f"[gemma_fast] score before the first shot (f{seed_frame}): "
              f"{confirmed_home}-{confirmed_away}"
              + ("  <- NOT 0-0; this clip starts mid-game"
                 if seeded != (0, 0) else ""))
    else:
        print(f"[gemma_fast] could not read the board before the first shot -- "
              f"the first reading during shot 1 will set the baseline instead")

    for i, (start, end, hoop) in enumerate(shots_sorted):
        # FIX 2: LOOK AT MORE THAN TWO FRAMES. The old pair (+0.5s, +1.5s) had
        # no way to tell a board it could read from one a referee was standing
        # in front of -- and TEST1's deciding frame is exactly that, a dark
        # diagonal across the digits. Spreading the reads over the window after
        # the shot means an obstruction that lasts a moment no longer decides
        # the answer, and agreement ACROSS frames replaces trust in any one of
        # them. Still cheap: these stop as soon as the score is settled.
        key_frames = [end + 15, end + 30, end + 60, end + 90, end + 120]
        zone = zones.get((start, end))

        score_from = None
        score_to = None
        score_change_frame = None
        score_change_time = None
        outcome = "unknown"

        last_home, last_away = confirmed_home, confirmed_away
        found_score_change = False
        errors = []                  # why a frame could not be read, if it could not
        misreads = []                # score moves basketball cannot produce
        unstable = []                # changes a second read would not confirm

        print(f"  Shot {i+1}/{len(shots_sorted)}: ", end="", flush=True)

        for frame_num in key_frames:
            if found_score_change:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ok, frame = cap.read()
            if not ok:
                continue

            # Crop to the marked scorebug and enlarge it (FIX 1).
            crop = scoreboard_crop(frame, calib)
            if crop is None:
                continue

            # Send to Gemma
            _, jpg = cv2.imencode(".jpg", crop)
            image_data = base64.standard_b64encode(jpg.tobytes()).decode("utf-8")

            try:
                response = client.models.generate_content(
                    model="gemma-4-26b-a4b-it",
                    contents=[
                        # See SCOREBOARD_PROMPT for why the wording matters.
                        SCOREBOARD_PROMPT,
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": image_data,
                            }
                        },
                    ],
                )
                text = response.text.strip()

                home, away = _parse_response(text)

                if home is not None and away is not None:
                    if last_home is None:
                        # FIRST EVER READING: this is the scoreboard as the clip
                        # found it, not a change. Seeing 15-12 for the first time
                        # says nothing about whether a basket was just made.
                        last_home, last_away = home, away
                        confirmed_home, confirmed_away = home, away
                        print("=", end="", flush=True)
                    elif (home, away) == (last_home, last_away):
                        last_home, last_away = home, away
                        print(".", end="", flush=True)
                    elif is_scoring_play(last_home, last_away, home, away, zone):
                        # CONFIRM BEFORE BELIEVING IT. A make is declared off a
                        # SINGLE reading of one frame, and the readings are not
                        # stable on the frames that matter: TEST1 frame 274 --
                        # the frame that decides this very shot -- read as 1, 1
                        # and 12 across three attempts (2026-08-02). The
                        # impossible-move guard cannot help, because 1 and 2 are
                        # both legal scores.
                        # So the deciding frame is read once more and the change
                        # only counts if the second read AGREES. Cost is one
                        # extra call per DETECTED CHANGE (a handful per game),
                        # not per frame -- the cheapest possible place to spend
                        # it, since a disagreement here is the difference
                        # between a confirmed make and a fabricated one.
                        second = _reread(client, image_data)
                        if second != (home, away):
                            unstable.append(
                                f"f{frame_num} read {[home, away]} then "
                                f"{list(second) if second else 'unreadable'}")
                            print("~", end="", flush=True)
                            continue
                        score_from = [last_home, last_away]
                        score_to = [home, away]
                        score_change_frame = frame_num
                        score_change_time = frame_num / fps
                        outcome = "candidate_make"
                        confirmed_home, confirmed_away = home, away
                        found_score_change = True
                        print(f"MAKE {score_from}->{score_to}")
                    else:
                        # The number moved in a way basketball cannot produce,
                        # so it is a MISREAD OF THE BOARD, not a basket. Ignored
                        # rather than confirmed -- see is_scoring_play.
                        why = "impossible score move"
                        dh, da = home - last_home, away - last_away
                        delta = dh or da
                        allowed = points_allowed(zone)
                        if not (dh and da) and delta in (1, 2, 3) and allowed:
                            why = (f"a {zone} shot cannot be worth {delta} "
                                   f"(only {'/'.join(map(str, allowed))})")
                        misreads.append(
                            f"f{frame_num} {[last_home, last_away]}->"
                            f"{[home, away]}: {why}")
                        print("?", end="", flush=True)

            except Exception as e:
                # NEVER swallow this. It used to bind `e` and print a bare "E",
                # so a run that read nothing at all looked identical to one
                # that read the board fine and simply saw no score change --
                # and "unknown" is a legitimate answer here, which is exactly
                # what made the difference invisible (2026-08-02). The reasons
                # are collected and reported once per shot rather than per
                # frame, so a rate limit does not spam a hundred lines.
                errors.append(f"{type(e).__name__}: {e}")
                print("E", end="", flush=True)

        if not found_score_change:
            if unstable:
                # The most dangerous case, so it is named plainly: we SAW a
                # legal-looking score change and could not reproduce it on a
                # second read. Before this check that would have shipped as a
                # confirmed make.
                print(f"unknown  <- score change NOT REPRODUCIBLE on a second "
                      f"read, so not confirmed: {'; '.join(unstable[:2])}")
            elif misreads:
                print(f"unknown  <- {len(misreads)} impossible score move(s) "
                      f"REFUSED (board misread, not a basket): "
                      f"{'; '.join(misreads[:2])}")
            elif errors:
                # Say WHY there is no answer: a board we could not reach is not
                # the same fact as a board that showed no points.
                seen = []
                for msg in errors:
                    if msg not in seen:
                        seen.append(msg)
                print(f"unknown  <- {len(errors)} read(s) FAILED, not "
                      f"'no score change': {'; '.join(seen[:2])}")
            else:
                print("unknown")

        results.append(
            ShotMakeResult(
                start_frame=start,
                end_frame=end,
                outcome=outcome,
                score_from=score_from,
                score_to=score_to,
                score_change_frame=score_change_frame,
                score_change_time_sec=score_change_time,
            )
        )

    cap.release()
    return results


def _parse_response(response: str) -> tuple[int | None, int | None]:
    """Parse Gemma response."""
    home = None
    away = None

    parts = response.split()
    for i, part in enumerate(parts):
        if part == "Home:" and i + 1 < len(parts):
            try:
                home = int(parts[i + 1])
            except (ValueError, IndexError):
                pass
        elif part == "Away:" and i + 1 < len(parts):
            try:
                away = int(parts[i + 1])
            except (ValueError, IndexError):
                pass

    return home, away


if __name__ == "__main__":
    import os

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not set")
        sys.exit(1)

    clip_name = "TEST1"
    shots_path = os.path.join(_HERE, "out", f"{clip_name}_shot_attempts.json")

    results = detect_makes_by_gemma_fast(clip_name, shots_path, api_key)
    print(f"\n[gemma_fast] Results for {clip_name}:")
    for r in results:
        print(f"  {r['start_frame']}-{r['end_frame']}: {r['outcome']}", end="")
        if r["score_from"]:
            print(f" ({r['score_from']} -> {r['score_to']})", end="")
        print()
