"""Phase 5 -- BALL TOUCHES: which player has the ball, frame by frame.

A TOUCH is ONE PLAYER holding the ball until she gives it up. It is NOT a
POSSESSION. A possession is a TEAM concept -- one team has the ball until they
score, turn it over, or are rebounded against -- and several touches happen
inside a single possession (pass, pass, drive, shoot = four touches, one
possession). phase2/possessions.py already owns the team-level meaning and is
untouched by this module; the two must never be conflated (DJ, 2026-07-27).

WHY THIS EXISTS: ball detection works and player tracking works, but nothing
ever joined them, so the system could say a body MOVED left and never that she
moved left WITH THE BALL. This module is that join. It needs no new model and
no new video pass -- every input is already cached on disk.

THE MEASUREMENT, in one line: the ball is HELD by the tracked body whose box it
is nearest to, measured in BODY HEIGHTS rather than pixels so the gate means
the same thing at both ends of the floor (the zoom-independence trick from the
TEST 16 pose rule -- the only signal in this project to survive a real
holdout). Distance is to the box, not the feet: the ball is in the HANDS.

THE HONEST RISK, ON THE RECORD BEFORE ANY RESULT: "nearest to the ball" is not
"has the ball". A pass flying close over a girl's head looks identical to a
hold for a handful of frames, and a rebound scrum puts four bodies within arm's
reach at once. The defences are the CONTESTED rule (a near-tie credits nobody)
and MIN_TOUCH_FRAMES (a blink is a pass going past, not a catch). Both are
HYPOTHESES until DJ eyeballs the overlay. Every threshold here was frozen in
tasks/todo.md BEFORE the first run and must not be tuned after seeing which
answers look nicer -- that is exactly how accel_y died (DECISIONS/TEST 11).

ABSTENTION, same discipline as the rest of the pipeline -- refusing to answer
beats answering wrong:
    no ball detected            -> nobody holds it, and it is NOT interpolated
    ball far from every body    -> nobody holds it (it is in the air)
    two bodies nearly tied      -> CONTESTED, nobody is credited
    run shorter than the floor  -> dropped, never counted
    identity not CONFIRMED      -> review item, never a bare stat

Nothing here writes into team_events or any Phase 1/2 artifact (ROADMAP
Principle 4: new layers sit BESIDE the spine, never inside it).
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "phase2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# REUSED, not rewritten: the ball leaves the HANDS, not the feet, so proximity
# to the BOX (0 when the ball is inside it) is the right test. shot_attempts
# already established this for release matching.
from shot_attempts import point_to_bbox_dist                      # noqa: E402

# --- FROZEN THRESHOLDS (declared in tasks/todo.md before the first run) ------
# Distances are fractions of the candidate's OWN bbox height, so a gate tuned
# on a near-court body still means the same thing on a far-court one.
HOLD_GATE_BODY_FRAC = 0.30   # ball within 30% of her own height of her box
MARGIN_BODY_FRAC = 0.15      # nearest must beat 2nd by this much, else CONTESTED
MIN_TOUCH_FRAMES = 6         # ~0.2s at 30fps; shorter runs are thrown away

# HOW LONG THE BALL MAY BE UNSEEN AND STILL BE HERS -- DJ's rule, ADOPTED
# 2026-07-28 after he watched the overlay end to end and confirmed the box
# stays on the right girl through every bridged stretch on TEST1:
#   "the girl last seen with the ball has the ball until proven otherwise"
# The ceiling is DJ's number and comes from basketball, not from a fit: no
# player holds the ball longer than this before passing, shooting or losing it.
# "Proven otherwise" is not a guess -- build_touches ends her run the instant
# a DIFFERENT holder proves itself (see the flicker guard below), so the rule
# only ever fills a hole where nobody else was seen with the ball.
# Expressed in SECONDS because that is the units DJ reasoned in; frames are
# derived per clip from its own fps.
# HONEST LIMIT ON THE EVIDENCE: eyeball-confirmed on TEST1 only. HARD's overlay
# has not been watched. And it raises the inferred share from ~14% to ~45%,
# which is exactly why observed_seconds and inferred_seconds are reported apart.
MAX_GAP_SECONDS = 15.0
MAX_GAP_FRAMES = 8           # mechanism default for direct/unit use only;
                             # analyze() derives the real value from fps


# ------------------------------------------------------------ per frame ----

def ball_position(detections, conf_floor):
    """Highest-confidence ball detection at conf >= conf_floor -> (cx, cy),
    or None. Same conf_floor every other ball stage uses -- no new floor is
    invented here. Multiple detections are NOT averaged: a mean of the real
    ball and a false positive is a position where the ball never was."""
    best = None
    for d in detections:
        if d["conf"] < conf_floor:
            continue
        if best is None or d["conf"] > best["conf"]:
            best = d
    if best is None:
        return None
    x1, y1, x2, y2 = best["bbox"]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def holder_at_frame(ball_xy, tracks, gate_frac=HOLD_GATE_BODY_FRAC,
                    margin_frac=MARGIN_BODY_FRAC, exclude=()):
    """Who is holding the ball in ONE frame?

    ball_xy: (cx, cy) or None.  tracks: [{"track_id":.., "bbox":[..]}, ...].
    exclude: track_ids that CANNOT hold the ball -- referees and bench players
    the human labelled (phase2/roster.load_ref_tracks). They are dropped from
    candidacy ENTIRELY, not merely from the winner slot, because a referee
    standing beside the ball-handler must not turn her real touch into a
    CONTESTED abstention either. Excluding them ADDS information (roster.py's
    own argument for the same filter in seeding).

    Returns a dict that ALWAYS explains itself:
        {"status": "held", "track_id":.., "dist_frac":.., "dist_px":..}
        {"status": "no_ball" | "no_tracks" | "too_far" | "contested", ...}
    Never returns a track_id it is not confident about."""
    if ball_xy is None:
        return {"status": "no_ball"}
    bx, by = ball_xy

    ranked = []                      # (dist_frac, track_id, dist_px)
    for t in tracks:
        if t["track_id"] in exclude:
            continue
        x1, y1, x2, y2 = t["bbox"]
        height = y2 - y1
        if height <= 0:              # degenerate box: carries no opinion
            continue
        d = point_to_bbox_dist(bx, by, t["bbox"])
        ranked.append((d / height, t["track_id"], d))
    if not ranked:
        return {"status": "no_tracks"}
    ranked.sort()

    best_frac, best_tid, best_px = ranked[0]
    if best_frac > gate_frac:
        return {"status": "too_far", "nearest_track_id": best_tid,
                "dist_frac": round(best_frac, 3)}
    if len(ranked) > 1 and (ranked[1][0] - best_frac) < margin_frac:
        return {"status": "contested", "track_ids": [best_tid, ranked[1][1]],
                "dist_fracs": [round(best_frac, 3), round(ranked[1][0], 3)]}
    return {"status": "held", "track_id": best_tid,
            "dist_frac": round(best_frac, 3), "dist_px": round(best_px, 1)}


# --------------------------------------------------------- runs of frames --

def build_touches(frame_holders, min_frames=MIN_TOUCH_FRAMES,
                  max_gap=MAX_GAP_FRAMES):
    """Squash per-frame holders into touches.

    frame_holders: [(frame, track_id or None), ...] in frame order.

    A touch runs while the SAME track keeps being credited. Frames where the
    system abstained do not end it unless the gap exceeds max_gap -- the ball
    detector drops frames and a dropout is not a turnover. end_frame is the
    last frame she was ACTUALLY credited, never the tail of a gap, so a touch
    is never reported as lasting longer than the evidence for it.

    Runs shorter than min_frames are DROPPED: that length is a pass flying
    past a body, not a catch.

    FLICKER GUARD (DJ, 2026-07-28): a change of holder must be SUSTAINED for
    min_frames before it counts as a real handover. Occlusion makes the credit
    jump to a passing body for a frame or two (A -> B -> A); measured on this
    footage at 1, 2 and 5 frames. A blip like that is not "proven otherwise"
    under DJ's own rule -- it is noise, and it must not (a) let B steal a
    touch, nor (b) chop A's genuine touch in half, which the old code did
    every single time even when B's blip was itself discarded.
    THE THRESHOLD IS min_frames ITSELF, deliberately -- no new tuned constant,
    and no number fitted to the flickers we happened to observe. The reasoning
    stands on its own: a run too short to prove she HELD the ball is equally
    too short to prove she TOOK it. Unconfirmed frames are credited to NOBODY.
    (Same shape as possessions.detect()'s HOLD_S, which already makes a
    court-side switch prove itself before it counts.)"""
    runs = []                        # [track_id, start, last_held, held_count]
    cur = None
    pend = None                      # [track_id, start, count] -- unconfirmed
    for (f, tid) in frame_holders:
        if cur is not None and f - cur[2] > max_gap:
            runs.append(cur)
            cur, pend = None, None
        if tid is None:
            continue
        if cur is None:
            cur, pend = [tid, f, f, 1], None
        elif tid == cur[0]:
            cur[2], cur[3] = f, cur[3] + 1
            pend = None              # she is back: the blip was noise
        else:
            if pend is None or pend[0] != tid:
                pend = [tid, f, 1]
            else:
                pend[2] += 1
            if pend[2] >= min_frames:            # handover PROVEN
                runs.append(cur)
                cur, pend = [pend[0], pend[1], f, pend[2]], None
    if cur is not None:
        runs.append(cur)

    return [{"track_id": tid, "start_frame": a, "end_frame": b,
             "held_frames": n, "span_frames": b - a + 1}
            for (tid, a, b, n) in runs if n >= min_frames]


# ------------------------------------------------------ identity + court ---

def attribute(touch, identity_by_ft, held_frames, numbers=None):
    """Join a touch to WHO she is. Strict on purpose: a touch is 'attributed'
    only if every credited frame agrees on ONE identity AND that identity was
    CONFIRMED at every one of them. Anything less is a review item -- a shaky
    identity cannot launder into a confident stat (stage5's rule, applied
    here unchanged).

    identity_by_ft: (frame, track_id) -> (window, identity_id, state).
    numbers: optional {(window, identity_id): roster_number} from the OCR
    registry ({clip}_ocr_confirms.json). WHY THIS MATTERS: an identity_id is a
    per-window internal COUNTER, not a jersey. TEST1 has a real #13 on the
    roster AND an identity_id 13 that belongs to a different girl -- printing
    the raw id would read as a jersey number to a human. jersey_number is None
    (reported as "unnamed") whenever the registry does not name her."""
    seen = [identity_by_ft.get((f, touch["track_id"])) for f in held_frames]
    known = [s for s in seen if s is not None]
    if not known:
        return {"status": "no_identity_data", "jersey_number": None,
                "reason": f"track {touch['track_id']} has no identity event in "
                          f"frames {touch['start_frame']}..{touch['end_frame']}"}
    keys = {(w, i) for (w, i, _st) in known}
    states = Counter(st for (_w, _i, st) in known)
    numbers = numbers or {}
    nums = {numbers.get(k) for k in keys}
    # TWO IDENTITY RECORDS WITH THE SAME JERSEY ARE ONE GIRL, not two people.
    # identity_id is scoped PER WINDOW, so a touch that crosses a window
    # boundary picks up a second record for the same player. Treating that as
    # an identity split refused to name girls we plainly know (TEST1's #32 lost
    # her number the moment a touch spanned windows 0 and 1). If any key is
    # unnamed we still refuse -- we cannot prove an unnamed record is her.
    same_girl = len(keys) == 1 or (len(nums) == 1 and None not in nums)
    one_key = sorted(keys)[0] if len(keys) == 1 else None
    rec = {"identity_key": list(one_key) if one_key else None,
           "identity_keys": [list(k) for k in sorted(keys)],
           "jersey_number": next(iter(nums)) if same_girl else None,
           "states": dict(states),
           "frames_with_identity": len(known), "frames_credited": len(held_frames)}
    if same_girl and len(known) == len(held_frames) and set(states) == {"confirmed"}:
        rec["status"] = "attributed"
    else:
        rec["status"] = "review_item"
        rec["reason"] = ("identity split across "
                         f"{len(keys)} identity/identities, states {dict(states)}, "
                         f"{len(known)}/{len(held_frames)} credited frames "
                         f"carry identity data")
    return rec


def shooter_from_touches(touches, arc_start_frame, max_back_frames=None):
    """WHO SHOT IT -- DJ's proposal (2026-07-27): the last player actually SEEN
    holding the ball before the arc began.

    Contrast with what the pipeline does today (shot_attempts.find_release):
    extend the arc's fitted parabola backwards and credit whoever was STANDING
    nearest the predicted release point. That is a guess about where bodies
    were positioned. This is a memory of who was watched holding the ball --
    strictly better evidence, but it can only answer when a touch was actually
    recorded, so it ABSTAINS more often. Neither is adopted; they are scored
    head to head in spikes/shooter_compare.py.

    A touch that is still in progress when the arc starts counts (she shot out
    of her own touch). max_back_frames bounds how stale a memory may be -- a
    touch that ended long before this shot says nothing about it; None = no
    bound. Returns the touch, or None when there is nothing to say."""
    best = None
    for t in touches:
        if t["start_frame"] > arc_start_frame:
            continue                      # began after the shot: cannot be it
        gap = arc_start_frame - t["end_frame"]
        if max_back_frames is not None and gap > max_back_frames:
            continue                      # too stale to speak to this shot
        if best is None or t["end_frame"] > best["end_frame"]:
            best = t
    return best


def jersey_numbers(registry_doc):
    """{(window, identity_id): roster_number} from {clip}_ocr_confirms.json --
    the SAME key the box score uses (stage8_box_score), so a touch and the box
    score can never disagree about who a given identity is."""
    return {(r["window"], r["identity_id"]): r["roster_number"]
            for r in registry_doc.get("identities", [])
            if r.get("roster_number") is not None}


def court_feet_at(oncourt_doc):
    """(frame, track_id) -> [x_ft, y_ft] from the on-court cache. Missing
    entries stay missing; a touch with no court position reports None rather
    than a guessed spot on the floor."""
    out = {}
    for fr in oncourt_doc["frames"]:
        f = fr["frame_index"]
        for tid_s, info in fr["tracks"].items():
            out[(f, int(tid_s))] = (info["court_feet"], info["on"])
    return out


# ---------------------------------------------------------------- assemble --

def analyze(det_doc, tracks_doc, oncourt_doc, events_doc, conf_floor,
            registry_doc=None, exclude_tracks=(),
            gate_frac=HOLD_GATE_BODY_FRAC, margin_frac=MARGIN_BODY_FRAC,
            min_frames=MIN_TOUCH_FRAMES, max_gap=None):
    """The whole join, as one pure function over already-loaded documents.

    Only the OVERLAP of the ball span and the tracks span can be answered: a
    frame with players but no ball detection run, or a ball position with no
    tracked bodies, is not evidence of anything."""
    exclude = frozenset(exclude_tracks)
    fps = det_doc.get("fps") or tracks_doc.get("fps") or 30.0
    if max_gap is None:                      # DJ's ceiling, in this clip's fps
        max_gap = int(round(MAX_GAP_SECONDS * fps))
    ball_by_frame = {fr["frame_index"]: ball_position(fr["detections"], conf_floor)
                     for fr in det_doc["frames"]}
    tracks_by_frame = {fr["frame_index"]: fr["tracks"] for fr in tracks_doc["frames"]}
    overlap = sorted(set(ball_by_frame) & set(tracks_by_frame))

    per_frame, reasons = [], Counter()
    for f in overlap:
        v = holder_at_frame(ball_by_frame[f], tracks_by_frame[f],
                            gate_frac, margin_frac, exclude)
        reasons[v["status"]] += 1
        per_frame.append((f, v.get("track_id")))

    touches = build_touches(per_frame, min_frames, max_gap)

    identity_by_ft = {}
    for e in events_doc["player_events"]:
        identity_by_ft[(e["frame"], e["track_id"])] = (e["window"],
                                                       e["identity_id"],
                                                       e["identity_state"])
    numbers = jersey_numbers(registry_doc) if registry_doc else {}
    feet = court_feet_at(oncourt_doc)
    credited = {f: tid for (f, tid) in per_frame if tid is not None}

    for t in touches:
        held = [f for f in range(t["start_frame"], t["end_frame"] + 1)
                if credited.get(f) == t["track_id"]]
        # SEEN vs FILLED IN, always kept apart (DJ, 2026-07-27). A touch spans
        # frames where the ball was visible in her hands AND frames bridged
        # across a detector dropout. Reporting only the total would let an
        # assumption launder into a hard number -- the same rule the identity
        # layer already applies with confirmed/candidate.
        t["observed_seconds"] = round(t["held_frames"] / fps, 2)
        t["inferred_seconds"] = round((t["span_frames"] - t["held_frames"]) / fps, 2)
        t["total_seconds"] = round(t["span_frames"] / fps, 2)
        t["identity"] = attribute(t, identity_by_ft, held, numbers)
        start_ft = feet.get((t["start_frame"], t["track_id"]))
        end_ft = feet.get((t["end_frame"], t["track_id"]))
        t["court_feet_start"] = start_ft[0] if start_ft else None
        t["court_feet_end"] = end_ft[0] if end_ft else None
        t["on_court"] = bool(start_ft and start_ft[1])

    return {"touches": touches, "frame_reasons": dict(reasons),
            "overlap_frames": len(overlap),
            "overlap_span": [overlap[0], overlap[-1]] if overlap else None,
            "excluded_tracks": sorted(exclude), "fps": fps}


def summary_lines(result, clip_name):
    """The RAW table, printed before any interpretation (the TEST 16 rule:
    report what was measured, then let a human judge it)."""
    r = result
    n = r["overlap_frames"]
    out = [f"BALL TOUCHES ({clip_name}) -- MEASURED, pending DJ review",
           f"  frames answerable (ball span AND tracks span): {n}"
           + (f"  [{r['overlap_span'][0]}..{r['overlap_span'][1]}]"
              if r["overlap_span"] else "")]
    for k in ("held", "no_ball", "too_far", "contested", "no_tracks"):
        c = r["frame_reasons"].get(k, 0)
        out.append(f"    {k:<10} {c:>6}  ({100 * c / max(n, 1):5.1f}%)")
    ts = r["touches"]
    obs = sum(t["observed_seconds"] for t in ts)
    inf = sum(t["inferred_seconds"] for t in ts)
    by_status = Counter(t["identity"]["status"] for t in ts)
    out.append(f"  touches (>= {MIN_TOUCH_FRAMES} credited frames): {len(ts)}"
               f"  totalling {obs + inf:.1f}s")
    out.append(f"    of that: {obs:.1f}s SEEN (ball visible in her hands)  +  "
               f"{inf:.1f}s FILLED IN (bridged across a dropout, "
               f"{100 * inf / max(obs + inf, 1e-9):.0f}% of the total)")
    out.append(f"  identity join: {dict(by_status)}")
    named = sum(1 for t in ts if t["identity"]["jersey_number"] is not None)
    out.append(f"  of those, {named} touch(es) belong to a girl we can NAME; "
               f"{len(ts) - named} are an unnamed body")
    if r.get("excluded_tracks"):
        out.append(f"  excluded from candidacy (human-labelled ref/bench): "
                   f"{r['excluded_tracks']}")
    off = [t for t in ts if not t["on_court"]]
    if off:
        # NOT auto-dropped: a real inbounds pass is thrown from behind the
        # baseline too, so filtering off-court would delete real touches along
        # with the junk. Flagged loudly instead, for a human to settle.
        out.append(f"  !! {len(off)} touch(es) sit OFF THE COURT "
                   f"({sum(t['total_seconds'] for t in off):.1f}s) -- either an "
                   f"inbounds pass or an unlabelled ref/bench body. NOT "
                   f"auto-dropped; marked 'off_court' below for a human call.")
    out.append("  every touch below is a CANDIDATE for review -- not a stat:")
    for t in ts:
        ident = t["identity"]
        # NEVER print a bare identity_id as if it were a jersey (it is an
        # internal per-window counter and reads as a number to a human).
        who = (f"#{ident['jersey_number']}" if ident["jersey_number"] is not None
               else "unnamed")
        ft = t["court_feet_start"]
        where = f"({ft[0]:.0f},{ft[1]:.0f})ft" if ft else "(no court pos)"
        flag = "" if t["on_court"] else "  <- OFF_COURT"
        out.append(f"    f{t['start_frame']:>5}..{t['end_frame']:<5} "
                   f"track {t['track_id']:>3}  "
                   f"{t['observed_seconds']:>4.1f}s seen +"
                   f"{t['inferred_seconds']:>4.1f}s filled  "
                   f"{who:<8} {ident['status']:<16} {where}{flag}")
    return out
