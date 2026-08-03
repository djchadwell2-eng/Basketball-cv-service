"""WHICH TEAM was holding the ball -- jersey colour attached to each TOUCH.

THE JOIN THIS MAKES. spikes/ball_touch.py already answers "which BODY has the
ball" (a TOUCH: one player holding it until she gives it up). It answers with a
track_id, which is a number the tracker made up -- it says nothing about teams.
This module puts a TEAM on each of those touches, so team_possessions.py can
chain them into possessions.

HOW, AND WHY IT IS SPLIT IN TWO STEPS. The coach types two jersey colours into
the setup form ("white", "green/yellow"). It is tempting to compare each crop
straight to those colours -- and that is exactly the mistake to avoid. A white
jersey under gym lights is grey; a dark green one is nearly black. Absolute
matching against a typed word would fail on real footage, and worse, it would
fail QUIETLY.

So the work is split:
  STEP 1 (measured, from the footage): cluster the bodies that actually held the
         ball into two groups by torso colour. The two centroids that come out
         are the REAL jersey colours on THIS video, under THIS lighting. No
         typed value is involved, so lighting is handled for free.
  STEP 2 (labelling only): use the typed colours to decide which of those two
         measured clusters is the home team and which is the away team. That is
         an ordering question with two possible answers, which is a far easier
         thing to get right than an absolute colour match.

The human cost is ZERO beyond what the setup form already collects. DJ's
constraint (2026-08-02): the pipeline already asks a human for two heavy things
(calibration clicks, roster) and must not ask for a third.

ABSTENTION, as everywhere else here -- every one of these returns None/no team
rather than a guess, because a wrong team does not produce a slightly-wrong
possession, it produces a possession attributed to the wrong side of the ball:
    no two parsable colours          -> the whole layer abstains
    fewer than 2 tracks with colour  -> cannot cluster, abstains
    the two clusters are not separated -> abstains (one colour on the floor)
    the two labellings score too close -> abstains (we can see two teams but
                                          cannot tell WHICH is which)
    a track with too few samples     -> that touch alone gets team None

Pure functions plus one video-sampling helper. Nothing here writes into
team_events or any Phase 1/2 artifact (ROADMAP Principle 4).
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np                                                # noqa: E402

# REUSED, not rewritten: the same "mean colour of a jersey crop" the OCR
# tiebreak already trusts, so two parts of the system can never disagree about
# what colour a given crop is.
from color_tiebreak import crop_color_signature                   # noqa: E402

# --- FROZEN THRESHOLDS (written down BEFORE the first run, per the rule that
# killed accel_y in DECISIONS/TEST 11 -- these must not be nudged afterwards to
# make a particular clip's answers look nicer) -------------------------------

# How far apart the two MEASURED cluster centroids must sit, as a plain BGR
# distance (the scale runs 0..441). Below this there is effectively one colour
# on the floor -- warm-ups, or both teams genuinely in similar shirts -- and
# splitting it in two would be inventing a distinction the pixels do not carry.
MIN_CENTROID_SEP = 30.0

# How far apart the two measured clusters must sit ALONG THE TEAM-COLOUR AXIS
# (see label_clusters for what that means). BGR units, same 0..441 scale. This
# is the confidence in the LABELLING specifically: below it we can see two
# groups but cannot say which is which, and a wrong answer swaps every team in
# the game rather than being slightly off.
#
# MEASURED, NOT FITTED: TEST1 comes out at 29.8 with a visually verified correct
# answer (one white jersey, three green -- crops eyeballed 2026-08-02), so 12.0
# sits well below a known-good case and well above the couple of units of noise
# you would get between two arbitrary groups of the same team. Still a FIRST
# GUESS until HARD and TEST2 are run; it must not be lowered to rescue a clip.
MIN_AXIS_SEP = 12.0

# The two colours typed at setup must themselves differ by at least this much,
# or there is no axis to project onto and the whole question is unanswerable.
MIN_REF_SEP = 40.0

# A track needs this many colour samples before its average means anything. One
# frame can be a blur, an occlusion, or a body half out of shot.
MIN_SAMPLES_PER_TRACK = 3

# Frames sampled per touch. Spread across the touch, so a single bad frame
# cannot decide a player's team.
SAMPLES_PER_TOUCH = 5


# ------------------------------------------------------------ the two teams --

def refs_from_teams(teams):
    """The two teams -> [{"name", "jersey_color", "bgr"}, ...], or None.

    Accepts BOTH shapes the project already stores a roster in, because the
    colours turned out to be recorded in both places already and neither needed
    a new field:
      - clip_config.Team objects (the hand-written clips: TEST1, HARD, TEST2)
      - registry dicts from clips/<NAME>.json (anything the web app uploaded)

    That is why this layer costs the coach NOTHING new. The setup form has been
    collecting jersey colours all along.

    None when there are not exactly two teams, when a colour cannot be read, or
    when both teams were given the same colour -- the caller must then abstain.
    """
    import clip_registry

    if not teams or len(teams) != 2:
        return None
    out = []
    for t in teams:
        if isinstance(t, dict):
            name, color = t.get("name"), t.get("jersey_color")
        else:
            name, color = getattr(t, "name", None), getattr(t, "jersey_color", None)
        bgr = clip_registry.parse_jersey_color(color)
        if bgr is None:
            return None
        out.append({"name": name, "jersey_color": color, "bgr": bgr})
    if out[0]["bgr"] == out[1]["bgr"]:
        return None
    return out


def sample_frames_for_touch(touch, n=SAMPLES_PER_TOUCH):
    """Which frames to look at for one touch -- up to n, spread evenly across
    it, so one blurred or occluded frame cannot decide a player's team."""
    a, b = touch["start_frame"], touch["end_frame"]
    if b <= a:
        return [a]
    n = max(1, min(n, b - a + 1))
    step = (b - a) / float(n - 1) if n > 1 else 0
    return sorted({int(round(a + i * step)) for i in range(n)})


# ------------------------------------------------------------ torso crop ----

def torso_box(bbox):
    """The chest region of a player box -> (x1, y1, x2, y2), or None.

    Same geometry src/team_assignment.py already uses: 15%..45% down the box and
    the central 40% of its width. That window is the jersey and little else --
    above it is head and hair, below it is shorts, and the left/right edges of a
    person box are mostly the floor behind her.
    """
    x1, y1, x2, y2 = [float(v) for v in bbox]
    h, w = y2 - y1, x2 - x1
    if h <= 0 or w <= 0:
        return None
    return (x1 + 0.30 * w, y1 + 0.15 * h, x1 + 0.70 * w, y1 + 0.45 * h)


def sample_torso(frame_bgr, bbox):
    """Mean (B, G, R) of one player's chest in one frame, or None if the crop
    lands outside the frame or comes back empty."""
    box = torso_box(bbox)
    if box is None:
        return None
    H, W = frame_bgr.shape[:2]
    x1 = max(0, int(box[0]))
    y1 = max(0, int(box[1]))
    x2 = min(W, int(box[2]))
    y2 = min(H, int(box[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    patch = frame_bgr[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    return crop_color_signature(patch)


# -------------------------------------------------------- step 1: measure ---

def average_colors(samples_by_track, min_samples=MIN_SAMPLES_PER_TRACK):
    """{track_id: [bgr, ...]} -> {track_id: mean_bgr}, dropping any track with
    too few samples to average honestly."""
    out = {}
    for tid, sigs in samples_by_track.items():
        good = [s for s in sigs if s is not None]
        if len(good) < min_samples:
            continue
        arr = np.array(good, dtype=np.float64)
        out[tid] = tuple(arr.mean(axis=0))
    return out


def _dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def cluster_two(track_colors, min_sep=MIN_CENTROID_SEP):
    """Split the tracks into the two jersey colours actually on this footage.

    {track_id: mean_bgr} -> {"cluster_of": {track_id: 0|1},
                             "centroids": [bgr, bgr], "separation": float}
    or None when it cannot honestly be done (fewer than two tracks, or two
    centroids too close together to be two different shirts).

    Deliberately NOT cv2.kmeans with a random seed: with k=2 on a handful of
    points, the answer is fully determined by the two most distant points, so
    this walks straight to it. Same answer every run, no seed to get lucky with.
    """
    tids = sorted(track_colors)
    if len(tids) < 2:
        return None
    pts = {t: np.array(track_colors[t], dtype=np.float64) for t in tids}

    # Seed with the two furthest-apart tracks: on a floor with two kits, those
    # are one of each.
    seed_a, seed_b, best = None, None, -1.0
    for i, ta in enumerate(tids):
        for tb in tids[i + 1:]:
            d = _dist(pts[ta], pts[tb])
            if d > best:
                seed_a, seed_b, best = ta, tb, d

    ca, cb = pts[seed_a].copy(), pts[seed_b].copy()
    assign = {}
    for _ in range(20):                      # converges in a handful of passes
        new = {t: (0 if _dist(pts[t], ca) <= _dist(pts[t], cb) else 1)
               for t in tids}
        if new == assign:
            break
        assign = new
        for lbl, ref in ((0, "a"), (1, "b")):
            members = [pts[t] for t in tids if assign[t] == lbl]
            if members:                      # an empty cluster keeps its centre
                c = np.mean(members, axis=0)
                if ref == "a":
                    ca = c
                else:
                    cb = c

    sep = _dist(ca, cb)
    if sep < min_sep:
        return None                          # one colour on the floor
    if not any(v == 0 for v in assign.values()) or \
       not any(v == 1 for v in assign.values()):
        return None                          # everything landed in one cluster
    return {"cluster_of": assign, "centroids": [tuple(ca), tuple(cb)],
            "separation": sep}


# --------------------------------------------------------- step 2: label ----

def label_clusters(centroids, refs, min_axis_sep=MIN_AXIS_SEP,
                   min_ref_sep=MIN_REF_SEP):
    """Decide which measured cluster is which team.

    centroids: [bgr, bgr] measured from the footage (cluster 0, cluster 1).
    refs:      [{"name":.., "bgr":..}, {"name":.., "bgr":..}] the typed colours.

    -> {"team_of_cluster": [name0, name1], "axis_sep": float} or None.

    WHY THIS IS NOT A NEAREST-COLOUR MATCH, and how that was found out. The
    obvious version -- score both pairings by distance from each centroid to
    each typed colour, take the better -- was built first and FAILED on TEST1
    (2026-08-02). The measured centroids there were (121,97,109) and (82,93,101):
    two muddy greys, nothing like "white/red" or "green/yellow". A torso crop is
    never pure jersey; it carries skin, shadow, the floor behind her and the gym
    lighting, and averaging drags every jersey toward the same middling grey. So
    the ABSOLUTE distances were meaningless (280 vs 318, a 1.14x margin -- it
    abstained, correctly, but it could never have answered).

    The key observation: that muddying is a COMMON OFFSET. It pushes both teams
    the same way, so it cancels out of a comparison BETWEEN them. The clustering
    step had in fact separated the players perfectly (verified by eye against the
    crops: one white jersey alone, three green together) -- only the naming was
    stuck.

    So this asks a relative question instead. Take the axis running from team
    A's typed colour to team B's typed colour. Project both measured centroids
    onto it. Whichever sits further toward B's end IS B. Only the DIRECTION of
    the difference is used, never the absolute position, so the common grey
    offset drops straight out of the arithmetic.

    On TEST1 this separates the two clusters by 29.8 along that axis and names
    them correctly.
    """
    if len(centroids) != 2 or len(refs) != 2:
        return None
    c0, c1 = np.array(centroids[0], dtype=np.float64), \
        np.array(centroids[1], dtype=np.float64)
    r0 = np.array(refs[0]["bgr"], dtype=np.float64)
    r1 = np.array(refs[1]["bgr"], dtype=np.float64)

    axis = r1 - r0
    mag = float(np.linalg.norm(axis))
    if mag < min_ref_sep:
        return None            # the two typed colours are too alike to order by
    unit = axis / mag

    # Position of each cluster along the team A -> team B direction.
    p0 = float(np.dot(c0, unit))
    p1 = float(np.dot(c1, unit))
    sep = abs(p1 - p0)
    if sep < min_axis_sep:
        return None            # two groups visible, but not tellable apart

    if p0 <= p1:               # cluster 0 sits toward team A's end
        names = [refs[0]["name"], refs[1]["name"]]
    else:
        names = [refs[1]["name"], refs[0]["name"]]
    return {"team_of_cluster": names, "axis_sep": round(sep, 2)}


# ---------------------------------------------------------------- assemble --

def team_of_tracks(track_colors, refs):
    """{track_id: mean_bgr} + the two typed teams -> (teams, reason, detail).

    teams  {track_id: team_name}, empty when the layer abstained
    reason why it abstained, or None on success -- so the caller can print WHY
           rather than silently producing a teamless game
    detail the two confidence numbers, for the summary and the JSON, returned
           here rather than recomputed by the caller
    """
    clustered = cluster_two(track_colors)
    if clustered is None:
        return {}, ("could not split the bodies into two jersey colours "
                    "(one colour on the floor, or too few tracks)"), None
    labelled = label_clusters(clustered["centroids"], refs)
    if labelled is None:
        return {}, ("two jersey colours are visible but they cannot be matched "
                    "to the two colours entered at setup -- check the jersey "
                    "colours on this game"), None
    names = labelled["team_of_cluster"]
    out = {t: names[c] for t, c in clustered["cluster_of"].items()}
    return out, None, {"separation": round(clustered["separation"], 2),
                       "axis_sep": labelled["axis_sep"]}


def attach_teams(touches, team_by_track):
    """Put a team on every touch. A track with no colour verdict leaves the
    touch at team None -- which team_possessions.py then SKIPS rather than
    treating as a change of possession."""
    for t in touches:
        t["team"] = team_by_track.get(t["track_id"])
    return touches


def summary_lines(touches, refs, detail=None):
    """The raw table first, interpretation second (the TEST 16 rule)."""
    named = [t for t in touches if t.get("team")]
    out = [f"TOUCH TEAMS -- {len(named)}/{len(touches)} touches have a team"]
    if refs:
        out.append("  colours entered at setup: "
                   + ", ".join(f"{r['name']} ({r['jersey_color']})" for r in refs))
    if detail:
        out.append(f"  measured cluster separation: {detail['separation']:.1f} "
                   f"(floor {MIN_CENTROID_SEP:.0f})")
        out.append(f"  labelling separation along the team-colour axis: "
                   f"{detail['axis_sep']:.1f} (floor {MIN_AXIS_SEP:.0f})")
    for r in (refs or []):
        n = sum(1 for t in named if t["team"] == r["name"])
        out.append(f"    {r['name']:<32} {n:>4} touch(es)")
    unknown = len(touches) - len(named)
    if unknown:
        out.append(f"    {'(no team -- abstained)':<32} {unknown:>4} touch(es)")
    return out
