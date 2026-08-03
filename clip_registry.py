"""ONE clip config the WEB APP can write, that BOTH Python config systems read.

THE PROBLEM THIS SOLVES. A clip needed two hand-edited Python files to exist:
  spikes/clips_config.py   calibration (keyframes, landmarks, exclude regions)
  clip_config.py           pipeline    (roster, spans, hoop anchors)
Both are Python SOURCE, so the app could only add a game by generating code --
and clip_config.py's own docstring has carried "(Merging the two configs is
future work.)" since it was written. That is the blocker to a browser setup
flow: a coach cannot edit Python.

THE FIX. A game uploaded through the app is ONE JSON document in
clips/<NAME>.json, holding both halves. The two Python configs merge these in
alongside their hand-written entries, so:
  - the hand-written clips (TEST1/HARD/TEST2/FULL_GAME*) are untouched and keep
    working exactly as before -- this adds a source, it does not replace one;
  - a JSON clip is a first-class clip everywhere the pipeline already looks.

A JSON clip may be PARTIAL. The setup flow fills it in stages (roster at
upload, landmarks after clicking, spans/hoops later), so anything missing is
simply absent and the consumer decides whether it has enough to run. Half a
config must never masquerade as a whole one.
"""

from __future__ import annotations

import json
import os
import re

_ROOT = os.path.dirname(os.path.abspath(__file__))
CLIPS_DIR = os.path.join(_ROOT, "clips")

# A clip name becomes a filename and a JSON artifact prefix all over this
# project, so it is restricted to the same characters the web app validates.
_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def valid_name(name: str) -> bool:
    return bool(_NAME_RE.match(name or ""))


def path_for(name: str) -> str:
    if not valid_name(name):
        raise ValueError(f"invalid clip name: {name!r}")
    return os.path.join(CLIPS_DIR, f"{name}.json")


def load(name: str) -> dict | None:
    """One registry clip, or None if it does not exist."""
    p = path_for(name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def load_all() -> dict:
    """Every registry clip, keyed by name. Missing directory = no clips."""
    out = {}
    if not os.path.isdir(CLIPS_DIR):
        return out
    for fn in sorted(os.listdir(CLIPS_DIR)):
        if not fn.endswith(".json"):
            continue
        name = fn[:-5]
        if not valid_name(name):
            continue
        try:
            with open(os.path.join(CLIPS_DIR, fn), encoding="utf-8") as fh:
                out[name] = json.load(fh)
        except (OSError, json.JSONDecodeError):
            # A malformed file must not take down every other clip's config.
            continue
    return out


def save(name: str, doc: dict) -> str:
    """Write (or overwrite) a registry clip. Returns the path."""
    p = path_for(name)
    os.makedirs(CLIPS_DIR, exist_ok=True)
    doc = dict(doc, name=name)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
    return p


def update(name: str, **fields) -> dict:
    """Merge fields into an existing clip (or create it). Returns the new doc.

    The setup flow writes a clip in stages, so this must MERGE rather than
    replace -- saving landmarks should never blank out the roster entered
    minutes earlier.
    """
    doc = load(name) or {"name": name}
    doc.update(fields)
    save(name, doc)
    return doc


# ---------------------------------------------------------------------------
# Adapters: registry JSON -> the shapes the two existing config systems expect.
# JSON has no tuples and no int keys, so the conversions live here, once.
# ---------------------------------------------------------------------------
def to_calibration_entry(doc: dict) -> dict:
    """Registry doc -> a spikes/clips_config.CLIPS entry.

    ONLY keyframes that actually carry marks are included. An unmarked keyframe
    contributes nothing to the solve and can only do harm: a dead frame (a black
    intro, say) shares no view with its neighbour, so the pair produces no
    homography at all, and that None used to travel into the optimiser and die
    as "'NoneType' object is not subscriptable". Skipping a frame nobody marked
    costs nothing and removes that whole failure.
    """
    landmarks = {}
    for frame, marks in (doc.get("landmarks") or {}).items():
        if not marks:
            continue
        landmarks[int(frame)] = [(m[0], float(m[1]), float(m[2])) for m in marks]
    keyframes = [k for k in (doc.get("keyframes") or []) if k in landmarks]
    return {
        "video_path": doc["video_path"],
        "keyframes": keyframes,
        "reference_pos": doc.get("reference_pos"),
        "exclude_regions": [tuple(r) for r in (doc.get("exclude_regions") or [])],
        # "auto" means solve the court from the marks rather than assume one --
        # the default for anything uploaded, since a new gym's floor is unknown.
        "court": doc.get("court", "auto"),
        "stills": keyframes,
        "landmarks": landmarks,
    }


def has_calibration(doc: dict) -> bool:
    """Enough clicked marks to attempt a calibration?"""
    lm = doc.get("landmarks") or {}
    return bool(doc.get("keyframes")) and any(lm.values())


def has_roster(doc: dict) -> bool:
    return any((t.get("numbers") or []) for t in (doc.get("teams") or []))


# ---------------------------------------------------------------------------
# Jersey colours: free text from the setup form -> a rough BGR reference.
#
# WHAT THIS IS FOR, and what it deliberately is NOT. The colour a coach types
# ("white", "green/yellow") is NOT the colour that jersey actually is on this
# footage -- a white jersey under gym lights reads grey, and a dark green one
# reads near-black. So these values must never be used as absolute targets to
# match a crop against.
#
# They are used for ONE job: putting the two teams in ORDER. The real colours
# are measured from the footage itself (phase2/touch_teams.py clusters the
# bodies actually on screen), which handles lighting for free; the typed names
# only decide WHICH of the two measured clusters is the home team and which is
# the away team. Getting that ordering right is a far easier question than
# matching an absolute colour, which is why it is split this way.
# ---------------------------------------------------------------------------

# BGR (OpenCV order), not RGB.
_COLOR_WORDS = {
    "white": (255, 255, 255), "silver": (192, 192, 192),
    "grey": (128, 128, 128), "gray": (128, 128, 128),
    "black": (0, 0, 0),
    "red": (0, 0, 255), "crimson": (60, 20, 220), "maroon": (0, 0, 128),
    "cardinal": (30, 20, 150), "burgundy": (32, 0, 128),
    "orange": (0, 165, 255), "yellow": (0, 255, 255), "gold": (0, 215, 255),
    "green": (0, 128, 0), "kelly": (60, 160, 70), "forest": (34, 80, 34),
    "teal": (128, 128, 0), "turquoise": (208, 224, 64),
    "blue": (255, 0, 0), "navy": (128, 0, 0), "royal": (225, 105, 65),
    "columbia": (235, 190, 130), "carolina": (235, 190, 130),
    "purple": (128, 0, 128), "violet": (211, 0, 138),
    "pink": (203, 192, 255), "brown": (42, 42, 165), "tan": (140, 180, 210),
}

# Modifiers that shift a colour rather than name one ("dark green", "light blue").
_DARK = 0.55
_LIGHT = 1.6


def parse_jersey_color(text):
    """Free-text jersey colour -> a rough (B, G, R), or None if unrecognisable.

    Handles the shapes the setup form actually produces: a single word
    ("white"), a modifier ("dark green"), and a school's two colours
    ("green/yellow", "white/red"). Several colours are AVERAGED -- the shirt is
    mostly the first colour with trim in the second, and for the ordering job
    described above an average separates two teams perfectly well.

    Returns None rather than a default when nothing is recognised. A guessed
    colour would silently mis-assign every possession in the game; refusing
    lets the caller say so out loud.
    """
    if not text:
        return None
    words = [w for w in re.split(r"[^a-z]+", str(text).lower()) if w]

    found, pending = [], 1.0
    for w in words:
        if w == "dark":
            pending = _DARK
            continue
        if w in ("light", "bright"):
            pending = _LIGHT
            continue
        bgr = _COLOR_WORDS.get(w)
        if bgr is None:
            pending = 1.0
            continue
        found.append(tuple(max(0.0, min(255.0, c * pending)) for c in bgr))
        pending = 1.0

    if not found:
        return None
    n = len(found)
    return tuple(sum(c[i] for c in found) / n for i in range(3))


def team_colors(doc: dict):
    """The two teams and their reference colours, or None if we cannot tell.

    -> [{"name":.., "jersey_color":.., "bgr":(b,g,r)}, ...] with exactly TWO
    entries, or None.

    None means the possession layer must ABSTAIN rather than guess: without two
    distinguishable colours there is no honest way to say which team has the
    ball. Cases that return None -- fewer or more than two teams, a missing or
    unrecognisable colour, or BOTH teams typed as the same colour (which is
    either a setup mistake or genuinely unplayable footage; either way we
    cannot separate them).
    """
    teams = doc.get("teams") or []
    if len(teams) != 2:
        return None
    out = []
    for t in teams:
        bgr = parse_jersey_color(t.get("jersey_color"))
        if bgr is None:
            return None
        out.append({"name": t.get("name"), "jersey_color": t.get("jersey_color"),
                    "bgr": bgr})
    if out[0]["bgr"] == out[1]["bgr"]:
        return None
    return out


def has_team_colors(doc: dict) -> bool:
    """Enough colour information to attempt possession detection?"""
    return team_colors(doc) is not None
