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
