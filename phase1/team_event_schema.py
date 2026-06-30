"""STAGE 2a -- the team_event schema (the reliable, IDENTITY-FREE spine).

One team_event per frame: where the on-court bodies are, in pixels and court feet,
plus how much we trust this frame's calibration. NOTHING about WHO a body is.

WHY identity-free is an ARCHITECTURE rule, not laziness
-------------------------------------------------------
Most Phase-1 team stats (zone occupancy, where play happens, pace) fall out of
POSITIONS alone. Identity (which player, which team) is a separate, harder, error-
prone layer. If we bake a guessed identity into the spine, a wrong guess corrupts
every downstream per-team number -- confident-wrong is worse than honest-unknown.
So the spine abstains: identity_state and team both default to "unknown", and
identity (tracks, player_events) bolts on LATER as NEW data with zero rework here.

IRON RULES (enforced by this schema's shape):
  * NO track_id, NO player number, NO field linking a detection across frames.
  * The detections list is ORDER-INDEPENDENT: detection N in frame A has NOTHING
    to do with detection N in frame B. Never read identity out of list position.
  * team defaults to "unknown". We do NOT guess home/away.

This module ONLY defines the records + JSON (de)serialization. It computes no
stats and reads no video.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict

# Small string so later schema changes don't silently break old JSON files.
SCHEMA_VERSION = "team_event.v1"

# Allowed abstention-aware vocab (kept tiny on purpose).
IDENTITY_STATES = ("unknown",)                 # Phase 1 never asserts identity
TEAM_LABELS = ("unknown",)                     # Phase 1 never guesses a team
CONF_STATES = ("ok", "low_confidence")


@dataclass
class Detection:
    """One on-court body in one frame. No identity, no cross-frame link."""
    pixel: tuple[float, float]          # feet ground-contact, native image px
    court_feet: tuple[float, float]     # (x, y) on the court, in feet (homography)
    identity_state: str = "unknown"     # abstention is first-class; never a track_id
    team: str = "unknown"              # do NOT guess home/away in Phase 1


@dataclass
class HomographyConfidence:
    """How much to trust THIS frame's court calibration (the frame-450 guardrail).

    Measured from the direct nearest-keyframe match (inliers + mean reproj px).
    state='low_confidence' means: flag it, don't silently feed it into stats.
    """
    inliers: int
    reproj_px: float
    state: str                          # "ok" | "low_confidence"
    nearest_keyframe: int               # which keyframe this frame was matched to


@dataclass
class TeamEvent:
    """One frame: trusted-ness of the calibration + the on-court bodies."""
    frame_index: int
    timestamp_s: float
    homography_confidence: HomographyConfidence
    detections: list[Detection] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION


# --- JSON (de)serialization: deterministic, plain dicts ----------------------

def event_to_dict(ev: TeamEvent) -> dict:
    return asdict(ev)


def event_from_dict(d: dict) -> TeamEvent:
    hc = HomographyConfidence(**d["homography_confidence"])
    dets = [Detection(pixel=tuple(x["pixel"]),
                      court_feet=tuple(x["court_feet"]),
                      identity_state=x.get("identity_state", "unknown"),
                      team=x.get("team", "unknown"))
            for x in d["detections"]]
    return TeamEvent(frame_index=d["frame_index"],
                     timestamp_s=d["timestamp_s"],
                     homography_confidence=hc,
                     detections=dets,
                     schema_version=d.get("schema_version", SCHEMA_VERSION))


def write_events(events: list[TeamEvent], path: str, *, clip: str, fps: float) -> None:
    """Persist a whole clip's events deterministically as one JSON document."""
    doc = {
        "schema_version": SCHEMA_VERSION,
        "clip": clip,
        "fps": fps,
        "frames": [event_to_dict(ev) for ev in events],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)


def read_events(path: str) -> tuple[list[TeamEvent], dict]:
    """Load a clip's events; returns (events, document-level metadata)."""
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    events = [event_from_dict(d) for d in doc["frames"]]
    meta = {k: doc[k] for k in ("schema_version", "clip", "fps") if k in doc}
    return events, meta


# --- Stage 2a deliverable: print the schema + one populated example ----------

def _example_event() -> TeamEvent:
    """A hand-built, fully-populated example (NOT from video) to show the shape."""
    return TeamEvent(
        frame_index=300,
        timestamp_s=10.0,
        homography_confidence=HomographyConfidence(
            inliers=2568, reproj_px=0.61, state="ok", nearest_keyframe=420),
        detections=[
            Detection(pixel=(742.0, 564.0), court_feet=(23.2, 25.0)),
            Detection(pixel=(540.5, 695.0), court_feet=(20.6, 15.4)),
            Detection(pixel=(905.0, 711.0), court_feet=(27.0, 16.9)),
        ],
    )


def main():
    print("=" * 70)
    print(f"TEAM_EVENT SCHEMA  ({SCHEMA_VERSION})")
    print("=" * 70)
    print(__doc__)
    print("Record shapes:")
    print("  TeamEvent             : frame_index, timestamp_s,")
    print("                          homography_confidence, detections[], schema_version")
    print("  HomographyConfidence  : inliers, reproj_px, state(ok|low_confidence),")
    print("                          nearest_keyframe")
    print("  Detection             : pixel(x,y), court_feet(x,y),")
    print("                          identity_state='unknown', team='unknown'")
    print("\n  NO track_id / NO cross-frame link / detections order-independent /")
    print("  team never guessed.\n")
    print("-" * 70)
    print("One fully-populated example team_event (round-tripped through JSON):")
    ev = _example_event()
    as_dict = event_to_dict(ev)
    text = json.dumps(as_dict, indent=2)
    print(text)
    # prove (de)serialization is lossless on the example
    back = event_from_dict(json.loads(text))
    assert event_to_dict(back) == as_dict, "round-trip mismatch"
    print("\nround-trip on example: OK (parsed back identical)")


if __name__ == "__main__":
    main()
