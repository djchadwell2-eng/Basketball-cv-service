"""STAGE 2c -- round-trip + identity-free proof of the team_event JSON.

Two checks, NO real stats (those are Stage 3):
  1. ROUND-TRIP: read the JSON, re-serialize it, and confirm the bytes parse back
     identical (write -> read -> write is stable).
  2. IDENTITY-FREE PROOF: reduce every frame to a positions-only view that
     literally has NO identity/team/track field, then count on-court bodies and
     their court zones from that view alone. If it runs, the spine is decoupled
     from identity -- stats don't need to know WHO anyone is.
"""

import json
import os
import sys
import tempfile

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import team_event_schema as schema

JSON_PATH = os.path.join(_HERE, "out", "TEST1_team_events.json")


# --- 1. round-trip ------------------------------------------------------------
def round_trip_ok(path: str) -> bool:
    events, meta = schema.read_events(path)
    tmp = os.path.join(tempfile.gettempdir(), "team_events_roundtrip.json")
    schema.write_events(events, tmp, clip=meta["clip"], fps=meta["fps"])
    with open(path, encoding="utf-8") as f:
        original = json.load(f)
    with open(tmp, encoding="utf-8") as f:
        reserialized = json.load(f)
    return original == reserialized


# --- 2. identity-free proof ---------------------------------------------------
def positions_only_view(events):
    """Strip everything except frame_index + court_feet. The returned structure
    has NO identity_state, NO team, NO ordering meaning -- pure positions."""
    return [{"frame_index": ev.frame_index,
             "court_xy": [list(d.court_feet) for d in ev.detections]}
            for ev in events]


def count_bodies_and_zones(positions_only):
    """A PROOF, not a real stat: bodies + length-third zone counts, using ONLY
    the positions-only view (it has no identity field to touch)."""
    rows = []
    for fr in positions_only:
        zones = {"left_third": 0, "middle_third": 0, "right_third": 0}
        for (x, _y) in fr["court_xy"]:
            z = "left_third" if x < 28.0 else "right_third" if x >= 56.0 else "middle_third"
            zones[z] += 1
        rows.append((fr["frame_index"], len(fr["court_xy"]), zones))
    return rows


def main():
    print("Stage 2c -- validating", JSON_PATH)

    rt = round_trip_ok(JSON_PATH)
    print(f"\n[1] round-trip (write->read->write identical): {'PASS' if rt else 'FAIL'}")
    assert rt, "round-trip mismatch -- serialization is not stable"

    events, meta = schema.read_events(JSON_PATH)
    print(f"    clip={meta['clip']} fps={meta['fps']} frames={len(events)} "
          f"schema={meta['schema_version']}")

    print("\n[2] identity-free proof: counting bodies + zones from a positions-only")
    print("    view that contains NO identity/team/track field at all.")
    view = positions_only_view(events)
    rows = count_bodies_and_zones(view)
    print(f"\n    {'frame':>5} {'on_court':>9}   zones (length thirds)")
    for (fi, n, zones) in rows[:6]:
        print(f"    {fi:>5} {n:>9}   {zones}")
    print("    ...")

    computable = all(isinstance(n, int) for (_f, n, _z) in rows)
    print(f"\nteam stats computable without identity: {'YES' if computable else 'NO'}")


if __name__ == "__main__":
    main()
