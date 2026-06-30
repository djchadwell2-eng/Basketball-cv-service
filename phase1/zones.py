"""STAGE 3a -- map a court-feet position to a named standard basketball zone.

Zones are HALF-COURT zones (paint, corners, wings, top-of-key, perimeter), folded
to the NEAREST basket -- because basketball zones are defined relative to a hoop,
and a pan clip puts action at one end at a time. A position at either end maps to
the same zone vocabulary.

Court coordinate system (from CONFIG, HS 84x50):
  x = 0..LENGTH along the court length (0 = left baseline, LENGTH = right baseline)
  y = 0..WIDTH  across the court width  (lane is centered on y = WIDTH/2)

Folding: nearest basket = left if x <= LENGTH/2 else right. `bx` = depth from that
basket's baseline (0 at baseline, grows toward midcourt).

LEFT / RIGHT are DIAGRAM-relative (not offense-relative, which would flip per end):
  LEFT  = lower sideline as drawn  (y < lane_y0)
  RIGHT = upper sideline as drawn  (y > lane_y1)

Depth bands are tied to real court features:
  PAINT      : inside the lane, depth <= free-throw line (ft_x)
  TOP_OF_KEY : straight-on (within lane width), ft_x < depth <= arc-top
  CORNER     : outside the lane, shallow (depth <= CORNER_DEPTH)
  WING       : outside the lane, CORNER_DEPTH < depth <= arc-top
  PERIMETER  : depth beyond the arc-top, out to midcourt (both ends)
(These are POSITIONAL zones, not 2pt/3pt shot zones.)
"""

from __future__ import annotations

import os
import sys

_SPIKES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "spikes")
sys.path.insert(0, _SPIKES)
import clips_config as cfg              # per-clip court dims (CONFIG-driven)

_COURT = cfg.active()["court"]
LENGTH = _COURT["length"]              # 84
WIDTH = _COURT["width"]               # 50
LANE_Y0 = _COURT["lane_y0"]           # 19
LANE_Y1 = _COURT["lane_y1"]           # 31
FT_X = _COURT["ft_x"]                 # 19 (free-throw line depth)

# Real-feature depths (HS 3-pt geometry): hoop 5.25 ft from baseline, arc r 19.75.
HOOP_DEPTH = 5.25
ARC_TOP_DEPTH = HOOP_DEPTH + 19.75    # 25.0 ft from baseline = top of the 3-pt arc
CORNER_DEPTH = 14.0                   # split between corner (shallow) and wing

ZONES = ("PAINT", "TOP_OF_KEY", "PERIMETER",
         "LEFT_CORNER", "RIGHT_CORNER", "LEFT_WING", "RIGHT_WING")


def zone_of(x: float, y: float) -> str:
    """Return the zone name for a court-feet position (folded to nearest basket)."""
    left_basket = x <= LENGTH / 2.0
    bx = x if left_basket else (LENGTH - x)      # depth from nearest baseline
    bx = max(0.0, bx)                            # behind-baseline clamps to the rim

    if LANE_Y0 <= y <= LANE_Y1:                  # straight-on (within lane width)
        if bx <= FT_X:
            return "PAINT"
        if bx <= ARC_TOP_DEPTH:
            return "TOP_OF_KEY"
        return "PERIMETER"

    side = "LEFT" if y < LANE_Y0 else "RIGHT"    # diagram-relative side
    if bx <= CORNER_DEPTH:
        return f"{side}_CORNER"
    if bx <= ARC_TOP_DEPTH:
        return f"{side}_WING"
    return "PERIMETER"


def _print_boundaries():
    print("Zone boundaries used (court-feet, folded to nearest basket):")
    print(f"  court           : {LENGTH:.0f} x {WIDTH:.0f} ft, lane y {LANE_Y0:.0f}-{LANE_Y1:.0f}")
    print(f"  depth bx        : 0 at nearest baseline -> {LENGTH/2:.0f} at midcourt")
    print(f"  PAINT           : y in [{LANE_Y0:.0f},{LANE_Y1:.0f}]  AND  bx <= {FT_X:.0f}")
    print(f"  TOP_OF_KEY      : y in [{LANE_Y0:.0f},{LANE_Y1:.0f}]  AND  {FT_X:.0f} < bx <= {ARC_TOP_DEPTH:.2f}")
    print(f"  LEFT/RIGHT CORNER: y<{LANE_Y0:.0f} / y>{LANE_Y1:.0f}  AND  bx <= {CORNER_DEPTH:.0f}")
    print(f"  LEFT/RIGHT WING  : y<{LANE_Y0:.0f} / y>{LANE_Y1:.0f}  AND  {CORNER_DEPTH:.0f} < bx <= {ARC_TOP_DEPTH:.2f}")
    print(f"  PERIMETER       : bx > {ARC_TOP_DEPTH:.2f}  (out to midcourt, either end)")
    print("  LEFT = lower sideline (y small), RIGHT = upper sideline (y large), as drawn.")


def main():
    _print_boundaries()

    # Known-position spot checks (so the user can verify the mapping by eye).
    checks = [
        ("left hoop / paint",      5.25, 25.0),
        ("left free-throw line",   19.0, 25.0),
        ("left low corner",        3.0,  6.0),
        ("left high corner",       3.0,  44.0),
        ("left low wing",          20.0, 8.0),
        ("top of key (left)",      23.0, 25.0),
        ("midcourt",               42.0, 25.0),
        ("right hoop / paint",     78.75, 25.0),
        ("right high corner",      81.0, 45.0),
    ]
    print("\nKnown-position spot checks:")
    for name, x, y in checks:
        print(f"  ({x:>5.1f},{y:>5.1f})  {name:<22} -> {zone_of(x, y)}")

    # A few real detections straight from the Stage-2 JSON.
    import json
    jp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out",
                      "TEST1_team_events.json")
    if os.path.exists(jp):
        doc = json.load(open(jp, encoding="utf-8"))
        fr = next(f for f in doc["frames"] if f["frame_index"] == 300)
        print(f"\nReal detections from frame {fr['frame_index']} (TEST1):")
        for det in fr["detections"][:6]:
            x, y = det["court_feet"]
            print(f"  court_feet=({x:>5.1f},{y:>5.1f}) -> {zone_of(x, y)}")


if __name__ == "__main__":
    main()
