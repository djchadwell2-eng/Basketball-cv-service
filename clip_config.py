"""ClipConfig -- the ONE per-clip config object for the CV pipeline.

Every clip-specific value the pipeline needs lives here in plain-English fields.
These are the same per-game inputs the web app will eventually send (roster entry
is a planned product feature), so this dataclass is effectively the CV service's
future API contract -- keep the field names human-readable.

NOTE: calibration inputs (keyframes, court dims, clicked landmarks, scorebug
exclude regions) still live in spikes/clips_config.py, selected by its ACTIVE
name. A ClipConfig.name MUST match a clips_config entry so calibration and the
downstream pipeline agree on the clip. (Merging the two configs is future work.)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_ROOT = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class Team:
    name: str                       # e.g. "Milford (white)"
    jersey_color: str               # e.g. "white"
    numbers: frozenset              # roster numbers legible for this team


@dataclass
class ClipConfig:
    name: str                       # MUST match a spikes/clips_config.py entry (calibration)
    video_path: str                 # source video
    event_frames: range             # frames Phase 1 samples for team_events
    render_sample_frames: range     # frames the Stage-1 mask eyeball render uses
    tracking_span_start: int        # first frame of the ByteTrack span
    tracking_span_len: int          # number of frames in the ByteTrack span
    teams: tuple                    # roster: per-team numbers + jersey color (Team, ...)
    seed_labels: dict               # track_id -> jersey number (hand-verified first signal)
    accumulation_window_seconds: float  # OCR temporal-accumulation window (demo stand-in = 2.0s)
    tracks_cache_path: str          # where ByteTrack tracks are written/read

    def roster_numbers(self) -> set:
        out = set()
        for t in self.teams:
            out |= set(t.numbers)
        return out


def _cache(name: str) -> str:
    return os.path.join(_ROOT, "phase2", "out", f"{name}_tracks_raw.json")


# --- TEST1: the exact values relocated from the old module-level constants -----
#   video_path .............. spikes/clips_config.py CLIPS["TEST1"]["video_path"]
#   event_frames ............ phase1/stage2_generate_events.GEN_FRAMES
#   render_sample_frames .... phase1/stage1_court_roi.SAMPLE_FRAMES
#   tracking_span_* ......... phase2/run_tracking.SPAN_START / SPAN_LEN
#   teams / seed_labels ..... phase2/roster.TEAMS / ROSTER_NUMBERS / SEED_LABELS["TEST1"]
#   accumulation_window ..... phase2/stage3..6 DEMO_WINDOW_SECONDS
TEST1_CLIP = ClipConfig(
    name="TEST1",
    video_path=r"C:\Users\djcha\Downloads\Test1.mp4",
    event_frames=range(120, 581, 10),
    render_sample_frames=range(120, 581, 30),
    tracking_span_start=300,
    tracking_span_len=120,
    teams=(
        Team("Milford (white)", "white", frozenset({13})),
        Team("Little Miami (green)", "green", frozenset({5, 24})),
    ),
    seed_labels={17: 13, 6: 5},
    accumulation_window_seconds=2.0,
    tracks_cache_path=_cache("TEST1"),
)

# The clip the stage scripts operate on when run standalone. The combined entry
# point (run_clip.py) sets this (and spikes/clips_config.ACTIVE) per run.
ACTIVE_CLIP = TEST1_CLIP
