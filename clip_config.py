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
    # Escape hatch for DELIBERATE rigs only (e.g. a stand-in roster whose seed
    # labels are off-roster). A real per-game config must never need this.
    allow_off_roster_seeds: bool = False

    def roster_numbers(self) -> set:
        out = set()
        for t in self.teams:
            out |= set(t.numbers)
        return out

    def validate(self) -> None:
        """Fail LOUD on a malformed config (bad path, empty roster, seed labels
        that can never confirm). Called by run_clip and the cache builders."""
        problems = []
        if not os.path.exists(self.video_path):
            problems.append(f"video not found: {self.video_path}")
        if self.tracking_span_len <= 0 or self.tracking_span_start < 0:
            problems.append(f"bad tracking span {self.tracking_span_start}..+{self.tracking_span_len}")
        if not self.teams or not self.roster_numbers():
            problems.append("empty roster (teams/numbers)")
        bad_nums = [n for n in self.roster_numbers()
                    if not isinstance(n, int) or not 0 <= n <= 99]
        if bad_nums:
            problems.append(f"roster numbers must be ints 0..99: {sorted(bad_nums)}")
        off = {t: n for t, n in self.seed_labels.items()
               if n is not None and n not in self.roster_numbers()}
        if off and not self.allow_off_roster_seeds:
            problems.append(
                f"seed labels off-roster (can NEVER be OCR-confirmed): {off} "
                f"vs roster {sorted(self.roster_numbers())} -- fix the roster/label, "
                f"or set allow_off_roster_seeds=True for a deliberate rig")
        if self.accumulation_window_seconds <= 0:
            problems.append("accumulation_window_seconds must be > 0")
        if problems:
            raise SystemExit(f"INVALID ClipConfig {self.name!r} -- refusing to run:\n  - "
                             + "\n  - ".join(problems))
        dup = set()
        for i, a in enumerate(self.teams):
            for b in self.teams[i + 1:]:
                dup |= set(a.numbers) & set(b.numbers)
        if dup:
            print(f"  WARNING: number(s) {sorted(dup)} appear on BOTH teams -- "
                  f"OCR alone cannot tell them apart (resolved by the jersey-"
                  f"color tiebreak in stage8_box_score.py where color evidence "
                  f"allows; the rest stay AMBIGUOUS, never guessed)")


def _cache(name: str) -> str:
    return os.path.join(_ROOT, "phase2", "out", f"{name}_tracks_raw.json")


# --- TEST1: the exact values relocated from the old module-level constants -----
#   video_path .............. spikes/clips_config.py CLIPS["TEST1"]["video_path"]
#   event_frames ............ phase1/stage2_generate_events.GEN_FRAMES
#   render_sample_frames .... phase1/stage1_court_roi.SAMPLE_FRAMES
#   tracking_span_* ......... phase2/run_tracking.SPAN_START / SPAN_LEN
#   teams / seed_labels ..... phase2/roster.TEAMS / ROSTER_NUMBERS / SEED_LABELS["TEST1"]
#   accumulation_window ..... phase2/stage3..6 DEMO_WINDOW_SECONDS
#   REAL ROSTER (user-entered from film, 2026-07-06) -- replaces the 3-number
#   stand-in {13 | 5, 24}. This is the roster half of the FAIR remeasure.
TEST1_CLIP = ClipConfig(
    name="TEST1",
    video_path=r"C:\Users\djcha\Downloads\Test1.mp4",
    event_frames=range(120, 581, 10),
    render_sample_frames=range(120, 581, 30),
    # Fair-remeasure sample WIDENING (2026-07-06): tracking span extended from
    # the 4s validation window (300..+120) to the FULL validated pan 120..580
    # (~15s, ~8 x 2.0s windows). Window size deliberately unchanged -- bigger
    # sample, same protocol. Prior span-300 numbers: DECISIONS.md 4a.
    tracking_span_start=120,
    tracking_span_len=461,
    teams=(
        Team("Milford (white/red)", "white/red", frozenset({3, 13, 23, 44, 10})),
        Team("Little Miami (green/yellow)", "green/yellow", frozenset({24, 5, 32, 14, 30})),
    ),
    seed_labels={17: 13, 6: 5},         # hand-verified by eye (still on-roster)
    accumulation_window_seconds=2.0,
    tracks_cache_path=_cache("TEST1"),
)

# --- HARD: REAL ROSTER (user-entered from film, 2026-07-06). Full-pan tracking
#   span 600..1200 (the calibrated keyframe range), matching TEST1's wide-sample
#   protocol. NOTE: #3 appears on BOTH teams (validate() warns; OCR alone cannot
#   team-attribute a #3 -- first real case for the future color tiebreak).
#   Seed labels RETIRED: the old hand-verified {4: 23, 10: 30} were keyed to the
#   span-800 tracking run's ids (ids do not survive re-tracking, see DECISIONS
#   4a-WIDE). #30 contradiction RESOLVED 2026-07-06: user re-checked film --
#   the old label was a misread; there is no #30 in HARD. Rosters stand.
#   Empty labels = a pure READ-RATE
#   measurement: confident reads land as no_position_hypothesis, confirms 0 by
#   construction until click/position-based seeding exists.
HARD_CLIP = ClipConfig(
    name="HARD",
    video_path=r"C:\Users\djcha\Downloads\HARD.mp4",
    event_frames=range(600, 1201, 20),         # P1 team-event sampling across the pan
    render_sample_frames=range(600, 1201, 60),
    tracking_span_start=600,                    # full calibrated pan (600..1200)
    tracking_span_len=601,
    teams=(
        # #23 added 2026-07-11: crop evidence (HARD_check24.png t7) showed a
        # white/red #23; user confirmed. NOTE: #3 AND #23 now exist on BOTH
        # teams -> two team-ambiguous numbers until the color tiebreak.
        Team("Milford (white/red)", "white/red", frozenset({1, 3, 13, 23, 24, 44})),
        Team("Winton Woods (black/green)", "black/green", frozenset({10, 3, 23, 0, 20})),
    ),
    seed_labels={},
    accumulation_window_seconds=2.0,
    tracks_cache_path=_cache("HARD"),
)

# The clip the stage scripts operate on when run standalone. The combined entry
# point (run_clip.py) sets this (and spikes/clips_config.ACTIVE) per run.
ACTIVE_CLIP = TEST1_CLIP
