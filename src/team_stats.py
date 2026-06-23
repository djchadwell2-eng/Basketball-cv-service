"""Team-level stats computed from court positions only (no identity).

Inputs are, per frame, the list of (team, court_x, court_y) for every on-court
body. From that we derive:
  - heatmap_data : raw court positions per team (binned into a PNG later)
  - court coverage + spacing per team
  - possession count + pace : an APPROXIMATE estimate from which half of the
    court the action sits on over time.

Everything here is intentionally simple. The possession/pace numbers especially
are rough first-pass estimates -- see the big comment on estimate_possessions().
"""

from __future__ import annotations

from .court_mapping import COURT_LENGTH_FT, COURT_WIDTH_FT

# Half-court line, in court feet (used to decide which end the action is on).
HALF_COURT_X = COURT_LENGTH_FT / 2.0


class TeamStatsAccumulator:
    """Collects per-frame team positions and turns them into team stats."""

    def __init__(self, fps: float):
        self.fps = fps
        # Flat list of [court_x, court_y] per team, across all frames (heatmaps).
        self.positions: dict[str, list[list[float]]] = {"team_a": [], "team_b": []}
        # Per-frame mean x of ALL on-court bodies -> drives possession side.
        # Each entry: (frame_index, mean_court_x) or (frame_index, None) if empty.
        self._frame_action_x: list[tuple[int, float | None]] = []
        # Per-frame per-team positions, kept to compute spacing.
        self._frame_team_positions: list[dict[str, list[tuple[float, float]]]] = []

    def add_frame(self, frame_index: int,
                  team_positions: list[tuple[str, float, float]]) -> None:
        """Record one frame's on-court bodies as (team, court_x, court_y)."""
        by_team: dict[str, list[tuple[float, float]]] = {"team_a": [], "team_b": []}
        all_x: list[float] = []
        for team, cx, cy in team_positions:
            if team in self.positions:
                # Round to 0.1 ft (~1 inch): far finer than the heatmap bins, but
                # keeps the JSON small enough to skim by hand over a full clip.
                self.positions[team].append([round(cx, 1), round(cy, 1)])
                by_team[team].append((cx, cy))
            all_x.append(cx)

        mean_x = sum(all_x) / len(all_x) if all_x else None
        self._frame_action_x.append((frame_index, mean_x))
        self._frame_team_positions.append(by_team)

    # ----- coverage & spacing -------------------------------------------------

    def court_coverage(self, team: str, cell_ft: float = 5.0) -> float:
        """Fraction of the court (in cell_ft x cell_ft cells) the team visited.

        A higher number means the team's bodies spread over more of the floor.
        """
        n_cols = max(1, int(COURT_LENGTH_FT / cell_ft))
        n_rows = max(1, int(COURT_WIDTH_FT / cell_ft))
        visited = set()
        for cx, cy in self.positions[team]:
            col = min(n_cols - 1, max(0, int(cx / cell_ft)))
            row = min(n_rows - 1, max(0, int(cy / cell_ft)))
            visited.add((col, row))
        return len(visited) / float(n_cols * n_rows)

    def avg_spacing(self, team: str) -> float:
        """Average distance (ft) between a team's bodies, averaged over frames.

        A rough "how spread out are they" number: small = bunched up, large =
        good floor spacing.
        """
        per_frame_means = []
        for by_team in self._frame_team_positions:
            pts = by_team.get(team, [])
            if len(pts) < 2:
                continue
            dists = []
            for i in range(len(pts)):
                for j in range(i + 1, len(pts)):
                    dx = pts[i][0] - pts[j][0]
                    dy = pts[i][1] - pts[j][1]
                    dists.append((dx * dx + dy * dy) ** 0.5)
            per_frame_means.append(sum(dists) / len(dists))
        if not per_frame_means:
            return 0.0
        return sum(per_frame_means) / len(per_frame_means)

    # ----- possessions & pace -------------------------------------------------

    def estimate_possessions(self, min_seconds_on_side: float = 4.0) -> list[dict]:
        """VERY approximate possession segmentation from court-side occupancy.

        Idea: in a possession, both teams cluster around one basket, so the mean
        x-position of all bodies sits on one half of the court. When the ball
        changes hands and play swings to the other basket, that mean crosses
        half court. We therefore count each sustained swing to the other side as
        a new possession.

        To avoid counting jitter around mid-court as dozens of fake possessions,
        a swing only counts once the action has stayed on the new side for
        `min_seconds_on_side`.

        TODO(later step): this is a placeholder. Real possession detection needs
        the ball and change-of-possession events. Treat these counts as a coarse
        pace proxy only.
        """
        min_frames = int(min_seconds_on_side * self.fps)

        events: list[dict] = []
        current_side: str | None = None        # "left" or "right"
        seg_start_frame: int | None = None
        # Candidate side we're waiting to confirm, and how long it has held.
        pending_side: str | None = None
        pending_count = 0

        def side_of(x: float | None) -> str | None:
            if x is None:
                return None
            return "left" if x < HALF_COURT_X else "right"

        for frame_index, mean_x in self._frame_action_x:
            side = side_of(mean_x)
            if side is None:
                continue  # empty frame, ignore

            if current_side is None:
                # First time we see anybody: start the first possession.
                current_side = side
                seg_start_frame = frame_index
                pending_side = side
                pending_count = 0
                continue

            if side == current_side:
                pending_side = side
                pending_count = 0
                continue

            # `side` differs from the current side -> maybe a swing.
            if side == pending_side:
                pending_count += 1
            else:
                pending_side = side
                pending_count = 1

            if pending_count >= min_frames:
                # Confirmed swing: close the current possession, open a new one.
                events.append(self._make_event(
                    len(events), current_side, seg_start_frame, frame_index))
                current_side = side
                seg_start_frame = frame_index
                pending_count = 0

        # Close the final open possession.
        if current_side is not None and seg_start_frame is not None:
            last_frame = self._frame_action_x[-1][0]
            events.append(self._make_event(
                len(events), current_side, seg_start_frame, last_frame))

        return events

    def _make_event(self, index, side, start_frame, end_frame) -> dict:
        return {
            "possession_index": index,
            "side": side,                 # which half the action sat on
            "start_frame": start_frame,
            "end_frame": end_frame,
            "start_time_s": round(start_frame / self.fps, 2),
            "end_time_s": round(end_frame / self.fps, 2),
            "approx": True,               # flag: rough estimate, see TODO above
        }

    def pace_per_minute(self, possessions: list[dict]) -> float:
        """Possessions per minute over the analyzed span (approximate)."""
        if not self._frame_action_x:
            return 0.0
        total_frames = self._frame_action_x[-1][0] - self._frame_action_x[0][0]
        minutes = (total_frames / self.fps) / 60.0 if self.fps else 0.0
        if minutes <= 0:
            return 0.0
        return round(len(possessions) / minutes, 2)
