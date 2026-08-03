"""Team-level stats computed from court positions only (no identity).

Inputs are, per frame, the list of (team, court_x, court_y) for every on-court
body. From that we derive:
  - heatmap_data : raw court positions per team (binned into a PNG later)
  - court coverage + spacing per team

NO POSSESSIONS, NO PACE. This module used to estimate both from which half of
the court the bodies were standing on. That was never basketball -- a team can
spend a whole possession in the backcourt, and both teams stand at the same end
on every made basket -- and its own TODO admitted it was a placeholder waiting
for "the ball and change-of-possession events". Those now exist
(phase2/team_possessions.py, built on real ball tracking + jersey colour), so
the guess was DELETED rather than left to be mistaken for a measurement
(DJ, 2026-08-02: "that's not how possessions work").
"""

from __future__ import annotations

from .court_mapping import COURT_LENGTH_FT, COURT_WIDTH_FT


class TeamStatsAccumulator:
    """Collects per-frame team positions and turns them into team stats."""

    def __init__(self, fps: float):
        self.fps = fps
        # Flat list of [court_x, court_y] per team, across all frames (heatmaps).
        self.positions: dict[str, list[list[float]]] = {"team_a": [], "team_b": []}
        # Per-frame per-team positions, kept to compute spacing.
        self._frame_team_positions: list[dict[str, list[tuple[float, float]]]] = []

    def add_frame(self, frame_index: int,
                  team_positions: list[tuple[str, float, float]]) -> None:
        """Record one frame's on-court bodies as (team, court_x, court_y)."""
        by_team: dict[str, list[tuple[float, float]]] = {"team_a": [], "team_b": []}
        for team, cx, cy in team_positions:
            if team in self.positions:
                # Round to 0.1 ft (~1 inch): far finer than the heatmap bins, but
                # keeps the JSON small enough to skim by hand over a full clip.
                self.positions[team].append([round(cx, 1), round(cy, 1)])
                by_team[team].append((cx, cy))

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
