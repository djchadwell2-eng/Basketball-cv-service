"""Render the team-stats JSON into heatmap PNGs and print a console summary.

Usage:
    python render_heatmaps.py --input out.json
    python render_heatmaps.py --input out.json --outdir heatmaps

For each team it writes <outdir>/<game_id>_<team>_heatmap.png showing where that
team's bodies spent time on the court, and prints the headline stats to console.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

# matplotlib only needed for rendering; import here so the core pipeline
# (process_game.py) doesn't depend on it.
import matplotlib
matplotlib.use("Agg")  # headless backend: write PNGs without needing a display
import matplotlib.pyplot as plt

# Court dimensions in feet -- kept in sync with src/court_mapping.py.
COURT_LENGTH_FT = 94.0
COURT_WIDTH_FT = 50.0


def _draw_court_outline(ax):
    """Draw a minimal court outline (boundary, half line, center circle)."""
    # Outer boundary.
    ax.plot([0, COURT_LENGTH_FT, COURT_LENGTH_FT, 0, 0],
            [0, 0, COURT_WIDTH_FT, COURT_WIDTH_FT, 0],
            color="white", linewidth=1.5)
    # Half-court line.
    ax.plot([COURT_LENGTH_FT / 2, COURT_LENGTH_FT / 2], [0, COURT_WIDTH_FT],
            color="white", linewidth=1.0)
    # Center circle (6 ft radius).
    circle = plt.Circle((COURT_LENGTH_FT / 2, COURT_WIDTH_FT / 2), 6.0,
                        fill=False, color="white", linewidth=1.0)
    ax.add_patch(circle)


def render_team_heatmap(positions, team_name, game_id, outdir, cell_ft=2.0):
    """Bin a team's positions into a 2D histogram and save it as a PNG."""
    out_path = os.path.join(outdir, f"{game_id}_{team_name}_heatmap.png")

    n_cols = max(1, int(COURT_LENGTH_FT / cell_ft))
    n_rows = max(1, int(COURT_WIDTH_FT / cell_ft))

    fig, ax = plt.subplots(figsize=(10, 5.5))

    if positions:
        arr = np.array(positions, dtype=float)
        # histogram2d expects (x, y); we want x along court length, y across width.
        heat, _, _ = np.histogram2d(
            arr[:, 0], arr[:, 1],
            bins=[n_cols, n_rows],
            range=[[0, COURT_LENGTH_FT], [0, COURT_WIDTH_FT]],
        )
        # Transpose so rows=y, cols=x for imshow; origin lower-left = court origin.
        ax.imshow(heat.T, origin="lower",
                  extent=[0, COURT_LENGTH_FT, 0, COURT_WIDTH_FT],
                  aspect="equal", cmap="hot", interpolation="gaussian")
    else:
        ax.text(COURT_LENGTH_FT / 2, COURT_WIDTH_FT / 2, "no positions",
                color="white", ha="center", va="center")
        ax.set_xlim(0, COURT_LENGTH_FT)
        ax.set_ylim(0, COURT_WIDTH_FT)
        ax.set_facecolor("black")

    _draw_court_outline(ax)
    ax.set_title(f"{game_id} -- {team_name} position heatmap")
    ax.set_xlabel("court length (ft)")
    ax.set_ylabel("court width (ft)")

    os.makedirs(outdir, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def print_summary(data):
    """Print the headline team stats to the console."""
    ts = data.get("team_stats", {})
    print("\n=========== TEAM STATS SUMMARY ===========")
    print(f"game_id            : {data.get('game_id')}")
    print(f"fps                : {data.get('fps')}")
    meta = data.get("meta", {})
    print(f"frames processed   : {meta.get('frames_processed')}")
    print(f"off-court dropped  : {meta.get('players_dropped_off_court')}")
    print("------------------------------------------")
    for team in ("team_a", "team_b"):
        t = ts.get(team, {})
        print(f"{team}:")
        print(f"    court coverage : {t.get('court_coverage')}")
        print(f"    avg spacing ft : {t.get('avg_spacing_ft')}")
        print(f"    position samples: {t.get('position_samples')}")
    print("------------------------------------------")
    print(f"approx possessions : {ts.get('possessions_total')}")
    print(f"pace (poss/min)    : {ts.get('pace_possessions_per_minute')}")
    print(f"note               : {ts.get('possession_estimate_note')}")
    print("==========================================\n")


def main():
    parser = argparse.ArgumentParser(description="Render team-stats heatmaps + summary.")
    parser.add_argument("--input", required=True, help="The JSON file from process_game.py.")
    parser.add_argument("--outdir", default="heatmaps", help="Directory for PNG output.")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    game_id = data.get("game_id", "game")
    heatmap_data = data.get("heatmap_data", {})

    print_summary(data)

    for team in ("team_a", "team_b"):
        positions = heatmap_data.get(team, [])
        path = render_team_heatmap(positions, team, game_id, args.outdir)
        print(f"Wrote heatmap: {path}  ({len(positions)} positions)")


if __name__ == "__main__":
    main()
