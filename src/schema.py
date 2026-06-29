"""Builds the output JSON document.

The shape is deliberately TEAM-LAYER ONLY for now. It is designed so that a
future "tracks" / "player_events" section can be ADDED as new top-level keys
WITHOUT restructuring anything that already exists here. Do not add those
sections now -- they are later steps.
"""

from __future__ import annotations

import json

# Bump this if the JSON shape ever changes in a breaking way, so downstream
# tooling (and future-you) can tell old files from new ones.
SCHEMA_VERSION = 1


def build_output(
    *,
    game_id: str,
    fps: float,
    team_events: list[dict],
    heatmap_data: dict[str, list],
    frames_processed: int,
    players_dropped_off_court: int,
    team_stats: dict,
    camera_motion_lost_frames: int = 0,
) -> dict:
    """Assemble the full output dictionary in the agreed schema."""
    return {
        "schema_version": SCHEMA_VERSION,
        "game_id": game_id,
        "fps": fps,
        "court_calibrated": True,
        # Approximate possession / pace segments (each flagged "approx": true).
        "team_events": team_events,
        # Raw on-court positions per team, for heatmap rendering.
        "heatmap_data": heatmap_data,
        # Extra derived team metrics (coverage, spacing, pace summary).
        "team_stats": team_stats,
        "meta": {
            "frames_processed": frames_processed,
            "players_dropped_off_court": players_dropped_off_court,
            # Frames where camera motion couldn't be measured (blur/few features)
            # and we assumed the camera held still. High counts => shakier map.
            "camera_motion_lost_frames": camera_motion_lost_frames,
        },
        # --- RESERVED FOR LATER STEPS -- do NOT populate now ------------------
        # A future step will add an individual-player layer. It will go in NEW
        # top-level keys like "tracks" and "player_events" so nothing above has
        # to change. Left as a documented placeholder on purpose.
        # "tracks": [...],
        # "player_events": [...],
    }


def write_output(output: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
