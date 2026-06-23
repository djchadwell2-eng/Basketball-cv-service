"""Entry point: video in -> team-stats JSON out.

Usage:
    python process_game.py --video path/to/game.mp4 --output path/to/out.json
    python process_game.py --video game.mp4 --output out.json --recalibrate
    python process_game.py --video game.mp4 --output out.json --max-frames 500

This is a STANDALONE program. It does not talk to any web app, database, or API.
It reads a video file and writes one JSON file you inspect by hand. Pair it with
render_heatmaps.py to turn that JSON into PNG heatmaps + a console summary.

Pipeline:
    1. calibrate the court ONCE (click landmarks; cached to a sidecar file)
    2. run YOLOv8m + ByteTrack over the frames (people only)
    3. map each player's feet to court feet; DROP anyone off the court
    4. cluster jersey colors into team A vs team B
    5. compute team heatmaps, coverage/spacing, and approximate possessions/pace
    6. write the schema JSON
"""

from __future__ import annotations

import argparse
import os

import cv2

from src.court_mapping import calibrate
from src.detection import iter_tracked_frames
from src.schema import build_output, write_output
from src.team_assignment import TeamColorModel
from src.team_stats import TeamStatsAccumulator


def get_video_fps(video_path: str) -> float:
    """Read frames-per-second from the video header (fallback to 30)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    # Some files report 0/NaN; fall back to a sane default so timing math works.
    if not fps or fps != fps:  # fps != fps catches NaN
        print("WARNING: could not read FPS from video; defaulting to 30.0")
        return 30.0
    return float(fps)


def main():
    parser = argparse.ArgumentParser(description="Basketball team-stats from video.")
    parser.add_argument("--video", required=True, help="Path to the input video file.")
    parser.add_argument("--output", required=True, help="Path to write the JSON output.")
    parser.add_argument("--recalibrate", action="store_true",
                        help="Force re-clicking the court landmarks (ignore cache).")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Optional: stop after N frames (handy for quick tests).")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        raise SystemExit(f"Video not found: {args.video}")

    game_id = os.path.splitext(os.path.basename(args.video))[0]
    fps = get_video_fps(args.video)
    print(f"Game: {game_id}   FPS: {fps:.2f}")

    # --- 1. Court calibration (click once, cached) ---------------------------
    mapper = calibrate(args.video, force_recalibrate=args.recalibrate)

    # --- 2-4. Detect/track, map to court, sample jersey colors ----------------
    team_colors = TeamColorModel()
    stats = TeamStatsAccumulator(fps=fps)

    # We need team labels (from color clustering) before we can attribute each
    # position to a team, but clustering needs all colors first. So we make a
    # single pass that stores per-frame (track_id, court_xy) for ON-court bodies
    # plus jersey-color samples, then resolve teams and replay the buffer.
    frames_processed = 0
    players_dropped_off_court = 0
    # Buffer of (frame_index, [(track_id, court_x, court_y), ...]) for on-court bodies.
    frame_buffer: list[tuple[int, list[tuple[int, float, float]]]] = []

    print("Running detection + tracking (this can take a while)...")
    for frame_index, frame_bgr, detections in iter_tracked_frames(args.video):
        frames_processed += 1

        on_court_here: list[tuple[int, float, float]] = []
        for det in detections:
            px, py = det.feet_pixel()
            cx, cy = mapper.pixel_to_court(px, py)

            # The court-bounds test is what removes bench / refs / crowd.
            if not mapper.is_on_court(cx, cy):
                players_dropped_off_court += 1
                continue

            on_court_here.append((det.track_id, cx, cy))
            # Sample jersey color only for on-court bodies (cleaner clusters).
            team_colors.add_sample(frame_bgr, det)

        frame_buffer.append((frame_index, on_court_here))

        if frames_processed % 100 == 0:
            print(f"  ...{frames_processed} frames")

        if args.max_frames is not None and frames_processed >= args.max_frames:
            print(f"Stopping early at --max-frames={args.max_frames}")
            break

    # --- 4b. Resolve teams, then replay the buffer into the stats accumulator --
    track_to_team = team_colors.finalize()
    print(f"Tracked bodies: {len(track_to_team)}  (clustered into team_a / team_b)")

    for frame_index, on_court_here in frame_buffer:
        team_positions = []
        for track_id, cx, cy in on_court_here:
            team = track_to_team.get(track_id)
            if team is None:
                continue  # no color sample -> unassignable, skip
            team_positions.append((team, cx, cy))
        stats.add_frame(frame_index, team_positions)

    # --- 5. Derive team stats -------------------------------------------------
    possessions = stats.estimate_possessions()
    team_stats = {
        "team_a": {
            "court_coverage": round(stats.court_coverage("team_a"), 3),
            "avg_spacing_ft": round(stats.avg_spacing("team_a"), 2),
            "position_samples": len(stats.positions["team_a"]),
        },
        "team_b": {
            "court_coverage": round(stats.court_coverage("team_b"), 3),
            "avg_spacing_ft": round(stats.avg_spacing("team_b"), 2),
            "position_samples": len(stats.positions["team_b"]),
        },
        "possessions_total": len(possessions),
        "pace_possessions_per_minute": stats.pace_per_minute(possessions),
        "possession_estimate_note": "APPROXIMATE -- derived from court-side "
                                    "occupancy, not ball tracking.",
    }

    # --- 6. Write JSON --------------------------------------------------------
    output = build_output(
        game_id=game_id,
        fps=fps,
        team_events=possessions,
        heatmap_data=stats.positions,
        frames_processed=frames_processed,
        players_dropped_off_court=players_dropped_off_court,
        team_stats=team_stats,
    )
    write_output(output, args.output)

    print(f"\nDone. Wrote {args.output}")
    print(f"  frames processed:            {frames_processed}")
    print(f"  off-court detections dropped: {players_dropped_off_court}")
    print(f"  approx possessions:          {len(possessions)} "
          f"({team_stats['pace_possessions_per_minute']}/min)")
    print("Render heatmaps with:  python render_heatmaps.py --input "
          f"{args.output}")


if __name__ == "__main__":
    main()
