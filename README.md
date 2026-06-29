# basketball-cv-service

Standalone computer-vision tool that turns a basketball game video into a
**team-level** stats JSON, plus heatmap PNGs you can eyeball.

It runs entirely on your machine. **No web app, no database, no API** — video in,
one JSON file out, you inspect the JSON by hand.

```
video.mp4  ──▶  process_game.py  ──▶  out.json  ──▶  render_heatmaps.py  ──▶  *.png + console stats
```

## What it does (this step)

This step is **player-position TEAM stats only**. It does *not* know which player
is which — only where bodies are on the court. That is the reliable thing to
build first.

1. **Detection / tracking** — YOLOv8m @ `imgsz=1280`, people only (`classes=0`),
   tracker `bytetrack.yaml`. This is the config already validated on real
   footage. See [src/detection.py](src/detection.py).
2. **Court mapping (homography)** — you click ~6 known court landmarks once per
   video in an OpenCV window; we build the pixel→court transform from those
   clicks and cache them so you don't re-click on reprocess. Anything mapping
   **outside the court bounds is dropped** (the "13-person rule" fix that removes
   bench / refs / crowd). See [src/court_mapping.py](src/court_mapping.py).
2b. **Camera tracking** — real footage pans/zooms to follow the ball, so the
   click-once map would slide off the court. The tracker measures background
   motion each frame (ORB features + RANSAC, players masked out) and updates the
   homography so the map **follows the camera**. Works for a single continuous
   shot; a hard cut to another angle would break it. See
   [src/camera_tracking.py](src/camera_tracking.py).
3. **Team stats from positions** (no identity):
   - team assignment by **jersey-color clustering** (just "team A vs team B"),
     [src/team_assignment.py](src/team_assignment.py)
   - per-team **position heatmaps**, **court coverage**, and **spacing**
   - **approximate possession count + pace** from court-side occupancy
   - See [src/team_stats.py](src/team_stats.py).

## What it deliberately does NOT do

These are later steps — building them now would be the mistake. They are left as
clearly-marked `TODO` comments where relevant:

- individual player identity, jersey-number OCR, roster matching
- seeding / re-seeding UI, abstention/confidence state machine
- ball detection, shot detection
- any web server / API / FastAPI, any database / Supabase, any Next.js hookup
- authentication, multi-user, job queues, model fine-tuning

## Install

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

The first run downloads the `yolov8m.pt` weights automatically (via ultralytics).

## Run

```bash
# 1) process a video -> JSON  (first run opens the click-to-calibrate window)
python process_game.py --video path/to/game.mp4 --output out.json

# quick smoke test on the first 500 frames only:
python process_game.py --video path/to/game.mp4 --output out.json --max-frames 500

# force re-clicking the court landmarks:
python process_game.py --video path/to/game.mp4 --output out.json --recalibrate

# 2) render heatmaps + print the stats summary
python render_heatmaps.py --input out.json --outdir heatmaps
```

### Calibration (the one manual step)

On the first run for a video, an OpenCV window shows the first frame and prompts
you to click court landmarks in order (corners, free-throw line intersections,
center). Click the ones you can see; **skip** the rest. You need **at least 4**.

| key       | action                                   |
|-----------|------------------------------------------|
| left-click| record the prompted landmark             |
| `u`       | undo the last click                      |
| `s`       | skip the current landmark (not visible)  |
| `ENTER`/`q` | finish (once ≥ 4 points are recorded)  |

Clicks are saved to `path/to/game.mp4.calib.json` so reprocessing the same clip
skips this step (delete that file or pass `--recalibrate` to redo it).

## Output JSON shape

```jsonc
{
  "schema_version": 1,
  "game_id": "game",
  "fps": 30.0,
  "court_calibrated": true,
  "team_events": [ /* approximate possession segments, each "approx": true */ ],
  "heatmap_data": { "team_a": [[x,y], ...], "team_b": [[x,y], ...] },
  "team_stats": { /* coverage, spacing, pace */ },
  "meta": { "frames_processed": 0, "players_dropped_off_court": 0 }
}
```

The schema is built so a future individual-player layer can be **added** as new
top-level keys (`tracks`, `player_events`) without restructuring what's here.
See [src/schema.py](src/schema.py).

## End-to-end test

There is no heavy test harness — the intended check is the real one: run it on an
actual clip and confirm the artifacts appear.

```bash
python process_game.py --video your_clip.mp4 --output out.json --max-frames 500
python render_heatmaps.py --input out.json --outdir heatmaps
```

Verify: `out.json` exists and is valid JSON with non-empty `heatmap_data`, and
`heatmaps/game_team_a_heatmap.png` / `..._team_b_heatmap.png` were written.

## Project layout

```
basketball-cv-service/
├── process_game.py        # entry point: video -> JSON
├── render_heatmaps.py     # JSON -> heatmap PNGs + console summary
├── requirements.txt
├── README.md
└── src/
    ├── detection.py       # YOLOv8m + ByteTrack (validated config)
    ├── court_mapping.py   # homography + on-court filtering
    ├── camera_tracking.py # follows a panning/zooming camera frame-to-frame
    ├── team_assignment.py # jersey-color clustering -> team A / team B
    ├── team_stats.py      # heatmaps, coverage, spacing, possessions/pace
    └── schema.py          # builds/writes the output JSON
```

## Footage requirements

- **Single continuous camera** (no hard cuts to other angles). Panning and
  zooming are fine — the camera tracker handles them.
- A reasonably stable wide/broadside view where you can identify court landmarks
  on the first frame to calibrate.
- Calibration is cached per video file (`<video>.calib.json`); delete it or pass
  `--recalibrate` to redo the clicks.
