"""Court mapping via homography.

WHY THIS MODULE EXISTS
----------------------
The detector gives us player positions in *pixel* coordinates, but pixels are
useless for team stats: a player near the camera occupies far more pixels and
moves faster on screen than a player at the far baseline. We need positions in
real *court* coordinates (feet) so that "where bodies spend time" and "which
half has the action" mean the same thing everywhere on the floor.

A basketball court is a flat plane, and the camera sees that plane through a
perspective projection. The mathematical map from one view of a plane to a
"top-down" view of the same plane is a HOMOGRAPHY: a single 3x3 matrix H such
that  [court_x, court_y, 1] ~ H @ [pixel_x, pixel_y, 1]  (up to scale).

To solve for H we need at least 4 point correspondences -- pairs of
(pixel location we clicked) <-> (known court location in feet). We ask the user
to click ~6 recognizable landmarks (corners, free-throw line intersections,
center) once per video. More than 4 makes the fit more robust to imprecise
clicks (cv2.findHomography least-squares averages out the slop).

A second, very useful side effect: once we can map any pixel to court feet, we
can DROP anything that maps outside the court rectangle. That is how we throw
out the bench, referees, and crowd without any person-classification -- the
"13-person rule" fix. Only bodies standing on the 94x50 ft floor count.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Court coordinate system.
#
# We use a full NBA-sized court measured in FEET, with the origin at one
# baseline/sideline corner:
#     x runs along the length of the court:  0 .. 94
#     y runs across the width of the court:  0 .. 50
#
# Key painted dimensions used for the landmarks below:
#   - free-throw line is 19 ft from the baseline
#   - the lane ("paint") is 16 ft wide, centered on the court -> y = 17 .. 33
#   - center line is at x = 47, center of court is (47, 25)
#
# These are approximate-but-consistent; what matters for team stats is that the
# same physical spot always maps to the same court coordinate.
# ---------------------------------------------------------------------------
COURT_LENGTH_FT = 94.0
COURT_WIDTH_FT = 50.0

# A small margin (feet) of slack outside the painted lines that we still count
# as "on court", because feet positions near the sideline can wobble a bit.
COURT_BOUNDS_MARGIN_FT = 3.0

# Ordered list of landmarks the user is asked to click. Each has a human label
# and its known court coordinate in feet. The user clicks the ones that are
# visible (skip the rest); we need at least 4 total.
#
# WHY this particular order: in real broadside/half-court footage the outer
# court corners are often hidden behind the scorer's table or crowd, while the
# painted lane ("the paint"/"the key") is almost always fully visible. So we
# LEAD with the four corners of the near lane plus center court -- five crisp,
# easy-to-click points that alone give a usable homography -- and only then ask
# for the harder full-court corners, which the user can skip when occluded.
#
# "near" = the sideline closest to the camera; "far" = the sideline by the crowd.
# The lane is modeled 16 ft wide (y = 17..33) and 19 ft deep (x = 0..19).
# TODO(later step): these are NBA-ish dimensions; a high-school court is ~84 ft
# long with a 12 ft lane. Good enough for approximate team stats now; revisit if
# absolute court coordinates need to be exact.
COURT_LANDMARKS = [
    # name (shown to user)                                         (court_x, court_y)
    ("LEFT paint: corner nearest the HOOP, NEAR (camera) side",    (0.0, 17.0)),
    ("LEFT paint: corner nearest the HOOP, FAR (crowd) side",      (0.0, 33.0)),
    ("LEFT paint: FREE-THROW-line corner, NEAR (camera) side",     (19.0, 17.0)),
    ("LEFT paint: FREE-THROW-line corner, FAR (crowd) side",       (19.0, 33.0)),
    ("CENTER court: middle of the center logo / circle",           (47.0, 25.0)),
    # --- everything below is optional: press 's' if hidden/off-screen ---------
    ("LEFT baseline @ NEAR court corner (skip if hidden)",         (0.0, 0.0)),
    ("LEFT baseline @ FAR court corner (skip if hidden)",          (0.0, 50.0)),
    ("CENTER line @ NEAR sideline (skip if hidden)",               (47.0, 0.0)),
    ("CENTER line @ FAR sideline (skip if hidden)",                (47.0, 50.0)),
    ("RIGHT paint: FREE-THROW-line corner, NEAR side (skip)",      (75.0, 17.0)),
    ("RIGHT paint: FREE-THROW-line corner, FAR side (skip)",       (75.0, 33.0)),
    ("RIGHT baseline @ NEAR court corner (skip if hidden)",        (94.0, 0.0)),
    ("RIGHT baseline @ FAR court corner (skip if hidden)",         (94.0, 50.0)),
]


class CourtMapper:
    """Holds a homography and maps pixel points -> court feet, with bounds test."""

    def __init__(self, homography: np.ndarray):
        # 3x3 matrix mapping pixel coords -> court coords.
        self.H = homography

    def pixel_to_court(self, px: float, py: float) -> tuple[float, float]:
        """Map a single pixel (px, py) to court coordinates in feet."""
        # Homogeneous coordinate [px, py, 1], transform, then divide by w.
        pt = np.array([px, py, 1.0], dtype=np.float64)
        mapped = self.H @ pt
        w = mapped[2] if mapped[2] != 0 else 1e-9
        return float(mapped[0] / w), float(mapped[1] / w)

    def is_on_court(self, court_x: float, court_y: float) -> bool:
        """True if a court coordinate is inside the floor (plus a small margin).

        This is the filter that removes bench/refs/crowd: their feet map to
        coordinates outside the 94x50 rectangle.
        """
        m = COURT_BOUNDS_MARGIN_FT
        return (
            -m <= court_x <= COURT_LENGTH_FT + m
            and -m <= court_y <= COURT_WIDTH_FT + m
        )


# ---------------------------------------------------------------------------
# Calibration: click landmarks once, cache them to a sidecar file.
# ---------------------------------------------------------------------------

def _calib_path_for(video_path: str) -> str:
    """Sidecar file next to the video, e.g. game.mp4 -> game.mp4.calib.json."""
    return video_path + ".calib.json"


def _grab_first_frame(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame from video: {video_path}")
    return frame


def _click_landmarks(frame: np.ndarray) -> list[dict]:
    """Open an OpenCV window and let the user click landmarks in order.

    Controls:
        left-click : record the next landmark's pixel location
        u          : undo the last click
        s          : skip the current landmark (not visible in this view)
        ENTER / q  : finish (allowed once >= 4 points are recorded)

    Returns a list of {"name", "court_xy", "pixel_xy"} dicts.
    """
    recorded: list[dict] = []
    # Index into COURT_LANDMARKS for the landmark we're currently asking for.
    idx = {"value": 0}

    window = "Click court landmarks (u=undo  s=skip  ENTER=done)"

    def redraw():
        disp = frame.copy()
        # Draw already-recorded points.
        for i, rec in enumerate(recorded):
            x, y = rec["pixel_xy"]
            cv2.circle(disp, (int(x), int(y)), 6, (0, 255, 0), -1)
            cv2.putText(disp, str(i + 1), (int(x) + 8, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # Prompt for the current landmark.
        if idx["value"] < len(COURT_LANDMARKS):
            name = COURT_LANDMARKS[idx["value"]][0]
            prompt = f"CLICK: {name}"
        else:
            prompt = "All landmarks done -- press ENTER to finish"
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(disp, prompt, (10, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(window, disp)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and idx["value"] < len(COURT_LANDMARKS):
            name, court_xy = COURT_LANDMARKS[idx["value"]]
            recorded.append({
                "name": name,
                "court_xy": list(court_xy),
                "pixel_xy": [float(x), float(y)],
            })
            idx["value"] += 1
            redraw()

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    redraw()

    print("\n--- COURT CALIBRATION ---")
    print("Click each prompted landmark on the court. Skip ones you can't see.")
    print("Keys:  u=undo  s=skip  ENTER/q=finish (need at least 4 points)\n")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, ord("q")):            # ENTER or q -> try to finish
            if len(recorded) >= 4:
                break
            print(f"Need at least 4 points (have {len(recorded)}).")
        elif key == ord("u") and recorded:   # undo
            recorded.pop()
            idx["value"] = max(0, idx["value"] - 1)
            redraw()
        elif key == ord("s") and idx["value"] < len(COURT_LANDMARKS):  # skip
            idx["value"] += 1
            redraw()

    cv2.destroyWindow(window)
    return recorded


def _homography_from_points(points: list[dict]) -> np.ndarray:
    """Least-squares homography mapping pixel -> court feet from clicked points."""
    pixel_pts = np.array([p["pixel_xy"] for p in points], dtype=np.float64)
    court_pts = np.array([p["court_xy"] for p in points], dtype=np.float64)
    # Direction matters: we want pixel -> court, so pixels are the source.
    H, _ = cv2.findHomography(pixel_pts, court_pts, method=0)
    if H is None:
        raise RuntimeError(
            "Failed to compute homography -- clicked points may be collinear. "
            "Re-run calibration and spread the points across the court."
        )
    return H


def calibrate(video_path: str, force_recalibrate: bool = False) -> CourtMapper:
    """Load cached calibration if present, otherwise prompt the user to click.

    The clicked landmark points are cached to a sidecar JSON file next to the
    video so reprocessing the same clip does not require re-clicking.
    """
    calib_file = _calib_path_for(video_path)

    if os.path.exists(calib_file) and not force_recalibrate:
        print(f"Loading saved court calibration: {calib_file}")
        with open(calib_file, "r", encoding="utf-8") as f:
            points = json.load(f)["points"]
    else:
        frame = _grab_first_frame(video_path)
        points = _click_landmarks(frame)
        with open(calib_file, "w", encoding="utf-8") as f:
            json.dump({"video": os.path.basename(video_path), "points": points},
                      f, indent=2)
        print(f"Saved court calibration to: {calib_file}")

    H = _homography_from_points(points)
    return CourtMapper(H)
