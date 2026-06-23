"""Team assignment by jersey color -- "team A vs team B", NOT player identity.

The only question here is: of the two jersey colors on the floor, which one is
this body wearing? We never ask *which player* -- just which of two clusters.

Approach (deliberately simple):
  1. For each detection, sample the torso patch (the jersey is the most
     color-distinctive part of a player) and record its average color.
  2. Average all samples per track_id, so each tracked body gets ONE stable
     color regardless of how many frames it appeared in.
  3. k-means (k=2) over those per-track colors -> two clusters = the two teams.

We label the clusters "team_a" and "team_b". Which physical team is A vs B is
arbitrary and stable for the whole video; team stats don't care about the label,
only that the two teams are separated.
"""

from __future__ import annotations

import cv2
import numpy as np

from .detection import Detection


def _sample_torso_color(frame_bgr: np.ndarray, bbox) -> np.ndarray | None:
    """Average HSV color of the torso region of a player box.

    We use HSV rather than BGR because Hue/Saturation separate jersey color from
    brightness, so the same jersey clusters together despite shadows/lighting.

    The torso patch is the central column of the box, upper portion -- this
    avoids the head/hair (top), the shorts/legs (bottom), and the background
    that bleeds in at the left/right edges of the box.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h = y2 - y1
    w = x2 - x1
    if h <= 0 or w <= 0:
        return None

    # Vertical: 15%..45% down the box (chest area). Horizontal: central 40%.
    ty1 = y1 + int(0.15 * h)
    ty2 = y1 + int(0.45 * h)
    tx1 = x1 + int(0.30 * w)
    tx2 = x1 + int(0.70 * w)

    # Clamp to the frame so we never index out of bounds.
    H, W = frame_bgr.shape[:2]
    ty1, ty2 = max(0, ty1), min(H, ty2)
    tx1, tx2 = max(0, tx1), min(W, tx2)
    if ty2 <= ty1 or tx2 <= tx1:
        return None

    patch = frame_bgr[ty1:ty2, tx1:tx2]
    if patch.size == 0:
        return None

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    return hsv.reshape(-1, 3).mean(axis=0)  # mean [H, S, V]


class TeamColorModel:
    """Accumulates per-track jersey colors, then clusters into two teams."""

    def __init__(self):
        # track_id -> list of sampled HSV colors across frames.
        self._samples: dict[int, list[np.ndarray]] = {}
        # Filled in by finalize(): track_id -> "team_a" / "team_b".
        self._team_of_track: dict[int, str] = {}

    def add_sample(self, frame_bgr: np.ndarray, det: Detection) -> None:
        color = _sample_torso_color(frame_bgr, det.bbox)
        if color is not None:
            self._samples.setdefault(det.track_id, []).append(color)

    def finalize(self) -> dict[int, str]:
        """Cluster per-track average colors into two teams. Returns track->team."""
        track_ids = list(self._samples.keys())
        if not track_ids:
            self._team_of_track = {}
            return self._team_of_track

        # One representative color per track.
        mean_colors = np.array(
            [np.mean(self._samples[tid], axis=0) for tid in track_ids],
            dtype=np.float32,
        )

        # Need at least 2 distinct tracks to form 2 clusters; otherwise everyone
        # is trivially "team_a".
        if len(track_ids) < 2:
            self._team_of_track = {track_ids[0]: "team_a"}
            return self._team_of_track

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, _ = cv2.kmeans(
            mean_colors, K=2, bestLabels=None, criteria=criteria,
            attempts=5, flags=cv2.KMEANS_PP_CENTERS,
        )
        labels = labels.flatten()

        self._team_of_track = {
            tid: ("team_a" if lbl == 0 else "team_b")
            for tid, lbl in zip(track_ids, labels)
        }
        return self._team_of_track

    def team_of(self, track_id: int) -> str | None:
        return self._team_of_track.get(track_id)
