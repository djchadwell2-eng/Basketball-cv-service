"""Make the click-once court map follow a moving (panning / zooming) camera.

WHY THIS MODULE EXISTS
----------------------
court_mapping.py builds ONE homography from the points you click on the first
frame. That is only valid while the camera stays still. Real game footage is
usually shot with a single camera that pans and zooms to follow the ball, so by
a few seconds later the floor is in a different place on screen and the click-
once map points at the wrong spot.

The court is a flat plane, though, and a camera that pans/tilts/zooms sees that
plane through a *changing but predictable* homography. So instead of re-clicking
every frame, we:

  1. anchor on frame 0, where you clicked            -> H0 (frame0 px -> court)
  2. each new frame, measure how the FIXED BACKGROUND (painted lines, logos,
     bleachers) shifted versus the previous frame, using ORB feature matching
     + RANSAC. That gives a small per-step homography g (frame_t px -> frame_{t-1} px).
  3. chain those steps to know how frame_t relates back to frame 0:
        A_t = A_{t-1} @ g_t          (frame_t px -> frame0 px)
  4. the court map for frame t is then  H_t = H0 @ A_t.

Players are MASKED OUT before feature matching: they move independently of the
camera, so their features would corrupt the "how did the camera move" estimate.
RANSAC also helps reject whatever foreground motion slips through.

LIMITATIONS (TODO for a later step):
  - Error accumulates step to step ("drift") over long clips. Fine for a few
    minutes of single-camera footage; a very long game may need re-anchoring.
  - Assumes ONE continuous shot. A hard CUT to another camera angle would break
    the chain -- this footage has none, but broadcast feeds with cuts would need
    shot-boundary detection + re-calibration per shot.
"""

from __future__ import annotations

import cv2
import numpy as np


class CameraTracker:
    """Propagates a base homography across a moving single-camera shot."""

    def __init__(self, base_homography: np.ndarray):
        self.H0 = base_homography                 # frame0 pixels -> court feet
        self.A = np.eye(3, dtype=np.float64)      # frame_t pixels -> frame0 pixels

        # ORB is a fast, free corner/feature detector. 2000 features is plenty of
        # background texture to lock onto without being slow.
        self.orb = cv2.ORB_create(2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Cache the previous frame's features so we only detect once per frame.
        self.prev_kp = None
        self.prev_des = None

        # Diagnostics the caller can read after a run.
        self.frames_tracked = 0
        self.frames_motion_lost = 0  # steps where we couldn't estimate motion

    # -- public ---------------------------------------------------------------

    def update(self, frame_bgr: np.ndarray, player_boxes) -> np.ndarray:
        """Advance one frame; return the court homography H_t for THIS frame.

        player_boxes: iterable of (x1, y1, x2, y2) pixel boxes to mask out.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mask = self._background_mask(gray.shape, player_boxes)
        kp, des = self.orb.detectAndCompute(gray, mask)

        # First frame (or no features yet): the map is just the click-time H0.
        if self.prev_des is None or des is None:
            self.prev_kp, self.prev_des = kp, des
            self.frames_tracked += 1
            return self._normalize(self.H0 @ self.A)

        g = self._estimate_step(kp, des)
        if g is not None:
            self.A = self.A @ g
        else:
            # Couldn't read the camera motion this step (blur, too few features).
            # Best guess: assume the camera didn't move -> keep A unchanged.
            self.frames_motion_lost += 1

        self.prev_kp, self.prev_des = kp, des
        self.frames_tracked += 1
        return self._normalize(self.H0 @ self.A)

    # -- internals ------------------------------------------------------------

    @staticmethod
    def _background_mask(shape, player_boxes) -> np.ndarray:
        """White (255) where we ALLOW features; black (0) over players."""
        h, w = shape[:2]
        mask = np.full((h, w), 255, np.uint8)
        for (x1, y1, x2, y2) in player_boxes:
            # Pad each box a little so motion-blur halos around a player are
            # excluded too.
            pw = 0.10 * (x2 - x1)
            ph = 0.10 * (y2 - y1)
            xa = max(0, int(x1 - pw)); ya = max(0, int(y1 - ph))
            xb = min(w, int(x2 + pw)); yb = min(h, int(y2 + ph))
            mask[ya:yb, xa:xb] = 0
        return mask

    def _estimate_step(self, kp, des) -> np.ndarray | None:
        """Homography mapping CURRENT frame px -> PREVIOUS frame px, or None."""
        if des is None or self.prev_des is None:
            return None
        if len(kp) < 12 or len(self.prev_kp) < 12:
            return None

        matches = self.matcher.match(des, self.prev_des)
        if len(matches) < 12:
            return None

        cur = np.float32([kp[m.queryIdx].pt for m in matches])
        prev = np.float32([self.prev_kp[m.trainIdx].pt for m in matches])

        # RANSAC throws out matches that don't agree with the dominant motion
        # (i.e. leftover players / outliers).
        H, inliers = cv2.findHomography(cur, prev, cv2.RANSAC, 3.0)
        if H is None or inliers is None or int(inliers.sum()) < 12:
            return None
        if not self._is_sane(H):
            return None
        return H.astype(np.float64)

    @staticmethod
    def _is_sane(H: np.ndarray) -> bool:
        """Reject obviously-broken per-step fits.

        Between consecutive frames the camera barely moves, so the implied scale
        change should be near 1. A wild scale means the fit latched onto garbage;
        better to drop it and assume no motion than to corrupt the chain.
        """
        if not np.isfinite(H).all():
            return False
        scale = np.sqrt(abs(np.linalg.det(H[:2, :2])))
        return 0.7 < scale < 1.4

    @staticmethod
    def _normalize(H: np.ndarray) -> np.ndarray:
        return H / H[2, 2] if H[2, 2] != 0 else H
