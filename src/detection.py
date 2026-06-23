"""Player detection + tracking.

This wraps the ONE validated configuration that already passed the user's
identity-stability test on real footage:

    YOLOv8m  @  imgsz=1280,  classes=[0] (people only),  tracker="bytetrack.yaml"

ByteTrack assigns a stable track_id to each person across frames. We do NOT try
to figure out *who* a track is (no jersey numbers, no roster) -- only that the
same body keeps the same id until it is honestly lost (e.g. behind an occlusion).
That is all the team-stats layer needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

# The validated config. Kept as module constants so the "magic numbers" live in
# exactly one obvious place.
MODEL_NAME = "yolov8m.pt"
IMG_SIZE = 1280
PERSON_CLASS = 0
TRACKER_CONFIG = "bytetrack.yaml"


@dataclass
class Detection:
    """One tracked person in one frame."""
    track_id: int
    # Pixel bounding box, corners (x1, y1) top-left and (x2, y2) bottom-right.
    bbox: tuple[float, float, float, float]

    def feet_pixel(self) -> tuple[float, float]:
        """Pixel location of the player's feet = bottom-center of the box.

        WHY the feet: homography maps the *ground plane*. A player's feet are on
        the floor; their head is well above it and would map to the wrong court
        spot. Bottom-center is our best single ground-contact estimate.
        """
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)


def iter_tracked_frames(video_path: str):
    """Yield (frame_index, frame_bgr, [Detection, ...]) for every video frame.

    Uses ultralytics streaming mode so we process frame-by-frame without loading
    the whole video into memory. `persist=True` tells the tracker that frames
    arrive in sequence so track_ids carry across calls.
    """
    model = YOLO(MODEL_NAME)

    results = model.track(
        source=video_path,
        classes=[PERSON_CLASS],
        imgsz=IMG_SIZE,
        tracker=TRACKER_CONFIG,
        persist=True,
        stream=True,     # generator: one result per frame
        verbose=False,
    )

    for frame_index, result in enumerate(results):
        frame_bgr = result.orig_img  # the original frame, for jersey-color sampling

        detections: list[Detection] = []
        boxes = result.boxes
        # boxes.id is None on frames where the tracker has no confirmed tracks.
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), tid in zip(xyxy, ids):
                detections.append(
                    Detection(track_id=int(tid),
                              bbox=(float(x1), float(y1), float(x2), float(y2)))
                )

        yield frame_index, frame_bgr, detections
