"""Phase 2 tracking -- YOLOv8m@1280 + ByteTrack (the validated config).

Ported from the proven spike (src/detection.py): person-only detection at
imgsz=1280 with ByteTrack assigning a stable track_id to each body across frames.
ByteTrack gives us CONTINUITY (the same body keeps the same id until it is honestly
lost); it does NOT tell us WHO a body is. Identity -- and the abstention-first
state machine that guards it -- is built on top of this in identity.py.

This layer is deliberately dumb: detect + track, emit (track_id, bbox) per frame.
No identity, no court mapping, no state. Those live in higher layers so the spine
stays simple and the safety-critical logic is isolated.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ultralytics import YOLO

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_NAME = os.path.join(_ROOT, "yolov8m.pt")
IMG_SIZE = 1280
PERSON_CLASS = 0
TRACKER_CONFIG = "bytetrack.yaml"


@dataclass
class Track:
    """One tracked body in one frame. Continuity only -- NOT an identity."""
    track_id: int
    bbox: tuple[float, float, float, float]     # (x1, y1, x2, y2) pixels

    def feet_pixel(self) -> tuple[float, float]:
        x1, _y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)             # ground contact = bottom-center


def iter_tracks(video_path: str, max_frames: int | None = None,
                tracker_config: str = TRACKER_CONFIG,
                model_name: str | None = None):
    """Yield (frame_index, frame_bgr, [Track, ...]) for each frame, in order.

    tracker_config defaults to the validated ByteTrack config; experiments
    (e.g. spikes/reid_fragment_probe.py) may pass an alternative yaml.
    model_name defaults to the validated detector (MODEL_NAME); a probe may
    pass a bigger model (e.g. yolov8x.pt, spikes/player_detector_probe.py) to
    MEASURE whether detector capacity reduces track fragmentation -- the real
    cache is never rebuilt from a probe, only compared against.
    """
    model = YOLO(model_name or MODEL_NAME)
    results = model.track(
        source=video_path, classes=[PERSON_CLASS], imgsz=IMG_SIZE,
        tracker=tracker_config, persist=True, stream=True, verbose=False,
    )
    for frame_index, result in enumerate(results):
        tracks: list[Track] = []
        boxes = result.boxes
        # boxes.id is None on frames where the tracker has no confirmed tracks.
        if boxes is not None and boxes.id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            ids = boxes.id.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), tid in zip(xyxy, ids):
                tracks.append(Track(int(tid),
                                    (float(x1), float(y1), float(x2), float(y2))))
        yield frame_index, result.orig_img, tracks
        if max_frames is not None and frame_index + 1 >= max_frames:
            break
