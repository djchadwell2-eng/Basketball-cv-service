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
    # __slots__ because there are MILLIONS of these: a 95-minute game holds ~34
    # bodies in each of 171,120 frames, and every identity stage builds one
    # object per body per frame. Without slots each carries its own attribute
    # dictionary, which costs more than the two values it exists to hold. No
    # field has a default, so declaring the slots by hand is safe here (a
    # default would collide with the slot descriptor) and works on any Python.
    # Nothing anywhere attaches extra attributes to a Track -- checked before
    # this was added, and slots will now say so loudly if anything tries.
    __slots__ = ("track_id", "bbox")
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
    yield from _emit(results, max_frames)


def iter_tracks_over(frames, tracker_config: str = TRACKER_CONFIG,
                     model_name: str | None = None):
    """Same tracking, over frames HANDED IN rather than a file path.

    WHY THIS EXISTS. The caller used to write the span to a temp mp4 first and
    track that file: decode the film, re-encode a copy, decode the copy. The
    encode alone measured 29.1 ms/frame -- roughly a third of a slice's cost --
    to produce something the detector then had to read back.

    And the copy is not the film. MEASURED on DJ's game: mp4v round-tripped
    frames differ from the source by a mean of 3.0 grey levels and up to 76, and
    running the detector on both, the COPY finds 1-3 FEWER people per frame with
    boxes shifted up to 39 px. So the temp file was costing money to make the
    detector's job harder.

    persist=True keeps the SAME ByteTrack state machine across calls, so the
    tracker behaves as it did; what changes is that it now sees the real pixels.
    """
    model = YOLO(model_name or MODEL_NAME)

    def _results():
        for img in frames:
            yield model.track(img, classes=[PERSON_CLASS], imgsz=IMG_SIZE,
                              tracker=tracker_config, persist=True,
                              verbose=False)[0]
    yield from _emit(_results())


def _emit(results, max_frames=None):
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
