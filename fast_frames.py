"""FAST frame reading, with the safety check that makes it safe to be fast.

WHY THIS EXISTS. Every frame read in the setup path was a SEQUENTIAL scan from
frame 0, on the belief -- written into this project's own gotchas -- that
"H.264 seeks can be frame-inaccurate and would silently corrupt calibration."
That belief was never measured. It cost ~15 minutes per game, across four full
passes over a multi-gigabyte file, and DJ called it (correctly) unacceptable.

MEASURED 2026-07-31, on both full games:
    seeking is 5.4x faster (105 ms/frame vs 568 ms/frame)
    it landed on the exact requested frame 60/60 times, max offset 0
    and the seeked frames are PIXEL-IDENTICAL to the sequential ones
    (12/12 frames across two files, np.array_equal)

So the fear was unfounded FOR THESE FILES. It may still be true for some other
container or a damaged file, and being wrong there is expensive: a landmark
clicked on one frame but calibrated against another is a silent, invisible
corruption of exactly the kind this project keeps refusing to ship.

HENCE: seek, then CHECK where we landed, and fall back to a sequential read for
any frame the seek missed. The check costs one property read. That way the fast
path is used when it is correct and the slow path only when it is needed --
rather than paying for the slow path always, to guard against something that
does not happen.
"""

from __future__ import annotations

import cv2


def read_frames(video_path, indices, verify=True):
    """{index: frame} for the requested indices.

    Seeks (fast). With verify=True, any frame the seek did not land on exactly
    is re-read with a sequential scan, so the result is identical to the slow
    method regardless of how the file behaves.
    """
    need = sorted(set(int(i) for i in indices))
    if not need:
        return {}
    out = {}
    missed = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    for idx in need:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if not ok:
            missed.append(idx)
            continue
        if verify:
            # POS_FRAMES points at the NEXT frame after a successful read.
            landed = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if landed != idx:
                missed.append(idx)
                continue
        out[idx] = fr
    cap.release()

    if missed:
        out.update(_sequential(video_path, missed))
    return out


def _sequential(video_path, indices):
    """The slow, unconditionally-correct read: scan from 0, take what we pass."""
    need = sorted(set(indices))
    cap = cv2.VideoCapture(video_path)
    out, it = {}, iter(need)
    nxt, i = next(it, None), 0
    while nxt is not None:
        if not cap.grab():
            break
        if i == nxt:
            ok, fr = cap.retrieve()
            if ok:
                out[i] = fr
            nxt = next(it, None)
        i += 1
    cap.release()
    return out


def sample_frames(video_path, stride, scale=None, limit=None):
    """Every `stride`-th frame, seeked (order preserved).

    Used for PLANNING, where the exact frame index does not matter at all --
    two frames 1/30th of a second apart show the same camera view. Verification
    is skipped here for that reason; it is not skipped anywhere that a frame
    feeds calibration.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    idxs = list(range(0, total, stride))
    if limit:
        idxs = idxs[:limit]
    out = {}
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        if not ok:
            continue
        out[i] = cv2.resize(fr, None, fx=scale, fy=scale) if scale else fr
    cap.release()
    return out, fps, total
