"""A reused worker must not re-track the PREVIOUS job's span.

THE BUG THIS LOCKS DOWN. phase2/run_tracking.py and phase2/oncourt.py bind
CLIP / SPAN_START / SPAN_LEN / OUT_JSON at MODULE IMPORT. A serverless worker
is reused between jobs, so the second job's `import` is a no-op and the module
keeps the first job's span. MEASURED 2026-09-10 on a real worker: a job asking
for frames 27,800..28,700 re-tracked 77,300..78,200 and wrote that header
instead. It was caught only by serverless_handler._cache_covers refusing to
publish it -- so the job FAILED rather than corrupting the game, but it failed
for a reason that had nothing to do with the film.

Why this is the normal case and not an edge case: run_chunked.run() sends TEN
spans of ONE clip to (at most) three workers -- so the shape tested here, the
same clip and a different span inside one process, IS production.

NOTHING REAL IS TOUCHED. Both tests write only into pytest's tmp_path, and
the detector is stubbed. An earlier draft of this file patched
run_tracking.main directly -- importlib.reload rebuilds the module namespace
and silently DISCARDED that patch, so the real tracker ran and overwrote
phase2/out/TEST1_tracks_raw.json (restored from git). The seams used here are
in modules that are NOT reloaded (cv2, phase2.tracking), so the stubs survive.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "phase2"), os.path.join(_ROOT, "phase1"),
           os.path.join(_ROOT, "spikes")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

N_FAKE_FRAMES = 4


class _FakeCap:
    """Enough cv2.VideoCapture for run_tracking.main; reads no film."""
    def get(self, _prop):
        return 30.0

    def set(self, _prop, _val):
        return True

    def read(self):
        return False, None

    def grab(self):
        return True

    def release(self):
        return None


@pytest.fixture
def stub_tracker(monkeypatch):
    """Stub the two heavy dependencies, in modules reload does not rebuild."""
    import cv2
    import tracking
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_a, **_k: _FakeCap())
    monkeypatch.setattr(
        tracking, "iter_tracks_over",
        lambda *_a, **_k: ((i, None, []) for i in range(N_FAKE_FRAMES)))


def _slice_of(config, start, length, path):
    """The same clip, a different slice, writing somewhere disposable."""
    return dataclasses.replace(config, tracking_span_start=start,
                               tracking_span_len=length,
                               tracks_cache_path=str(path))


def test_run_tracking_rebinds_span_on_a_reused_worker(tmp_path, stub_tracker):
    """Two jobs, one process, same clip, DIFFERENT spans."""
    import clip_config
    import cache_tracks

    base = clip_config.get_clip("TEST1")
    first = _slice_of(base, 300, 150, tmp_path / "first.json")
    second = _slice_of(base, 600, 180, tmp_path / "second.json")

    cache_tracks.cache(first)
    cache_tracks.cache(second)          # the warm-worker second job

    head_a = json.load(open(first.tracks_cache_path, encoding="utf-8"))
    head_b = json.load(open(second.tracks_cache_path, encoding="utf-8"))

    assert head_a["span_start"] == 300
    assert head_b["span_start"] == 600, (
        f"warm worker wrote span_start={head_b['span_start']} for a job that "
        f"asked for 600 -- it re-tracked the previous job's frames")


def test_oncourt_rebinds_span_on_a_reused_worker(tmp_path):
    """Same trap in the on-court builder, which binds CLIP at import.

    No stubbing needed: oncourt.build's own stale-cache guard fires first, and
    the span it reports as WANTED is exactly the binding under test."""
    import clip_config
    import cache_oncourt

    base = clip_config.get_clip("TEST1")
    real_tracks = base.tracks_cache_path        # read-only here; span 120..461
    first = _slice_of(base, 300, 150, real_tracks)
    second = _slice_of(base, 600, 180, real_tracks)

    with pytest.raises(SystemExit) as e1:
        cache_oncourt.cache(first)
    with pytest.raises(SystemExit) as e2:       # the warm-worker second job
        cache_oncourt.cache(second)

    assert "300, 150" in str(e1.value)
    assert "600, 180" in str(e2.value), (
        f"warm worker still wanted the previous job's span: {e2.value}")
