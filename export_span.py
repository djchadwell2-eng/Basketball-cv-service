"""export_span.py -- write a clip's TRACKING span to spikes/out/{clip}_span.mp4.

The vision pass (Gemini WATCHES the clip while holding the CV box score) must
watch the SAME window the box score covers -- i.e. the tracking span, not the
whole video. This saves that window as a persistent mp4 the web app hands to
Gemini. Called by analyze_clip.py at the end of a run; also runnable alone:

    .venv/Scripts/python export_span.py HARD
"""

from __future__ import annotations

import os
import shutil
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "phase2"))


def export(clip):
    import clip_config
    config = getattr(clip_config, f"{clip}_CLIP")
    clip_config.ACTIVE_CLIP = config          # bind BEFORE importing run_tracking
    import run_tracking
    tmp, fps, n = run_tracking.extract_subclip(
        config.video_path, config.tracking_span_start, config.tracking_span_len)
    out = os.path.join(_ROOT, "spikes", "out", f"{clip}_span.mp4")
    shutil.copy(tmp, out)
    print(f"[export_span] {clip}: {n} frames @ {fps:.0f}fps "
          f"(tracking span {config.tracking_span_start}..+{config.tracking_span_len}) -> {out}")
    return out


def main():
    export(sys.argv[1] if len(sys.argv) > 1 else "HARD")


if __name__ == "__main__":
    main()
