"""cache_oncourt.py -- classify each cached track ON/OFF court, once per clip.

Companion to cache_tracks.py: run AFTER the tracks cache exists. Uses the
validated calibration to map every cached track's feet to court feet and writes
phase2/out/{clip}_oncourt.json, which the identity stages read to seed (and
OCR, and queue) ONLY on-court bodies -- the ROI-mask half of the fair remeasure.

Usage (no CLI config layer -- pass a ClipConfig object):
    python -c "import cache_oncourt, clip_config; cache_oncourt.cache(clip_config.HARD_CLIP)"
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (_ROOT, os.path.join(_ROOT, "spikes"),
           os.path.join(_ROOT, "phase1"), os.path.join(_ROOT, "phase2")):
    sys.path.insert(0, _p)


def cache(config):
    """Classify the config's cached tracks; write {clip}_oncourt.json."""
    config.validate()                         # malformed config: refuse before classifying
    import clip_config
    clip_config.ACTIVE_CLIP = config          # set BEFORE imports (bind at import)
    import clips_config as cc
    cc.ACTIVE = config.name                    # builder CALIBRATES -> sync BOTH configs
    import oncourt                             # binds ACTIVE_CLIP at import -> this config
    print(f"[cache_oncourt] {config.name}: tracks {config.tracks_cache_path}")
    oncourt.build()


if __name__ == "__main__":
    import clip_config
    cache(clip_config.ACTIVE_CLIP)
