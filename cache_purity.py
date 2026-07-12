"""cache_purity.py -- OCR-sweep each labelable track's lifespan, once per clip.

Companion to cache_tracks.py / cache_oncourt.py: run AFTER both exist. Writes
phase2/out/{clip}_purity.json -- per-track verdicts (consistent / spliced /
no_evidence). SPLICED tracks (two different confident numbers on one track_id
= the tracker jumped players) are quarantined from labeling and their seed
labels are refused.

Usage (no CLI config layer -- pass a ClipConfig object):
    python -c "import cache_purity, clip_config; cache_purity.cache(clip_config.HARD_CLIP)"
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (_ROOT, os.path.join(_ROOT, "spikes"),
           os.path.join(_ROOT, "phase1"), os.path.join(_ROOT, "phase2")):
    sys.path.insert(0, _p)


def cache(config):
    """OCR-sweep the config's labelable tracks; write {clip}_purity.json."""
    config.validate()
    import clip_config
    clip_config.ACTIVE_CLIP = config          # set BEFORE imports (bind at import)
    import clips_config as cc
    cc.ACTIVE = config.name
    import purity
    print(f"[cache_purity] {config.name}")
    purity.build()


if __name__ == "__main__":
    import clip_config
    cache(clip_config.ACTIVE_CLIP)
