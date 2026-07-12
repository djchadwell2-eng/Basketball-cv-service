"""cache_tracks.py -- run ByteTrack ONCE over a clip's tracking span, write the cache.

The combined pipeline (run_clip.py) READS this cache and never tracks inline, so the
slow CPU detection happens here, once. We cache ONLY the config's tracking span (a
~120-frame window inside the calibrated pan), not the whole clip -- that keeps it far
under the full-clip ~90 min.

Usage (no CLI config layer -- pass a ClipConfig object):
    python -c "import cache_tracks, clip_config; cache_tracks.cache(clip_config.HARD_CLIP)"
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "phase2"))


def cache(config):
    """Track the config's span and write tracks to config.tracks_cache_path."""
    config.validate()                         # malformed config: refuse before tracking
    import clip_config
    clip_config.ACTIVE_CLIP = config          # set BEFORE importing run_tracking (binds at import)
    import run_tracking                        # reads ACTIVE_CLIP at import -> this config
    print(f"[cache_tracks] {config.name}: span {config.tracking_span_start}.."
          f"{config.tracking_span_start + config.tracking_span_len} "
          f"-> {config.tracks_cache_path}")
    run_tracking.main()


if __name__ == "__main__":
    import clip_config
    cache(clip_config.ACTIVE_CLIP)
