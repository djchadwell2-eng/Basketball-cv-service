"""analyze_clip.py -- the ONE command the web app invokes to make CV analyze a
clip (Phase 7 L1: "teach the app to run CV"). Runs the full pipeline
(run_clip) for a SET-UP clip, then bundles the web-facing measured-stats
contract (measured_stats). One entry point so the app just shells out to:

    .venv/Scripts/python analyze_clip.py HARD

Prints STAGE markers to stdout so the app can show progress. Exits non-zero
on failure. The clip MUST already be a configured ClipConfig with its caches
(a brand-new upload needs setup first -- Phase 7 L4, the browser calibration).
This deliberately reuses run_clip verbatim (CV owns the numbers); it adds no
new analysis, only the run+bundle orchestration.
"""

from __future__ import annotations

import os
import sys
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)


def main():
    clip = sys.argv[1] if len(sys.argv) > 1 else "HARD"
    t0 = time.time()
    print(f"[analyze_clip] STAGE start clip={clip}", flush=True)

    import clip_config
    config = getattr(clip_config, f"{clip}_CLIP", None)
    if config is None:
        print(f"[analyze_clip] FAILED: no ClipConfig named {clip}_CLIP -- the clip "
              f"must be set up first (Phase 7 L4).", flush=True)
        raise SystemExit(2)

    print("[analyze_clip] STAGE run_clip (calibration -> tracking -> team events "
          "-> identity -> box score -> ball/shot layer) ...", flush=True)
    import run_clip
    run_clip.run(config)

    print("[analyze_clip] STAGE bundling measured-stats web contract ...", flush=True)
    import measured_stats
    out = measured_stats.generate(clip)

    print("[analyze_clip] STAGE exporting tracking-span video (for the vision pass) ...", flush=True)
    import export_span
    export_span.export(clip)

    dt = time.time() - t0
    print(f"[analyze_clip] STAGE done clip={clip} in {dt:.0f}s "
          f"({len(out['box_score'])} players, {len(out['shots'])} shot(s) located)",
          flush=True)


if __name__ == "__main__":
    main()
