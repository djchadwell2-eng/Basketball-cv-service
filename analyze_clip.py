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

    # get_clip, NOT getattr(clip_config, f"{clip}_CLIP"): the attribute lookup
    # only ever finds the hand-written baselines (TEST1/HARD/...), so a game set
    # up in the browser -- which lives in clips/<NAME>.json -- could never be
    # found and every web upload died here. get_clip checks both sources.
    import clip_config
    config = clip_config.get_clip(clip)
    if config is None:
        import clip_registry
        doc = clip_registry.load(clip)
        if doc is None:
            print(f"[analyze_clip] FAILED: no clip named {clip} -- not a built-in "
                  f"and no clips/{clip}.json.", flush=True)
        else:
            # The clip EXISTS but is only half a config: the browser flow fills
            # in the calibration half, and nothing has chosen what stretch of
            # the game to analyse yet.
            missing = [f for f in ("tracking_span_start", "tracking_span_len")
                       if doc.get(f) is None]
            print(f"[analyze_clip] FAILED: {clip} is set up but has no analysis span "
                  f"yet (missing: {', '.join(missing) or 'roster'}).", flush=True)
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
