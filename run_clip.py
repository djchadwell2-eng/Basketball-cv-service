"""run_clip.py -- ONE entry point that runs the whole pipeline from a ClipConfig.

    calibration  ->  tracking (READ from cache)  ->  P1 team events  ->
    P1 coach output (heatmap/zones)  ->  P2 identity  ->  box score

Built fresh (NOT on the rejected World-B process_game.py). It reads the tracks
cache produced by cache_tracks.py and never tracks inline.

Clip identity comes from TWO objects today: the downstream pipeline reads
ACTIVE_CLIP (clip_config), while the calibration + stage2's video path/label still
read spikes/clips_config.ACTIVE (see phase2/DECISIONS.md KNOWN DEBT). run_clip
SYNCHRONIZES them (clips_config.ACTIVE = config.name) and then GUARDS with a loud
assertion, so a cross-clip desync fails loud rather than extracting team_events
from the wrong video -- the same abstention-first principle as the identity layer.

Usage (no CLI config layer -- pass a ClipConfig):
    python -c "import run_clip, clip_config; run_clip.run(clip_config.HARD_CLIP)"
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (_ROOT, os.path.join(_ROOT, "spikes"),
           os.path.join(_ROOT, "phase1"), os.path.join(_ROOT, "phase2")):
    sys.path.insert(0, _p)


def _sync_and_guard(config):
    """Point BOTH clip selectors at this config, then FAIL LOUD if they disagree."""
    import clip_config
    clip_config.ACTIVE_CLIP = config                 # downstream pipeline selector
    import clips_config as cc                          # spikes: calibration selector
    cc.ACTIVE = config.name                            # sync (patch, not a root fix)
    if cc.ACTIVE != config.name:                       # guardrail: must not run desynced
        raise RuntimeError(
            f"CLIP DESYNC -- refusing to run.\n"
            f"  spikes/clips_config.ACTIVE = {cc.ACTIVE!r}\n"
            f"  ClipConfig.name            = {config.name!r}\n"
            f"team_events would extract from the WRONG video. Aborting "
            f"(abstention-first: fail loud, never run wrong).")
    print(f"[run_clip] clip selectors in sync: {config.name!r}")


def _section(title):
    print("\n" + "=" * 72 + f"\n== {title}\n" + "=" * 72)


def run(config):
    _sync_and_guard(config)

    # --- CALIBRATION (config-driven; must hold before anything downstream) ---
    _section(f"CALIBRATION -- {config.name}")
    import refit_keyframes as rk
    import stage4_courtmap as s4
    import numpy as np
    KF, ref_pos, Hs0, L0, tags, obs, corr = rk._setup()
    before = rk.keyframe_consistency(Hs0, corr)
    Hs1, L1, _res = rk._solve(KF, ref_pos, Hs0, L0, tags, obs, corr, rk.CORR_WEIGHT)
    after = rk.keyframe_consistency(Hs1, corr)
    _hc, _per, mean_ft, max_ft = s4.compute_H_court(L1, tags)
    print(f"keyframe mutual-consistency: mean "
          f"{np.mean([b[2] for b in before]):.1f} -> {np.mean([a[2] for a in after]):.1f} px")
    print(f"landmark court-fit: mean {mean_ft:.2f} ft / max {max_ft:.2f} ft")

    # --- TRACKING: read the cache, never track inline ---
    _section("TRACKING (read from cache)")
    if not os.path.exists(config.tracks_cache_path):
        raise SystemExit(
            f"No tracks cache at {config.tracks_cache_path}.\n"
            f"Run:  python -c \"import cache_tracks, clip_config; "
            f"cache_tracks.cache(clip_config.{config.name}_CLIP)\"  first.")
    import json
    doc = json.load(open(config.tracks_cache_path, encoding="utf-8"))
    fidx = [f["frame_index"] for f in doc["frames"]]
    nids = len({t["track_id"] for f in doc["frames"] for t in f["tracks"]})
    print(f"cache: {len(doc['frames'])} frames {min(fidx)}..{max(fidx)}, "
          f"{nids} track_ids  (span {doc['span_start']}..+{doc['span_len']})")

    # --- P1: team events + coach output (heatmap/zones) ---
    _section("PHASE 1 -- team events")
    import stage2_generate_events as gen
    gen.main()
    _section("PHASE 1 -- coach output (zones + heatmap)")
    import stage3_team_stats as ts
    ts.main()

    # --- P2: identity (containment, seeding+queue, box score, OCR confirm) ---
    _section("PHASE 2 -- containment (per-window)")
    import stage3_windows as win
    win.main()
    _section("PHASE 2 -- seeding + review queue")
    import stage4_seed_queue as q
    q.main()
    _section("PHASE 2 -- player_events + box score")
    import stage5_player_events as pe
    pe.main()
    _section("PHASE 2 -- OCR second signal (auto-confirm)")
    import stage6_ocr_confirm as ocr
    ocr.main()

    # --- INTEGRITY REPORT ---
    _section("INTEGRITY")
    pe_doc = json.load(open(pe.OUT_JSON, encoding="utf-8"))
    evs = pe_doc["player_events"]
    stamped = all("identity_state" in e for e in evs)
    box = pe_doc["box_score_confirmed_frames"]
    review = pe_doc["review_items"]
    print(f"player_events: {len(evs)}  every event stamped with identity_state: {stamped}")
    print(f"box score trusts CONFIRMED only: {len(box)} identities with confirmed frames")
    print(f"candidate/unknown surfaced for review (not counted): {len(review)} identities")
    print(f"\n[run_clip] DONE -- {config.name} ran end-to-end.")


if __name__ == "__main__":
    import clip_config
    run(clip_config.ACTIVE_CLIP)
