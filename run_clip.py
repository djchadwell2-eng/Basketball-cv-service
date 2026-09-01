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


def _load_and_check_cache(config):
    """Load the tracks cache and REFUSE if it doesn't describe this config's
    clip + span. A stale/wrong cache means stages 4/6 would crop OCR evidence
    from frames the tracker never saw -- silently wrong output. Runs BEFORE the
    slow calibration solve so a bad cache fails in a second, not after minutes."""
    import json
    if not os.path.exists(config.tracks_cache_path):
        raise SystemExit(
            f"No tracks cache at {config.tracks_cache_path}.\n"
            f"Run:  python -c \"import cache_tracks, clip_config; "
            f"cache_tracks.cache(clip_config.{config.name}_CLIP)\"  first.")
    doc = json.load(open(config.tracks_cache_path, encoding="utf-8"))
    got = (doc.get("clip"), doc.get("span_start"), doc.get("span_len"))
    want = (config.name, config.tracking_span_start, config.tracking_span_len)
    if got != want or not doc["frames"]:
        hint = ("(0 frames = the extract failed at cache time; bad video path?)\n"
                if not doc["frames"] else "")
        raise SystemExit(
            f"STALE/WRONG TRACKS CACHE -- refusing to run.\n"
            f"  cache file:   {config.tracks_cache_path}\n"
            f"  cache says:   clip={got[0]!r} span={got[1]}..+{got[2]}  "
            f"frames={len(doc['frames'])}\n"
            f"  config wants: clip={want[0]!r} span={want[1]}..+{want[2]}\n"
            f"{hint}"
            f"Re-run:  python -c \"import cache_tracks, clip_config; "
            f"cache_tracks.cache(clip_config.{config.name}_CLIP)\"")
    return doc


def run(config):
    config.validate()                        # malformed config: refuse before any work
    _sync_and_guard(config)
    doc = _load_and_check_cache(config)      # fail loud BEFORE calibrating
    import oncourt
    oncourt.load_checked(config)             # on-court cache too (ROI-mask seeding)

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
    import json
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
    _section("PHASE 2 -- retroactive stat merge (agree-triggered only)")
    import stage7_merge as merge
    merge.main()
    _section("PHASE 2 -- jersey-keyed box score")
    import stage8_box_score as box
    box.main()

    # --- P5: ball/shot layer (sits BESIDE the spine -- never touches
    #     team_events or any Phase 1/2 artifact, ROADMAP Principle 4) ---
    sa_json = None
    if config.ball_span_len:
        import ball_stages as bs
        _section("PHASE 5 -- ball detection (own fine-tuned weights)")
        det_json = bs.stage_ball_detect(config)
        # ARCS BEFORE THE RIM, deliberately. The arcs are built from ball
        # detections alone and never look at the hoop; the hoop is consulted
        # only inside shot_attempts, per arc segment. Carrying the rim through
        # every frame first meant computing ~165,000 rim positions no shot would
        # ever ask about -- 19.6 hours at a MEASURED 0.412 s/frame, which is the
        # single reason a full game could not have shots. Same rim, same shots.
        _section("PHASE 5 -- ball trajectory (physics-gated arcs)")
        arcs_json = bs.stage_ball_trajectory(config, det_json)
        _section("PHASE 5 -- hoop anchor carry (only where an arc is)")
        hoop_json = bs.stage_hoop_anchor(config, only_frames=bs.arc_frames(arcs_json))
        _section("PHASE 5 -- shot attempts")
        sa_json = bs.stage_shot_attempts(config, arcs_json, hoop_json, det_json)
        _section("PHASE 5 -- shot locations + chart")
        loc_json = bs.stage_shot_location(config, sa_json)
        _section("PHASE 5 -- shot outcomes (candidate labels, review-first)")
        outc_json = bs.stage_shot_outcome(config, sa_json, arcs_json, hoop_json, det_json)
        # Prints its own full summary; nothing downstream consumes it yet, so
        # the path is deliberately not captured.
        _section("PHASE 5 -- ball touches (who has the ball; NOT possessions)")
        touches_json = bs.stage_ball_touches(config, det_json)
        _section("PHASE 5 -- team possessions (whose ball, from when to when)")
        bs.stage_team_possessions(config, touches_json)
    else:
        _section("PHASE 5 -- ball/shot layer")
        print(f"[run_clip] ball layer not configured for {config.name} "
              f"(ball_span_len=0) -- skipped, not silently faked.")

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
    if os.path.exists(ocr.OUT_JSON):
        oc = json.load(open(ocr.OUT_JSON, encoding="utf-8"))
        print(f"OCR second signal: {len(oc['outcomes']['agree'])} auto-confirmed, "
              f"{len(oc['outcomes']['disagree'])} disagreement flag(s)")
    if os.path.exists(merge.OUT_JSON):
        md = json.load(open(merge.OUT_JSON, encoding="utf-8"))
        sc = md["state_counts"]
        print(f"retroactive merge: {len(md['merges'])} span(s) re-credited, "
              f"{len(md['contradictions'])} contradiction flag(s)")
        print(f"final event states: confirmed={sc.get('confirmed', 0)} "
              f"retro={sc.get('confirmed_retroactive', 0)} "
              f"candidate={sc.get('candidate', 0)} unknown={sc.get('unknown', 0)}  "
              f"(canonical artifact: {os.path.basename(merge.OUT_JSON)})")
    if sa_json is not None:
        sa = json.load(open(sa_json, encoding="utf-8"))
        loc = json.load(open(loc_json, encoding="utf-8"))
        outc = json.load(open(outc_json, encoding="utf-8"))
        n_att = sum(1 for r in sa["attempts"] if r["verdict"] == "shot_attempt")
        n_loc = sum(1 for r in loc["locations"] if r["status"] == "located")
        oc = {}
        for r in outc["outcomes"]:
            oc[r["outcome"]] = oc.get(r["outcome"], 0) + 1
        print(f"shot layer: {n_att} attempt(s), {n_loc} located, outcomes {oc} "
              f"(candidate labels feeding review, never bare stats)")
    print(f"\n[run_clip] DONE -- {config.name} ran end-to-end.")


if __name__ == "__main__":
    import clip_config
    run(clip_config.ACTIVE_CLIP)
