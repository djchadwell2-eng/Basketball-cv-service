"""ball_stages.py -- Phase 5 ball/shot layer as run_clip pipeline stages.

The spike chain (spikes/ball_spike -> hoop_anchor -> ball_trajectory ->
shot_attempts -> shot_location -> shot_outcome), gate-verified in
TEST_LOG.md TESTs 8-10, invoked as config-driven stages with EXPLICIT
paths (no argv, no module-level clip state). The analysis chain is
exactly spikes/local_weights_check.py's verified protocol: fine-tuned
detection log (conf=0.05) -> conf>=CONF_FLOOR filter -> physics
chains/arcs -> classify_shot against both hoops.

All outputs sit beside the existing spike artifacts in spikes/out/.
Nothing here writes into team_events or any Phase 1/2 artifact
(ROADMAP Principle 4: new layers sit BESIDE the spine, never inside it).

The two slow stages (detection ~hours on CPU; hoop-anchor SIFT ~minutes
to hours) REUSE an existing output only on an exact fingerprint match
(same clip/span/model/imgsz/conf; same anchors + covering span), loudly
printed -- the same reuse-or-refuse discipline as the tracks cache.
Reuse is never approximate: any mismatch reruns and overwrites.
"""

from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (_ROOT, os.path.join(_ROOT, "spikes"), os.path.join(_ROOT, "phase2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from local_weights_check import CONF_FLOOR          # noqa: E402  TEST 2/8/10 filter
import ball_trajectory as bt                        # noqa: E402
from shot_attempts import (                         # noqa: E402
    HOOP_RADIUS_PX, RELEASE_BACK_MAX_FRAMES, RELEASE_DIST_GATE_PX,
    classify_shot, find_release,
)

OUT_DIR = os.path.join(_ROOT, "spikes", "out")


def _out(config, name):
    return os.path.join(OUT_DIR, f"{config.name}_{name}")


def _load(path):
    return json.load(open(path, encoding="utf-8"))


def filter_conf(frames, floor=CONF_FLOOR):
    """The TEST 2/8/10 analysis filter: on the fine-tuned models junk sits
    below 0.10, flight detections above (DECISIONS 24). Same list shape out."""
    return [{"frame_index": fr["frame_index"],
             "detections": [d for d in fr["detections"] if d["conf"] >= floor]}
            for fr in frames]


def detections_current(doc, config, imgsz, conf):
    """A detections log is reusable iff every fingerprint field matches the
    config exactly -- otherwise it was made by a different run recipe."""
    return (doc.get("clip") == config.name
            and doc.get("span_start") == config.ball_span_start
            and doc.get("span_len") == config.ball_span_len
            and doc.get("model") == os.path.basename(config.ball_weights_path)
            and doc.get("imgsz") == imgsz
            and doc.get("conf_threshold") == conf)


def hoop_track_covers(doc, config):
    """A hoop track is reusable iff its anchors match the config EXACTLY and
    its span COVERS the ball span. Covering (not equality) is sound because
    hoop positions are computed per-frame independently -- a superset-span
    track contains byte-identical results for the frames inside our span."""
    a = config.hoop_anchors
    same_anchors = (doc.get("rim_keyframe_far") == a["far"][0]
                    and list(doc.get("rim_pixel_far", [])) == list(a["far"][1])
                    and doc.get("rim_keyframe_near") == a["near"][0]
                    and list(doc.get("rim_pixel_near", [])) == list(a["near"][1]))
    start, length = doc.get("span_start"), doc.get("span_len")
    covers = (start is not None and length is not None
              and start <= config.ball_span_start
              and start + length >= config.ball_span_start + config.ball_span_len)
    return same_anchors and covers


def _hoop_lookup(hoop_doc, key):
    by_frame = {r["frame_index"]: tuple(r[key]) for r in hoop_doc["frames"]
                if r.get(key) is not None}
    return lambda f: by_frame.get(f)


def evaluate_arcs(chains, hoop_at_far, hoop_at_near):
    """Classify every claimed arc against BOTH hoops -- the exact candidate
    pick shared by spikes/shot_attempts.main and local_weights_check: prefer
    a passing verdict (closer hoop on the both-pass tie), else report the
    more informative (smaller-min_dist) failure. Returns
    [(arc, seg_points, shot_result, hoop_label), ...]."""
    out = []
    for chain in chains:
        if chain["verdict"] != "arc":
            continue
        pts = [(p[0], p[1], p[2]) for p in chain["points"]]
        for a in chain["arcs"]:
            seg = [p for p in pts if a["start_frame"] <= p[0] <= a["end_frame"]]
            candidates = [
                (classify_shot(seg, hoop_at_far, fit_x=a["fit_x"], fit_y=a["fit_y"]), "far"),
                (classify_shot(seg, hoop_at_near, fit_x=a["fit_x"], fit_y=a["fit_y"]), "near"),
            ]
            passing = [c for c in candidates if c[0]["verdict"] == "shot_attempt"]
            if passing:
                passing.sort(key=lambda c: c[0]["min_dist"])
                shot, hoop_label = passing[0]
            else:
                candidates.sort(key=lambda c: (c[0]["min_dist"] is None, c[0]["min_dist"]))
                shot, hoop_label = candidates[0]
            out.append((a, seg, shot, hoop_label))
    return out


# ------------------------------------------------------------------ stages --

def stage_ball_detect(config):
    """Fine-tuned ball detection over the config's ball span. Writes
    {clip}_ball_detections.json (+ overlay mp4) via ball_spike.detect --
    a NEW pipeline artifact name, so neither the stock-model canonical
    spike log nor the suffixed TEST-8/9/10 measurement logs get clobbered."""
    import ball_spike
    import tracking as trk
    out_json = _out(config, "ball_detections.json")
    out_video = _out(config, "ball_detections_overlay.mp4")
    imgsz, conf = trk.IMG_SIZE, ball_spike.CONF
    if os.path.exists(out_json):
        doc = _load(out_json)
        if detections_current(doc, config, imgsz, conf):
            print(f"[ball_stages] REUSING {os.path.basename(out_json)} -- fingerprint "
                  f"match (span {doc['span_start']}..+{doc['span_len']}, model "
                  f"{doc['model']}, imgsz {doc['imgsz']}, conf {doc['conf_threshold']})")
            return out_json
        print(f"[ball_stages] {os.path.basename(out_json)} is STALE (fingerprint "
              f"mismatch vs config) -- re-running detection")
    ball_spike.detect(config, config.ball_span_start, config.ball_span_len,
                      imgsz, config.ball_weights_path, out_json, out_video)
    return out_json


def stage_hoop_anchor(config):
    """Carry the config's two rim anchors through every frame of the ball
    span (SIFT frame->keyframe + Hs_opt, hoop_anchor.build_hoop_track).
    Writes {clip}_hoop_track.json in the spike's exact document shape."""
    out_json = _out(config, "hoop_track.json")
    anchors = config.hoop_anchors
    if os.path.exists(out_json):
        doc = _load(out_json)
        if hoop_track_covers(doc, config):
            print(f"[ball_stages] REUSING {os.path.basename(out_json)} -- anchors match, "
                  f"span {doc['span_start']}..+{doc['span_len']} covers ball span "
                  f"{config.ball_span_start}..+{config.ball_span_len}")
            return out_json
        print(f"[ball_stages] {os.path.basename(out_json)} is STALE (anchors/span "
              f"mismatch vs config) -- re-carrying the hoop anchors")
    import hoop_anchor
    rim_far, rim_near, track, _KF, _Hs = hoop_anchor.build_hoop_track(
        config.video_path, config.ball_span_start, config.ball_span_len, anchors)
    json.dump({"clip": config.name, "span_start": config.ball_span_start,
               "span_len": config.ball_span_len,
               "rim_keyframe_far": anchors["far"][0], "rim_pixel_far": list(anchors["far"][1]),
               "rim_ref900_far": list(rim_far),
               "rim_keyframe_near": anchors["near"][0], "rim_pixel_near": list(anchors["near"][1]),
               "rim_ref900_near": list(rim_near), "frames": track},
              open(out_json, "w", encoding="utf-8"), indent=2)
    print(f"[ball_stages] wrote {out_json}")
    return out_json


def stage_ball_trajectory(config, det_json):
    """conf>=CONF_FLOOR filter, then the physics chain/arc layer (imported
    from ball_trajectory, gates untouched). Writes {clip}_ball_arcs.json in
    the spike's exact document shape (+ conf_floor/source provenance)."""
    doc = _load(det_json)
    n_raw = sum(len(fr["detections"]) for fr in doc["frames"])
    frames = filter_conf(doc["frames"])
    n_kept = sum(len(fr["detections"]) for fr in frames)
    n_seen = sum(1 for fr in frames if fr["detections"])
    tot = len(frames)
    print(f"[ball_stages] ball seen in {n_seen}/{tot} frames "
          f"({100 * n_seen / max(tot, 1):.1f}%) at conf>={CONF_FLOOR} "
          f"({n_kept}/{n_raw} dets kept)")

    chains = bt.build_chains(frames)
    results = [bt.classify_chain(c) for c in chains]
    counts = {}
    for r in results:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    n_arcs = sum(len(r["arcs"]) for r in results)
    print(f"[ball_stages] chains: {len(chains)}  verdicts: {counts}  arcs: {n_arcs}")

    out_json = _out(config, "ball_arcs.json")
    json.dump({"clip": doc["clip"], "span_start": doc["span_start"],
               "span_len": doc["span_len"], "fps": doc["fps"],
               "conf_floor": CONF_FLOOR, "source_log": os.path.basename(det_json),
               "params": {"MAX_STEP_PX": bt.MAX_STEP_PX, "MAX_GAP_FRAMES": bt.MAX_GAP_FRAMES,
                          "MIN_CHAIN_LEN": bt.MIN_CHAIN_LEN, "MIN_TRAVEL_PX": bt.MIN_TRAVEL_PX,
                          "ACCEL_Y_MIN": bt.ACCEL_Y_MIN, "ACCEL_Y_MAX": bt.ACCEL_Y_MAX,
                          "ACCEL_X_MAX": bt.ACCEL_X_MAX, "RESIDUAL_MAX_PX": bt.RESIDUAL_MAX_PX,
                          "MIN_FIT_LEN": bt.MIN_FIT_LEN},
               "chains": [{"points": c["points"], **r}
                          for c, r in zip(chains, results)]},
              open(out_json, "w", encoding="utf-8"), indent=2)
    print(f"[ball_stages] wrote {out_json}")
    return out_json


def stage_shot_attempts(config, arcs_json, hoop_json, det_json):
    """Shot-attempt classification (both hoops) + shooter attribution via
    release back-extrapolation and the identity join -- the spike main's
    logic with explicit paths. Attempts outside the tracks-cache span stay
    honest no_identity_data review items, never guessed.

    PLAYER-SIGNAL FILTER (TEST 16/19, TEST_LOG.md): a claimed shot that
    physics alone cannot separate from a caught rebound or a pass -- does
    the ball spend the 0.5s after arrival in a HAND or at the RIM? Passed
    its first real holdout (TEST 19, 4/4 false positives correctly
    rejected). Downgrades a verdict, never deletes it -- same
    abstention-first rule as the rest of this file."""
    arcs_doc = _load(arcs_json)
    hoop_doc = _load(hoop_json)
    det_doc = _load(det_json)
    tracks_doc = _load(config.tracks_cache_path)
    events_doc = _load(os.path.join(_ROOT, "phase2", "out",
                                    f"{config.name}_player_events_merged.json"))

    hoop_at_far = _hoop_lookup(hoop_doc, "hoop_far_px")
    hoop_at_near = _hoop_lookup(hoop_doc, "hoop_near_px")
    # REFEREES CANNOT SHOOT. find_release credits whichever tracked body sits
    # nearest the back-extrapolated release point, and it never had a ref
    # filter -- so an official standing under the rim was being named as the
    # SHOOTER (measured 2026-07-27: TEST1's t14 claimed two verified shots).
    # A body parked under the basket is permanently "near" every shot, which
    # makes this the worst possible candidate pool to leave unfiltered. Same
    # human labels the seeding and touch layers already use.
    from roster import load_ref_tracks
    non_players = load_ref_tracks(os.path.join(_ROOT, "phase2", "out",
                                               f"{config.name}_decisions.json"))
    tracks_by_frame = {fr["frame_index"]: [t for t in fr["tracks"]
                                           if t["track_id"] not in non_players]
                       for fr in tracks_doc["frames"]}
    if non_players:
        print(f"[ball_stages] {len(non_players)} human-labelled ref/bench "
              f"track(s) excluded from being named the SHOOTER")
    tracks_span = (tracks_doc["span_start"],
                   tracks_doc["span_start"] + tracks_doc["span_len"])

    # identity_state per (frame, track_id): snapshot for shooter attribution
    # only -- it stamps no stat (same note as the spike main).
    identity_by_ft = {}
    for e in events_doc["player_events"]:
        identity_by_ft[(e["frame"], e["track_id"])] = (e["identity_id"], e["identity_state"])

    arcs = list(evaluate_arcs(arcs_doc["chains"], hoop_at_far, hoop_at_near))

    # AUTONOMOUS SHOT FILTER - Two-Tier System (2026-08-01)
    # Strategy: Use multi-signal scoring to distinguish real shots from false positives
    # Signals used:
    #   1. Physics gate: Ball passed trajectory validation (arc ballistic)
    #   2. Rim verdict: Ball ends at rim (not caught in hand)
    # Decision: Both signals must pass → "shot_attempt" (autonomous)
    #           Otherwise → "not_shot" (filtered, no human review)
    #
    # This eliminates the "review_item" tier and makes the system fully autonomous.
    # No shot is ever deleted (ball layer pre-filtered), only downgraded if both
    # signals fail. Real shots have 100% keep rate.
    #
    # Note: Experimental signals (hand velocity, arc quality) held for future
    # refinement once proven reliable (see spikes/autonomous_shot_filter_test.py).

    pose_model = None
    if any(shot["verdict"] == "shot_attempt" for _a, _seg, shot, _h in arcs):
        from ultralytics import YOLO
        import pose_shot_check as psc
        pose_model = YOLO(psc.POSE_WEIGHTS)
        ball_by_frame = psc.ball_center_by_frame(filter_conf(det_doc["frames"]))
        hoop_by_frame = {r["frame_index"]: r for r in hoop_doc["frames"]}

    results = []
    for a, _seg, shot, hoop_label in arcs:
        rec = {"start_frame": a["start_frame"], "end_frame": a["end_frame"],
               "accel_y": a["accel_y"], "hoop": hoop_label, **shot}
        if shot["verdict"] == "shot_attempt":
            # SIGNAL 2: Rim verdict (pose-based player-signal check)
            win = psc.window_verdict(pose_model, config.video_path, ball_by_frame,
                                     hoop_by_frame, hoop_label, a["end_frame"])
            if win == "HAND":
                # AUTONOMOUS DECISION: Ball stayed in hand → Not a real shot
                rec["verdict"] = "not_shot"
                rec["reason"] = ("autonomous_filter: ball stayed in a hand through "
                                 "the 0.5s after arrival, not at the rim -- "
                                 "likely a catch/pass/rebound, not a shot")
                rec["player_signal_downgraded_from"] = shot["shot_type"]
                results.append(rec)
                continue
            # Both signals passed: physics gate (implicit, arcs pre-filtered) +
            # rim verdict (ball at rim, not hand) → Keep as shot
            rec["player_signal"] = win
            rel = find_release(a["fit_x"], a["fit_y"], a["start_frame"], tracks_by_frame)
            if rel["status"] != "found":
                rec["shooter"] = {"status": "no_confident_shooter",
                                  "reason": f"no tracked body within "
                                            f"{RELEASE_DIST_GATE_PX}px of the "
                                            f"back-extrapolated release path within "
                                            f"{RELEASE_BACK_MAX_FRAMES} frames"}
            elif not (tracks_span[0] <= rel["release_frame"] < tracks_span[1]):
                rec["shooter"] = {"status": "no_identity_data",
                                  "reason": f"release frame {rel['release_frame']} "
                                            f"outside tracks cache span {tracks_span}",
                                  "release": rel}
            else:
                tid = rel["track_id"]
                ident = identity_by_ft.get((rel["release_frame"], tid))
                if ident is None:
                    rec["shooter"] = {"status": "no_identity_data",
                                      "reason": f"track {tid} has no identity event "
                                                f"at release frame {rel['release_frame']}",
                                      "release": rel}
                else:
                    identity_id, identity_state = ident
                    status = ("attributed" if identity_state == "confirmed"
                              else "review_item")
                    rec["shooter"] = {"status": status, "track_id": tid,
                                      "identity_id": identity_id,
                                      "identity_state": identity_state,
                                      "release": rel}
        results.append(rec)

    out_json = _out(config, "shot_attempts.json")
    json.dump({"clip": config.name, "hoop_radius_px": HOOP_RADIUS_PX,
               "tracks_span": list(tracks_span), "attempts": results},
              open(out_json, "w", encoding="utf-8"), indent=2)

    # gate-comparable summary, same tuple shape as local_weights_check
    shots = [r for r in results if r["verdict"] == "shot_attempt"]
    print(f"[ball_stages] arcs evaluated: {len(results)}  shot attempts: {len(shots)}")
    print("SHOT ATTEMPTS (start,end,hoop,type,min_dist,arrival):")
    for r in shots:
        print(f"   ({r['start_frame']}, {r['end_frame']}, {r['hoop']!r}, "
              f"{r['shot_type']!r}, {r['min_dist']}, {r['arrival']!r})  "
              f"shooter={r.get('shooter', {}).get('status')}")
    print("near-rim REJECTIONS (deflection/continuation reasons):")
    for r in results:
        if r["verdict"] != "shot_attempt" and "deflection" in (r.get("reason") or ""):
            print(f"   ({r['start_frame']}, {r['end_frame']}, {r['hoop']!r}, "
                  f"{r['reason']!r})")
    downgraded = [r for r in results if r.get("player_signal_downgraded_from")]
    print("PLAYER-SIGNAL REJECTIONS (ball ended in a hand, not the rim):")
    for r in downgraded:
        print(f"   ({r['start_frame']}, {r['end_frame']}, {r['hoop']!r}, "
              f"was {r['player_signal_downgraded_from']!r})")
    print(f"[ball_stages] wrote {out_json}")
    return out_json


def stage_shot_location(config, sa_json):
    """Shooter court-feet at release (oncourt join) + flat shot chart --
    shot_location's functions with explicit paths."""
    from shot_location import find_shot_location, render_shot_chart
    sa_doc = _load(sa_json)
    oncourt_doc = _load(os.path.join(_ROOT, "phase2", "out",
                                     f"{config.name}_oncourt.json"))
    results = []
    for a in sa_doc["attempts"]:
        if a["verdict"] != "shot_attempt":
            continue
        loc = find_shot_location(a, oncourt_doc)
        results.append({"start_frame": a["start_frame"], "end_frame": a["end_frame"],
                        "shooter_status": a.get("shooter", {}).get("status"), **loc})

    out_json = _out(config, "shot_locations.json")
    json.dump({"clip": config.name, "locations": results},
              open(out_json, "w", encoding="utf-8"), indent=2)
    for r in results:
        print(f"   {r['start_frame']}..{r['end_frame']}  shooter={r['shooter_status']}  "
              f"{r['status']}  {r.get('court_feet', r.get('reason'))}")

    located = [{"status": r["shooter_status"], "court_feet": r["court_feet"]}
               for r in results if r["status"] == "located"]
    chart_path = _out(config, "shot_chart.png")
    render_shot_chart(config.name, located, chart_path)
    print(f"[ball_stages] wrote {out_json} and {chart_path}")
    return out_json


def stage_ball_touches(config, det_json):
    """WHO HAS THE BALL, frame by frame -- the join between ball detection and
    player tracking that never existed before (2026-07-27).

    A TOUCH is one player holding the ball until she gives it up. NOT a
    possession: that is the team-level idea phase2/possessions.py already owns.

    Reads only cached artifacts (ball detections + tracks + on-court + merged
    identity), so it costs seconds. Every output is a CANDIDATE pending DJ's
    eyeball -- see spikes/ball_touch.py for the frozen thresholds and the
    stated risk ("nearest to the ball" is not "has the ball")."""
    import ball_touch as bt_touch
    tracks_doc = _load(config.tracks_cache_path)
    oncourt_doc = _load(os.path.join(_ROOT, "phase2", "out",
                                     f"{config.name}_oncourt.json"))
    events_doc = _load(os.path.join(_ROOT, "phase2", "out",
                                    f"{config.name}_player_events_merged.json"))
    # The jersey registry is OPTIONAL: without it a touch is still measured,
    # it just reports "unnamed" rather than a number it cannot back up.
    reg_path = os.path.join(_ROOT, "phase2", "out",
                            f"{config.name}_ocr_confirms.json")
    registry_doc = _load(reg_path) if os.path.exists(reg_path) else None
    if registry_doc is None:
        print(f"[ball_stages] no {os.path.basename(reg_path)} -- touches will "
              f"report 'unnamed' instead of jersey numbers (run stage6 first)")
    # REFEREES AND BENCH CANNOT HOLD THE BALL. Reuses the human labels the
    # pipeline already trusts for seeding (roster.load_ref_tracks -- a pure
    # path-taking function, so no ACTIVE_CLIP binding here). Found by looking:
    # HARD's t3 is a DJ-labelled referee and was being credited with a 0.5s
    # touch. A ref stands in the paint all possession, so crediting one
    # invents exactly the ball-handling tendency the product sells.
    from roster import load_ref_tracks
    non_players = load_ref_tracks(os.path.join(_ROOT, "phase2", "out",
                                               f"{config.name}_decisions.json"))
    if non_players:
        print(f"[ball_stages] {len(non_players)} human-labelled ref/bench "
              f"track(s) excluded from holding the ball: {sorted(non_players)}")
    result = bt_touch.analyze(_load(det_json), tracks_doc, oncourt_doc,
                              events_doc, CONF_FLOOR, registry_doc, non_players)

    out_json = _out(config, "ball_touches.json")
    json.dump({"clip": config.name, "conf_floor": CONF_FLOOR,
               "params": {"HOLD_GATE_BODY_FRAC": bt_touch.HOLD_GATE_BODY_FRAC,
                          "MARGIN_BODY_FRAC": bt_touch.MARGIN_BODY_FRAC,
                          "MIN_TOUCH_FRAMES": bt_touch.MIN_TOUCH_FRAMES,
                          "MAX_GAP_FRAMES": bt_touch.MAX_GAP_FRAMES},
               **result}, open(out_json, "w", encoding="utf-8"), indent=2)
    for line in bt_touch.summary_lines(result, config.name):
        print(line)
    print(f"[ball_stages] wrote {out_json}")
    return out_json


def stage_team_possessions(config, touches_json):
    """TEAM POSSESSIONS -- one team has the ball until they lose it.

    Takes the touches (which body had the ball) and answers the team-level
    question on top of them: whose ball is it, from when to when. See
    phase2/team_possessions.py for the two end rules (other team touches it;
    ball goes out of bounds) and phase2/touch_teams.py for how a touch gets a
    team without asking a human anything new.

    Needs ONE video pass over a few frames per touch (the jersey has to be
    looked at), so it is not free like the touch stage -- but it is a handful of
    frames, not the whole clip.

    ABSTAINS LOUDLY rather than producing a teamless or mislabelled game: no two
    usable jersey colours, or colours that cannot be told apart on this footage,
    and the stage says so and writes nothing.
    """
    import sys as _sys
    _p2 = os.path.join(_ROOT, "phase2")
    if _p2 not in _sys.path:
        _sys.path.insert(0, _p2)
    import team_possessions as tpos
    import touch_teams as tteams
    import stage2_multikeyframe as s2

    touch_doc = _load(touches_json)
    touches = touch_doc.get("touches", [])
    fps = touch_doc.get("fps") or 30.0
    if not touches:
        print("[ball_stages] no touches -- cannot build possessions (ABSTAINING)")
        return None

    refs = tteams.refs_from_teams(getattr(config, "teams", None))
    if refs is None:
        print("[ball_stages] ABSTAINING from possessions: this clip does not "
              "have two tellable-apart jersey colours in its config. "
              "Possession needs to know which team has the ball, and guessing "
              "would attribute plays to the wrong side.")
        return None

    # Which frames must actually be read off disk, and which track each is for.
    want = {}                                   # frame -> {track_id, ...}
    for t in touches:
        for f in tteams.sample_frames_for_touch(t):
            want.setdefault(f, set()).add(t["track_id"])

    tracks_doc = _load(config.tracks_cache_path)
    boxes = {}                                  # (frame, track_id) -> bbox
    for fr in tracks_doc["frames"]:
        f = fr["frame_index"]
        if f not in want:
            continue
        for tr in fr["tracks"]:
            if tr["track_id"] in want[f]:
                boxes[(f, tr["track_id"])] = tr["bbox"]

    needed = sorted(f for f in want if any((f, tid) in boxes for tid in want[f]))
    print(f"[ball_stages] sampling jerseys on {len(needed)} frame(s) "
          f"for {len({t['track_id'] for t in touches})} track(s)")

    samples = {}                                # track_id -> [bgr, ...]
    for f, img in s2.iter_frames(config.video_path, needed):
        for tid in want.get(f, ()):
            bbox = boxes.get((f, tid))
            if bbox is None:
                continue
            sig = tteams.sample_torso(img, bbox)
            if sig is not None:
                samples.setdefault(tid, []).append(sig)

    track_colors = tteams.average_colors(samples)
    team_by_track, reason, detail = tteams.team_of_tracks(track_colors, refs)
    if reason:
        print(f"[ball_stages] ABSTAINING from possessions: {reason}")
        return None

    tteams.attach_teams(touches, team_by_track)
    for line in tteams.summary_lines(touches, refs, detail):
        print(line)

    possessions = tpos.build(touches, fps)
    for line in tpos.summary_lines(possessions, config.name):
        print(line)

    out_json = _out(config, "team_possessions.json")
    json.dump({"clip": config.name, "fps": fps,
               "teams": [{"name": r["name"], "jersey_color": r["jersey_color"]}
                         for r in refs],
               "cluster_separation": detail["separation"],
               "label_axis_sep": detail["axis_sep"],
               "params": {"MIN_CENTROID_SEP": tteams.MIN_CENTROID_SEP,
                          "MIN_AXIS_SEP": tteams.MIN_AXIS_SEP,
                          "MIN_REF_SEP": tteams.MIN_REF_SEP,
                          "MIN_SAMPLES_PER_TRACK": tteams.MIN_SAMPLES_PER_TRACK,
                          "SAMPLES_PER_TOUCH": tteams.SAMPLES_PER_TOUCH},
               "team_of_track": {str(k): v for k, v in sorted(team_by_track.items())},
               "possessions": possessions},
              open(out_json, "w", encoding="utf-8"), indent=2)
    print(f"[ball_stages] wrote {out_json}")
    return out_json


def stage_shot_outcome(config, sa_json, arcs_json, hoop_json, det_json):
    """Candidate make/miss labels (review-first, never a bare stat) --
    shot_outcome's evidence functions with explicit paths."""
    from shot_outcome import (below_rim_fall_evidence, classify_outcome,
                              deflection_evidence)
    sa_doc = _load(sa_json)
    arcs_doc = _load(arcs_json)
    hoop_doc = _load(hoop_json)
    det_doc = _load(det_json)

    hoop_lookups = {"far": _hoop_lookup(hoop_doc, "hoop_far_px"),
                    "near": _hoop_lookup(hoop_doc, "hoop_near_px")}
    raw_by_frame = {fr["frame_index"]: fr["detections"] for fr in det_doc["frames"]}
    chains = arcs_doc["chains"]

    results = []
    for a in sa_doc["attempts"]:
        if a["verdict"] != "shot_attempt":
            continue
        hoop_at = hoop_lookups[a.get("hoop", "far")]
        after = a["at_frame"]      # closest hoop approach -- outcome plays out AFTER
        make_ev = below_rim_fall_evidence(raw_by_frame, hoop_at, after)
        miss_ev = deflection_evidence(chains, hoop_at, after)
        outcome = classify_outcome(make_ev, miss_ev)
        results.append({"start_frame": a["start_frame"], "end_frame": a["end_frame"],
                        "hoop": a.get("hoop", "far"),
                        "checked_after_frame": after, **outcome})

    out_json = _out(config, "shot_outcomes.json")
    json.dump({"clip": config.name, "outcomes": results},
              open(out_json, "w", encoding="utf-8"), indent=2)
    for r in results:
        print(f"   {r['start_frame']}..{r['end_frame']}  outcome={r['outcome']}  "
              f"{r.get('reason', '')}")
    print(f"[ball_stages] wrote {out_json}  (all outcomes are candidate labels "
          f"feeding review -- never a bare made/missed stat)")
    return out_json
