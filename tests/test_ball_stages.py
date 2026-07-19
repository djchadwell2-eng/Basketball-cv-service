"""ball_stages.py -- the run_clip Phase 5 glue. Tests cover exactly the
places integration could shift behavior: the conf-floor analysis filter
(the TEST 2/8/10 protocol), the reuse-or-rerun fingerprints for the two
slow stages, the new ClipConfig ball-layer validation, and one LITERAL-DATA
regression locking the integrated chain to TEST_LOG TEST 10's measured
TEST1 result (4/5 verified attempts incl. both target layups; shot B
missed -- v2's known blind spot, expected). The physics gates themselves
are NOT re-tested here -- tests/test_ball_trajectory.py owns those and is
deliberately untouched.
"""

import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "spikes"))

import ball_stages as bs  # noqa: E402
from clip_config import ClipConfig, Team  # noqa: E402


# ------------------------------------------------------ conf-floor filter --

def _frames(*dets_per_frame):
    return [{"frame_index": 100 + i,
             "detections": [{"bbox": [0, 0, 10, 10], "conf": c} for c in confs]}
            for i, confs in enumerate(dets_per_frame)]


def test_conf_floor_drops_below_keeps_at_and_above():
    out = bs.filter_conf(_frames([0.09, 0.10, 0.50], [0.05], []))
    assert [d["conf"] for d in out[0]["detections"]] == [0.10, 0.50]
    assert out[1]["detections"] == []          # frame kept, dets emptied
    assert out[2]["detections"] == []
    assert [fr["frame_index"] for fr in out] == [100, 101, 102]


def test_conf_floor_value_is_the_verified_protocols():
    assert bs.CONF_FLOOR == 0.10               # TEST 2/8/10 analysis filter


# ------------------------------------------------- detections fingerprint --

def _cfg(**over):
    base = dict(
        name="TEST1",
        video_path=os.path.abspath(__file__),   # exists; validate() only stats it
        event_frames=range(0, 10),
        render_sample_frames=range(0, 10),
        tracking_span_start=0, tracking_span_len=10,
        teams=(Team("A", "white", frozenset({1})),),
        seed_labels={}, accumulation_window_seconds=2.0,
        tracks_cache_path="unused",
        ball_weights_path=os.path.abspath(__file__),
        ball_span_start=0, ball_span_len=605,
        hoop_anchors={"far": (120, (582.0, 143.0)), "near": (580, (1377.0, 233.0))},
    )
    base.update(over)
    return ClipConfig(**base)


def _det_doc(**over):
    doc = {"clip": "TEST1", "span_start": 0, "span_len": 605,
           "model": os.path.basename(os.path.abspath(__file__)),
           "imgsz": 1280, "conf_threshold": 0.05}
    doc.update(over)
    return doc


def test_detections_fingerprint_matches_exact():
    assert bs.detections_current(_det_doc(), _cfg(), 1280, 0.05)


@pytest.mark.parametrize("field,value", [
    ("clip", "HARD"), ("span_start", 1), ("span_len", 604),
    ("model", "other_weights.pt"), ("imgsz", 1920), ("conf_threshold", 0.10),
])
def test_detections_fingerprint_refuses_any_mismatch(field, value):
    assert not bs.detections_current(_det_doc(**{field: value}), _cfg(), 1280, 0.05)


# -------------------------------------------------- hoop-track fingerprint --

def _hoop_doc(**over):
    doc = {"span_start": 0, "span_len": 1299,
           "rim_keyframe_far": 120, "rim_pixel_far": [582.0, 143.0],
           "rim_keyframe_near": 580, "rim_pixel_near": [1377.0, 233.0]}
    doc.update(over)
    return doc


def test_hoop_track_superset_span_with_same_anchors_is_reusable():
    # per-frame results are independent -> a covering span holds identical
    # data for the frames inside the ball span (the real TEST1 case: track
    # 0..1299 covers ball span 0..605)
    assert bs.hoop_track_covers(_hoop_doc(), _cfg())


def test_hoop_track_refuses_wrong_anchor_pixel():
    assert not bs.hoop_track_covers(_hoop_doc(rim_pixel_near=[1377.0, 234.0]), _cfg())


def test_hoop_track_refuses_wrong_anchor_keyframe():
    assert not bs.hoop_track_covers(_hoop_doc(rim_keyframe_far=121), _cfg())


def test_hoop_track_refuses_non_covering_span():
    assert not bs.hoop_track_covers(_hoop_doc(span_len=600), _cfg())      # ends early
    assert not bs.hoop_track_covers(_hoop_doc(span_start=1), _cfg())      # starts late


# --------------------------------------------- ClipConfig ball validation --

def test_validate_passes_with_ball_layer_configured():
    _cfg().validate()


def test_validate_ignores_ball_fields_when_not_configured():
    _cfg(ball_span_len=0, hoop_anchors=None,
         ball_weights_path="does_not_exist.pt").validate()


def test_validate_refuses_missing_weights():
    with pytest.raises(SystemExit, match="ball weights not found"):
        _cfg(ball_weights_path="does_not_exist.pt").validate()


def test_validate_refuses_incomplete_anchors():
    with pytest.raises(SystemExit, match="hoop_anchors"):
        _cfg(hoop_anchors={"far": (120, (582.0, 143.0))}).validate()
    with pytest.raises(SystemExit, match="hoop_anchors"):
        _cfg(hoop_anchors=None).validate()


def test_validate_refuses_negative_ball_span_start():
    with pytest.raises(SystemExit, match="bad ball span"):
        _cfg(ball_span_start=-1).validate()


# ------------------------------------- LITERAL-DATA regression (TEST 10) --

_V2_LOG = os.path.join(_ROOT, "spikes", "out",
                       "TEST1_ball_spike_log_ball_finetuned_v2.json")
_HOOP_TRACK = os.path.join(_ROOT, "spikes", "out", "TEST1_hoop_track.json")


@pytest.mark.skipif(not (os.path.exists(_V2_LOG) and os.path.exists(_HOOP_TRACK)),
                    reason="TEST1 v2 measurement artifacts not on this machine")
def test_integrated_chain_reproduces_test10_on_the_saved_test1_v2_log():
    """Feed the SAVED TEST1 v2 detection log through the integrated analysis
    chain (filter_conf -> build_chains/classify_chain -> evaluate_arcs) and
    require TEST_LOG TEST 10's exact result. Any drift here means the
    integration changed the verified behavior -- fix the integration, never
    this expectation."""
    import ball_trajectory as bt
    log_doc = json.load(open(_V2_LOG, encoding="utf-8"))
    hoop_doc = json.load(open(_HOOP_TRACK, encoding="utf-8"))

    frames = bs.filter_conf(log_doc["frames"])
    chains = bt.build_chains(frames)
    results = [bt.classify_chain(c) for c in chains]
    merged = [{"points": c["points"], **r} for c, r in zip(chains, results)]
    evaluated = bs.evaluate_arcs(merged,
                                 bs._hoop_lookup(hoop_doc, "hoop_far_px"),
                                 bs._hoop_lookup(hoop_doc, "hoop_near_px"))

    shots = [(a["start_frame"], a["end_frame"], hoop, s["shot_type"],
              s["min_dist"], s["arrival"])
             for a, _seg, s, hoop in evaluated if s["verdict"] == "shot_attempt"]
    assert sorted(shots) == [
        (58, 70, "far", "jumpshot", 118.1, "extrapolated"),
        (164, 184, "far", "layup", 15.3, "observed"),
        (236, 250, "far", "layup", 27.2, "observed"),
        (581, 589, "near", "layup", 18.4, "observed"),
    ]
    # shot B (315-327) is v2's known blind spot -- MISSED is the measured
    # result; a claim appearing there would be a behavior change too.
    assert not any(s[0] >= 305 and s[1] <= 337 for s in shots)
    rejections = [(a["start_frame"], a["end_frame"])
                  for a, _seg, s, _h in evaluated
                  if s["verdict"] != "shot_attempt"
                  and "deflection" in (s.get("reason") or "")]
    assert sorted(rejections) == [(103, 110), (188, 202)]
