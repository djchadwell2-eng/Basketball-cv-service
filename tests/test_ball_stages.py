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


def _test1_v2_shots():
    """Run the SAVED TEST1 v2 detection log through the integrated analysis
    chain (filter_conf -> build_chains/classify_chain -> evaluate_arcs) and
    return (claimed_arcs, rejected_arcs) as [(start, end), ...]."""
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
    claimed = sorted((a["start_frame"], a["end_frame"])
                     for a, _s, sh, _h in evaluated
                     if sh["verdict"] == "shot_attempt")
    rejected = sorted((a["start_frame"], a["end_frame"])
                      for a, _s, sh, _h in evaluated
                      if sh["verdict"] != "shot_attempt")
    return claimed, rejected


def _overlaps(a, b):
    return a[0] <= b[1] and b[0] <= a[1]


# WHY THIS GATE WAS REWRITTEN (2026-07-29). It used to assert TEST 10's EXACT
# output tuples, down to min_dist to one decimal. That is a snapshot of what the
# code did on one day, not a statement of what it must do -- so DJ's deliberate
# chain-fragmentation fix (_merge_gapped_chains, landed 2026-07-29) broke it
# even though it IMPROVED the result: shot A's arrival went from EXTRAPOLATED at
# 118.1px to OBSERVED at 61.4px, recovering real flight the old chaining threw
# away. A gate that fires on improvement trains people to edit the gate, which
# is exactly how a real regression eventually slips through.
#
# These assert the REQUIREMENT instead, against DJ's own ground truth
# (local_weights_check.GROUND_TRUTH -- the same source every clip gate uses):
# find his verified shots, do not claim his confirmed non-shots, keep v2's known
# blind spot, and never weld two shots into one. That is strictly stronger on
# what matters and immune to legitimate improvement.
# THE EXACT NUMBERS ARE NOT LOST -- they are recorded in TEST_LOG.md TEST 28.

@pytest.mark.skipif(not (os.path.exists(_V2_LOG) and os.path.exists(_HOOP_TRACK)),
                    reason="TEST1 v2 measurement artifacts not on this machine")
def test_test1_v2_still_finds_every_verified_shot_except_its_known_blind_spot():
    from local_weights_check import GROUND_TRUTH
    claimed, _rejected = _test1_v2_shots()
    truth = GROUND_TRUTH["TEST1"]["shots"]
    blind_spot = (315, 327)                     # shot B: v2's measured miss

    for (a, b, _hoop) in truth:
        found = any(_overlaps((a, b), c) for c in claimed)
        if (a, b) == blind_spot:
            assert not found, ("shot B is v2's KNOWN blind spot; suddenly "
                               "finding it is a behavior change to investigate, "
                               "not a silent win")
        else:
            assert found, f"verified shot {a}..{b} is no longer claimed"


@pytest.mark.skipif(not (os.path.exists(_V2_LOG) and os.path.exists(_HOOP_TRACK)),
                    reason="TEST1 v2 measurement artifacts not on this machine")
def test_no_claimed_arc_welds_two_verified_shots_together():
    """The real risk the chain merge introduces. Stitching chains across a gap
    recovers truncated flights, but a merge gap that is too generous would join
    one shot to the NEXT one -- costing an attempt and inventing a monster arc.
    Bound stated as a property, not a fitted frame count."""
    from local_weights_check import GROUND_TRUTH
    claimed, _rejected = _test1_v2_shots()
    truth = GROUND_TRUTH["TEST1"]["shots"]
    for c in claimed:
        hit = [(a, b) for (a, b, _h) in truth if _overlaps((a, b), c)]
        assert len(hit) <= 1, (f"claimed arc {c} spans {len(hit)} verified "
                               f"shots {hit} -- the chain merge welded them")


def test_the_weld_guard_can_actually_fail():
    """PROOF THE GUARD IS NOT VACUOUS. On TEST1's real data no merge gap welds
    two shots -- checked at 40, 120 and 300 frames, all clean, which is real
    evidence for the chain-merge's own claim that the PHYSICS FIT (not the gap
    size) is what keeps unrelated flights apart. But a guard that never fires on
    the data it guards proves nothing about itself, and this project has shipped
    a tautological metric before ("wrong-player time 0.0s"). So the detection
    logic is exercised directly on a synthetic weld."""
    truth = [(55, 74, "far"), (165, 184, "far")]
    welded = (57, 180)                    # one arc swallowing BOTH shots
    hit = [(a, b) for (a, b, _h) in truth if _overlaps((a, b), welded)]
    assert len(hit) == 2                  # the guard's condition would fail
    clean = (57, 93)                       # the real current claim
    assert len([(a, b) for (a, b, _h) in truth if _overlaps((a, b), clean)]) == 1


@pytest.mark.skipif(not (os.path.exists(_V2_LOG) and os.path.exists(_HOOP_TRACK)),
                    reason="TEST1 v2 measurement artifacts not on this machine")
def test_the_two_near_rim_non_shots_are_still_rejected():
    """(103,110) and (188,202) are near-rim arcs that are NOT shots. They were
    frozen as a rejection list before; asserted here as the thing that matters
    -- they must never become CLAIMS."""
    claimed, rejected = _test1_v2_shots()
    for span in ((103, 110), (188, 202)):
        assert not any(_overlaps(span, c) for c in claimed), \
            f"non-shot {span} is now being claimed as a shot attempt"
        assert any(_overlaps(span, r) for r in rejected), \
            f"non-shot {span} vanished entirely -- it should be evaluated and rejected"
