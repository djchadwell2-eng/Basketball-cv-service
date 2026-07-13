"""Ball trajectory layer -- pure geometry tests (synthetic detections, no
video). The safety property: an ARC claim requires physics consistency
(smooth parabolic motion, downward accel in a plausible band); everything
else must stay a NO-CLAIM chain or junk -- never guessed into a ball claim.
DECISIONS.md 13 measured why: detector confidence CANNOT gate ball claims
(the real shot arc never crossed conf 0.5), so physics is the only gate.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "spikes"))

from ball_trajectory import (  # noqa: E402
    ACCEL_Y_MAX, ACCEL_Y_MIN, MIN_CHAIN_LEN, MIN_FIT_LEN,
    build_chains, classify_chain,
)


def det(frame, cx, cy, conf=0.10):
    """One synthetic detection in the spike-log shape (center-based helper)."""
    return {"frame_index": frame, "cx": cx, "cy": cy, "conf": conf}


def parabola(frames, x0=100.0, vx=8.0, y0=300.0, vy=-9.0, ay=0.5):
    """Ballistic points: constant-velocity x, downward-accelerating y (image
    +y = down; ay=0.5 px/frame^2 is inside the band measured on HARD)."""
    t0 = frames[0]
    return [det(f, x0 + vx * (f - t0),
                y0 + vy * (f - t0) + 0.5 * ay * (f - t0) ** 2)
            for f in frames]


def frames_doc(dets):
    """Wrap flat detections into the spike log's frames[] shape."""
    by_frame = {}
    for d in dets:
        by_frame.setdefault(d["frame_index"], []).append(
            {"bbox": [d["cx"] - 10, d["cy"] - 10, d["cx"] + 10, d["cy"] + 10],
             "conf": d["conf"]})
    return [{"frame_index": f, "detections": by_frame[f]}
            for f in sorted(by_frame)]


# ---------------------------------------------------------------- chaining

def test_smooth_motion_forms_one_chain():
    chains = build_chains(frames_doc(parabola(range(0, 20))))
    assert len(chains) == 1
    assert len(chains[0]["points"]) == 20


def test_gap_over_tolerance_splits_the_chain():
    dets = parabola(range(0, 10)) + parabola(range(16, 26), x0=500.0, y0=300.0)
    chains = build_chains(frames_doc(dets))
    assert len(chains) == 2


def test_small_gap_is_bridged():
    pts = parabola(range(0, 20))
    dets = [d for d in pts if d["frame_index"] not in (9, 10)]   # 2-frame hole
    chains = build_chains(frames_doc(dets))
    assert len(chains) == 1
    assert len(chains[0]["points"]) == 18


def test_two_distant_simultaneous_balls_stay_separate_chains():
    dets = parabola(range(0, 15)) + parabola(range(0, 15), x0=1500.0, y0=300.0)
    chains = build_chains(frames_doc(dets))
    assert len(chains) == 2


def test_teleport_step_starts_a_new_chain():
    dets = parabola(range(0, 8)) + parabola(range(8, 16), x0=900.0, y0=300.0)
    chains = build_chains(frames_doc(dets))
    assert len(chains) == 2


# ------------------------------------------------------------ classification

def test_clean_parabola_is_claimed_as_arc():
    chain = build_chains(frames_doc(parabola(range(0, 20))))[0]
    out = classify_chain(chain)
    assert out["verdict"] == "arc"
    assert len(out["arcs"]) == 1
    a = out["arcs"][0]
    assert a["start_frame"] == 0 and a["end_frame"] == 19
    assert ACCEL_Y_MIN <= a["accel_y"] <= ACCEL_Y_MAX


def test_static_glare_is_junk_never_fit():
    dets = [det(f, 400.0 + (0.4 if f % 2 else -0.4), 500.0) for f in range(20)]
    chain = build_chains(frames_doc(dets))[0]
    assert classify_chain(chain)["verdict"] == "static_junk"


def test_too_short_chain_is_never_claimed():
    chain = build_chains(frames_doc(parabola(range(0, MIN_CHAIN_LEN - 1))))[0]
    assert classify_chain(chain)["verdict"] == "too_short"


def test_jittery_random_walk_fails_physics_and_stays_no_claim():
    import random
    rng = random.Random(7)
    x, y, dets = 400.0, 400.0, []
    for f in range(24):
        x += rng.uniform(-15, 15) + 8      # drifts so it isn't static junk
        y += rng.uniform(-15, 15)
        dets.append(det(f, x, y))
    chain = build_chains(frames_doc(dets))[0]
    out = classify_chain(chain)
    assert out["verdict"] == "no_claim"
    assert out["arcs"] == []


def test_upward_accelerating_motion_is_physically_impossible_and_refused():
    chain = build_chains(frames_doc(parabola(range(0, 20), vy=2.0, ay=-0.5)))[0]
    assert classify_chain(chain)["verdict"] == "no_claim"


def test_absurdly_strong_accel_is_outside_the_band_and_refused():
    chain = build_chains(frames_doc(parabola(range(0, 20), ay=ACCEL_Y_MAX * 4)))[0]
    assert classify_chain(chain)["verdict"] == "no_claim"


def test_arc_inside_a_longer_chain_is_found_without_claiming_the_rest():
    """Flight then held-by-a-player: the parabolic prefix must be claimed,
    the erratic tail must NOT be swallowed into the claim."""
    import random
    rng = random.Random(3)
    flight = parabola(range(0, 14))
    x = flight[-1]["cx"]
    y = flight[-1]["cy"]
    tail = []
    for f in range(14, 28):
        x += rng.uniform(-12, 12)
        y += rng.uniform(-12, 12)
        tail.append(det(f, x, y))
    chain = build_chains(frames_doc(flight + tail))[0]
    out = classify_chain(chain)
    assert out["verdict"] == "arc"
    assert len(out["arcs"]) == 1
    a = out["arcs"][0]
    assert a["start_frame"] == 0
    assert a["end_frame"] >= MIN_FIT_LEN - 1
    assert a["end_frame"] <= 16          # small overhang ok; tail NOT swallowed
