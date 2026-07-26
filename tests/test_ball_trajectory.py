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


# --------------------------------------------- robust (outlier-tolerant) fit

def _chain_of(dets):
    """Hand-built chain dict (bypasses build_chains): the unit under test
    here is the CLASSIFIER's robust fallback on a chain that already
    contains junk members -- exactly how the real fine-tuned-detector
    chains arrive (junk that survived the association gate)."""
    return {"points": [(d["frame_index"], d["cx"], d["cy"], d["conf"])
                       for d in dets]}


def test_robust_fit_recovers_parabola_corrupted_by_outliers():
    """DECISIONS 26: a denser detector adds occasional junk members to an
    otherwise clean chain; the plain fitter rejects the whole chain. The
    robust fallback drops the worst-residual points (bounded) and recovers
    the arc, recording what it dropped."""
    pts = parabola(range(0, 12))
    pts[5] = det(5, pts[5]["cx"] - 18, pts[5]["cy"] + 15)      # mid outlier
    pts.append(det(12, pts[-1]["cx"] + 35, pts[-1]["cy"] + 25))  # tail junk
    out = classify_chain(_chain_of(pts))
    assert out["verdict"] == "arc"
    assert len(out["arcs"]) == 1
    a = out["arcs"][0]
    assert a.get("n_dropped", 0) >= 1
    assert ACCEL_Y_MIN <= a["accel_y"] <= ACCEL_Y_MAX


def test_robust_fit_never_rescues_a_random_walk():
    """The robust path must not turn junk into an arc: dropping 25% of a
    random walk still leaves a random walk."""
    import random
    rng = random.Random(7)
    x, y, dets = 400.0, 400.0, []
    for f in range(24):
        x += rng.uniform(-15, 15) + 8
        y += rng.uniform(-15, 15)
        dets.append(det(f, x, y))
    out = classify_chain(_chain_of(dets))
    assert out["verdict"] == "no_claim"
    assert out["arcs"] == []


def test_robust_fit_drop_budget_is_bounded():
    """More corruption than the drop budget (25% = 3 of 12) must stay
    no_claim -- the fallback is bounded, not 'drop until it fits'."""
    pts = parabola(range(0, 12))
    for i, f in enumerate([2, 5, 7, 9]):                       # 4 of 12 corrupted
        pts[f] = det(f, pts[f]["cx"] + (18 + 6 * i) * (-1) ** i,
                     pts[f]["cy"] + 15 + 6 * i)
    out = classify_chain(_chain_of(pts))
    assert out["verdict"] != "arc"


def test_robust_fit_absolute_drop_cap_on_long_chains():
    """Dense fine-tuned logs produce 100+ point chains; the drop budget is
    capped at 4 ABSOLUTE (not just 25%) both to keep the subset search
    bounded (an uncapped analysis run hung for 2.4 CPU-hours) and because a
    chain needing >4 junk drops isn't a trustworthy rescue. Big outliers
    every 7 frames so every 8-point growth window contains one (growth finds
    nothing; only the robust path could rescue): 5 outliers (>cap) -> no
    arc; 4 outliers (=cap) -> arc with n_dropped == 4."""
    def corrupted(n_frames, outlier_frames):
        pts = [det(d["frame_index"], d["cx"], d["cy"])
               for d in parabola(range(0, n_frames), vy=-11.0, ay=0.55)]
        for f in outlier_frames:
            pts[f] = det(f, pts[f]["cx"] + 45, pts[f]["cy"] + 45)
        return pts

    out5 = classify_chain(_chain_of(corrupted(40, [5, 12, 19, 26, 33])))
    assert out5["verdict"] != "arc"

    out4 = classify_chain(_chain_of(corrupted(34, [5, 12, 19, 26])))
    assert out4["verdict"] == "arc"
    assert out4["arcs"][0].get("n_dropped", 0) == 4


def test_robust_path_respects_the_min_fit_len_evidence_gate():
    """REGRESSION for a gate violation caught during Test 1's side-effect
    check: the robust path must NOT rescue chains shorter than MIN_FIT_LEN
    (§14: <8 points = too little evidence for any physics claim). A clean
    6-point parabola stays no_claim."""
    pts = parabola(range(0, 6), vy=-11.0, ay=0.55)
    out = classify_chain(_chain_of(pts))
    assert out["verdict"] != "arc"


def test_robust_fit_recovers_the_real_shot_b_chain():
    """REGRESSION, real TEST1 fine-tuned-ball data (DECISIONS 26): the
    user-verified 10.5s jump shot's chain -- 11 points, 3 junk members
    (f=314 head, f=323 x-backtrack, f=329 tail jump) that fail the plain
    fitter. The robust fallback must recover it as an arc."""
    raw = [(314, 308.0, 346.0, 0.15), (315, 331.5, 319.0, 0.23),
           (318, 344.0, 274.0, 0.21), (320, 348.5, 256.0, 0.78),
           (322, 357.0, 236.0, 0.78), (323, 378.0, 230.0, 0.12),
           (324, 368.0, 230.0, 0.78), (325, 379.0, 226.0, 0.70),
           (326, 379.0, 224.0, 0.74), (327, 382.0, 223.0, 0.51),
           (329, 424.0, 236.0, 0.10)]
    chain = build_chains(frames_doc([det(f, x, y, c) for f, x, y, c in raw]))[0]
    out = classify_chain(chain)
    assert out["verdict"] == "arc"
    a = out["arcs"][0]
    assert 314 <= a["start_frame"] <= 320
    assert 325 <= a["end_frame"] <= 329


def test_upward_accelerating_motion_is_physically_impossible_and_refused():
    chain = build_chains(frames_doc(parabola(range(0, 20), vy=2.0, ay=-0.5)))[0]
    assert classify_chain(chain)["verdict"] == "no_claim"


def test_absurdly_strong_accel_is_outside_the_band_and_refused():
    chain = build_chains(frames_doc(parabola(range(0, 20), ay=ACCEL_Y_MAX * 4)))[0]
    assert classify_chain(chain)["verdict"] == "no_claim"


def test_pan_dragged_glare_drift_is_never_claimed():
    """REGRESSION, real data: this is the actual HARD glare chain (frames
    1115-1157, camera pan dragging a floor highlight left-down) whose 8-frame
    slice fit at accel_y=0.309 and was claimed before the MIN_Y_RANGE gate.
    Near-flat drift must never earn an arc claim."""
    glare = [(1115, 556, 648), (1116, 550, 649), (1117, 546, 650),
             (1118, 543, 651), (1119, 541, 652), (1120, 539, 652),
             (1121, 538, 653), (1122, 544, 651), (1124, 528, 654),
             (1125, 526, 655), (1126, 523, 655), (1130, 504, 659),
             (1133, 489, 661), (1134, 486, 663), (1135, 480, 664),
             (1136, 474, 664), (1137, 468, 666), (1138, 462, 668),
             (1139, 453, 670), (1140, 444, 672), (1141, 436, 674),
             (1142, 427, 677), (1143, 417, 679), (1144, 407, 682),
             (1145, 396, 684), (1146, 386, 687), (1147, 375, 690),
             (1148, 365, 693), (1149, 354, 696), (1150, 343, 698),
             (1151, 333, 702), (1152, 324, 704), (1153, 314, 706),
             (1154, 305, 709), (1155, 295, 712), (1156, 286, 714),
             (1157, 278, 717)]
    chains = build_chains(frames_doc([det(f, x, y) for f, x, y in glare]))
    assert len(chains) == 1
    out = classify_chain(chains[0])
    assert out["verdict"] == "no_claim"
    assert out["arcs"] == []


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


# TEST4 frames 4485-4518 (TEST 19): DJ's MADE 3-pointer at 2:27, verbatim from
# the v3 detection log -- one smooth 33-point flight that the greedy growth
# loop carved into (4485,4508) + (4509,4518), so the shot layer claimed the
# same shot TWICE. Literal real data, same convention as the shot-B chain
# regression above: a synthetic parabola cannot reproduce this, because the
# split is caused by real perspective bend (whole-span rms_y 5.31 > the gate
# while both halves pass).
TEST4_SPLIT_FLIGHT = [
    (4485, 1225.1, 381.6), (4486, 1231.1, 353.5), (4488, 1242.6, 294.7),
    (4489, 1243.0, 266.3), (4490, 1251.5, 247.3), (4491, 1255.5, 228.6),
    (4492, 1260.5, 209.1), (4493, 1265.8, 192.4), (4494, 1269.0, 177.3),
    (4495, 1273.1, 163.3), (4496, 1276.9, 151.2), (4497, 1281.5, 138.6),
    (4498, 1285.3, 130.0), (4499, 1291.2, 122.5), (4500, 1294.5, 113.7),
    (4501, 1297.6, 109.1), (4502, 1300.8, 103.7), (4503, 1302.9, 101.4),
    (4504, 1306.4, 99.7), (4505, 1309.0, 99.1), (4506, 1311.8, 99.4),
    (4507, 1313.5, 100.6), (4508, 1316.3, 102.3), (4509, 1318.4, 106.4),
    (4510, 1320.7, 111.3), (4511, 1322.5, 117.9), (4512, 1324.6, 125.1),
    (4513, 1326.9, 132.1), (4514, 1329.8, 140.0), (4515, 1332.7, 148.8),
    (4516, 1333.6, 158.7), (4517, 1337.0, 168.3), (4518, 1338.5, 180.2),
]


def test_one_real_flight_is_not_claimed_as_two_shots():
    """TEST 19's double-count bug: one flight must produce ONE arc, not two."""
    chain = build_chains(frames_doc([det(f, x, y) for f, x, y in TEST4_SPLIT_FLIGHT]))[0]
    out = classify_chain(chain)
    assert out["verdict"] == "arc"
    assert len(out["arcs"]) == 1, (
        f"one flight claimed as {len(out['arcs'])} arcs: "
        f"{[(a['start_frame'], a['end_frame']) for a in out['arcs']]}")
    a = out["arcs"][0]
    # the merged arc must actually span the flight, not just be one half
    assert a["start_frame"] <= 4490 and a["end_frame"] >= 4515


def test_merge_never_joins_two_arcs_separated_by_a_real_gap():
    """The merge is only for a carved-up single flight. Two flights with a
    hole between them must stay two arcs -- otherwise a shot and a later,
    unrelated flight could be welded into one claim."""
    first = parabola(range(0, 14))
    second = parabola(range(60, 74))       # far beyond MAX_GAP_FRAMES
    chains = build_chains(frames_doc(first + second))
    total = sum(len(classify_chain(c)["arcs"]) for c in chains)
    assert total == 2, f"expected 2 separate arcs, got {total}"
