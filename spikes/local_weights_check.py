"""READ-ONLY check: do LOCALLY-TRAINED fine-tuned weights reproduce the
hosted model's TEST_LOG.md TEST 2 result? (Milestone 1 adoption diligence)

TEST 2 proved the HOSTED basketball-players-fy4c2 v25 model reproduces
HARD's 2 verified shots and rejects its deflections -- via the Roboflow
API (~72k calls/game, won't scale). This harness runs the IDENTICAL
analysis (conf>=0.10 filter -> ball_trajectory chains/physics gates ->
shot_attempts classifier, both hoops) over a ball_spike.py log produced
by LOCAL weights, and prints the same ground-truth check.

Writes nothing -- raw stdout is the artifact (TEST_LOG protocol).

Run:  .venv/Scripts/python spikes/local_weights_check.py HARD _ball_finetuned_v1
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)

from ball_trajectory import build_chains, classify_chain
from shot_attempts import classify_shot

CONF_FLOOR = 0.10          # TEST 2's analysis filter (junk sits <0.10, DECISIONS 24)

# HARD ground truth (TEST_LOG.md TEST 2 success condition); TEST1 spans are
# the USER-VERIFIED attempts from TEST 1's fine-tuned run (shot A 55-74,
# shot B 315-327 both verified jumpshots; 3 verified layups) -- the
# unverified 84-98 candidate is deliberately NOT ground truth.
GROUND_TRUTH = {
    # "deflections" = any confirmed NON-shot near-rim arc, not just literal
    # deflections: (418,438)/(1217,1250) are rim deflections; (1352,1378) is
    # a cross-court PASS that happens to arc near the far hoop pixel (DJ-
    # confirmed 2026-07-23, TEST 12 follow-up); (2234,2250) is a player
    # HOLDING the ball on an inbounds play (no pass until frame ~2340) --
    # not real ball motion at all, a bogus arc fit to detection jitter.
    # (395,445) is the ORIGINAL DJ-refuted false positive -- a rebound caught
    # and dished out to the perimeter (TEST 10, 2026-07-17, DJ eyeball: "NO
    # shot attempt"), which v2, v3 and the TEST 17 control all claim as a
    # layup. It was discussed in every test since and was NEVER ADDED HERE,
    # so the harness has been silently checking 4 of the 5 known non-shots.
    # Found 2026-07-25 when the control run claimed it and the check said
    # nothing. Ground truth is only protection if it is actually written down.
    "HARD": {"shots": [(356, 381, "near"), (1188, 1211, "far")],
             "deflections": [(395, 445), (418, 438), (1217, 1250),
                             (1352, 1378), (2234, 2250)]},
    "TEST1": {"shots": [(55, 74, "far"), (165, 184, "far"), (242, 250, "far"),
                        (315, 327, "far"), (581, 592, "near")],
              "deflections": []},
}


def main():
    clip = sys.argv[1] if len(sys.argv) > 1 else "HARD"
    suffix = sys.argv[2] if len(sys.argv) > 2 else "_ball_finetuned_v1"
    log_path = os.path.join(_HERE, "out", f"{clip}_ball_spike_log{suffix}.json")
    hoop_path = os.path.join(_HERE, "out", f"{clip}_hoop_track.json")

    doc = json.load(open(log_path, encoding="utf-8"))
    n_raw = sum(len(fr["detections"]) for fr in doc["frames"])
    frames = [{"frame_index": fr["frame_index"],
               "detections": [d for d in fr["detections"] if d["conf"] >= CONF_FLOOR]}
              for fr in doc["frames"]]
    n_kept = sum(len(fr["detections"]) for fr in frames)
    n_seen = sum(1 for fr in frames if fr["detections"])
    tot = len(frames)
    print(f"[check] {clip} log={os.path.basename(log_path)} "
          f"model={doc.get('model')} frames={tot}")
    print(f"[check] ball seen in {n_seen}/{tot} frames ({100*n_seen/max(tot,1):.1f}%) "
          f"at conf>={CONF_FLOOR} ({n_kept}/{n_raw} dets kept)")

    chains = build_chains(frames)
    results = [classify_chain(c) for c in chains]
    arcs = []
    for c, r in zip(chains, results):
        if r["verdict"] != "arc":
            continue
        pts = [(p[0], p[1], p[2]) for p in c["points"]]
        for a in r["arcs"]:
            arcs.append((a, pts))
    arcs.sort(key=lambda ar: ar[0]["start_frame"])

    hoop_doc = json.load(open(hoop_path, encoding="utf-8"))

    def _hoop_lookup(key):
        by_frame = {r["frame_index"]: tuple(r[key]) for r in hoop_doc["frames"]
                    if r.get(key) is not None}
        return lambda f: by_frame.get(f)

    hoop_at = {"far": _hoop_lookup("hoop_far_px"), "near": _hoop_lookup("hoop_near_px")}

    print(f"\nframes analyzed: {tot}   arcs: {len(arcs)}")
    print("arcs (start,end,n_dropped):",
          [(a["start_frame"], a["end_frame"], a.get("n_dropped", 0)) for a, _ in arcs])

    shots, rejections = [], []
    for a, pts in arcs:
        seg = [p for p in pts if a["start_frame"] <= p[0] <= a["end_frame"]]
        cands = []
        for label in ("far", "near"):
            s = classify_shot(seg, hoop_at[label], fit_x=a["fit_x"], fit_y=a["fit_y"])
            cands.append((s, label))
        passing = [c for c in cands if c[0]["verdict"] == "shot_attempt"]
        if passing:
            passing.sort(key=lambda c: c[0]["min_dist"])
            s, label = passing[0]
            shots.append((a["start_frame"], a["end_frame"], label, s["shot_type"],
                          s["min_dist"], s["arrival"], a.get("n_dropped", 0)))
        else:
            cands.sort(key=lambda c: (c[0]["min_dist"] is None, c[0]["min_dist"]))
            s, label = cands[0]
            if "deflection" in (s.get("reason") or ""):
                rejections.append((a["start_frame"], a["end_frame"], label, s["reason"]))

    print("\nSHOT ATTEMPTS (start,end,hoop,type,min_dist,arrival,n_dropped):")
    for s in shots:
        print("  ", s)
    print("\nnear-rim REJECTIONS (deflection/continuation reasons):")
    for r in rejections:
        print("  ", r)

    gt = GROUND_TRUTH.get(clip)
    if gt:
        print("\nGROUND TRUTH CHECK:")
        for (gs, ge, hoop) in gt["shots"]:
            # same shot = a claimed attempt at the right hoop covering the
            # BULK of the verified span. Overlap, not endpoint proximity:
            # the old "both endpoints within 10 frames" test reported TEST 11's
            # v3 far shot as MISSED because the arc started 11 frames early
            # while fully containing the verified span -- a harness false
            # alarm that needed a hand-correction in the log, and it fired
            # again on the TEST 17 control. Flagged in TEST 11, fixed here.
            hits = []
            for s in shots:
                if s[2] != hoop:
                    continue
                covered = min(s[1], ge) - max(s[0], gs)
                if covered >= 0.5 * (ge - gs):
                    hits.append(s)
            print(f"  verified shot {gs}-{ge} ({hoop}): "
                  f"{'REPRODUCED ' + str(hits[0]) if hits else '*** MISSED ***'}")
        for (ds, de) in gt["deflections"]:
            # a deflection is wrongly claimed only if a SHOT's own span lies
            # mostly inside the deflection span (not the +/-10 adjacency that
            # false-alarmed in TEST 2's check -- see TEST_LOG note)
            claimed = [s for s in shots if s[0] >= ds - 5 and s[1] <= de + 5]
            print(f"  deflection {ds}-{de}: "
                  f"{'*** CLAIMED AS SHOT: ' + str(claimed) + ' ***' if claimed else 'correctly NOT a shot'}")


if __name__ == "__main__":
    main()
