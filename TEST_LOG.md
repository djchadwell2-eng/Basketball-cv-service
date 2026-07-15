# TEST_LOG — gated ball-detection optimization tests (protocol: DJ, 2026-07-15)

Rules in force: raw output only (no paraphrase), timestamped entries, suite
must be green before each test, stop on any break or contradiction, NOTHING
adopted — every result is "MEASURED — pending DJ review". Stop after Test 2.

---

## TEST 1 — Robust arc fitting (outlier-tolerant parabola fitter)

Success condition: shot B (TEST1 frames ~315-327, user-verified jump shot)
forms a valid arc from fine-tuned-model detections, AND all 158 regression
tests stay green.

### [2026-07-15 12:33:21] Pre-test gate: regression suite

```
$ .venv/Scripts/python -m pytest tests/ -q
........................................................................ [ 91%]
..............                                                           [100%]
158 passed in 2.28s
```

GATE: GREEN — proceeding with Test 1.

### Baseline (pre-change), for reference — from diagnosis run [2026-07-15 ~12:20]

Shot B's chain DOES form from fine-tuned detections (conf>=0.10) but is
rejected by the physics fitter (verdict no_claim). Raw chain points
(frame, cx, cy, conf):

```
chain 1: 11 pts, verdict=no_claim
    (314, 308, 346, 0.15)
    (315, 331, 319, 0.23)
    (318, 344, 274, 0.21)
    (320, 349, 256, 0.78)
    (322, 357, 236, 0.78)
    (323, 378, 230, 0.12)
    (324, 368, 230, 0.78)
    (325, 379, 226, 0.7)
    (326, 379, 224, 0.74)
    (327, 382, 223, 0.51)
    (329, 424, 236, 0.1)
```

Corrupting members visible: f=323 (378,230) conf 0.12 x-backtracks vs f=324
(368,230) conf 0.78; f=329 (424,236) conf 0.10 is a +42px tail jump; f=314
conf 0.15 head point. Fine-tuned arcs near shot B (300-340): [] (none).
Top-1-per-frame filtering alone did NOT fix it (arcs near shot B still []).

### [2026-07-15 12:41 – 17:16] Implementation notes (raw trail, incl. 2 issues found & fixed mid-test)

Change made: `spikes/ball_trajectory.py` gained a bounded robust fallback
(`_robust_whole_chain_fit`): when the ordinary growth loop finds NO arcs in a
chain, try dropping subsets of the worst-residual candidate pool (pool =
budget+3 points; budget = ceil(25% of chain) hard-capped at 4 and floored so
kept >= MIN_FIT_LEN=8), smallest subsets first; physics gates unchanged; arcs
found this way carry n_dropped + dropped_frames.

Issue 1 (found by wall-clock): first version had NO absolute drop cap; on
dense fine-tuned logs (100+ point chains) the subset search went
combinatorial — measured 2.4 CPU-HOURS hung before kill. Fixed:
ROBUST_MAX_DROPS_ABS = 4. Also: greedy and 1-step-lookahead drop orders both
FAILED to recover shot B (greedy trace below); bounded subset search fixed it.

```
greedy trace on real shot-B chain (11 pts, budget 3):
iter 0: n=11 ok=False accel_y=1.724 rms=(8.62,2.49)  -> drop f=329
iter 1: n=10 ok=False accel_y=1.645 rms=(5.76,2.46)  -> drop f=323
iter 2: n=9  ok=False accel_y=1.628 rms=(4.83,2.55)  -> drop f=315
iter 3: n=8  ok=False accel_y=1.683 rms=(3.2,2.0)    [FAILS: rms_x 3.2 > 3.0]
correct 3-drop set {329,323,314}: n=8 ok=True accel_y=1.425 rms=(2.37,1.41)
```

Issue 2 (found by the side-effect check, NOT by the unit suite): robust path
initially ran the whole-chain fit BEFORE checking MIN_FIT_LEN, letting a
6-point chain claim an arc (stock TEST1 gained a bogus (806,811) arc) —
violates §14's "<8 points = too little evidence" gate. Fixed with a length
guard + regression test (test_robust_path_respects_the_min_fit_len_evidence_gate).

New tests added: 6 (robust recovery synthetic, random-walk non-rescue, drop
budget bound, absolute cap on long chains, min-fit-len gate, literal real
shot-B chain regression). Suite 158 -> 164.

### [2026-07-15 17:16:01] TEST 1 RESULT — raw output of the final verification run

```
===== TEST1 fine-tuned (roboflow log, conf>=0.10) =====
arcs (start,end,n_dropped): [(0, 8, 0), (20, 30, 0), (55, 74, 0), (84, 98, 0), (165, 184, 0), (188, 202, 0), (242, 250, 0), (257, 264, 0), (315, 327, 3), (354, 372, 0), (375, 385, 0), (388, 407, 0), (409, 416, 0), (419, 430, 0), (437, 457, 0), (507, 519, 0), (581, 592, 0)]
shots (start,end,hoop,type,min_dist,arrival,n_dropped):
   (55, 74, 'far', 'jumpshot', 77.8, 'extrapolated', 0)
   (84, 98, 'far', 'jumpshot', 59.9, 'observed', 0)
   (165, 184, 'far', 'layup', 12.6, 'observed', 0)
   (242, 250, 'far', 'layup', 24.9, 'observed', 0)
   (315, 327, 'far', 'jumpshot', 96.6, 'observed', 3)
   (581, 592, 'near', 'layup', 1.1, 'observed', 0)

===== TEST1 STOCK =====
arcs: [(59, 71, 0), (315, 327, 0), (354, 371, 0), (375, 386, 0), (387, 400, 0), (439, 457, 0), (718, 741, 0), (745, 762, 0), (769, 783, 0), (793, 800, 0), (809, 828, 0), (944, 957, 0), (959, 974, 0)]
shots:
   (59, 71, 'far', 'jumpshot', 87.2, 'extrapolated', 0)
   (315, 327, 'far', 'jumpshot', 101.5, 'observed', 0)

===== HARD STOCK (ground truth: ONLY 356-381 near + 1188-1211 far) =====
arcs: [(15, 25, 0), (29, 38, 0), (61, 78, 0), (79, 91, 0), (195, 210, 0), (293, 304, 0), (306, 316, 0), (327, 337, 0), (356, 381, 0), (418, 438, 0), (439, 453, 0), (652, 661, 0), (677, 687, 0), (688, 698, 0), (1007, 1020, 0), (1031, 1038, 0), (1092, 1101, 0), (1136, 1149, 0), (1188, 1211, 0), (1217, 1250, 0), (1273, 1287, 0), (1292, 1306, 0), (1310, 1322, 0), (1394, 1406, 0), (1479, 1488, 0), (1489, 1501, 0), (2380, 2388, 0), (2456, 2466, 0), (2593, 2604, 0), (2611, 2633, 0)]
shots:
   (356, 381, 'near', 'jumpshot', 19.4, 'observed', 0)
   (1188, 1211, 'far', 'jumpshot', 54.6, 'observed', 0)
(total analysis wall time: 0.5s)
```

### [2026-07-15 17:16] Post-change regression suite

```
$ .venv/Scripts/python -m pytest tests/ -q
....................                                                     [100%]
164 passed in 1.39s
```
(All 158 pre-existing tests green; 164 = 158 + 6 new robust-fitter tests.)

### TEST 1 VERDICT — MEASURED — pending DJ review

- SUCCESS CONDITION MET: shot B (user-verified 10.5s jump shot) now forms a
  valid arc from fine-tuned detections — arc (315, 327) with n_dropped=3
  (dropped junk frames were the low-conf members f=314/323/329), classified
  shot_attempt/jumpshot, min_dist 96.6px observed. All 158 pre-existing
  regression tests green.
- Side-effect check: STOCK results on both clips are IDENTICAL to pre-change
  (same arcs, same 2+2 ground-truth shots; the bogus 6-pt arc that appeared
  mid-test was a bug in the new code, caught and fixed with its own test).
- Fine-tuned TEST1 shot list after Test 1: 6 attempts — shot A (verified),
  84-98 (UNVERIFIED — needs DJ eyeball), shot B (verified, recovered),
  3 layups (all user-verified on film).
- NOT ADOPTED: no default/config changed. Stock yolov8m remains the ball
  layer's committed detector; the robust fitter is in the code path but its
  effect on stock artifacts is nil (verified above).
- Contradiction check (rule 5): no contradiction with DECISIONS ground
  truth. One PRE-EXISTING known error stands (flagged in-session
  2026-07-15, before these tests): DECISIONS §25's sentence "the 2 jump
  shots PLUS 3 layups" is wrong about which jump shots — it was shot A +
  unverified 84-98, with shot B missing (now recovered by Test 1).
  Correction of that sentence is left for DJ review, per protocol.
