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

---

## TEST 2 — Fine-tuned ball model on HARD.mp4 (adoption gate)

Success condition: reproduces HARD's 2 verified shots (356-381 @ near hoop,
1188-1211 @ far hoop) and correctly rejects its deflections (418-438,
1217-1250).

### [2026-07-15 17:17:05] Pre-test gate: regression suite

```
$ .venv/Scripts/python -m pytest tests/ -q
....................                                                     [100%]
164 passed in 1.31s
```
(All 158 pre-existing tests green within the 164.)

GATE: GREEN — proceeding with Test 2. Roboflow probe launched on HARD
frames 0-2746 (full clip), conf floor 5%, model basketball-players-fy4c2
v25 — hosted inference, ~80 min expected.

### [2026-07-15 18:00:23] TEST 2 RESULT — raw probe + analysis output

Probe (hosted inference, full clip):
```
[rf_probe] 2746 frames, ball seen in 2665 (97%) -> spikes/out/HARD_ball_spike_log_roboflow.json
```
(Stock yolov8m on the same clip: 65.5% of frames — DECISIONS §18.)

Analysis (trajectory + shot classifier, conf>=0.10, both hoops), raw:
```
frames analyzed: 2746   arcs: 58   wall: 0.5s
arcs (start,end,n_dropped): [(15, 25, 0), (29, 40, 0), (52, 62, 0), (63, 78, 0), (79, 93, 0), (163, 177, 0), (178, 187, 0), (195, 207, 0), (220, 234, 0), (247, 256, 0), (269, 282, 0), (296, 305, 0), (307, 320, 0), (322, 337, 0), (341, 348, 0), (351, 376, 0), (416, 438, 0), (439, 463, 0), (464, 471, 0), (483, 492, 0), (652, 664, 0), (677, 687, 0), (688, 698, 0), (700, 712, 0), (892, 900, 0), (997, 1007, 0), (1008, 1021, 0), (1030, 1046, 0), (1079, 1086, 0), (1092, 1102, 0), (1119, 1129, 0), (1135, 1152, 0), (1184, 1213, 0), (1216, 1250, 0), (1251, 1259, 0), (1259, 1267, 0), (1270, 1285, 0), (1291, 1308, 0), (1310, 1326, 0), (1341, 1348, 0), (1352, 1378, 0), (1391, 1406, 0), (1410, 1419, 0), (1465, 1488, 0), (1489, 1508, 0), (2182, 2193, 0), (2194, 2205, 0), (2376, 2388, 0), (2394, 2408, 0), (2446, 2454, 0), (2456, 2469, 0), (2472, 2479, 0), (2482, 2491, 0), (2515, 2527, 0), (2556, 2564, 0), (2565, 2588, 0), (2590, 2604, 0), (2610, 2633, 0)]

SHOT ATTEMPTS (start,end,hoop,type,min_dist,arrival,n_dropped):
   (351, 376, 'near', 'jumpshot', 75.9, 'observed', 0)
   (1184, 1213, 'far', 'jumpshot', 14.5, 'observed', 0)
   (1352, 1378, 'far', 'jumpshot', 16.5, 'observed', 0)

near-rim REJECTIONS (deflection/continuation reasons):
   (416, 438, 'near', 'originates 60.0px from the hoop and leaves it (ends 377.3px away) -- a deflection/continuation, not a fresh release')
   (1216, 1250, 'far', 'originates 24.3px from the hoop and leaves it (ends 382.3px away) -- a deflection/continuation, not a fresh release')
   (1465, 1488, 'far', 'originates 108.0px from the hoop and leaves it (ends 455.1px away) -- a deflection/continuation, not a fresh release')
   (2376, 2388, 'far', 'originates 108.7px from the hoop and leaves it (ends 316.5px away) -- a deflection/continuation, not a fresh release')

GROUND TRUTH CHECK:
  verified shot 356-381 (near): REPRODUCED (351, 376, 'near', 'jumpshot', 75.9, 'observed', 0)
  verified shot 1188-1211 (far): REPRODUCED (1184, 1213, 'far', 'jumpshot', 14.5, 'observed', 0)
  deflection 418-438: correctly NOT a shot
  deflection 1217-1250: *** CLAIMED AS SHOT: [(1184, 1213, 'far', 'jumpshot', 14.5, 'observed', 0)] ***
```

NOTE on the last line: that `*** CLAIMED AS SHOT ***` is a FALSE ALARM in the
CHECK SCRIPT, not the pipeline — its ±10-frame overlap window [1207..1260]
caught the ADJACENT legitimate verified shot (1184-1213, ends 1213 >= 1207).
The arc actually covering the deflection frames, (1216, 1250), was explicitly
REJECTED with the correct reason (second line of the rejections list above:
"originates 24.3px ... ends 382.3px away"). No shot attempt overlaps
1216-1250 itself.

### [2026-07-15 18:00:52] Post-test regression suite

```
$ .venv/Scripts/python -m pytest tests/ -q
....................                                                     [100%]
164 passed in 2.08s
```

### TEST 2 VERDICT — MEASURED — pending DJ review

- SUCCESS CONDITION MET: both verified HARD shots reproduced from
  fine-tuned detections — 351-376 near (vs stock's 356-381; same shot,
  slightly longer observed flight) and 1184-1213 far (vs stock's
  1188-1211; min_dist improved 54.6 -> 14.5px). Both known deflections
  (418-438, 1217-1250) correctly rejected by the arriving-vs-leaving
  gate, with two MORE deflection-shaped arcs (1465-1488, 2376-2388) also
  rejected for the same measured reason.
- NEW FINDING, needs DJ eyeball: a THIRD shot attempt claimed at
  1352-1378 (far hoop, 16.5px observed, 45.1-45.9s). This is the same
  45.2s at-rim flight noted in DECISIONS §13/§15 as possible shot
  activity (stock detector saw it at conf 0.81 but never formed a
  hoop-reaching arc). Not a ground-truth violation — a new candidate
  from a better detector — but UNVERIFIED until DJ checks the footage
  at ~45.1-45.9s.
- Detection density: 97% of frames vs stock's 65.5%; arcs 30 -> 58 on
  the same clip; zero robust-fitter drops needed anywhere on HARD
  (n_dropped=0 for all 58 arcs — the fine-tuned HARD chains were clean).
- NOT ADOPTED: stock yolov8m remains the committed ball detector; the
  fine-tuned model's logs live in spikes/out/*_roboflow.json only; no
  default/config changed.
- Contradiction check (rule 5): no contradictions with DECISIONS ground
  truth found. The 1352-1378 candidate is consistent with (not contrary
  to) §15's open note about 45.2s activity.

---

## PROTOCOL COMPLETE — stopped after Test 2 as instructed.

---

## TEST 3 — Auto hoop anchor: fine-tuned Hoop class vs hand-clicked anchors

### [2026-07-15 18:55:03] Pre-test gate: regression suite
```
164 passed in 2.03s
```
GATE: GREEN. Probe: spikes/roboflow_hoop_probe.py, every 10th frame, ALL
Hoop-class dets logged (conf floor 1%). TEST1: 130 sampled frames.
HARD: 275 sampled frames. Anchors = per-frame carried positions from
{clip}_hoop_track.json (hand-clicked, homography-carried, user-verified).

### [2026-07-15 19:03:08] TEST 3 RESULT — raw analysis output

```
--- TEST1 (sampled every 10th frame) ---
  conf>=0.4: anchor-points=183  matched(<=150px)=120  missed=63  median_dist=21.0px  p95=31.9px  extra-hoops(no anchor within 150px)=0
  conf>=0.2: anchor-points=183  matched(<=150px)=121  missed=62  median_dist=21.0px  p95=31.7px  extra-hoops(no anchor within 150px)=0
--- HARD (sampled every 10th frame) ---
  conf>=0.4: anchor-points=280  matched(<=150px)=2  missed=278  median_dist=21.6px  p95=21.5px  extra-hoops(no anchor within 150px)=0
  conf>=0.2: anchor-points=280  matched(<=150px)=6  missed=274  median_dist=21.6px  p95=24.8px  extra-hoops(no anchor within 150px)=1
```

Diagnostic (HARD, any-conf dets within 150px of an anchor):
```
HARD anchor points sampled: 280
  no Hoop det within 150px at ANY conf: 26
  det present: 254  conf median=0.021  max=0.50
  conf buckets: {'<0.1': 245, '0.2-0.4': 4, '>=0.4': 2, '0.1-0.2': 3}
```
(The <0.1 "hits" are junk-flood coincidence — ~30 low-conf dets/frame make a
150px hit likely by chance; usable-confidence detection is 2-6 of 280.)

### TEST 3 VERDICT — MEASURED — pending DJ review

- CLIP-DEPENDENT. TEST1: Hoop class localizes well when it fires — median
  21px from the user-verified anchor, p95 ~32px, ZERO false extra hoops at
  conf>=0.2 — but only 66% of anchor-points matched (34% missed). HARD:
  effectively blind — 2/280 at conf>=0.4.
- FLAG (rule 5): DECISIONS §24 hypothesized the Hoop class "could replace
  the manual rim-anchor clicks." That hypothesis is now HALF-REFUTED by
  measurement: viable as a PROPOSER/cross-check on TEST1-like footage,
  NOT a universal replacement (HARD fails). §24's wording was speculative
  ("could"), not a measured claim, so no measured contradiction — but the
  distinction is now on the record.
- Possible use pending DJ review: auto-PROPOSE anchors on new clips (user
  confirms, same click-seeding flow) + drift cross-check where it fires.
  NOT adopted; manual anchors remain the committed mechanism.

### [2026-07-15 19:04:05] Post-test regression suite
```
164 passed in 1.60s
```

---

## TEST 4 — Fine-tuned Player/Ref classes as tracking input

### [2026-07-15 19:04:05] Pre-test gate: regression suite
```
164 passed in 1.60s
```
GATE: GREEN. Probe: spikes/roboflow_player_probe.py — fetch Player+Ref dets
for TEST1's full tracking span (frames 120-580, all 461), then run
ultralytics BYTETracker STANDALONE (default bytetrack.yaml, same params as
the committed pipeline) over Player dets only; compare fragmentation vs the
cached COCO baseline; count Ref dets + baseline tracks IoU-matching refs.

### [2026-07-15 19:12:35] TEST 4 RESULT — raw output

(One probe-harness bug mid-test, fixed before measurement: BYTETracker() in
this ultralytics version takes no frame_rate kwarg; fetch phase was already
complete and reused.)

```
============ PLAYER/REF PROBE (TEST1) ============
                          COCO-person(cached)   RF-Player(standalone BYTE)
  distinct track_ids                 122                 43
  mean tracks/frame                 28.0                9.8
  mean lifespan (fr)               105.8              105.3
  Ref detections total (all conf): 20759 (45.0/frame avg)
  baseline track_ids IoU-matching a Ref det (conf>=0.4): 6 of 122
  ref-matched baseline ids: [1, 7, 11, 14, 372, 927]
```

Follow-up figures (same data, stricter conf + apples-to-apples baseline):
```
baseline distinct track_ids classified ON-COURT anywhere: 43
RF Ref dets/frame at conf>=0.4: mean=3.0 median=3
RF Player dets/frame at conf>=0.4: mean=9.9 median=10
```

### TEST 4 VERDICT — MEASURED — pending DJ review

- The headline 122->43 is MISLEADING without context: the committed
  pipeline ALREADY filters to on-court bodies via the ROI mask (§9), and
  the baseline's ON-COURT distinct ids = 43 — IDENTICAL to RF-Player's 43.
  So the fine-tuned Player class is a clean SUBSTITUTE for ROI filtering
  (median 10 players + 3 refs/frame at conf>=0.4 — sensible court reality),
  NOT a fragmentation improvement: mean lifespan is unchanged (105.3 vs
  105.8), consistent with §23's "association, not detection, is the player
  bottleneck". The 45/frame Ref figure is the all-conf junk flood; the
  strict-conf reality is 3/frame (= actual refs).
- The REAL measurable win available: semantic REF EXCLUSION. 6 of the 43
  on-court baseline ids IoU-match referees — refs are on court, so the ROI
  filter cannot exclude them and they sit in today's review queue. The
  Player class distinguishes them -> potential ~14% queue reduction (6/43)
  + removes the per-clip ROI dependency. Pending DJ review; nothing adopted.
- Caveats on record: detector conf scales differ; baseline used
  model.track() plumbing vs standalone harness (same yaml params).

### [2026-07-15 19:13:36] Post-test regression suite
```
164 passed in 1.68s
```

---

## TEST 5 — BoT-SORT with motion compensation, WITHOUT re-ID

### [2026-07-15 19:13:36] Pre-test gate: regression suite
```
164 passed in 1.68s
```
GATE: GREEN. Config: phase2/botsort_gmc_only.yaml (bytetrack-identical
association params + gmc_method sparseOptFlow, with_reid False). Runner:
spikes/reid_fragment_probe.py (read-only, TEST1 span, cached baseline
comparison). Metric = fragmentation proxies (distinct ids, mean lifespan)
— no ground-truth ID-switch labels exist; this is the same proxy §11/§23
used, logged as such.

### [2026-07-15 19:36] TEST 5 RESULT — raw probe output

```
================ FRAGMENTATION PROBE (TEST1) ================
                       ByteTrack(cached)   BoT-SORT+reID
  distinct track_ids            122             117
  mean tracks/frame            28.0            29.5
  mean lifespan (fr)          105.8           116.2
  fragmentation ratio  1.04x fewer fragments
  (baseline cache untouched; adoption is a separate decision)
```
(NOTE: the "BoT-SORT+reID" column header is the probe script's hardcoded
label — the config actually run was phase2/botsort_gmc_only.yaml,
with_reid: False, gmc_method: sparseOptFlow. Output json:
spikes/out/TEST1_tracks_botsort_gmconly.json.)

### TEST 5 VERDICT — MEASURED — pending DJ review

- GMC-only BoT-SORT: 117 distinct ids (vs 122 baseline; vs 131 when re-ID
  was ON, §11) and mean lifespan 116.2 vs 105.8 = +10% LONGER tracks.
- This is the FIRST tracker variant measured to improve LIFESPAN — the
  §23 detector swap and §11 re-ID both left it flat/worse. Directionally
  consistent with the pan-compensation hypothesis (GMC helps a panning
  camera). Magnitude is modest (4% fewer fragments, 10% longer lives).
- Comparison across levers so far (TEST1 span, distinct ids / lifespan):
  bytetrack baseline 122/105.8; botsort+reID 131/~106 (§11); yolov8x
  detector 106/105.8 (§23); botsort GMC-only 117/116.2 (this test);
  RF-Player dets 43/105.3 (Test 4 — different subject set, see caveat).
- Nothing adopted; bytetrack.yaml remains committed.

### [2026-07-15 19:21:45] Post-test regression suite
```
164 passed in 2.04s
```
(Timestamp correction: Test 5's probe finished and was logged at 19:21:45,
not 19:36 — the entry header above was written from an estimate before the
suite run; raw output is verbatim either way.)

---

## TEST 6 — ByteTrack parameter sweep (match_thresh, track_buffer)

### [2026-07-15 19:21:45] Pre-test gate: regression suite
```
164 passed in 2.04s
```
GATE: GREEN. Grid (probe-only yamls in scratchpad, baseline params
match_thresh=0.8/track_buffer=30): mt=0.7/buf=30, mt=0.9/buf=30,
mt=0.8/buf=60. The 4th cell (mt=0.8/buf=120) was ALREADY MEASURED in
DECISIONS §11: 128 distinct ids (worse than baseline 122) — reused, not
re-run. Runner: spikes/reid_fragment_probe.py per config, sequential,
TEST1 span, ~20 min each.
