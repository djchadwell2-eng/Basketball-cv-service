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

### [2026-07-15 ~20:20] TEST 6 RESULT — raw probe outputs (3 runs, sequential)

(Column header "BoT-SORT+reID" is the probe script's hardcoded label; the
configs actually run are ByteTrack variants as titled. Baseline params:
match_thresh=0.8, track_buffer=30.)

match_thresh=0.7, track_buffer=30:
```
                       ByteTrack(cached)   [mt=0.7]
  distinct track_ids            122             256
  mean tracks/frame            28.0            25.8
  mean lifespan (fr)          105.8            46.5
```

match_thresh=0.9, track_buffer=30:
```
                       ByteTrack(cached)   [mt=0.9]
  distinct track_ids            122              93
  mean tracks/frame            28.0            28.9
  mean lifespan (fr)          105.8           143.1
```

match_thresh=0.8, track_buffer=60:
```
                       ByteTrack(cached)   [buf=60]
  distinct track_ids            122             120
  mean tracks/frame            28.0            28.0
  mean lifespan (fr)          105.8           107.6
```

4th grid cell reused from DECISIONS §11 (not re-run): match_thresh=0.8,
track_buffer=120 -> 128 distinct ids (worse than baseline), lifespan flat.

### TEST 6 VERDICT — MEASURED — pending DJ review

- match_thresh is a LIVE lever; track_buffer is DEAD (60 and 120 both ~flat).
  mt=0.7 (stricter association): catastrophic — 2.1x MORE fragments,
  lifespan halved. mt=0.9 (looser): the LARGEST tracking gain of any lever
  measured to date — 122 -> 93 distinct ids (-24%) and lifespan 105.8 ->
  143.1 (+35%), beating yolov8x (§23: 106/flat) and GMC-only (Test 5:
  117/116.2).
- CRITICAL SAFETY CAVEAT (why this is NOT adoptable from these numbers
  alone): a looser match threshold can also MERGE DIFFERENT PLAYERS into
  one track (ID switches). Fragmentation metrics CANNOT see that failure —
  a wrongly-merged track looks "better" on ids/lifespan while being
  confidently-wrong for identity, exactly the §7a/§8 splice class the
  purity machinery exists to catch. Any adoption path must first run the
  purity checks (§8 detector B / disputed frames) + an eyeball pass on
  mt=0.9 tracks. NOT done here (outside this test's scope), NOT adopted.
- Rule-5 note: DECISIONS §11's guidance "fragmentation levers now: footage
  zoom/4K, then span-prioritized queue cutoff" was written after testing
  only track_buffer; match_thresh had never been measured. This result
  NUANCES (does not contradict) that entry — no prior match_thresh
  measurement existed to contradict.

### [2026-07-15 19:43:53] Post-test regression suite
```
164 passed in 2.15s
```

---

## TEST 7 — Third-party trackers (OC-SORT, StrongSORT via boxmot)

### [2026-07-15 19:43:53] Pre-test gate: regression suite
```
164 passed in 2.15s
```
GATE: GREEN. Isolation: boxmot 19.0.0 installed in a SEPARATE venv
(.venv-boxmot) to protect the main env — main suite verified green after
install. Phase A (main venv): dump raw yolov8m person dets for the TEST1
span at conf>=0.1 (= bytetrack.yaml track_low_thresh) so all trackers see
identical detections. Phase B (.venv-boxmot): OC-SORT (motion-only) +
StrongSORT (appearance re-ID, default model) fed those dets; fragmentation
stats vs cached ByteTrack baseline (122 / 28.0 / 105.8).

### [2026-07-16 18:14] TEST 7 RESULT — raw output

Phase A (main venv): raw yolov8m person dets, TEST1 span, conf>=0.1 ->
spikes/out/TEST1_rawdets_person.json (461 frames).

Phase B (.venv-boxmot, boxmot 19.0.0), raw:
```
  RESULT ocsort: distinct_ids=107  mean_tracks/frame=26.6  mean_lifespan=114.6
  RESULT strongsort: NOT MEASURED — ModuleNotFoundError: No module named 'boxmot.data'
  (baseline ByteTrack cached: 122 / 28.0 / 105.8 — same span,
   same detector family; dets here re-generated at conf>=0.1)
```

Probe-harness issues hit + handled during this test (raw trail):
1. boxmot's base tracker requires an img ndarray even for motion-only
   trackers -> harness now always passes frames.
2. First phase-B run lost OC-SORT's completed result when StrongSORT
   raised (results held in memory; launcher's own `| tail` also truncated
   output) -> harness rewritten with per-tracker isolation + immediate
   stats printing; OC-SORT re-run cleanly.
3. StrongSORT is UNRUNNABLE in boxmot 19.0.0 as installed: its ReID
   loader import chain crosses `boxmot.data`, a module MISSING from the
   wheel (upstream packaging bug). Timeboxed one fix attempt: pip upgrade
   resolves back to 19.0.0 in this env (20/21/22 requirements
   incompatible) -> NOT MEASURED, reason logged.

### TEST 7 VERDICT — MEASURED (OC-SORT) / NOT MEASURED (StrongSORT) — pending DJ review

- OC-SORT (motion-only): 107 distinct ids / 114.6 mean lifespan vs
  baseline 122 / 105.8 — a modest improvement, in the same league as
  GMC-only BoT-SORT (Test 5: 117 / 116.2) and yolov8x (§23: 106 / 105.8),
  and WELL BEHIND ByteTrack mt=0.9 (Test 6: 93 / 143.1, caveats there).
- StrongSORT: not measured (upstream packaging bug, above). Information
  loss judged small: §11 already measured appearance re-ID as
  COUNTERPRODUCTIVE on this footage (identical uniforms; 122->131), and
  StrongSORT's distinguishing feature is exactly appearance re-ID.
- Nothing adopted. Isolated .venv-boxmot left in place for potential
  future re-measurement (e.g., when upstream fixes the wheel).

---

## PROTOCOL COMPLETE — Tests 3-7 done, stopped as instructed.

Cross-test tracking summary (TEST1 span, distinct ids / mean lifespan,
baseline ByteTrack = 122 / 105.8; ALL pending DJ review, NOTHING adopted):
```
  ByteTrack mt=0.9        93 / 143.1   (Test 6 — best, but ID-switch safety caveat unresolved)
  yolov8x detector       106 / 105.8   (§23)
  OC-SORT                107 / 114.6   (Test 7)
  BoT-SORT GMC-only      117 / 116.2   (Test 5)
  ByteTrack buf=60       120 / 107.6   (Test 6)
  BoT-SORT + re-ID       131 / ~106    (§11 — worse)
  ByteTrack mt=0.7       256 /  46.5   (Test 6 — much worse)
  RF-Player dets          43 / 105.3   (Test 4 — different subject set: on-court only,
                                        = baseline's own on-court 43; not comparable directly)
```

---

## TEST 8 — LOCALLY-TRAINED fine-tuned weights vs hosted API (Milestone 1: own the ball model)

Success condition: weights trained from scratch on the same public dataset
(basketball-players-fy4c2 v25) pass the SAME gate TEST 2's hosted model
passed — reproduce HARD's 2 verified shots (356-381 near, 1188-1211 far),
reject its deflections (418-438, 1217-1250) — with zero API dependency.

### [2026-07-17] Setup (raw trail)

- RunPod RTX 4090 pod (persistent Network Volume). Dataset v25 downloaded
  (1140 train / 32 valid images). Trained yolov8m.pt @ imgsz=1280,
  100 epochs, AutoBatch (settled at batch=3). ~35s/epoch, ~1h total.
- Final val (pod, 32 images): all-class mAP50 0.877. Per-class Ball is the
  WEAKEST: P=1.0 R=0.567 mAP50=0.642 (18 instances — small sample, flagged
  honestly). Hoop 0.906, Player 0.903.
- Weights pulled to models/ball_finetuned_v1.pt (52,109,074 bytes, md5-safe
  transfer). Loads under the project .venv; class list Ball[0]..Time
  Remaining[8].
- spikes/ball_spike.py: ball-class id now resolved from the loaded model's
  own names ("ball"/"sports ball"), hardcoded COCO 32 kept as fallback —
  stock behavior unchanged (suite green pre/post: 164 passed).
- Detection run: pod GPU over the EXACT extracted subclip the local spike
  uses (HARD_span_0_2746.mp4, md5 b73fd420... verified identical after an
  interrupted-then-resumed upload), same conf=0.05/imgsz=1280/stream loop
  as ball_spike.py (pod script mirrors it minus overlay). Local CPU run of
  the identical command (spikes/ball_spike.py HARD 0 2746 1280
  models/ball_finetuned_v1.pt) completed later same day: coverage 54.3%
  identical, and the harness output MATCHES the GPU log exactly — same 3
  attempts (11.8/35.1/32.9px), same 2 rejections, same ground-truth
  passes. GPU-vs-CPU consistency cross-check: PASSED.
- Analysis harness: spikes/local_weights_check.py — identical chain to
  TEST 2's analysis (conf>=0.10 filter -> ball_trajectory chains/physics
  -> shot_attempts classifier, both hoops). VALIDATED first: run over the
  saved HOSTED log it reproduces TEST 2's published numbers EXACTLY (58
  arcs; 3 attempts 75.9/14.5/16.5px; 4 rejections), and its overlap check
  correctly clears TEST 2's known ±10-window false alarm on 1217-1250.

### [2026-07-17 18:05] TEST 8 RESULT — raw harness output (local weights, GPU log)

```
[check] HARD log=HARD_ball_spike_log_ball_finetuned_v1_gpu.json model=ball_finetuned_v1.pt frames=2746
[check] ball seen in 1438/2746 frames (52.4%) at conf>=0.1 (1575/1692 dets kept)

frames analyzed: 2746   arcs: 44

SHOT ATTEMPTS (start,end,hoop,type,min_dist,arrival,n_dropped):
   (353, 382, 'near', 'jumpshot', 11.8, 'observed', 0)
   (1187, 1212, 'far', 'jumpshot', 35.1, 'observed', 0)
   (1352, 1375, 'far', 'jumpshot', 32.9, 'observed', 0)

near-rim REJECTIONS (deflection/continuation reasons):
   (422, 438, 'near', 'originates 101.8px from the hoop and leaves it (ends 375.6px away) -- a deflection/continuation, not a fresh release')
   (1227, 1250, 'far', 'originates 83.8px from the hoop and leaves it (ends 382.6px away) -- a deflection/continuation, not a fresh release')

GROUND TRUTH CHECK:
  verified shot 356-381 (near): REPRODUCED (353, 382, 'near', 'jumpshot', 11.8, 'observed', 0)
  verified shot 1188-1211 (far): REPRODUCED (1187, 1212, 'far', 'jumpshot', 35.1, 'observed', 0)
  deflection 418-438: correctly NOT a shot
  deflection 1217-1250: correctly NOT a shot
```

### [2026-07-17 18:06] Post-test regression suite

```
164 passed in 3.10s
```

### TEST 8 VERDICT — MEASURED — pending DJ review

- SUCCESS CONDITION MET: both verified HARD shots REPRODUCED and both
  known deflections correctly rejected — same gate TEST 2's hosted model
  passed, now with locally-owned weights and ZERO API calls.
- Same third candidate attempt claimed at 1352-1375 (far) as the hosted
  model's 1352-1378 — the ~45.1-45.9s activity still awaiting DJ's
  eyeball from TEST 2. Independent training run converging on the same
  unverified candidate is corroborating, not confirming.
- Character difference vs hosted, on the record: local model is QUIETER —
  52.4% frame coverage vs hosted 97%/stock 65.5%, but 93% of its dets sit
  above conf 0.10 (1575/1692) vs hosted's 48% (3548/7379). Matches its
  val profile (Ball P=1.0/R=0.567): high precision, lower recall. Arcs
  44 vs hosted 58. §13/§20's raw-coverage-anti-correlates lesson holds
  again — the gate outcome, not coverage, is the metric.
- Near-shot min_dist: near shot IMPROVED (11.8px vs hosted 75.9 / stock
  19.4); far shot 35.1px (hosted 14.5 / stock 54.6).
- LIMITATION on the record: gate = 2 verified shots + 2 deflections on
  ONE clip. TEST1's shots (esp. layups, the fine-tune's raison d'être
  per §24) NOT yet run against these weights. Hosted-vs-local per-frame
  agreement beyond the gate not measured.
- NOT ADOPTED: stock yolov8m remains the committed ball detector;
  models/ball_finetuned_v1.pt + the _gpu log are artifacts only. Adoption
  (and any lifecycle question: hosted API retirement, GPU-vs-CPU runtime
  for production) is DJ's call.

---

## TEST 9 — Local weights on TEST1 (layup coverage) + coverage-gap diagnosis

Success condition (exploratory, DJ-directed "how do we close 52%->97%"):
measure the local-vs-hosted gap where it MATTERS (flight frames), and
whether local weights reproduce TEST1's 5 user-verified attempts (esp.
the 3 layups, the fine-tune's raison d'etre per §24).

### [2026-07-17 18:30] Gap diagnosis on HARD (raw analysis output)

Of 982 frames where hosted(conf>=0.10) sees a ball and local doesn't:
```
both see ball: 1384  hosted-only: 982  local-only: 54  neither: 326
hosted ball size when BOTH see it: median 27px; when ONLY hosted: median 27px
local conf when it DOES see: median 0.84
hosted conf in hosted-only frames: median 0.19 (p25 0.13, p75 0.38)
local near-miss (det at 0.05-0.10) in those frames: 44/982
hosted-only frames inside a hosted ARC (flight): 150/982
hosted-only frames OUTSIDE any arc (dribble/held/junk): 832/982
```
READING: 85% of the raw-coverage gap is NON-FLIGHT (dribble/held/junk at
hosted conf ~0.19) — invisible to the shot pipeline by design. NOT a
small-ball problem (same 27px in hits and misses). The product-relevant
gap = 150 flight frames (44 vs 58 arcs). §13/§20's lesson holds: raw
coverage is the wrong target; FLIGHT coverage is the real one.

### [2026-07-17 18:38] TEST 9 RESULT — TEST1 0-605, local weights (raw)

```
[check] ball seen in 272/605 frames (45.0%) at conf>=0.1 (291/319 dets kept)
frames analyzed: 605   arcs: 10

SHOT ATTEMPTS (start,end,hoop,type,min_dist,arrival,n_dropped):
   (318, 328, 'far', 'jumpshot', 98.7, 'observed', 0)
   (581, 589, 'near', 'layup', 18.0, 'observed', 0)

GROUND TRUTH CHECK:
  verified shot 55-74 (far): *** MISSED ***
  verified shot 165-184 (far): *** MISSED ***
  verified shot 242-250 (far): *** MISSED ***
  verified shot 315-327 (far): REPRODUCED (318, 328, 'far', 'jumpshot', 98.7, 'observed', 0)
  verified shot 581-592 (near): REPRODUCED (581, 589, 'near', 'layup', 18.0, 'observed', 0)
```

Per-attempt detection density (frames with a conf>=0.10 ball det):
```
shot A  (55-74,  20f): local 11, hosted 20  -> arc 58-70 formed but truncated, failed hoop gate
layup 1 (165-184, 20f): local  5, hosted 20  -> below MIN_FIT_LEN evidence, no arc
layup 2 (242-250,  9f): local  5, hosted  9  -> below MIN_FIT_LEN evidence, no arc
shot B  (315-327, 13f): local  8, hosted 11  -> caught (318-328)
layup 3 (581-592, 12f): local  9, hosted 10  -> caught (581-589, layup, 18.0px)
```

### TEST 9 VERDICT — MEASURED — pending DJ review

- CLIP-DEPENDENT, same shape as TEST 3's hoop finding: local weights
  match the hosted model on HARD (TEST 8) but reproduce only 2/5 verified
  TEST1 attempts. Root cause is detection DENSITY during flight: local
  sees ~half the frames per attempt; hosted sees ~all. TEST1's smaller
  ball (24px median vs HARD 39px) is where the local model's lower recall
  (val Ball R=0.567) bites. The physics gate's >=8-point evidence bar is
  working as designed — the failure is upstream (too few sightings), not
  the gate.
- Chasing hosted's 97% RAW coverage is the wrong goal (85% of that gap
  is non-flight mumble at conf ~0.19). The right goal: flight-frame
  recall on small/far balls. Exact miss frames are now enumerable from
  the logs on both clips = a targeted Milestone-2 harvest list.
- Levers, unmeasured, for DJ: (a) training recipe (longer/bigger model/
  small-object augmentation; v1 was a default-recipe first pass at
  batch=3) — zero labeling effort, unknown ceiling; (b) Milestone 2
  own-footage labels aimed at the enumerated miss frames — targets the
  product's actual footage distribution, needs a DJ labeling session;
  (c) both, (a) then (b) from the better base. Roboflow's hosted model
  is likely NOT vanilla yolov8m + default recipe, so (a) alone may not
  fully close it.
- Suites green all session (164 pre/mid/post; 18:39:26 bookend).
- NOT ADOPTED, nothing changed in committed defaults.

---

## TEST 10 — Recipe-retrained v2 weights (Option A of the DJ-approved A-then-B plan)

Success condition: HARD gate stays perfect (2 verified shots + 2
deflections) AND TEST1 improves on v1's 2/5 (the 2 missed layups are the
target — the fine-tune's raison d'etre).

### [2026-07-17 ~19:50-21:35] Setup (raw trail)

- Recipe v2 vs v1's defaults: yolov8l.pt (vs m), epochs 200 + patience 75
  (vs 100), cos_lr=True, scale=0.7 (stronger size augmentation for small
  balls), same imgsz=1280 / batch=-1 / dataset v25. New 4090 pod (prior
  pod's host lost its GPUs while Stopped — second occurrence; also hit a
  cuda>=12.8-template-vs-host-driver mismatch, fixed by picking the
  PyTorch 2.4/cu124 template). Early-stopped at epoch 121 (1.67h).
- Pod val (32 imgs): all mAP50 0.879; Ball P=1.0 R=0.541 mAP50=0.653 —
  statistically indistinguishable from v1's val Ball (R=0.567/0.642).
  VAL DID NOT PREDICT THE CLIP RESULT (below) in either direction —
  18-instance val set is too small to steer by; the clip gates are the
  instrument. Weights -> models/ball_finetuned_v2.pt (87,734,579 bytes).
- Detection runs: HARD on pod GPU (same uploaded md5-verified subclip,
  same mirrored script; doc's "model" field still says v1 — cosmetic sed
  miss, actual weights are v2 per run dir), TEST1 local CPU via
  spikes/ball_spike.py TEST1 0 605 1280 models/ball_finetuned_v2.pt.

### [2026-07-17 ~22:05] TEST 10 RESULT — raw harness output

HARD (v2, GPU log):
```
[check] ball seen in 1315/2746 frames (47.9%) at conf>=0.1 (1453/1639 dets kept)
frames analyzed: 2746   arcs: 48
SHOT ATTEMPTS:
   (351, 375, 'near', 'jumpshot', 88.5, 'observed', 0)
   (401, 415, 'near', 'layup', 1.6, 'observed', 0)
   (1188, 1213, 'far', 'jumpshot', 17.7, 'observed', 0)
   (1352, 1381, 'far', 'jumpshot', 31.3, 'observed', 0)
near-rim REJECTIONS:
   (416, 438, 'near', 'originates 59.6px ... ends 373.1px away')
   (1216, 1250, 'far', 'originates 25.3px ... ends 381.1px away')
GROUND TRUTH CHECK:
  verified shot 356-381 (near): REPRODUCED (351, 375)
  verified shot 1188-1211 (far): REPRODUCED (1188, 1213)
  deflection 418-438: correctly NOT a shot
  deflection 1217-1250: correctly NOT a shot
```

TEST1 (v2, CPU log):
```
[check] ball seen in 309/605 frames (51.1%) at conf>=0.1 (337/422 dets kept)
frames analyzed: 605   arcs: 13
SHOT ATTEMPTS:
   (58, 70, 'far', 'jumpshot', 118.1, 'extrapolated', 0)
   (164, 184, 'far', 'layup', 15.3, 'observed', 0)
   (236, 250, 'far', 'layup', 27.2, 'observed', 0)
   (581, 589, 'near', 'layup', 18.4, 'observed', 0)
near-rim REJECTIONS:
   (103, 110, 'far', ...)   (188, 202, 'far', ...)
GROUND TRUTH CHECK:
  verified shot 55-74 (far): REPRODUCED (58, 70, extrapolated 118.1px)
  verified shot 165-184 (far): REPRODUCED (164, 184, layup, 15.3px)
  verified shot 242-250 (far): REPRODUCED (236, 250, layup, 27.2px)
  verified shot 315-327 (far): *** MISSED ***
  verified shot 581-592 (near): REPRODUCED (581, 589, layup, 18.4px)
```

Per-attempt flight-frame density (frames with conf>=0.10 det / span):
```
           shotA   layup1  layup2  shotB   layup3
  v1       11/20    5/20    5/9     8/13    9/12
  v2       11/20   20/20    9/9     4/13    8/12
  hosted   20/20   20/20    9/9    11/13   10/12
```

### TEST 10 VERDICT — MEASURED — pending DJ review

- SUCCESS CONDITION MET: HARD stays perfect; TEST1 2/5 -> 4/5. Both
  target layups RECOVERED with hosted-equal flight density (20/20, 9/9).
- REGRESSION inside the win, on the record: shot B (315-327), caught by
  v1 (8/13 density), drops to 4/13 under v2 -> below MIN_FIT_LEN -> lost.
  v1 and v2 have COMPLEMENTARY blind spots (union = 5/5). Options, all
  unmeasured: (a) Milestone-2 harvest now targets shot B's span + HARD
  misses; (b) two-model union at 2x compute (echoes §20's deferred
  multi-res ensemble; same warning applies); (c) accept 4/5 from one
  model. DJ's call after B.
- Two NEW unverified candidates flagged at log time; ONE RESOLVED SAME
  SESSION BY DJ EYEBALL: HARD 401-415 near-hoop 'layup' 1.6px
  (~13.4-13.8s) is REFUTED — DJ confirms the play was rebound -> dish
  out to the perimeter, NO shot attempt. v2 therefore carries ONE
  CONFIRMED FALSE-POSITIVE shot claim on HARD (a rebound/dish flight
  arriving near the rim reads as a layup to the arriving-vs-leaving
  gate). Adoption math for v2-alone is now: HARD 2/2 verified + 1 FP;
  TEST1 4/5. The FP span (395-445) goes on the Milestone-2 harvest list
  (correct labels through the rebound/dish may resolve the chain into a
  continuation the gate refuses; not guaranteed — re-measure after B).
  Still open for DJ eyeball: shotA's marginal pass (118.1px extrapolated
  vs 125 radius) and the twice-claimed 1352-13xx far candidate
  (~45.1-45.9s, from TEST 2). TEST1's two new rejections look right
  (188-202 = layup-1's rebound continuation, correctly refused).
- shotA density note: v2 caught it at the SAME 11/20 density v1 missed
  it at — the arc formed with different member frames; small-sample
  fragility, not a robustness claim.
- Suite green (164, 22:11:05). NOT ADOPTED: stock yolov8m remains the
  committed default; v1/v2 both artifacts in models/ pending DJ review.

---

## TEST 11 — v3: DJ's own-footage labels (230 imgs) merged into Milestone-2, retrained from v2

Success condition: recover TEST1 shot B (v2's regression) without losing
anything v2 had, AND resolve (or at least not worsen) the DJ-refuted
HARD rebound/dish false positive.

### [2026-07-22] Setup (raw trail)

- New pod (prior one's host lost GPU stock while Stopped, again — third
  occurrence overall; template picked correctly this time, PyTorch
  2.4/cu124, no driver mismatch).
- DJ labeled all 230 harvested frames in Roboflow project
  "my-footage-ball" (single class "ball"), generated Version 1 (230
  images, no augmentation), took ~2h hands-on time (not the ~30min
  originally estimated).
- Merge script (spikes-adjacent, pod-side): downloaded DJ's v1 via
  RF_KEY, remapped its single "ball" class id (0) onto v25's Ball id (0,
  same value, confirmed not assumed), copied all 230 into
  dataset/train/{images,labels} prefixed own_ (own footage has no
  separate held-out split — v25's existing 32-image valid set remains
  the only validation, by design; own labels are pure signal-add).
  Final counts: train 1140->1370, valid 32 (unchanged).
- Training: model=/workspace/runs/ball_finetune_v2/weights/best.pt (NOT
  stock — continues from v2, keeping everything v2 already learned),
  same recipe knobs as v2 (epochs 150, patience 60, cos_lr, scale=0.7),
  early-stopped at epoch 131 (2.16h). Pod val (still the 32-img public
  set only): all mAP50 0.877; Ball P=0.99 R=0.556 mAP50=0.618 —
  val Ball recall UNCHANGED from v1/v2 (own-footage signal doesn't show
  up in a val set that contains none of it; expected, not a red flag).

### [2026-07-22 17:05-17:10] TEST 11 RESULT — raw harness output

HARD (v3, GPU log):
```
[check] ball seen in 2181/2746 frames (79.4%) at conf>=0.1 (3577/4278 dets kept)
frames analyzed: 2746   arcs: 57

SHOT ATTEMPTS:
   (351, 375, 'near', 'jumpshot', 88.2, 'observed', 0)
   (403, 415, 'near', 'layup', 0.9, 'observed', 0)
   (1177, 1214, 'far', 'jumpshot', 10.4, 'observed', 0)
   (1352, 1377, 'far', 'jumpshot', 15.6, 'observed', 0)
   (2234, 2250, 'far', 'layup', 58.5, 'observed', 0)

near-rim REJECTIONS:
   (416, 438, 'near', ...)   (1215, 1250, 'far', ...)   (2377, 2405, 'far', ...)

GROUND TRUTH CHECK (harness output, then hand-corrected below):
  verified shot 356-381 (near): REPRODUCED (351, 375)
  verified shot 1188-1211 (far): *** MISSED *** [HARNESS FALSE ALARM, see below]
  deflection 418-438: correctly NOT a shot
  deflection 1217-1250: correctly NOT a shot
```
HAND-VERIFIED CORRECTION: the "MISSED" far shot is a harness tolerance
bug, not a model failure. Raw per-frame dets 1184-1215 show a clean,
confident (0.68-0.80) detection run through the ENTIRE verified span
1188-1211; the claimed arc (1177, 1214) fully covers it. The check
script's REPRODUCED test requires both endpoints within +/-10 frames;
arc start 1177 vs verified 1188 = 11 frames, one over the cutoff. ACTUAL
RESULT: REPRODUCED (both HARD shots caught, both deflections rejected).
Harness bug logged, not yet fixed (widen tolerance or match-by-overlap
instead of both-endpoints — cosmetic, doesn't change any verdict to date).

TEST1 (v3, CPU log):
```
[check] ball seen in 415/605 frames (68.6%) at conf>=0.1 (597/699 dets kept)
frames analyzed: 605   arcs: 15

SHOT ATTEMPTS:
   (58, 77, 'far', 'jumpshot', 47.2, 'extrapolated', 0)
   (166, 184, 'far', 'layup', 14.0, 'observed', 0)
   (236, 250, 'far', 'layup', 25.9, 'observed', 0)
   (314, 327, 'far', 'jumpshot', 97.3, 'observed', 0)
   (571, 589, 'near', 'layup', 18.2, 'observed', 0)

near-rim REJECTIONS:
   (188, 202, 'far', 'originates 31.4px ... ends 132.3px away')

GROUND TRUTH CHECK:
  verified shot 55-74 (far): REPRODUCED
  verified shot 165-184 (far): REPRODUCED
  verified shot 242-250 (far): REPRODUCED
  verified shot 315-327 (far): REPRODUCED  <-- v2's regression, RECOVERED
  verified shot 581-592 (near): REPRODUCED
```

### [2026-07-22 17:13:26] Post-test regression suite

```
204 passed in 2.16s
```
(164 -> 204 reflects the shot-layer-into-run_clip integration done in a
parallel session 2026-07-19, not this test.)

### TEST 11 VERDICT — MEASURED — pending DJ review

- SUCCESS CONDITION PART 1 MET: TEST1 is now 5/5 -- full ground-truth
  recovery, no regressions, v1/v2's complementary-blind-spot problem is
  GONE (one model does it all). HARD coverage 79.4% (conf>=0.10), closing
  most of the gap to hosted's 86.2% (v1/v2 both sat ~48-52%).
- SUCCESS CONDITION PART 2 NOT MET: the DJ-refuted rebound/dish sequence
  (real play: rebound -> outlet pass, NOT a shot) is STILL claimed as a
  layup, and MORE confidently than v2 (0.9px min-dist vs v2's 1.6px).
  Own-footage labels did not fix this, and the harvest DID target this
  exact span (395-445) — meaning either DJ's labeling queue hadn't
  reached those specific frames yet (3/4 done at generate-time), or
  (more likely per the physics) THIS ISN'T A LABELING PROBLEM: a caught
  rebound genuinely, physically arrives at the rim, same as a made shot,
  so the arriving-vs-leaving classify_shot() gate (DECISIONS 25) cannot
  distinguish them on ball-position alone. More/better ball labels teach
  the DETECTOR to see the ball better; they cannot teach the CLASSIFIER
  a distinction the ball's position doesn't encode. Real fix candidates,
  unmeasured, for DJ: (a) rebound/board-contact signal (a shot that goes
  in or clean-misses doesn't bounce off iron the way a rebound-then-grab
  does -- may show in the trajectory's velocity/direction change);
  (b) player-signal cross-check (Milestone-2 player labels + tracking:
  a real shot has a release near a raised-arm shooter well before the
  arrival; a rebound has a player's hands already AT the rim) -- ties
  directly to player-tracker plan item 3, now with a concrete use case,
  not just fragmentation.
- NEW unverified candidate for DJ eyeball: HARD 2234-2250 far-hoop
  'layup' 58.5px (~74.5-75.0s) -- never claimed by v1/v2/hosted, real or
  detector noise from denser own-footage-informed detections. Confidence
  profile (0.1-0.66, noisier than the clean rebound/dish run) suggests
  worth a look but lower priority than the recurring 1352-1377 candidate.
- 1352-1377 candidate (far, ~45.1-45.9s) claimed AGAIN, 4th model in a
  row (hosted/v2/v3, v1 too per TEST 9 -- check) to independently flag
  it. Escalating from "worth a look" to "should actually be checked
  before shipping" given the consistency.
- Suite 204 green pre/post. NOT ADOPTED: stock yolov8m remains the
  committed default in run_clip's ball stage; v3 is the strongest
  candidate so far but the rebound/dish false-positive is a real
  correctness gap, not cosmetic -- do not adopt until resolved or
  explicitly accepted as a known limitation by DJ.

### [2026-07-22, same session] Follow-up investigation: is the rebound/dish separable by trajectory shape alone?

Hypothesis tested (no new labels needed -- pure analysis of existing
detection logs): a caught rebound might show a DIFFERENT velocity/accel
signature than a real shot even though both "arrive" at the rim.

Raw per-frame velocity (best-conf detection, HARD v3 log): the verified
near shot (351-385) shows a textbook smooth parabola -- monotonic
deceleration from ~18px/frame to ~0.5px/frame at the apex (f=366-369),
then smooth symmetric reacceleration. The disputed span (395-425) shows
~17 frames of near-CONSTANT 5-10px/frame motion (no apex, no
deceleration) followed by two erratic jumps (speed 38, then 52) --
consistent with a catch/hand-change discontinuity, not flight.

Fitted-arc comparison (the same accel_y/travel_y the classifier itself
computes), across the ONLY samples that exist to date:
```
                          accel_y   travel_y
Disputed (403-415)         0.655      67px
Verified HARD near shot    0.967     134px
Verified HARD far shot     0.848     177px
Verified TEST1 layup 1     1.230      72px
Verified TEST1 layup 2     1.343      40px
Verified TEST1 layup 3     1.069      51px
Known deflection 1         1.058     251px
Known deflection 2         1.159     351px
Known deflection 3         0.937      92px
```
- travel_y does NOT separate the disputed case (67px sits inside the
  real-layup range 40-72px) -- ruled out as a discriminator.
- accel_y is SUGGESTIVE: disputed (0.655) sits below every real shot
  measured (0.848-1.343) with a real gap. BUT known deflections (already
  a solved case via the arriving/leaving gate) sit at 0.937-1.159 --
  INSIDE the real-shot range, not low like the disputed case. So "low
  accel_y" is NOT a general non-shot signature (deflections disprove
  that) -- at most it might be specific to THIS play's soft-catch motion
  style. n=1 disputed example is not enough to tell a real pattern from
  noise.

VERDICT: measured, NOT adopted, NOT a fix -- one suggestive data point,
explicitly not generalizable on current evidence. Two live paths forward,
neither built: (a) harvest more near-rim non-shot examples (rebounds,
tips, deliberate catches) across both clips specifically to grow this
n past 1 and see if the accel_y gap holds or vanishes; (b) the
player-signal cross-check (Milestone player-labeling in progress ties
in here) -- structurally more promising since it targets WHY the motion
differs (a shooter's release vs a rebounder's hands already at the rim)
rather than hoping the symptom (accel_y) reliably tracks the cause.

### [2026-07-22, same session] Follow-up part 2: exhaustive near-rim arc sweep (path (a) above)

Reran classify_shot on EVERY arc in both clips (not just the known ones),
keeping only arcs that actually came within 200px of a hoop (i.e. the
ones the shot detector had to make a real call on) -- 12 on HARD, 8 on
TEST1, n=20 total near-rim judgment calls to date. Full table logged in
this session's raw output. Result: accel_y now cleanly separates into
two clusters with NO overlap:
```
CONFIRMED REAL shots/layups (7): 0.848, 0.967, 1.069, 1.188, 1.230, 1.343, 1.935
DISPUTED/suspected non-shot (2): 0.655 (rebound/dish, DJ-refuted)
                                  0.455 (HARD 2234-2250, the new
                                         unverified candidate from TEST 11)
```
Gap: every confirmed real shot is >=0.848; both disputed/suspected cases
are <=0.655 -- a clean floor with margin, same shape as every other gate
this project has built (origin gate, arrival gate, y-range gate). BUT
n=2 on the low side, both from the SAME clip (HARD), and one of the two
isn't even confirmed non-shot yet (2234-2250 awaits DJ eyeball -- it's
merely suspected because of this pattern, which is circular until
checked against real footage). The recurring 1352-1377 candidate (now
flagged by 4 model runs across TEST 2/8/10/11) has accel_y=1.388 --
squarely inside the REAL cluster, which is corroborating (not proof)
that it's a genuine shot worth DJ's eyeball, not detector noise.

VERDICT: STILL not adopted, still not a built gate -- but upgraded from
"one suggestive point" to "a real candidate gate pending one ground-truth
check." NEXT ACTION (cheap, no GPU/labeling needed): DJ eyeballs
HARD ~74.5-75.0s (frames 2234-2250) and confirms/denies it's a shot. If
CONFIRMED non-shot: n=2 clean floor becomes real evidence, worth
prototyping accel_y>=0.8 as a second gate on TOP of the existing
arriving/leaving gate (never replacing it) with a regression test using
these exact 9 measured points. If CONFIRMED real shot: the pattern breaks
(back to n=1, inconclusive), and the player-signal cross-check becomes
the only live path.

---

## TEST 12 — ByteTrack match_thresh=0.9 ID-switch eyeball check (Test 6's outstanding requirement)

Success condition: Test 6 measured match_thresh 0.8->0.9 as the single
biggest fragmentation win to date (122->93 ids, +35% lifespan) but flagged
it as NOT ADOPTABLE without an eyeball pass for ID switches (a looser
matcher can silently merge two different players into one track — a
failure fragmentation metrics cannot see). This test performs that pass.

### [2026-07-22] Setup

- phase2/bytetrack_mt09.yaml created (stock bytetrack.yaml + match_thresh
  0.9, everything else identical) -- the probe-only yaml TEST 6 used lived
  in scratchpad and no longer exists; recreated and RE-VERIFIED it
  reproduces TEST 6's exact numbers before trusting it further.
- spikes/reid_fragment_probe.py phase2/bytetrack_mt09.yaml
  spikes/out/TEST1_tracks_mt09.json (TEST1 span 120-581, CPU ~17 min).
- spikes/render_tracker_overlay.py (new): renders a per-track colored-box
  + id-number video from any tracks json, half real-time speed. Rendered
  BOTH the mt09 candidate and the current mt=0.8 baseline (from the
  committed tracks cache) for side-by-side comparison.

### [2026-07-22] Recreation check

```
distinct track_ids: 93  mean tracks/frame: 28.9  mean lifespan: 143.1
```
Matches TEST 6's original mt=0.9 measurement exactly (93 / 143.1) --
confirms the recreated yaml is faithful to what was originally tested.

### [2026-07-22] DJ eyeball result — raw

DJ watched both overlay videos in full, looking specifically for a
track's ID NUMBER appearing on a DIFFERENT physical player (not just a
new number appearing on someone, which is normal fragmentation, not a
switch). DJ's report: "I noticed 2 confident switches however they were
the same 2 switches found in the other video... no new swaps. And the
players do seem to hold their box better" -- i.e. mt=0.9 does NOT
introduce new switches beyond what the current committed system already
has at these two moments, and independently confirms the measured
lifespan improvement is visually real.

DJ then flagged one for a closer look: "id 17 switches players" near the
start of the clip. Investigated with exact frame data (not re-guessed):
- mt09's id=17 tracks one player continuously and smoothly, f=120 (4.00s)
  to f=196 (6.53s) -- consistent position cluster, no jumps.
- Goes silent for an 0.8s gap (f=196 to f=221 -- occlusion/lost, inside
  track_buffer=30's tolerance).
- Reappears at f=221 (7.37s) with a visibly different box size/aspect.
- Baseline's own id=17 (SAME player, identical positions where they
  overlap) cleanly ENDS at f=145 (4.83s) and is never reused -- baseline
  does not carry this identity through the gap at all; whatever DJ saw
  "switch" in the baseline video at this same real-world moment must have
  been a DIFFERENT id number, since raw id=17 data proves it stops clean
  there.
- Extracted actual video stills (not inference from bbox math) at f=196
  and f=221/223 with id=17's box highlighted, both frames showing all
  other tracked boxes for context. VISUALLY CONFIRMED: f=196's id=17 box
  is on a WHITE (Milford) jersey player. f=221/223's id=17 box is on a
  GREEN (Little Miami) jersey player -- a DIFFERENT PLAYER, on the
  OPPOSING TEAM, in the same court location the confusion started.

### TEST 12 VERDICT — MEASURED, DJ-CONFIRMED — pending DJ review

- CONFIRMED, with picture evidence: match_thresh=0.9 DOES produce a real
  ID switch (not just a theoretical risk) -- id=17 silently carried a
  track through an 0.8s occlusion gap and reattached to a different
  player on the OPPOSING team at the reappearance point. This is the
  exact "confidently wrong" failure Test 6's safety caveat existed to
  catch, now caught with an exact timestamp and stills, not a guess.
- CRITICAL NUANCE, also DJ-confirmed: this switch is NOT new. DJ
  independently observed the same 2 problem spots in BOTH videos, no new
  switches introduced by the looser setting. The current COMMITTED system
  already mishandles this same real-world moment -- it just fails
  differently (ends the track cleanly rather than silently reattaching
  it), which is a real behavioral distinction: today's failure mode
  (a new, unconfirmed id starting fresh) is structurally SAFER than
  mt=0.9's failure mode (a possibly-already-confirmed id silently
  continuing onto the wrong body) even though both are "wrong" at this
  moment, because a confirmed identity's label only ever rides on a
  track_id that keeps going -- baseline breaking the id_17 track here
  would force re-confirmation; mt=0.9 continuing it would not.
- NOT ADOPTED (unchanged verdict, now with hard evidence instead of a
  hypothetical): match_thresh=0.9 should NOT replace 0.8 as-is. The
  underlying win (35% longer tracks, confirmed both by numbers AND DJ's
  visual read of "players hold their box better") is real and worth
  pursuing, but needs a safeguard at the RE-ATTACH moment specifically --
  e.g. requiring the reappearing detection's jersey/team color to be
  consistent with the track's last-known color before accepting a match
  across a gap (ties directly to the shipped color-tiebreak machinery
  already in the identity layer) -- unmeasured, next candidate step.
- This session's spikes/bytetrack_mt09.yaml and
  spikes/render_tracker_overlay.py are now reusable tools for the
  player-tracker plan's item 2 (ID-switch ground truth) -- this single
  confirmed switch (TEST1, id=17, players at f=196 vs f=221-223, court
  position near the near hoop) is the first real entry in that ground
  truth, not just a proxy metric.

---

## TEST 12 follow-up — DJ ground truth on the two open ball candidates (2026-07-23)

Both candidates flagged across TEST 8-11 (§ "NEW unverified candidate" /
"recurring candidate") now have DJ-confirmed answers, and they REVISE the
follow-up accel_y investigation's conclusion, not just extend it.

DJ's raw report:
```
~74.5-75.0s: "This isnt a shot or a rebound its just a person holding the
ball on an inbounds pass play also she dosent even pass the ball in
until 78s"
~45.1-45.9s: "This is a pass to another player across the court."
```

Both CONFIRMED NON-SHOTS. Added to local_weights_check.py's HARD ground
truth as known non-shot spans (1352,1378) and (2234,2250), alongside the
two literal deflections -- so every future model version (v4+) gets
auto-checked against all 4, permanently, the same protection the two
original deflections have had since TEST 2.

REVISES the accel_y follow-up (two entries up in this log): that
investigation found confirmed-real shots cluster at accel_y 0.848-1.935
and flagged 1352-1378's accel_y=1.388 as "inside the real cluster,
corroborating it's a genuine shot." DJ's ground truth proves that
inference WRONG -- 1352-1378 is a pass, not a shot, despite a
completely normal, shot-shaped accel_y. Root cause is obvious in
hindsight: a pass thrown across court and a jump shot thrown at the hoop
are BOTH clean ballistic throws by a human arm -- physics cannot tell
them apart when the pass's flight path happens to pass near the hoop's
screen position. The 2234-2250 case, by contrast, fits the earlier
low-accel_y pattern (0.455) exactly, because it isn't real ball flight at
all -- a mostly-stationary held ball produces detection jitter, not a
ballistic arc, so its "arc" is a bogus fit, not a rebound-catch example.

REVISED VERDICT ON accel_y AS A GATE: DOWNGRADED from "candidate gate
pending one check" to "catches at most ONE of at least TWO distinct
false-positive categories, blind to the other." Category A (rebound
catch / held ball -- low accel_y OR a bogus/jittery fit): accel_y may
help here, still only n=2 confirmed (403-415 rebound/dish, 2234-2250
held ball), both LOW. Category B (legitimate pass that geometrically
arcs near the hoop pixel -- normal shot-like accel_y): accel_y CANNOT
help here by construction, proven by 1352-1378. Building an accel_y
gate now would give false confidence -- it would silently let category-B
false positives straight through while appearing to have "solved" the
false-positive problem.

WHAT ACTUALLY GENERALIZES (unchanged, now with a second confirmed
reason to prioritize it): the player-signal cross-check. A rebound/held-
ball has a player's hands already at/near the ball with no release
motion; a cross-court pass has its APPARENT "shooter" (nearest player at
the arc's start) standing far from the hoop in a passing posture, not a
shooting one, AND the ball's actual destination is another PLAYER's
hands, not the rim -- neither category-A nor category-B fools a check
that asks "does this look like a person shooting AT the hoop," because
both fail that question for reasons a pure ball-trajectory gate cannot
see. This is now the ONLY measured path that covers both known failure
categories, not just a preference.

NOT ADOPTED, nothing built yet. NEXT ACTION: no more ball-physics-only
gates should be attempted on this data -- the player-tracking-based
cross-check (player-tracker plan item 3, which DJ's in-progress player
labeling directly feeds) is the next real step for this problem
specifically, not a new ball detector or a new physics threshold.

---

## TEST 13 — Color-consistency safeguard for tracker reattach (mt=0.9's missing piece)

Success condition: a jersey-color check at the moment a lost track
reattaches should catch TEST 12's confirmed real switch (TEST1 id=17,
white Milford at f=196 -> green Little Miami at f=221) without any
roster/OCR input, using only unsupervised color clustering on this
clip's own crops.

### [2026-07-23] Setup

spikes/tracker_color_reattach_check.py (new, read-only): reuses
color_tiebreak.crop_color_signature and ocr_reader.jersey_crop (no new
crop logic). Builds 2 color clusters via simple k=2 k-means over ~2700
sampled torso crops from across the clip (no team labels, no roster --
basketball has ~2 jersey colors, so unsupervised clustering finds them
the same way color_tiebreak already does with team-labeled crops, just
without the labels). For every track that goes silent >1 frame and
reattaches under the SAME id, classifies the jersey color just before
the gap and just after; DIFFERENT clusters (with the same margin-based
abstention as color_tiebreak.COLOR_MARGIN_RATIO=1.4) = SUSPECT.

### [2026-07-23] TEST 13 RESULT — raw output (TEST1, mt=0.9 tracks)

```
[reattach-check] built 2 color clusters from 2682 sampled crops:
  (64, 53, 57) vs (123, 112, 121)   <- dark (green/red?) vs light (white?) jerseys

109 reattach events total: 8 SUSPECT, 84 safe, 17 abstained

  id=17  gap 196->221 (25f, 0.83s)  before_cluster=1 after_cluster=0  SUSPECT  <<<<<
  [+ 7 other SUSPECT events: id=13 (x2), id=19, id=36, id=45, id=92, id=215]
```

### [2026-07-23 19:14:45] Regression suite

```
204 passed in 3.35s
```

### TEST 13 VERDICT — MEASURED — pending DJ review

- SUCCESS CONDITION MET: the check correctly flags TEST 12's confirmed
  real switch (id=17) as SUSPECT, using nothing but unsupervised color
  clustering -- no roster, no OCR, no manual color entry. This is the
  first validated evidence that a color-consistency safeguard is
  technically workable, not just a plausible idea.
- HONEST LIMITATION, on the record: this validates the method caught the
  ONE case we already know is real (n=1 true positive). The other 7
  SUSPECT flags are UNVERIFIED -- could be more real switches (which
  would be great, more free ground truth) or could be false alarms from
  lighting/motion-blur/angle changes fooling the color clusterer during
  a genuinely safe reattach. Without DJ eyeballing those 7, the method's
  real PRECISION is unmeasured -- same discipline as every other gate in
  this project: one confirmed catch is promising, not proof.
- Also unmeasured: FALSE NEGATIVE rate (real switches this method would
  MISS, e.g. two players in identical jerseys swapping -- color alone
  cannot see that, same blind spot §11 already found for appearance
  re-ID generally). Cross-team switches (like TEST 12's) are the easy
  case for a color check; same-team switches are the hard case still
  unaddressed by this alone.
- NOT ADOPTED -- nowhere near wired into the actual tracker yet, this is
  a standalone probe. NEXT ACTION: DJ eyeballs the 7 unverified SUSPECT
  moments (frame numbers in the raw output above) the same way id=17 was
  checked in TEST 12 (extract stills, compare jerseys) to learn real
  precision before any adoption conversation. If precision holds up,
  the natural next step is wiring this as a POST-PROCESS split (break a
  track_id in two at a color-mismatch reattach, never touching the
  underlying ByteTrack matcher itself) rather than a tracker-internal
  patch -- simpler, isolated, matches this project's "new layer beside,
  not a rewrite" pattern used everywhere else (color_tiebreak itself,
  retroactive merge, etc.).

### [2026-07-23, same session] Eyeball follow-up: all 7 unverified SUSPECT flags checked

Extracted before/after stills (bboxes drawn) for all 7 and inspected
directly, same method as TEST 12's id=17 stills.

```
id=45 gap 214->224:  CONFIRMED real switch -- green jersey (#14 area) at
                      f=214 tangled with a white player; by f=224 the box
                      has moved onto the WHITE player, green player now
                      only partially adjacent. Second confirmed switch.
id=36 gap 256->282:  AMBIGUOUS -- two players (white + green) closely
                      overlapping in both stills; box position is
                      plausibly consistent OR plausibly swapped, cannot
                      call it either way from these frames.
id=13 gap 358->367:  NOT A PLAYER -- box sits on the animated scoreboard
                      graphic in the bottom-left corner in BOTH stills.
id=13 gap 369->373:  same scoreboard-graphic false detection.
id=19 gap 316->325:  same scoreboard-graphic false detection.
id=215 gap 358->361: same scoreboard-graphic false detection.
id=92 gap 318->335:  NOT A PLAYER -- box sits on a REFEREE (striped
                      shirt), not a tracked athlete.
```

REVISED TEST 13 PRECISION (now measured, not just n=1): of 8 total
SUSPECT flags across this session (id=17 + the 7 above), **2 confirmed
real switches** (17, 45), **1 ambiguous** (36), **5 flagged moments
weren't real player-to-player switches at all** -- they were the color
classifier correctly noticing a color change on a BOGUS detection (the
scoreboard graphic animating, or a referee's uniform reading differently
than expected). Precision on "is this a real player-identity switch":
2/8 confirmed (25%), up to 3/8 if the ambiguous case counts.

BONUS FINDING, unplanned: this probe incidentally surfaced a SEPARATE,
previously-unknown detector bug -- the person detector occasionally
fires on the animated scoreboard graphic overlay and on referees as if
they were tracked players (4 of 8 flags were this, not jersey-color
noise). Not investigated further this session; worth a note for whoever
next touches player detection filtering (a simple frame-region exclusion
for the scoreboard's fixed screen position, and/or leaning harder on the
already-measured Player-vs-Ref distinction from TEST 4, would likely
clear both at once).

REVISED TEST 13 VERDICT: the color-consistency idea is NOT a clean,
high-precision switch detector on its own (2/8-3/8 true positive rate on
this sample) -- but it is NOT useless either: it caught both real
switches found in this session (100% recall on the n=2 known cases,
though n=2 is small) at the cost of several false alarms that turned out
to be a different bug entirely. STILL NOT ADOPTED. Two honest paths
forward, neither built: (a) filter out non-player detections (scoreboard
region, refs) BEFORE running the color check, which would likely lift
precision substantially given 4 of the 6 false alarms trace to exactly
those two sources; (b) treat this as one signal among several (combine
with a position/motion plausibility check) rather than a standalone
gate. Given today's precision, shipping this as an automatic track-
splitter would create nearly as many new (wrong) splits as it fixes --
NOT safe to adopt as-is.

---

## TEST 14 — Scoreboard OCR as an independent make/miss second signal (DJ's idea, 2026-07-23)

DJ's proposal: the scoreboard graphic sits at a fixed screen position and
is the OFFICIAL scoring record -- read it in a completely separate pass
from player/ball detection (never feeding into it), match score-change
timestamps to shot attempts, and get real make/miss ground truth for
free. Bonus: keeps this OCR work from ever confusing the player detector
again (TEST 13 found it sometimes tracks the scoreboard graphic itself
as a "player").

### Setup + validation of the core assumption

Confirmed by eye (3 widely-spaced HARD frames, f=0/1200/2700): the
graphic sits at the IDENTICAL screen position for the entire clip
regardless of camera pan/zoom -- it's a screen-locked broadcast overlay,
not a world-locked one. Free bonus: HARD's score read "15-12" at all
three checked frames -- the score literally never changes in this clip,
independently confirming (via a totally different method) what careful
trajectory analysis already concluded: both of HARD's known shot
attempts were misses.

### Build 1 (spikes/scoreboard_ocr_probe.py) -- naive version, FAILED informatively

Sampled the fixed corner region every 15 frames, ran EasyOCR (reused
phase2/ocr_reader.py's engine, digit-only), picked the two TALLEST
digit detections per frame as home/away score (score digits render
larger than foul-counts/period-number on the graphic -- no hand-tuned
sub-crop needed). Raw run on HARD: constant stream of spurious "changes"
(15-25, 12-23, 15-891...) from single-frame misreads. Added a
naive fix (require 3 confident reads in a row) -- worked cleanly on
HARD (locked "15-12" at 3s, zero false changes for the rest of the
clip) but on TEST1 took 26.5 SECONDS to ever achieve a first lock,
silently leaving EVERY one of TEST1's 5 verified shots (all before
19.6s) with zero reliable score coverage. Caught this by testing
generalization deliberately (not just trusting the first clip's clean
result) -- if this had shipped, the shot-matcher would have quietly
reported "5/5 misses" backed by NO real data, the exact confident-wrong
failure this project's whole discipline exists to prevent.

Root cause, diagnosed with raw per-frame OCR dumps (not guessed): (1)
a genuinely-correct score digit sometimes reads at LOW individual
confidence (0.11-0.5) on compressed/blurry video -- the naive >=0.5
per-digit threshold was rejecting valid reads outright; (2) occasional
taller spurious detections (a jersey number, an edge artifact) can
outrank a real score digit for one frame, and a strict N-in-a-row streak
means ONE bad frame resets the count to zero, so recovery after any
noise is slow.

### Build 2 -- sliding-window majority vote + monotonicity guard, PASSED

Fixes: (a) lowered the per-digit confidence floor to 0.15 (near-zero =
nothing digit-shaped found; a real-but-blurry digit still clears this)
and rely on temporal voting to reject noise instead of a strict per-
frame confidence gate; (b) replaced the N-in-a-row streak with a 7-frame
sliding window, majority vote (>=4/7) -- one bad frame only costs one
vote, doesn't reset everything; (c) a MONOTONICITY guard -- basketball
scores never decrease, so any winning "majority" that would lower either
score is proof of continued noise, not a real event, and gets rejected
+ the window cleared rather than accepted. (c) was added after (a)+(b)
alone still produced a physically-impossible 0-2 -> 2-2 -> 0-2 -> 2-2
oscillation on TEST1 -- caught by eye, not assumed correct just because
the code ran.

### RESULT (both clips re-verified after the fix)

```
HARD:  first lock "15-12" @ 3s, ZERO events for the full 91.5s clip.
       Matches known ground truth exactly (both verified shots = misses).
TEST1: first lock "0-2" @ 22.5s, ONE real event: 27.0s  0-2 -> 2-2.
       Monotonic (no decreases), physically plausible.
```

Suite: 204 passed (20:29:58 bookend).

### TEST 14 VERDICT — MEASURED — pending DJ review

- CORE IDEA VALIDATED: the scoreboard IS reliably readable as an
  independent second signal, confirmed against HARD's known ground truth
  and internally consistent (monotonic) on TEST1.
- HONEST LIMITATION, not yet solved: TEMPORAL RESOLUTION. The 0.5s
  sample stride + 7-frame voting window means a real basket is only
  confirmed roughly 1-3 seconds after it happens, and TEST1's away team
  reached 2 points SOMEWHERE between frame 0 (confirmed 0-0 by eye) and
  the first lock at 22.5s -- a window spanning FOUR of TEST1's five
  verified shot attempts (shot A, layup 1, layup 2, shot B all end
  before 11s; only layup 3 at 19.6s is close to the lock). We now know
  at least one of those four was very likely a MAKE (the score did
  reach 2), but CANNOT yet say which one from this pass alone.
  spikes/match_shots_to_score.py (built, tested against HARD only) would
  currently mis-report all four as MISS if pointed at this data, because
  none of their 6-second post-shot windows reach 22.5s -- NOT run on
  TEST1 for this reason; would have been a confident-wrong result.
- NOT ADOPTED, nothing wired into run_clip. NEXT ACTION options, neither
  built: (a) after a shot attempt is detected, sample the scoreboard
  region MUCH more densely (every frame, not every 15th) for a short
  window right after that specific attempt -- turns "was there a make
  somewhere in this flurry" into a per-shot precise timestamp; (b) also
  read the PERIOD/game-clock digits the same way (already visible in
  every still) as a free cross-check and to help order events. (a) is
  the natural next step and directly closes today's gap.
- Bonus finding carried over from TEST 13: the scoreboard region is a
  known source of false player detections. Once this OCR pass is wired
  in, its region is a natural, already-computed exclusion zone for the
  player detector -- two birds.

---

## TEST 15 — Junk-detection filter (scoreboard graphic + referees) before the colour check

Success condition (set before running, tasks/todo.md PART B): the 4
scoreboard flags + the 1 referee flag from TEST 13 disappear, and BOTH
confirmed real switches (id=17, id=45) are STILL flagged -- i.e. precision
2/8 -> 2/3 with recall intact. Stated failure mode to watch: "if a real
switch stops being flagged, the filter is too aggressive -- report it, do
not tune until it passes."

### [2026-07-25] Setup

- spikes/tracker_color_reattach_check.py gained `apply_junk_filter()`: drops
  a detection whose box CENTRE falls in the clip's `exclude_regions` (the
  scorebug rectangle ALREADY in spikes/clips_config.py for the calibration
  engine -- no new constant, no new hand-tuning), and drops boxes that
  IoU>=0.5 match a Ref-class detection. `--nofilter` reproduces TEST 13.
- Centre, not overlap, on purpose: a real player standing near the scorebug
  corner still has their centre outside it.

### [2026-07-25] Recreation check (before trusting the modified script)

```
$ .venv/Scripts/python spikes/tracker_color_reattach_check.py TEST1 \
      spikes/out/TEST1_tracks_mt09.json --nofilter
[reattach-check] 109 reattach events: 8 SUSPECT, 84 safe, 17 abstained
```
EXACTLY TEST 13's numbers (109 / 8 / 84 / 17) -- the edit is faithful.

### [2026-07-25] TEST 15 RESULT — raw output (scoreboard filter on)

```
[reattach-check] junk filter ON: dropped 842 scoreboard-region detections + 0 referee detections
[reattach-check] built 2 color clusters from 2509 sampled crops: (127, 115, 125) vs (66, 54, 59)
  id=17  gap 196->221 (25f, 0.83s)  before_cluster=0 after_cluster=1  SUSPECT  <<<<<
  id=36  gap 256->282 (26f, 0.87s)  before_cluster=1 after_cluster=0  SUSPECT  <<<<<
  id=92  gap 318->335 (17f, 0.57s)  before_cluster=1 after_cluster=0  SUSPECT  <<<<<

104 reattach events: 3 SUSPECT, 83 safe, 18 abstained
```
All FOUR scoreboard false alarms (id=13 x2, id=19, id=215) are GONE. id=17,
the cleanly-confirmed switch, survives. No NEW flags appeared.

### [2026-07-25] The id=45 regression — measured, not guessed

id=45 (TEST 13's SECOND confirmed real switch) is no longer SUSPECT. First
check was whether the filter ate it: it did not -- its boxes sit at centre
(694,476) and (720,461), nowhere near the scorebug rect (0,810,415,1080).
It became ABSTAIN. Per-crop margin ratios (threshold 1.4):

```
                                        f=214            f=224
id=17  confirmed switch (clean stills)   2.210 DECIDES   1.739 DECIDES
id=45  confirmed switch (tangled pair)   1.395 ABSTAINS  1.877 DECIDES
id=36  AMBIGUOUS by DJ eyeball           3.496 DECIDES   1.574 DECIDES
```
id=45's f=214 crop misses the abstention line by 0.005. Cause: removing 173
scorebug crops from the k-means sample shifts the centroids slightly, and
this crop sits exactly on the margin. Its signature (105,92,91) is a muddy
mid-tone -- neither the light centroid (127,115,125) nor the dark one
(66,54,59) -- which matches TEST 13's own eyeball note that at f=214 the
green player was "tangled with a white player": the box contains BOTH
jerseys, so the colour reading is mush.

### [2026-07-25] Referee half — a real bug in my own filter, found and fixed

spikes/ref_boxes.py (new): v3's Ref class over TEST1's tracking span,
conf>=0.4, imgsz=1280 -> 895 boxes across 460/461 frames (1.9/frame; TEST 4
measured 3.0/frame with the hosted model, so the same order).

FIRST IMPLEMENTATION WAS WRONG and the numbers said so before any conclusion
was drawn: dropping ref-matched detections ONE AT A TIME raised the event
count 104 -> 126. Filtering cannot create events; that was the tell. Cause:
a referee's track only matches a Ref box on the frames the Ref detector
actually fires, so per-detection removal punches holes in that track, and
every hole becomes a fabricated "reattach event" the tracker never
experienced -- and the referee track stayed flagged anyway, now full of
artificial gaps. FIXED by excluding refs WHOLE-TRACK (a track >=50% ref-
matched is a referee) instead of per-detection. Event count returned to 104.

Corroboration the ref detection is right, not just self-consistent: the two
tracks dropped whole are id=7 and id=14 -- both in TEST 4's independently
measured ref-matched id list [1, 7, 11, 14, 372, 927].

### [2026-07-25] TEST 15 FINAL RESULT — raw output (both filters)

```
[reattach-check] referee tracks dropped whole: [7, 14]
[reattach-check] junk filter ON: dropped 842 scoreboard-region detections + 919 referee detections
[reattach-check] built 2 color clusters from 2324 sampled crops: (122, 111, 121) vs (60, 48, 53)
  id=36  gap 256->282 (26f, 0.87s)  SUSPECT
  id=92  gap 318->335 (17f, 0.57s)  SUSPECT
  id=17  gap 196->221 (25f, 0.83s)  ABSTAIN     <- CONFIRMED real switch
  id=45  gap 214->224 (10f, 0.33s)  ABSTAIN     <- CONFIRMED real switch
104 reattach events: 2 SUSPECT, 72 safe, 30 abstained
```

Per-crop margin ratios under the final (correctly filtered) colour scale:
```
                                  before-crop        after-crop
id=17  CONFIRMED switch #1        2.946 DECIDES      1.338 ABSTAINS
id=45  CONFIRMED switch #2        1.861 DECIDES      1.385 ABSTAINS
id=36  ambiguous by DJ eyeball    2.655 DECIDES      2.124 DECIDES
```

### TEST 15 VERDICT — MEASURED — pending DJ review

- **HEADLINE, and it is a decisive NEGATIVE for TEST 13's idea: with the junk
  detections correctly removed, the colour check flags NEITHER confirmed real
  switch (0/2 -- both abstain), while still flagging the one case a human
  could not call.** TEST 13's apparent "100% recall on n=2" was riding on a
  CONTAMINATED colour reference: the scoreboard graphic and the referees'
  black-and-white stripes are extreme colour outliers, and once they stop
  polluting the 2-cluster fit, both real switches fall inside the abstention
  margin. The method was always this weak; TEST 13 could not see it.
- **The structural reason, which is why no threshold tweak rescues this:**
  look at the margin table -- both confirmed switches read CONFIDENTLY on the
  before-crop and MUSHILY on the after-crop. That is not bad luck, it is
  causal. An ID switch happens BECAUSE two players occluded each other, so
  the box that reattaches sits on a partially-occluded, half-covered body
  whose colour is a blend. Colour is least readable exactly at the moments it
  is needed, and most confident (id=36: 2.655 / 2.124) exactly where it is
  not. The signal is anti-correlated with its own use case.
- Margins are 1.338 and 1.385 against a 1.40 line. Lowering MARGIN_RATIO to
  1.30 would "recover" both and was NOT done: that is fitting a threshold to
  n=2 to reach a desired answer, the accel_y mistake, and it would also
  promote an unknown number of the 72 currently-safe events into flags.
- SEPARATELY, the junk filter itself is a CLEAN WIN and stands on its own
  merits regardless of the colour idea dying: 842 detections on an animated
  graphic and 2 whole referee tracks were being silently carried through the
  tracker into everything downstream. That is a real detector bug, now
  measured and cheaply fixable, and it was found by accident.
- Open, minor: id=92 was called a referee by DJ's TEST 13 eyeball but v3's
  Ref class does not claim its track. Either the eyeball or the Ref class is
  wrong on that one; not resolved, and it does not affect the verdict.
- NOT ADOPTED, and the recommendation is now stronger than "not yet": on this
  evidence the colour-consistency safeguard should NOT be pursued further as
  a switch detector. The mt=0.9 tracking win (TEST 6/12) therefore remains
  UNSAFE and unadopted, and needs a different safeguard -- the honest
  candidate is TEST 19-style ground truth (player-tracker plan item 2), not
  another appearance heuristic. §11 already measured appearance re-ID as
  counterproductive on this footage; this is the second independent finding
  in the same direction.
- Superseded: the earlier partial verdict written mid-test (scoreboard-only
  result, "8 -> 3 flags, precision 2/3") described a real intermediate
  measurement but is NOT the conclusion -- it predates the referee half and
  the margin table above.

#### (intermediate, scoreboard-only measurement -- superseded by the above)

- The scoreboard half is an unambiguous win:
  842 junk detections dropped, 4/4 scoreboard false alarms eliminated, zero
  new false alarms, and the robust confirmed switch untouched. Flags 8 -> 3.
- FAILURE MODE FIRED, exactly as the plan said to watch for: id=45 went
  SUSPECT -> ABSTAIN. Note the direction -- it abstains ("I can't tell"),
  it does NOT declare the reattach safe, which is the less dangerous of the
  two ways to be wrong. MARGIN_RATIO was deliberately NOT lowered to 1.39
  to recover it: that is fitting a threshold to n=2, the exact accel_y
  mistake this log already recorded once.
- MORE IMPORTANT FINDING, unplanned, and it revises TEST 13: the margin
  table above shows colour CONFIDENCE DOES NOT TRACK SWITCH TRUTH. id=36 --
  the case a human could not call from stills -- is flagged more confidently
  (3.496) than id=45, a CONFIRMED switch (1.395). TEST 13's headline claim
  of "100% recall on the n=2 known cases" is therefore weaker than it read:
  only ONE of the two (id=17) is a clean colour catch; the other was always
  marginal and its flag was closer to luck than to signal.
- STANDING: precision on real player-identity switches is now 1 confirmed +
  1 ambiguous + 1 referee out of 3 flags (the referee is filterable, ref-box
  pass still running at write time). Recall on the 2 known switches: 1 clean
  catch, 1 abstention. NOT ADOPTED -- unchanged. The colour check is a
  useful NOISE reducer once junk detections are gone, but on this evidence
  it is not a switch detector, and nothing should be wired to it yet.
- The scoreboard filter itself, though, is a clean and separable win worth
  keeping regardless of what happens to the colour idea -- it fixes a REAL
  detector bug (842 detections on an animated graphic) that has nothing to
  do with tracking safety and was silently polluting everything downstream.
- Suite: 223 passed (up from TEST 13/14's 204 -- the extra 19 are the
  parallel calibration session's tests, not this track's).

---

## TEST 16 — Pose estimation vs the three confirmed false positives (the headline test)

Success condition: a visible separation between the 7 DJ-confirmed real
shots and the 3 DJ-confirmed non-shots, using an off-the-shelf COCO-
pretrained pose model and ZERO new labels. Explicit discipline set before
running: print the raw table FIRST, propose no rule until after, and treat
anything found as a HYPOTHESIS needing a holdout clip (the accel_y lesson).

### [2026-07-25] Setup

spikes/pose_shot_check.py (new, read-only). yolo11x-pose.pt (COCO, 17
keypoints), imgsz=1280, keypoint conf floor 0.30. Spans are the arcs v3
ACTUALLY claimed (TEST 11 raw output), so we measure what the classifier
saw. At the arc's first frame (claimed release) and last frame (claimed
arrival) it measures the distance from the ball to the nearest wrist,
whether that wrist is above its own shoulder, and the distance to the rim.
ALL distances normalised by that person's bbox height, so a distant player
and a near player compare fairly -- and so the hand and rim columns are in
the SAME units, which is the only way "did this end at the rim or in
someone's hands?" is a fair question rather than pixels vs body-lengths.

### [2026-07-25] TEST 16 RESULT — raw output

```
truth clip  span         rel.hand  armUp  arr.hand  arr.rim  ENDS AT  what it really is
FAKE  HARD  403-415          0.72   True      0.10     0.28     HAND  rebound caught -> dished out
FAKE  HARD  1352-1377        0.13   True      0.38     3.29     HAND  cross-court pass
FAKE  HARD  2234-2250        0.31  False      0.16     0.34     HAND  player HOLDING ball, inbounds
REAL  HARD  351-375          0.08   True      1.59     0.61      rim  verified jump shot
REAL  HARD  1177-1214        1.09  False      1.09     0.08      rim  verified jump shot
REAL  TEST1 58-77            0.33   True      1.22     0.80      rim  verified jump shot A
REAL  TEST1 166-184          0.19   True      0.59     0.08      rim  verified layup 1
REAL  TEST1 236-250          0.43   True      0.60     0.17      rim  verified layup 2
REAL  TEST1 314-327          0.10  False      0.13     0.77     HAND  verified jump shot B
REAL  TEST1 571-589          0.37   True      0.94     0.11      rim  verified layup 3
```

### TEST 16 VERDICT — MEASURED — pending DJ review

- MY STATED HYPOTHESIS WAS WRONG, on the record. I predicted the separator
  would be the RELEASE: "a shooter has arms extended overhead with the ball
  leaving their hands." It does not separate at all. rel.hand REAL spans
  0.08-1.09 and FAKE spans 0.13-0.72 -- fully overlapping. armUp is True for
  5/7 REAL and 2/3 FAKE -- no signal. Every release-side measurement failed.
- WHAT DOES SEPARATE IS THE DESTINATION, not the posture: at the end of the
  flight, is the ball nearer a HAND or nearer the RIM? 3/3 FAKE end at a
  hand; 6/7 REAL end at the rim. 9 of 10, with NO fitted threshold at all --
  the rule is a comparison between two measured distances, not a magic
  number, which is a materially different kind of claim than accel_y's
  ">= 0.848" was.
- THE HAND SIGNAL IS GENUINELY NEW INFORMATION, not a repackaged min_dist:
  rim distance ALONE does not separate these events. Two FAKEs (0.28, 0.34)
  end CLOSER to the rim than three REALs (0.61, 0.77, 0.80). The classifier
  already had the rim distance and it is exactly why these three got
  through; the wrist is what the ball layer could not see.
- Why this survives the case that killed accel_y: the cross-court pass
  (1352-1377) has a perfectly shot-shaped trajectory -- and lands 3.29 body-
  heights from the rim, in a teammate's hands. Physics cannot see that; the
  destination can.
- THE ONE MISS, and it is in the dangerous direction: TEST1 shot B (314-327)
  is a REAL shot that ends at a hand (0.13 vs rim 0.77) -- a FALSE NEGATIVE,
  i.e. this rule would SUPPRESS a genuine shot. Cause is visible in the
  data: v3's arc for shot B stops short of the rim (TEST 11: min_dist 97.3px
  'observed'), so its "arrival" is not the rim arrival at all -- it is where
  the detections ran out, and a player's hands happened to be there. This is
  the same shot that v1 caught, v2 lost and v3 recovered; it has been the
  fragile one at every stage.
- TWO OBVIOUS CONFOUNDS CHECKED AND CLEARED (before anyone gets excited):
  (a) CLIP IMBALANCE -- all 3 fakes are HARD, so "ends at hand" could have
      been a HARD artefact. Within HARD ALONE, same camera, same clip:
      2 REAL both end at the rim, 3 FAKE all end at a hand. Clean separation
      without leaving the clip. Not a clip artefact.
  (b) EVENT DURATION -- a short event might trivially keep the ball near the
      thrower's hands. FAKE lengths are 12/16/25 frames, sitting INSIDE the
      REAL range of 13-37. Not a duration artefact.
  Worth noting the one miss (shot B) is the shortest REAL event at 13
  frames, which fits the "detections ran out before the rim" explanation
  rather than contradicting it.
- **STABILITY CHECK -- THE MOST IMPORTANT CAVEAT, and it is a real weakness.**
  Re-measured every event with the arc's last frame shifted by +/-3 frames
  (0.1s), since shot B's failure was itself an endpoint artefact:
```
  truth span        ENDS AT at end-3 / end / end+3   stable?
  REAL  351-375     rim  / rim  / rim                yes
  REAL  1177-1214   rim  / rim  / rim                yes
  REAL  58-77       rim  / rim  / rim                yes
  REAL  166-184     rim  / rim  / rim                yes
  REAL  236-250     rim  / rim  / rim                yes
  REAL  314-327     HAND / HAND / HAND               yes  (the known miss)
  REAL  571-589     rim  / rim  / HAND               NO -- FLIPS
  FAKE  403-415     rim  / HAND / HAND               NO -- FLIPS
  FAKE  1352-1377   HAND / HAND / HAND               yes
  FAKE  2234-2250   HAND / HAND / rim                NO -- FLIPS
```
  THREE OF TEN events change their answer when the arc endpoint moves by a
  tenth of a second. The 9/10 headline is measured at the TRUE arc endpoint
  (the "end" column -- what the classifier would actually feed it), and the
  flips go in both directions, so this is noise rather than a systematic
  bias. But it means the signal is real yet FRAGILE at the single-frame
  level, and a single-frame reading is not a safe basis for a gate.
- CANDIDATE FIX, explicitly UNTESTED and NOT to be adopted quietly: read a
  WINDOW after the arc ends rather than one frame -- a caught/held ball
  STAYS in someone's hands for many frames, whereas a ball that reaches the
  rim does not linger in anyone's grip. That formulation has the same causal
  story with far less single-frame sensitivity. It must be specified BEFORE
  TEST 19 runs, not tuned afterwards to whatever makes the holdout pass.
- NOT ADOPTED. NOT A GATE. This rule was derived AFTER seeing these 10
  events, which is precisely the shape of the accel_y error (clean-looking
  at n=9, destroyed by ground truth on a 10th). It is a HYPOTHESIS with a
  causal story, and it is frozen as stated above until TEST 19 runs it
  against a clip it was not built on. If it holds there, it is the first
  real answer to the false-positive problem; if it does not, it dies the
  same way accel_y did and that is a result too.
- Pose viability on this footage, the stated risk going in: NOT a problem.
  Wrists resolved at usable confidence on every one of the 10 events,
  including the far-hoop TEST1 layups. The "small distant players" worry did
  not materialise at imgsz=1280.
- Cost: zero labelling, zero GPU, ~1 hour. Suite 223 green.

---

## TEST 17 — Newer-architecture bracket (IN PROGRESS, started 2026-07-25)

Success condition per candidate: the SAME gate v3 passed -- HARD's 2 verified
shots reproduced, all known non-shots rejected, TEST1 5/5. Dataset unchanged
(1370 train imgs incl. DJ's 230, 32 valid). Recipe = v3's (imgsz 1280,
epochs 150, patience 60, cos_lr, scale 0.7). Pod: fresh RTX 4090.

### TWO HARNESS BUGS FOUND AND FIXED (2026-07-25) — both were hiding results

1. **The original DJ-refuted false positive was never in the ground truth.**
   The rebound-caught-and-dished play (TEST 10, 2026-07-17, DJ: "NO shot
   attempt") has been discussed in every test since and was never added to
   local_weights_check.py's GROUND_TRUTH. The harness has therefore been
   auto-checking 4 of the 5 known non-shots, silently. Found because the
   TEST 17 control claimed it as a layup and the check said nothing. Added
   as (395,445). CONSEQUENCE FOR THE RECORD: **v3's real gate result is 2/2
   verified shots + 5/5 TEST1 with THREE false positives, not two.**
2. **The +/-10 both-endpoints match test** reported a shot as MISSED when the
   claimed arc fully CONTAINED the verified span but started 11 frames early.
   TEST 11 hit this, hand-corrected it in the log, and left it unfixed; it
   fired again on the control. Replaced with overlap matching (a claim must
   cover >=50% of the verified span). VERIFIED against v3's saved logs: HARD
   now reports both shots REPRODUCED automatically (matching TEST 11's
   hand-correction) and TEST1 stays 5/5.

### RUN 1 — bracket_yolov8l_stock (THE CONTROL) — COMPLETE

114 epochs, 1.98h, batch=2 (AutoBatch). Val (the 32-image public set):
```
all-mAP50 0.892   Ball P 0.797  R 0.667  mAP50 0.707
(v3 for comparison: Ball P 0.99  R 0.556  mAP50 0.618)
```
Val says the from-stock control BEATS v3 on Ball. **Ignore that**: this val
set has 18 ball instances, contains none of DJ's footage, and TEST 10
recorded that it failed to predict the clip result in either direction. The
gate is the instrument.

HARD gate (raw, 84.5% raw coverage vs v3's 79.4%):
```
  verified shot 356-381 (near): REPRODUCED (351, 377)
  verified shot 1188-1211 (far): REPRODUCED (1177, 1213)
  deflection 395-445:   *** CLAIMED AS SHOT (403, 415, 'layup', 1.3px) ***
  deflection 418-438:   correctly NOT a shot
  deflection 1217-1250: correctly NOT a shot
  deflection 1352-1378: *** CLAIMED AS SHOT (1352, 1380, 'jumpshot', 17.7px) ***
  deflection 2234-2250: *** CLAIMED AS SHOT (2235, 2250, 'layup', 59.2px) ***
  NEW unverified claim: (1749, 1770, 'far', 'jumpshot', 111.2px) -- never
  claimed by hosted/v1/v2/v3; needs a DJ eyeball (~58.3-59.0s).
```
TEST1 gate (72.4% raw coverage): all 5 verified shots REPRODUCED -- but with
TWO extra claims v3 does not make:
```
   (85, 98,  'far', 'jumpshot', 61.0)  <- the long-standing unverified
                                          84-98 candidate (TEST 1)
   (188, 195,'far', 'layup',    31.8)  <- v3 REJECTS this span (188-202) as
                                          "originates 31.4px from the hoop and
                                          leaves it -- a deflection/
                                          continuation". It directly follows
                                          verified layup 1 (165-184), i.e. it
                                          looks like that layup's rebound.
                                          Unverified by DJ, but the reading is
                                          consistent with TEST 10's note.
```

### RUN 1 VERDICT — the CONTROL LOSES TO v3, and val said the opposite

```
                     verified shots        extra / false claims
  v3 (current best)  HARD 2/2, TEST1 5/5   3 confirmed FPs
  control (stock)    HARD 2/2, TEST1 5/5   the same 3 confirmed FPs
                                           + (1749,1770) new, unverified
                                           + (188,195) a rebound continuation
                                             v3 correctly refuses
                                           + (85,98) unverified candidate
```
- IDENTICAL recall (every verified shot on both clips), STRICTLY WORSE
  precision. v3 stays the best model available.
- THE VAL SET WAS ACTIVELY MISLEADING, for the second logged time: it scored
  the control ABOVE v3 on Ball (0.707 vs 0.618, recall 0.667 vs 0.556) and the
  control is the worse model where it counts. Adopting on val would have been
  a regression. This is now the strongest argument yet for TEST 18 (an
  own-footage validation set) -- the current one does not merely fail to help,
  it points the wrong way.
- RAW COVERAGE ALSO POINTED THE WRONG WAY AGAIN (control 84.5%/72.4% vs v3
  79.4%/68.6%): more coverage, more false claims, same real shots. Fourth
  independent repeat of the DECISIONS 13/18/20 lesson.
- MOST IMPORTANT READING: a completely independent training run, from stock
  weights, on the same data, lands on the SAME THREE false positives. That is
  strong evidence these are not a detector deficiency at all -- no amount of
  detector quality separates a caught rebound, a cross-court pass or a held
  ball from a shot, because the ball's position genuinely does not encode the
  difference. It is the same conclusion TEST 16 reached from the opposite
  direction (pose), now corroborated by a training experiment that was
  designed to test something else entirely.

### TEST 19 — TEST4 HOLDOUT: PRE-REGISTERED PREDICTIONS (2026-07-26, BEFORE DJ's ground truth)

Test4.mp4 = the 5-minute clip DJ supplied 2026-07-26. 9022 frames, 1920x1080
30fps, a gym/teams v3 has never seen. THE FIRST REAL GENERALISATION TEST in
this project -- every threshold and gate to date was built on HARD + TEST1.

**Rim without calibration, measured not assumed.** Shot classification needs a
per-frame rim, which normally waits on the calibration chain (hoop_anchor
carries a clicked rim through the pan). TEST 3 measured the alternative on the
hosted model and shelved it as clip-dependent (HARD: 2/280 frames usable). On
TEST4, v3's Hoop class is a different story entirely:
```
  dets/frame at conf>=0.40: {0: 740, 1: 8247, 2: 35}
  frame-to-frame jump of the best hoop: median 0.5px  p99 21.4px
  jumps over 100px: 0 of 8262
```
Rim resolved on 8282/9022 frames with no clicks and no calibration
(spikes/hoop_track_from_dets.py, which refuses the job if the stability
numbers are bad on some future clip). So this holdout did NOT have to wait on
calibration -- and the far/near LABEL in the output below is meaningless here
by construction (one rim stream), it says "a rim", not "which end".

**v3's raw result on the holdout:** ball seen in 76.8% of frames (vs 79.4% on
HARD, 68.6% on TEST1 -- it generalises), 271 arcs, **17 shot attempts claimed**
and 8 near-rim arcs rejected as deflections/continuations.

**THE PRE-REGISTERED PART.** TEST 16's rule was frozen before this clip
existed: at the end of the flight, is the ball nearer a HAND or nearer the RIM
(in body-height units)? Plus the pre-specified window variant (majority vote
over the 0.5s following the arc), written down in TEST 16's verdict precisely
so it could not be invented afterwards. Both have now been applied to all 17
claims WITHOUT DJ's ground truth, and both are recorded here before it exists:

```
  claim @        single-frame   window     PREDICTION
  0:15.1-0:15.6  rim            rim        real shot
  0:15.7-0:16.3  rim            rim        real shot
  0:53.4-0:54.5  rim            rim        real shot
  1:20.5-1:20.9  rim            rim        real shot
  1:21.2-1:21.7  HAND           HAND       NOT a shot
  1:34.2-1:34.7  HAND           HAND       NOT a shot
  1:36.1-1:36.7  HAND           HAND       NOT a shot
  2:06.3-2:06.8  HAND           HAND       NOT a shot
  2:29.5-2:30.3  rim            rim        real shot
  2:30.3-2:30.6  rim            rim        real shot
  3:07.0-3:08.0  rim            rim        real shot
  3:40.1-3:40.8  rim            rim        real shot
  4:05.8-4:06.5  HAND           HAND       NOT a shot
  4:11.2-4:11.6  HAND           HAND       NOT a shot
  4:34.8-4:35.0  rim            HAND       *** THE TWO VARIANTS DISAGREE ***
  4:35.9-4:36.8  HAND           HAND       NOT a shot
  4:37.5-4:38.4  rim            rim        real shot
```
PREDICTION IN ONE LINE: **of v3's 17 claimed shots, 7 are false positives**
(8 if the window variant is right about 4:34.8). The two variants agree on 16
of 17 -- the window version is materially more stable than the single-frame
version was on the training clips, which is what it was designed for.

FALSIFIABLE OUTCOMES, stated in advance:
- If DJ's ground truth says ~7 of these 17 are non-shots AND they are these
  specific ones, the rule survives a clip it was not built on -- the first
  thing in this project to beat the accel_y trap.
- If the non-shots are a DIFFERENT subset, the rule dies exactly as accel_y
  did, and the player-labelling path is the only remaining route.
- If v3 also MISSED real shots (ones in neither the 17 nor the 8 rejections),
  that is a separate recall failure the rule cannot speak to at all.

DJ IS BEING ASKED TO VERIFY BLIND -- the predictions above are deliberately
NOT being shown to him before he reports what each moment actually is, so the
check cannot be primed by knowing the expected answer.

### [2026-07-26] DJ's GROUND TRUTH (delivered blind, predictions already logged)

```
 #1 jumpshot 3pt   13-17s    Miss
 #2 jumpshot 3pt   52-55s    Miss
 #3 layup        1:18-1:22   Miss
 #4 jumpshot 3pt 1:30-1:34   Miss
 #5 layup        2:01-2:04   MADE
 #6 jumpshot 3pt 2:27-2:30   MADE
 #7 layup        3:05-3:09   MADE
 #8 jumpshot 3pt 3:39-3:41   Miss
 #9 layup        4:31-4:36   MADE  ("happened off screen / CV probably didn't see it")
```
Scored with +/-1.5s tolerance, because these are human-estimated ranges.

### TEST 19 RESULT 1 — RECALL: v3 GENERALISES. 8 of 9 shots found on an unseen gym.

```
  #1 -> 2 claims    #4 -> 1 claim     #7 -> 1 claim
  #2 -> 1 claim     #5 -> *** NO CLAIM: MISSED ***
  #3 -> 2 claims    #6 -> 2 claims    #8 -> 1 claim     #9 -> 3 claims
```
Every shot type, both makes and misses, on a gym/teams/uniforms the model has
never seen -- caught, except #5 (a MADE layup at 2:01-2:04). This is the first
generalisation evidence the project has ever had, and it is good news: the
detector half is not overfitted to HARD + TEST1.

### TEST 19 RESULT 2 — PRECISION: 17 claims for 9 shots. The dominant cause is NOT what we were chasing.

Two DIFFERENT failure modes, and separating them matters:

**(a) SPLIT ARCS -- a plain bug, and the bigger contributor.** Consecutive
claim gaps:
```
    454-469  ->   470-488    gap 1 frame   <- ONE flight, split into two arcs
   4485-4508 ->  4509-4518   gap 1 frame   <- ONE flight, split into two arcs
   2414-2426 ->  2437-2452   gap 11 frames
   8243-8251 -> 8278-8303 -> 8324-8351     gaps 27 and 21 frames
```
A single ball flight is being cut into two arcs and EACH half is then
classified as its own shot attempt. Both members of a 1-frame-gap pair are
claimed. That is not a perception failure and no player signal can fix it --
it is arc assembly, and it accounts for at least 4 of the 8 excess claims.
NOT investigated further today; it is now the cheapest known precision win.

**(b) GENUINE FALSE POSITIVES -- 4 claims match nothing at all:**
```
   1:36.1-1:36.7    2:06.3-2:06.8    4:05.8-4:06.5    4:11.2-4:11.6
```

### TEST 19 RESULT 3 — THE PRE-REGISTERED POSE RULE: 4/4 on the clean cases

Scoring only the cases where the truth is UNAMBIGUOUS:
```
  the 4 claims that match NO ground-truth shot (true false positives):
     1:36.1  predicted HAND (not a shot)   CORRECT
     2:06.3  predicted HAND (not a shot)   CORRECT
     4:05.8  predicted HAND (not a shot)   CORRECT
     4:11.2  predicted HAND (not a shot)   CORRECT
                                           -> 4 / 4

  the 4 real shots with a single unambiguous claim:
     0:53.4 (#2)  predicted rim (real)     CORRECT
     3:07.0 (#7)  predicted rim (real)     CORRECT
     3:40.1 (#8)  predicted rim (real)     CORRECT
     1:34.2 (#4)  predicted HAND           WRONG-or-ambiguous, see below
                                           -> 3 / 4
```
**THE RULE PASSED THE TEST THAT KILLED accel_y.** Predictions were registered
before the ground truth existed, on a clip the rule was not built on, and it
correctly rejected 4 of 4 genuine false positives while correctly keeping 3 of
4 clean real shots. accel_y failed this same kind of test immediately. n=8 is
small and this is NOT adoption -- but it is the first signal in this project
to survive a holdout, and it is now the leading candidate.

The single-frame and window variants agreed on 16 of 17 claims (differing only
at 4:34.8), so the window variant's extra stability cost nothing here.

**HONEST AMBIGUITIES, not resolved in the rule's favour:**
- 1:34.2-1:34.7 sits just AFTER DJ's #4 window (1:30-1:34) on a MISSED 3pt.
  It is equally readable as the shot (with slop in a human timestamp) or as
  the rebound off that miss. If it is the rebound, the rule was RIGHT and the
  score is 4/4 -- but that must not be assumed to make the result look better.
  Needs DJ.
- 2:06.3 was scored a false positive, but #5 (the MISSED shot, 2:01-2:04) is
  2.3s earlier. If that claim is actually #5 with timestamp slop, then v3 has
  9/9 recall and the rule wrongly suppressed a real shot. Needs DJ.
- The 3 claims at 4:34.8-4:38.4 fall inside #9's window, but DJ says #9
  happened OFF SCREEN. If the shot was not visible, those 3 claims cannot be
  it, and they are 3 more false positives (of which the rule flags 1-2).
  Needs DJ.

### TEST 19 STANDING VERDICT — MEASURED, NOT ADOPTED

- Recall generalises (8/9 on a new gym). Precision does not: 17 claims for 9
  shots, i.e. ~47% of claims are excess. A coach shown this today would see
  roughly twice the real number of shots.
- The two causes are now SEPARATED and they need different fixes: split arcs
  (a bug, cheap, no new data) and true false positives (the pose rule, which
  just passed its first holdout).
- Suite 223 green.

## RECALL DIAGNOSIS — why v3 missed 2 of 9 shots on TEST4 (2026-07-27)

TEST 19 found v3 misses shot #4 (3pt, 1:30-1:34, MISS) and #5 (layup, 2:01-
2:04, MADE). Both are now traced to root cause using only data already on
disk -- no GPU, no new footage. THE TWO MISSES HAVE DIFFERENT CAUSES.

### #4 (the missed 3-pointer): CHAIN FRAGMENTATION, not a detection failure

The ball IS seen throughout -- 141/160 frames at conf>=0.10 across the span.
But it never forms ONE continuous chain from release to arrival. Frame-by-
frame distance to the (correctly tracked, stable) hoop:
```
  f2654  515px  (release, far from hoop)
  f2702  923px  (still receding -- normal early flight/wind-up)
  ...  chain BREAKS here (gap to f2711), BREAKS AGAIN (gap to f2790) ...
  f2802   54px  (true arrival)
  f2826  151px  (this is the RIM BOUNCE, DJ-confirmed non-shot per TEST 19)
```
The true flight is real and coherent (515px -> 54px, monotonic approach) but
the chain BUILDER split it into 3 separate chains at the two gaps. Only the
LAST piece (2800-2824) is a fittable arc, and because IT starts already
36px from the hoop (the earlier, far-away portion belongs to a different,
rejected chain), classify_shot's arriving-vs-leaving heuristic (DECISIONS 25)
reads it as "originates near the hoop and leaves" -- a deflection -- and
rejects it. The heuristic cannot currently tell "the early part of my own
flight went missing" from "this genuinely started near the rim". Both
gaps (2695->2711, 16 frames; 2756->2790, 34 frames) exceed MAX_GAP_FRAMES=3
by a wide margin -- a real continuity loss, not a borderline case.

### #5 (the missed made layup): the KNOWN short-flight problem, reconfirmed on a 3rd clip

The ball reaches 25px from the hoop at f3704 (2:03.5, squarely in DJ's
window) -- a genuine arrival. But the chain containing it gets verdict
NO_CLAIM: every possible 8-point window inside it fails the physics gate,
and the whole-chain robust fallback also fails. The y-trace shows why:
```
  201 -> 194 -> 189 -> 186 -> 184 -> 183 -> 185 -> 187 -> 188 -> 193 -> 197
  -> 204 -> 206 -> 202 -> 197 -> 194 -> 193 -> 191 -> 191 -> 191 -> 193
  -> 196 -> 201 -> 205 -> 212 -> 211 -> 210 -> 210 -> 210
```
Down, up, down, up, down, flat -- a wobble/roll near the rim, not one clean
parabola. No 8-point stretch is monotonic enough to pass accel_y/rms_y.

THIS IS NOT A NEW FINDING -- it is DECISIONS 22 (2026-07-14) reconfirmed on a
third, independent clip: "layups fail the arc tracker: short + occluded ball
flight -> no parabola -> no arc." That measurement concluded a layup detector
should NOT be built on ball-arc data alone, and named the two live paths as
footage (out of scope) and pose estimation. TEST 16's pose rule already exists
but currently only REJECTS false positives (ball ends at hand vs rim) on
arcs the physics layer already claimed -- it has no role in cases where the
physics layer claims NOTHING. Using pose as a POSITIVE trigger (detect a
release motion directly, independent of the arc gate) is a different,
larger, unbuilt task.

### STANDING READ

Two different bugs need two different fixes, and neither is a detector-
quality problem (consistent with everything TEST 17's control run already
showed): #4 is fixable in the association/chain-merge layer (the split-arc
merge already built this session is the right FAMILY of fix, but does not
cover a gap this large). #5 is the same short-flight wall this project hit
in Milestone 2 and pose-as-trigger is now a two-clip-independent case for
being the next real lever, not a one-off suggestion.
NOT ADOPTED, nothing changed. Diagnosis only.

---

## TEST 20 — DEAD-BALL DETECTION VIA THE CLOCK'S RHYTHM (Time_out.mp4, 2026-07-26)

DJ's framing: "all the clocks are different, we have to figure out a universal
sign." Correct, and this test answers it -- the universal sign is not what the
clock LOOKS like, it is that **every basketball clock ticks once per second**.
So the state of play can be read WITHOUT OCR, without reading a single digit,
by asking only "is this patch of screen changing at 1 Hz?".

Clip: Time_out.mp4, 1920x1080 30fps, 155.6s, end of a quarter (Carroll 24 /
Milford 38, period 4). A THIRD scorebug style -- a broadcast overlay, unlike
TEST4's LED gym board and unlike HARD's. The rhythm method does not care.

### Step 1 — the clock LOCATES ITSELF

Grid the scorebug corner into 8px blocks, record which frames each block
changes in, keep blocks whose MEDIAN gap between changes is ~30 frames:
```
  blocks ticking at 1 Hz (median gap 26-34 frames): 14
  clustered at x 184-200, y 990-1014
```
Nothing else on a scoreboard behaves that way -- the score changes rarely,
fouls rarely, the animated logo changes continuously rather than in
one-second steps. Confirmed against a zoom of the graphic: that cluster sits
exactly on the clock's SECONDS digits. No per-gym model, no training, no
hand-typed coordinates.

### Step 2 — ground truth, read off the clock by eye

```
   0s, 20s   02.9    (frozen -- dead ball with 2.9s left in the quarter)
   40s-115s  08:00   (frozen -- the between-quarters break)
   125s      07:57   } running: 25s of game time across 25s of real time,
   150s      07:32   } exactly 1:1
```

### Step 3 — RESULT: pixel-change + temporal voting vs that ground truth

Changed-pixel count in the clock region per frame, thresholded, then "ticked
in >=3 of the last 4 seconds" (the same temporal-voting discipline TEST 14
used on the score, and for the same reason -- one bad second must not flip
the verdict):
```
     0s  ........................####................................
    60s  ............................................................
   120s  ....###############################
         # = live play    . = dead ball
  transitions: 24s -> LIVE, 28s -> DEAD, 124s -> LIVE
```
**Correct on 151 of 155 seconds.** The real break (0-123s) and the real
resumption (124s onward) are both found, and the resumption is located to
within ~2s of the true 122s.

### The one error, and it is instructive

24-27s reads LIVE and is not. That is the moment the board RESETS from 02.9
to 08:00 between quarters -- a genuine change in the clock region that is not
the clock running. Two honest notes:
- It is not noise and no threshold removes it; it is a real display event.
  Distinguishing "reset" from "running" needs either a longer sustained-tick
  requirement (real play lasts far longer than a graphic transition) or
  actually reading the digits.
- Deliberately NOT tuned away on this one clip. Lengthening the vote window
  until this clip passes is the accel_y mistake in miniature. The safe
  direction is noted instead: requiring a longer run makes the system call
  MORE things dead, which is the conservative error for our purpose.

### Second-order finding: the overlay is SEMI-TRANSPARENT

Court and players show faintly through the scorebug, so raw pixel differences
fire even when the digits are static -- visible as isolated one-second blips
throughout the frozen stretch. The 3-of-4 vote removes them because a running
clock ticks EVERY second while bleed-through is sporadic. Worth knowing for
any future work that reads this graphic: the background is never fully masked.

### PHANTOM-SHOT COUNT (2026-07-27) — and a finding that OVERTURNS the plan

Full detection + shot pipeline run over all 4667 frames. 6 shot claims total,
ALL of them inside 0.3s-20.4s -- before the clock even reaches its 08:00
reset. ZERO claims anywhere in the 40-115s huddle window, and zero in the
resumed-play tail (122-155.6s).

**Checked the actual frames before calling any of these phantoms, and it is
a good thing this was checked: all 6 are FREE THROWS.** Frame f=8 shows both
teams lined up along the free-throw lane in textbook formation, a referee
holding the ball at the top of the key, fouls 2-4 on the board. Frame f=209
shows a player mid-release at the line with the ball already near the rim.
Clock reads 02.9 (frozen) in both -- because THAT IS CORRECT: free throws are
played with a stopped clock, always, every level of the sport.

**This overturns the plan as stated in TEST 20's own writeup, not just
extends it.** "Clock frozen = dead ball = suppress shot claims" was this
session's working rule for the timeout-detection use case. Applied
BLANKET-STYLE to shot suppression, it would have DELETED EVERY FREE THROW IN
THE GAME -- a categorically worse error than the phantom-shot problem it
would have solved, since free throws are a real, common, scored shot type
and this project's whole discipline exists to avoid exactly this kind of
confident deletion.

**The real, narrower finding underneath it is still good news:** the actual
between-quarters BREAK (40-115s, the stretch DJ's ground-truth frame showed
players huddled at the bench with an empty court) produced ZERO shot claims
on this clip. So on this one sample, the pipeline did not need dead-ball
suppression at all -- the false-positive problem TEST 20 set out to solve did
not actually occur here. n=1 clip, not a general claim.

**REVISED UNDERSTANDING for any future dead-ball work:** the clock-rhythm
signal is still sound for what TEST 20 built it for (locating quarter/
timeout breaks). It must NEVER be used alone to gate shot claims -- a frozen
clock is the NORMAL, correct state during free throws, which are exactly
the moments a shot detector most needs to keep working. Any real dead-ball
shot-filter needs a second signal that free throws specifically produce and
scrambles/genuine breaks do not (candidates, unmeasured: the free-throw-line
player FORMATION itself; PERIOD/foul-count staying fixed distinguishes a
free throw from a period change; or simply: only suppress within the 40-115s
STYLE of window -- sustained clock-freeze on the order of a minute+ -- never
a a few-second freeze, since free throws take seconds and a real break takes
much longer).

### STATUS

NOT ADOPTED. The player-clustering cross-check (the 80s frame shows the
whole squad huddled at the bench with an empty court) is not built yet, and
now looks LESS urgent given the zero-phantom result above -- though still
worth a cheap look given n=1.

---

## SCOREBOARD PRESENCE ON TEST4 (2026-07-26) — and a false negative I walked into

DJ reported the scorebug FADES IN AND OUT on two clips and proposed demoting
the scoreboard to a secondary signal. This measured it on TEST4.

### FIRST RESULT WAS A FALSE NEGATIVE, and it is worth recording as such

spikes/scoreboard_presence.py watches v3's own scoreboard classes (Team
Points / Time Remaining / Period / Shot Clock) and returned:
```
  0/602 sampled frames have the scorebug (0.0%)
  1 blackout, longest 301.0s   <- i.e. "the graphic is never on screen"
```
That is WRONG, and it is exactly the failure mode DJ had just finished
describing. Extracting frames and LOOKING shows the scoreboard is plainly
there in the bottom-left of every frame checked. The 0/602 means "the
detector cannot see this scoreboard", not "there is no scoreboard".

I built a tool that concluded from absence on the same day I wrote the rule
saying never conclude from absence. Logged rather than quietly fixed, because
the lesson is the point: a null detection is only evidence once the detector
has been shown to work on that input.

### WHY IT WAS BLIND: the scorebug STYLE is completely different per clip

- HARD: a clean BROADCAST OVERLAY graphic ("15 - 12"), which is what the
  public training set contains and what TEST 14's reader was built for.
- TEST4: a rendered LED SCOREBOARD (7-segment amber/red/green digits, gym
  scoreboard styling) carrying game clock, PERIOD, FOULS, BONUS and **TOL
  (timeouts left)**.
v3's scoreboard classes fire on the former and are 100% blind to the latter.

### ACTUAL PRESENCE ON TEST4: 100%, no fade whatsoever

Re-measured WITHOUT the model -- crop the fixed bottom-left region and count
strongly-saturated bright pixels (the LED digits); court floor is neither:
```
  602 samples | LED-pixel count median 16828, min 5788, max 19332
  absent (below a generous 4207 threshold): 0 / 602  (0.0%)
```
The graphic never leaves the screen on this clip. So DJ's fade is CLIP-
SPECIFIC (it must be the broadcast-overlay clips), not a universal property.

### WHAT THIS CHANGES

- The scoreboard's real problem is GENERALISATION, not fading. Two clips, two
  entirely different scorebug styles, one detector that handles one of them.
  Any reader is per-clip until proven otherwise -- the same clip-dependence
  that TEST 3 found for the Hoop class and TEST 15 found for jersey colour.
  This is the third independent instance of the same lesson.
- DJ's "CONFIRM, NEVER DENY" rule is upheld and is now doubly load-bearing:
  it makes BOTH failure modes (a fade, and a reader blind to a new style)
  cost coverage instead of correctness.
- BONUS for the dead-ball work: TEST4's board shows **TOL (timeouts left)**.
  A decrementing TOL is a discrete, unambiguous "a timeout was called" marker
  -- stronger than inferring it from clock behaviour, though it marks the
  start and not the end. Only useful on boards of this style. Unmeasured.

---

## TEST 19 FINAL SCORE — DJ resolved the three ambiguities (2026-07-26)

DJ's rulings, given AFTER the predictions were logged:
```
  1:34      bounce off the rim              -> NOT a shot
  2:06      BLOB pass (baseline inbounds)   -> NOT a shot
  4:34-4:38 real MADE layup, 2 pts, but the shot went up OFF SCREEN
```

### These REVISE RECALL DOWNWARD -- v3 is 7/9, not 8/9

The claim at 1:34 was provisionally credited to DJ's shot #4 (the 1:30-1:34
missed 3pt). It is not that shot -- it is the rim bounce afterwards. So #4 has
NO valid claim and was MISSED. Corrected recall:
```
  #1 found   #2 found   #3 found   #4 *** MISSED ***   #5 *** MISSED ***
  #6 found   #7 found   #8 found   #9 found (descent only -- release was off screen)
  = 7 of 9
```
Both misses (#4 a missed 3pt, #5 a made layup) are on the record as the
detector's real recall gap on unseen footage. The earlier "8/9" in this log
was based on the unresolved 1:34 claim and is superseded.

### THE POSE RULE: 9 of 9 on every unambiguous case

Scored only where the truth is now certain, predictions registered in advance:
```
  CONFIRMED NON-SHOTS (5)                       predicted   result
    1:36.1  matches no real shot                 HAND        CORRECT
    2:06.3  BLOB pass          (DJ-confirmed)    HAND        CORRECT
    4:05.8  matches no real shot                 HAND        CORRECT
    4:11.2  matches no real shot                 HAND        CORRECT
    1:34.2  bounce off the rim (DJ-confirmed)    HAND        CORRECT

  CONFIRMED REAL SHOTS, single clean claim (4)  predicted   result
    0:53.4  (#2)                                 rim         CORRECT
    2:29.5  (#6, post-merge)                     rim         CORRECT
    3:07.0  (#7)                                 rim         CORRECT
    3:40.1  (#8)                                 rim         CORRECT
                                                             = 9 / 9
```
DJ's two rulings BOTH went the rule's way, and neither was knowable when the
prediction was made. This is the second holdout the rule has survived and the
first one where the hard cases (a rim bounce and an inbounds pass -- the exact
two categories ball physics provably cannot separate) were adjudicated by a
human against a locked-in prediction.

### STILL NOT CLEAN — what the 9/9 does NOT cover

- **A PROBABLE MISS the rule got wrong:** claim 0:15.7-0:16.3 is, by the same
  y-motion reading DJ just independently confirmed at 1:34 (descend, reverse,
  descend), the rim bounce of shot #1 -- and the rule called it "rim" (a
  shot). Not DJ-adjudicated, so not scored, but it should be assumed WRONG
  rather than quietly left out: that would make the honest tally 9/10.
- **The #9 cluster (3 claims, 4:34.8-4:38.4) is unresolved by construction.**
  DJ confirms one real made layup there whose release was OFF SCREEN, so at
  most one claim is the shot and >=2 are excess. The rule splits them
  (rim / HAND / rim), which cannot be right for a single made basket.
- n is still small: 9 adjudicated events on one holdout clip.

### STANDING VERDICT

Recall 7/9 and precision 8 excess claims out of 16 -- the detector half is
NOT ready, and the recall misses are the more serious of the two because no
downstream rule can recover a shot that was never seen. The pose rule is now
the strongest candidate this project has produced (9/9 pre-registered on a
holdout, vs accel_y which died on first contact), but it is STILL NOT ADOPTED:
it has a probable miss on rim bounces, no answer for the off-screen cluster,
and has never been wired into run_clip.

---

## SPLIT-ARC FIX (2026-07-26, from TEST 19's double-counted shot)

### First, a CORRECTION to TEST 19's write-up

TEST 19 said the adjacent-claim pairs were "one flight split into two arcs...
at least 4 of the 8 excess claims". THAT WAS TOO HASTY and is wrong. Looking
at the actual ball positions, the two pairs are DIFFERENT things:

- **2:27 (4485-4508 / 4509-4518) -- a genuine split.** One smooth 33-point
  parabola: ball rises (y 434->99), apex, falls (99->180), inside ONE chain
  (609). The greedy growth loop cut it mid-descent.
- **0:15 (454-469 / 470-488) -- NOT a split.** The ball descends (y 30->152),
  then RISES (152->98), then descends again. That is a shot arriving, bouncing
  off the rim, and rebounding -- two genuinely different motions, and the
  chain builder had ALREADY put them in separate chains (41 and 42) because
  the direction reversal breaks its motion prediction.

So: ONE split-arc bug, not four. The rest of the excess claims are the
rebound/bounce family (what TEST 16's pose rule targets).

### Root cause

`classify_chain`'s growth loop is greedy: it takes the longest parabola it can
fit from point i, then RESTARTS at the next point (`i = i + len(seg)`). A real
flight is not a perfect parabola in image space over a long span -- camera pan
and perspective bend it -- so a long flight can fail the residual gate as a
WHOLE while both halves pass. Measured on the 2:27 flight: whole-span
accel_y 1.273 (fine), accel_x -0.124 (fine), **rms_y 5.31 (over the gate)**;
the robust fitter recovers the whole thing by dropping 3 points.

### The fix (spikes/ball_trajectory.py, `_merge_split_arcs`)

After the growth loop, merge consecutive arcs WITHIN a chain when the merged
span still fits (ordinary fit, else the existing robust fitter). Two guards:
- gap <= MAX_GAP_FRAMES+1 (the same continuity tolerance the chain builder
  uses), so two flights with a real hole between them are never welded;
- the merged span must actually fit, else they stay separate.

NO direction/bounce check was needed, and that is the point: the chain builder
already separates a bounce into its own chain, so a bounce never reaches this
function. The fix is safe by construction rather than by a special case.

SECOND, SMALLER BUG FOUND WHILE FIXING THE FIRST: taking the robust fit's
`kept[]` range as the arc's extent (the existing convention) moved the merged
arc's apparent RELEASE POINT 25px closer to the rim -- flipping DJ's 3-pointer
to "layup" -- and discarded the frame nearest the rim (min_dist 39 -> 52). A
dropped point is an outlier for fitting a CURVE but is still a real detection
that defines where the flight began and ended. The merge now keeps the full
span's extent and uses the robust fit only for the curve coefficients.

### Verification

```
  suite:  228 passed (226 + 2 new)
  TEST4:  (4485,4508)+(4509,4518) -> ONE claim (4485, 4518, 'far',
          'jumpshot', 39.1, 'observed', 3) -- correct span, correctly typed
          a jumpshot (DJ: made 3pt), best rim distance kept
  HARD:   BYTE-IDENTICAL to pre-fix (same 2/2 shots, same spans, same
          min_dists, same 3 false positives)
  TEST1:  BYTE-IDENTICAL to pre-fix (same 5/5, same spans, same min_dists)
```
Two regression tests added, both using the LITERAL 33-point TEST4 chain (same
convention as TEST 1's real shot-B chain -- a synthetic parabola cannot
reproduce this bug, because the split is caused by real perspective bend).
VERIFIED THE TESTS CATCH THE BUG: with `_merge_split_arcs` stubbed out, the
flight splits back into 2 arcs and the test fails as intended.

### Honest scale of the win

```
                     before fix   after fix
  claims                17           16
  real shots found       8/9          8/9
  excess claims           9            8
```
It removed exactly ONE duplicate -- real, permanent, and cheap, but modest,
exactly as the corrected diagnosis above predicts. The remaining 8 excess
claims are: 4 matching no real shot at all (pose rule flagged 4/4), 2 rebounds
immediately after real shots (#1, #3), and 3 clustered around #9 which DJ says
happened OFF SCREEN and are still unresolved.

---

### [2026-07-26 05:00] TEST 17 PAUSED (DJ stopping the pod overnight)

State at pause, so this is resumable without re-deriving anything:
```
  bracket_yolov8l_stock (CONTROL)  DONE   114 ep, 1.98h  -- gated, LOSES to v3
  bracket_yolo11l_b2               DONE   172.6 min      -- val below, NOT yet gated
  bracket_yolo12l_b2               KILLED at epoch 123/150 (last.pt saved -> resumable)
  bracket_yolo26l_b2               NEVER STARTED
```
yolo11l val (32-img public set -- remember this set scored the control ABOVE
v3 and the control turned out worse, so this predicts nothing):
```
  all-mAP50 0.884   Ball P 1.0  R 0.581  mAP50 0.692
  (control  0.892          P 0.797 R 0.667 mAP50 0.707)
  (v3       0.877          P 0.99  R 0.556 mAP50 0.618)
```
Weights pulled local: models/bracket_yolov8l_stock.pt, models/bracket_yolo11l.pt
-- so yolo11l's clip gate can be run on CPU with NO pod. Everything else lives
on the network volume (/workspace/runs/*), which survives Stop and Terminate.
GPU verified idle (0%, 2MiB) before the pod was stopped.

RESUME NOTES: yolo12l can be continued with resume=True from
/workspace/runs/bracket_yolo12l_b2/weights/last.pt; yolo26l needs a fresh run.
Both are LOW priority now -- TEST 19 showed the dominant precision problem on
a real 5-minute clip is SPLIT ARCS (a bug, no GPU needed), not detector
quality, and the control already demonstrated a fresh training run lands on
the same false positives.

### RUN 2 — RESTARTED for a confound (cost ~1.1h of compute)

AutoBatch chose batch=1 for yolo11l but batch=2 for the control. Batch size
changes training dynamics, so an architecture compared at batch 1 against a
control at batch 2 is confounded -- exactly the comparison the control exists
to make, made meaningless. Killed at epoch ~24 and restarted with batch PINNED
at 2 for all remaining candidates (bracket2.py). The completed control ran at
batch 2 and stays valid. Queue: yolo11l_b2 -> yolo12l_b2 -> yolo26l_b2.

---

## TEST 21 — BALL-TO-PLAYER TOUCHES (the join that never existed)

Purpose: the system saw the ball and saw the players and never said WHICH
PLAYER HAD THE BALL. This builds that join. A TOUCH is one player holding the
ball until she gives it up — NOT a possession, which is the team-level concept
phase2/possessions.py already owns (DJ's correction, 2026-07-27).

Thresholds FROZEN in tasks/todo.md BEFORE the first run (the accel_y guard),
and asserted by a test so they cannot drift:
    HOLD_GATE_BODY_FRAC 0.30   MARGIN_BODY_FRAC 0.15
    MIN_TOUCH_FRAMES    6      MAX_GAP_FRAMES   8
Distances are in BODY HEIGHTS, not pixels, so the gate means the same thing at
both ends of the floor. Distance is to the BOX, not the feet — the ball is in
the hands. point_to_bbox_dist reused from shot_attempts.py, not rewritten.

### [2026-07-27] Gate: regression suite before and after

```
$ .venv/Scripts/python.exe -m pytest tests/ -q
261 passed in 4.03s
```
(230 before this work + 31 new ball-touch tests.)

### [2026-07-27] MEASURED — pending DJ review

```
                              TEST1            HARD
  answerable frames           461 (120..580)   601 (600..1200)
  held                        40.1%            30.3%
  no_ball                     47.7%            51.6%
  too_far                      4.1%             9.2%
  contested                    8.0%             9.0%
  TOUCHES (>=6 frames)        8  (5.2s)        10 (5.4s)
  ...nameable                 3                6
  identity join               1 attributed     6 attributed
                              7 review_item    4 review_item
```

TEST1 touches:
```
  f179..198  t14   0.6s  unnamed  review_item  (-4,42)ft  <- OFF_COURT
  f233..262  t14   0.6s  unnamed  review_item  (-2,38)ft  <- OFF_COURT
  f345..399  t67   1.8s  #32      review_item  (6,34)ft
  f408..438  t395  1.0s  unnamed  review_item  (17,9)ft
  f453..476  t67   0.5s  #32      review_item  (30,24)ft
  f505..516  t67   0.2s  #32      attributed   (41,27)ft
  f520..525  t875  0.2s  unnamed  review_item  (51,32)ft
  f533..539  t875  0.2s  unnamed  review_item  (59,31)ft
```

HARD touches:
```
  f616..658   t14    1.2s  unnamed  attributed   (8,29)ft
  f671..682   t4     0.3s  #10      attributed   (12,15)ft
  f808..816   t55    0.3s  #1       review_item  (-0,16)ft
  f888..906   t930   0.6s  #1       attributed   (-1,14)ft
  f1005..1013 t1748  0.3s  unnamed  review_item  (3,16)ft
  f1017..1052 t1963  1.0s  #3       attributed   (9,14)ft
  f1063..1077 t2475  0.2s  #3       attributed   (24,5)ft
  f1082..1100 t2475  0.6s  unnamed  review_item  (34,3)ft
  f1116..1123 t2475  0.3s  #3       attributed   (53,9)ft
  f1147..1164 t1502  0.6s  unnamed  review_item  (66,41)ft
```

### TWO BUGS FOUND BY RENDERING THE VIDEO, NOT BY THE NUMBERS

1. REFEREES CREDITED WITH THE BALL. HARD t3 — a referee DJ labelled himself —
   held a 0.5s "touch" while the ball was up at the rim on a shot. Fixed with
   the filter the pipeline ALREADY uses for seeding (roster.load_ref_tracks,
   DJ's own ref/bench labels), excluded from candidacy entirely so a ref
   standing beside a handler cannot force a CONTESTED abstention either.
   NOT a threshold change — the frozen numbers are untouched.
   EFFECT ON HARD: −1 fake touch, +2 REAL touches revealed (f808 #1 appeared;
   #3's f1017 touch grew 0.8s → 1.0s). Nameable touches 5 → 6.
   This reproduces TEST 15's finding in a new place: junk detections were
   being carried silently into a downstream layer.

2. identity_id WAS ABOUT TO BE PRINTED AS A JERSEY NUMBER. The first TEST1 run
   reported "identity 13" / "identity 39" — internal per-window counters. The
   girl is actually #32, and TEST1 has a real #13 on the roster, so the output
   would have read as a confident, wrong jersey call. Fixed by joining
   (window, identity_id) → roster_number via {clip}_ocr_confirms.json, the
   same key stage8_box_score uses. Unnamed identities print "unnamed" and
   never leak a raw id. Pinned by a test.

### FLAGGED, NOT SILENTLY FIXED
TEST1 still credits 1.2s (2 of 8 touches) to t14, an UNLABELLED referee.
Reported loudly as OFF_COURT rather than deleted: dropping all off-court
touches would also drop real INBOUNDS PASSES, thrown from behind the baseline
by real players. DJ's call. Cheapest fix = DJ marks t14 'ref' (one click in
the review bundle he already has) and the existing filter handles it.

### NOT DONE ON PURPOSE
No "ball near the rim = nobody holds it" rule. It would have cleaned up the
referee case, which is exactly why it is dangerous — a heuristic invented
AFTER seeing a bad result is the accel_y mistake. Candidate for a future
gated test, not slipped in.

### OPEN
DJ has not watched the overlays yet. Nothing here is adopted as a stat.
    spikes/out/TEST1_ball_touches_overlay.mp4   (15.4s)
    spikes/out/HARD_ball_touches_overlay.mp4    (20.0s)
Colour key: yellow dot = ball; GREEN = credited this frame; DARK GREEN = still
her touch but ball not visible (bridged, not evidence); ORANGE = looked like a
hold, too brief to count, thrown away; RED = contested, nobody credited.

### NEW EVIDENCE FOR AN EXISTING DECISION
~50% of frames on BOTH clips have no ball detection at all. That is now the
measured bottleneck on this whole feature, and it is a fresh argument in the
v3-weights adoption question.

---

## TEST 22 — SEEN vs FILLED IN, and WHO SHOT IT (DJ's "guessing" proposal)

DJ, after confirming the TEST 21 overlays are right: "the girl last seen with
the ball has the ball until proven otherwise." He asked for two builds and
pushed back on two others. Suite 261 → 268 green.

### [2026-07-27] BUILT: observed vs inferred seconds kept apart

Every touch now reports observed_seconds / inferred_seconds / total_seconds.
Rationale: an assumption must never share a bucket with an observation — the
same rule the identity layer already applies with confirmed/candidate.

```
TEST1  6.2s total = 5.2s SEEN + 1.0s FILLED IN (16% inferred)
HARD   6.3s total = 5.4s SEEN + 0.9s FILLED IN (14% inferred)
```
At today's gap the feature is ~85% direct observation. This number is now the
dial that shows how much any future bridging change costs in honesty.

### [2026-07-27] BUILT AS A COMPARISON (NOT ADOPTED): who shot it

Two methods:
  TODAY    shot_attempts.find_release — extrapolate the arc's parabola back
           10 frames, credit the nearest body within 120px. Knows SPACE only.
  PROPOSED ball_touch.shooter_from_touches — the last player actually SEEN
           holding the ball before the arc. Knows TIME as well as space.

```
WHO SHOT IT -- HARD
          arc  DJ-verified   TODAY (standing)   PROPOSED (held it)   verdict
     1188..1213        YES              t1502                t1502     AGREE
  (3 other arcs fall outside the tracks span — neither method can answer)

WHO SHOT IT -- TEST1
     164..184          YES                t14    no touch recorded  only TODAY
     236..250          YES                t14                  t14     AGREE
  (2 other arcs outside the tracks span)
```

HONEST LIMIT, FOUND WHILE BUILDING: GROUND_TRUTH records WHICH arcs are real
shots, NOT WHO TOOK THEM. This project has no shooter ground truth, so no
winner can be declared. The script surfaces disagreements for DJ; his answers
would create that ground truth.

### THE FINDING THAT MATTERS — and it is not about which method wins

**t14 is the UNLABELLED REFEREE** (same track flagged OFF_COURT in TEST 21).
BOTH methods credit a referee with TEST1 shots. find_release has no ref filter
at all, so this contamination is ALREADY IN THE SHIPPED SHOT LAYER — a third
place one unlabelled track is poisoning. Cheapest fix remains one DJ click.

MECHANISM WORTH TESTING FURTHER: on arc 164..184 the proposed method REFUSED
the referee, because his only proximity to the ball came AFTER the shot went
up (the ball arrived at the rim where he stands). find_release cannot refuse
that — a body parked under the rim is forever "near" a shot. Knowing about
TIME is a real advantage over knowing only about SPACE. n=1: a mechanism to
test, not a result to bank.

### A COUNTING BUG IN MY OWN COMPARISON (found and fixed before reporting)
Out-of-span rows were being tallied as DISAGREE even though neither method can
answer outside the tracks span. Fixed; the printed tallies now skip them.

### WITHDRAWN: "a contested frame should break the bridge" — my proposal, wrong
DJ pushed back and was right.
1. The "proven otherwise" rule ALREADY EXISTS: build_touches ends a run the
   instant a DIFFERENT holder is credited. Evidence of a change, not suspicion.
2. A scrum is ambiguity, not proof. Breaking on it deletes real time in the
   common case and adds nothing in the case it was meant to catch.
3. My "6 of 9 bridges cross a contested frame" figure counted bridges passing
   NEAR ambiguity, not bridges that got it WRONG. I presented a risk indicator
   as an error rate. Bad analysis.
RESIDUAL RISK (real, stated): a DOUBLE invisible handover (A→B→A) credits A
with B's seconds. Requires two invisible changes in one gap; rare, and its
likelihood grows with gap length.

### MY TEST OF "WAIT LONGER" ANSWERED THE WRONG QUESTION
TEST 21's sweep measured HOW MUCH would be filled in, not whether the fill-in
is CORRECT. Calling the extra 3.4s "invented" presumed the answer: if she
really did have the ball, those seconds are TRUE and the old setting was
UNDERCOUNTING. The table cannot separate those cases and should not have been
presented as settling it.
RIGHT TEST (cheap, renderer already exists): render the long-bridge version and
have DJ watch whether the box stays on the RIGHT girl through the gaps.
NEEDED EITHER WAY: a ceiling drawn from BASKETBALL — no possession lasts 30
seconds — not from an analyst's caution.

---

## TEST 23 — DJ's "until proven otherwise" rule at his 15s ceiling

DJ set the ceiling himself, from basketball: 15 seconds. Built as
spikes/long_bridge_test.py with SUFFIXED artifacts so the canonical outputs are
never clobbered. Nothing adopted. Suite 268 → 271 green.

WHAT 15s MEANS ON THESE CLIPS: TEST1's answerable window is 15.4s, HARD's is
20.0s — so a 15s ceiling is EFFECTIVELY NO LIMIT here. Deliberately the worst
case: if the box still follows the right girl under no limit at all, the rule
is safe at any smaller number.

### [2026-07-27] MEASURED

```
TEST1                touches   SEEN s  FILLED s  % filled
  now (0.27s wait)         8      5.2       1.0       16%
  DJ's rule (15s)          7      5.3       4.4       45%
HARD
  now (0.27s wait)        10      5.4       0.9       14%
  DJ's rule (15s)         10      5.7       3.5       38%
```

The SEEN column barely moves (5.2→5.3, 5.4→5.7). The rule does not find new
evidence; it extends credit ACROSS existing evidence. Whether that credit is
CORRECT is what the video answers and the table cannot — which was the whole
point of TEST 22's correction to my own TEST 21 analysis.

Biggest single stretch, HARD: t4 (#10) f671..738 = 0.4s seen + 1.9s filled.

### THE FINDING THAT CHANGES THE ORDER OF WORK

Rendered TEST1 at 15s and inspected f290: the box sits on the REFEREE (t14) for
2.5 straight seconds while the ball is nowhere near him and the play has moved
down court. His fake touch grows 0.6s → 3.2s.

**DJ's rule AMPLIFIES whatever error is already in the input.** It is not wrong
in principle — it is a multiplier, and one of the things it currently
multiplies is a known, already-diagnosed bug.

CONSEQUENCE: judging the 15s video before labelling t14 means judging the
referee bug, not the rule. Correct order: label the ref → re-render → judge.

### TWO BUGS IN MY OWN CODE, found while building this

1. WINDOW-CROSSING TOUCHES LOST THEIR JERSEY. identity_id is scoped PER WINDOW,
   so a touch spanning a window boundary picks up a second identity record for
   the SAME girl. attribute() treated that as an identity split and refused to
   name her — TEST1's #32 lost her number the moment a merged touch spanned
   windows 0 and 1. Fixed: two records carrying the SAME jersey are one girl;
   different jerseys still refuse; a named record plus an unnamed one still
   refuses (an unnamed record cannot be proven to be her).
2. THE RENDERER DISAGREED WITH ITS OWN DATA. render_ball_touches recomputed
   per-frame verdicts WITHOUT the ref exclusions the touches were built with,
   so the video could show a state the JSON did not contain. Fixed by passing
   the exclusion set through; it now defaults to the touches doc's own
   excluded_tracks.
Both pinned by tests.

### EVIDENCE PRODUCED FOR DJ
    spikes/out/TEST1_track14_who_is_this.png   — three crops of t14: striped
    shirt, standing under the basket; at f250 the ball is above his head at the
    rim, which is exactly why the system credited him.
    spikes/out/{TEST1,HARD}_ball_touches_overlay_gap15s.mp4

### OPEN
DJ to confirm t14 is a referee (he does not know the review-bundle workflow, so
the offer is: he confirms from the crop and the single label gets written).
Then re-run and judge the 15s videos.

---

## TEST 24 — referee labelled, shot layer fixed, and a FAIR test of DJ's rule

DJ confirmed from spikes/out/TEST1_track14_who_is_this.png that t14 is a
referee (2026-07-28). Label written to TEST1_decisions.json after a backup
(TEST1_decisions.backup-20260728-pre-t14ref.json); all 20 prior labels verified
byte-identical afterwards. Suite 271 green throughout.

### [2026-07-28] FIX 1 — the shot layer had NO referee filter at all

find_release credits whichever tracked body sits nearest the back-extrapolated
release point, and never filtered non-players. A body parked under the basket
is permanently "near" every shot, making it the worst possible unfiltered
candidate pool. TEST1's referee was being named as the SHOOTER on two
DJ-verified shots. Fixed in ball_stages.stage_shot_attempts using the same
human ref/bench labels the seeding and touch layers already use.

```
TEST1 shooters, BEFORE          TEST1 shooters, AFTER
  164..184  -> t14 (REFEREE)      164..184  -> t38 (a real player)
  236..250  -> t14 (REFEREE)      236..250  -> t49 (a real player)
```

### [2026-07-28] FIX 2 — the referee is out of the touch layer

```
TEST1 touches   BEFORE  8 touches, 6.2s (5.2 seen + 1.0 filled)
                AFTER   6 touches, 4.5s (4.0 seen + 0.5 filled)
```
The total went DOWN by 1.7s. That is the fix working — the removed time was
fiction, and the OFF_COURT warning is now gone entirely. (Standing constraint:
CORRECTNESS OUTRANKS COVERAGE; a number falling after a fix is normal here.)

### [2026-07-28] DJ's 15s rule, re-tested on CLEAN input

```
TEST1                touches   SEEN s  FILLED s  % filled
  now (0.27s wait)         6      4.0       0.5       12%
  DJ's rule (15s)          6      4.2       2.6       38%
HARD
  now (0.27s wait)        10      5.4       0.9       14%
  DJ's rule (15s)         10      5.7       3.5       38%
```
Touch COUNT is now identical under both settings on both clips — the rule no
longer merges anything on TEST1, it only extends credit. SEEN barely moves
(4.0→4.2, 5.4→5.7), confirming again that the rule finds no new evidence.

### EYEBALL OF THE TWO RISKIEST BRIDGES (TEST1, 15s)
  f490, inside #32's 1.4s bridge:  **CORRECT.** The box is on green #32 mid-
      drive with the ball visibly at her hip and white #13 defending her. The
      detector lost the ball; the rule held the right girl. This is DJ's rule
      doing exactly what he said it would.
  f220, inside t49's 0.7s-filled / 0.2s-seen bridge:  **GENUINELY AMBIGUOUS.**
      A rebound scrum in the paint, four bodies tangled, ball in the middle.
      Not obviously right or wrong even to a human. Note the system reports it
      as "unnamed body / review_item" — it is not claiming a named stat.
So the rule's best case is clearly good and its worst case is a scrum, which is
the failure mode DJ predicted himself. Ratio matters: 0.2s of evidence
supporting 0.7s of credit is the shape to watch, not the absolute gap length.

### OPEN
DJ to watch {TEST1,HARD}_ball_touches_overlay_gap15s.mp4 end to end and decide
whether 15s is adopted, or a smaller ceiling, or an evidence-ratio rule instead.

---

## TEST 25 — FLICKER GUARD (DJ's occlusion worry, measured then built)

DJ (2026-07-28): "if a girl puts up a shot but it flickers to another girl,
does the one it flickers to get credited with the shot?"

Measured FIRST with spikes/flicker_check.py. A flicker has a precise signature:
the credited holder goes A → B → A. The ball did not change hands twice in a
fifth of a second; B's body passed between the camera and the ball.

### [2026-07-28] MEASURED — how long does a real flicker last?

```
TEST1   3 flickers:  1 frame ×2,  2 frames ×1
HARD    3 flickers:  1 frame ×2,  5 frames ×1
```

**HARM 1 (DJ's question — can B steal a touch?): does NOT happen today.**
Every flicker on both clips is shorter than MIN_TOUCH_FRAMES=6, so the existing
floor already deletes all of them. One flicker sat within the shooter window
before a claimed shot (TEST1 f532, 1 frame, before the arc at f581) and was too
short to steal it.

**BUT THE MARGIN IS ONE FRAME.** HARD's longest flicker is 5; the floor is 6.
That is luck, not safety — a 6- or 7-frame occlusion on another clip sails
straight through. DJ's instinct to want a guard was correct even though the
failure has not fired yet.

**HARM 2 (not previously flagged, and it fires every time):** a flicker that is
itself discarded STILL ended the real holder's run and started a new one.
Measured 3 times per clip. It cost credited time on both sides of the blip.

### [2026-07-28] BUILT — a handover must prove itself

A change of holder is now only accepted once the new holder has been credited
for MIN_TOUCH_FRAMES. Below that it is noise: it cannot start a touch, and it
cannot break the current one. Unconfirmed frames are credited to NOBODY —
abstention, not a gift to the first girl.

THE THRESHOLD IS MIN_TOUCH_FRAMES ITSELF — no new constant, and nothing fitted
to the flickers just observed (the accel_y guard). The reasoning stands alone:
**a run too short to prove she HELD the ball is too short to prove she TOOK
it.** Same shape as possessions.detect()'s HOLD_S, which already makes a
court-side switch prove itself before counting.

```
                   touches        SEEN s        totalling
TEST1   before        6             4.0            4.5s
        after         5             3.8            4.6s
HARD    before       10             5.4            6.3s
        after         8             5.0            5.8s
```
The touch COUNT falling is the HARM 2 fix: spuriously split touches rejoined
(TEST1's two t875 fragments became one; HARD's t2475 f1063..1077 + f1082..1100
became one f1063..1100). HARD's total falls 0.5s because frames that were
flickering between two candidates now credit nobody instead of the first one.

Shooter attribution unchanged by the guard on both clips (HARD still AGREE on
1188..1213; TEST1 unchanged). Suite 271 → 276 green.

### STILL OPEN
The guard makes a flicker structurally unable to steal a touch AT ANY LENGTH
BELOW 6 FRAMES. A sustained mis-credit (≥6 frames of the wrong body) would
still get through — that is not a flicker, it is a tracker/occlusion failure of
a different class, and nothing here claims to catch it.

---

## TEST 26 — DJ's 15s rule ADOPTED, and run_clip verified end to end

DJ watched TEST1_ball_touches_overlay_gap15s.mp4 end to end (2026-07-28):
"Yes its on the right girl the whole time." That is the eyeball gate this
project requires, so MAX_GAP_SECONDS = 15.0 is now the pipeline default
(expressed in SECONDS because that is the unit DJ reasoned in; frames derived
per clip from its own fps). Suite 276 → 277.

### [2026-07-28] run_clip TEST1 end to end — FIRST full run since ball touches
### was added to the pipeline. Exit 0.

```
calibration:        keyframe consistency 7.7 -> 0.6 px
                    landmark court-fit mean 0.15 ft / max 0.35 ft
player_events:      12904, all identity-stamped
shot layer:         4 attempts, 3 located
ball touches:       5 touches, 8.0s = 4.2s SEEN + 3.7s FILLED (47%)
```
The whole chain works with the new stage in it. Verified rather than assumed —
the stage had only ever been exercised standalone.

### A REAL COST OF THE 15s RULE, surfaced by this run

```
identity join, short setting:  {review_item: 4, attributed: 1}
identity join, 15s setting:    {review_item: 5, attributed: 0}
```
**Adopting DJ's rule cost TEST1 its only confidently-attributed touch.**
Mechanism, and it is not a bug: at 15s, touches MERGE (f453..476 + f505..516
became one f453..516). A longer touch covers more frames, so it is more likely
to span a stretch where the identity was not CONFIRMED — and attribute() only
says "attributed" when every credited frame agrees on one girl AND every one of
them was confirmed.

Arguably the merged answer is MORE honest: it is one continuous hold, part of
it is not confidently identified, so the whole hold is uncertain. But it is a
genuine trade — DJ's rule buys longer, truer-looking touches and pays in
confident attribution. Both halves of that must stay visible; the
observed/inferred split already makes the first half visible, and this line
makes the second half visible.

NOT CHANGED IN RESPONSE: attribute() was not loosened to recover the number.
Loosening a confidence rule because a separate change made it bite is exactly
how a stat launders.

### WEB-APP GAP (measured, not assumed)
The app's contract is spikes/out/{clip}_measured_stats.json, built by
measured_stats.generate() from box_score + shot_locations + shot_attempts and
read by lib/measuredStats.ts. **It contains no touches.** So everything built
in TESTs 21-26 is currently invisible to the app. Wiring touches into that
contract is the one piece between here and a demo that shows this work.

---

## TEST 27 — HOW MUCH CLICKING IS NECESSARY? (TEST1 says a lot less; HARD says no)

Motivation (measured 2026-07-28): calibration clicking is the binding constraint
on the whole project, not accuracy. TEST1 58 marks → 15.4s; HARD 59 → 20.0s;
TEST2 70 → 12.0s. ~10 clicks per 3 seconds. A 2-min clip ≈ 360 clicks, a full
game ≈ 5,700. Every stat the product wants needs LENGTH.

The ~100-frame keyframe spacing was a WORKFLOW CONVENTION
(tasks/calibration_generalization_log.md: "keyframes ~100–150 apart"), never a
measured requirement. spikes/keyframe_thinning_test.py is the first measurement.

METHOD (true holdout, the LOO discipline from the court-dimension work): refit
using only a SUBSET of keyframes — dropping a keyframe from s2.KEYFRAMES
automatically drops its clicked marks from the fit — then, for each DROPPED
keyframe, do exactly what the live path does (SIFT-match it to the nearest KEPT
keyframe, compose) and project ITS OWN held-back marks to court feet against
the rulebook position. "If DJ had not clicked here, how far off would the court
be?" Bar: 0.29 ft is the "glued" benchmark, 0.94 ft is what DJ called BROKEN.

### [2026-07-29] TEST1 — looks like a big win

```
keep [120,320,500,580]  40/58 clicks (31% fewer)  worst held-out 0.33 ft  PASS
keep [120,420,580]      29/58 clicks (50% fewer)  worst held-out 0.30 ft  PASS
keep [120,580]          17/58 clicks (71% fewer)  worst held-out 0.33 ft  PASS
```

### [2026-07-29] HARD — DOES NOT GENERALIZE. The win is clip-specific.

```
keep [600,800,1000,1200]  32/59 (46% fewer)  worst 0.99 ft   MARGINAL
keep [600,900,1200]       25/59 (58% fewer)  worst 0.95 ft   MARGINAL
keep [600,1200]           14/59 (76% fewer)  worst 39.14 ft  CATASTROPHIC
```
At two keyframes HARD's in-sample fit itself collapses to 26.79 ft and the
project's OWN pre-existing guardrail fires: "WEAK PAIR FLAG: 600->1200 inlier
ratio 0.039 < 0.6". TEST1's equivalent pair scored 0.317 — weak but survivable.

**CONCLUSION: clicking CANNOT be safely halved as a blanket rule.** Even the
mildest thinning puts HARD at ~1 ft, the error DJ judged broken by eye. This is
the project's own recurring lesson landing again — one clip is not evidence.

### WHY THEY DIFFER, and what it implies
HARD pans 3.6 px/frame median vs TEST1's 0.8 (4.5x, measured in the individual-
tracker work). More camera motion means two keyframes the same number of FRAMES
apart are much further apart in VIEW, so SIFT degrades faster. The right spacing
is therefore driven by CAMERA MOTION, not by a frame count — and this project
already owns the instrument that measures it: the adjacent-pair inlier ratio,
with a 0.6 threshold already coded in stage2_multikeyframe.py.

HYPOTHESIS FOR A FUTURE GATED TEST, deliberately NOT built now (it is a rule
invented after seeing results — the accel_y trap): space keyframes as far apart
as the inlier ratio allows rather than every ~100 frames, and let each clip
self-tune. On TEST1 that would cut clicks materially; on HARD it would cut few
or none. It must be FROZEN and tested on a clip it was not built on.

---

## TEST 28 — A GATE BROKE FROM OUTSIDE THIS TRACK (not a change made here)

`spikes/ball_trajectory.py` was modified at 2026-07-29 00:23 — after this
track's last green suite run and by different work (the CHAIN-FRAGMENTATION fix
the handoff listed as open: `_merge_gapped_chains`, MAX_CHAIN_MERGE_GAP_FRAMES
= 40). Nothing in the touches/app work edited that file.

It breaks `test_integrated_chain_reproduces_test10_on_the_saved_test1_v2_log`,
whose docstring says: "Any drift here means the integration changed the verified
behavior -- fix the integration, never this expectation." The expectation was
NOT changed and the other track's file was NOT reverted; this is a report.

```
                    GATE (TEST 10)                     NOW
 shot A   (58, 70, jumpshot, 118.1, extrapolated)  (57, 93, jumpshot, 61.4, observed)
 shot B   (164,184, layup, 15.3, observed)         unchanged
 shot C   (236,250, layup, 27.2, observed)         unchanged
 shot D   (581,589, layup, 18.4, observed)         (581, 601, layup, 18.4, observed)
```

TWO READINGS, both fair, which is why a human should settle it:
  BETTER — shot A's arrival is now OBSERVED at 61.4 px instead of EXTRAPOLATED
  at 118.1. The merge recovered real flight the old chaining threw away, which
  is exactly what the fix was built to do.
  WORSE — both merged spans now run PAST DJ's ground truth. Ground truth has
  shot A at 55..74 (new end 93, +19 frames) and shot D at 581..592 (new end
  601, +9). A 40-frame merge gap may be welding a real flight to whatever came
  next. Worth checking against the clips that fix was actually built on (TEST4).

CONSEQUENCE FOR THE TOUCH WORK: none observed — shooter attribution on both
clips is unchanged. But the shot layer feeds shooter attribution, so this
belongs in front of whoever owns the recall fix.

---

## TEST 29 — SMART KEYFRAME SPACING WORKS. My TEST 27 conclusion was WRONG.

DJ asked for the "smarter court clicking thing". Built spikes/plan_keyframes.py
(greedy walk: keep the largest jump whose SIFT inlier ratio holds >= 0.6, the
project's own weak-pair threshold) and spikes/validate_keyframe_plan.py.

### THE CORRECTION, first, because I told DJ the opposite

TEST 27 concluded "clicking CANNOT be safely halved" because HARD's thinned
subsets scored ~0.95 ft against an absolute 0.5 ft bar. **I never measured
HARD's BASELINE.** Control run, dropping just ONE keyframe from the full seven:

```
HARD, drop only kf 900  (48 of 59 marks kept):  held-out 0.69 ft
HARD, drop only kf 1000 (48 of 59 marks kept):  held-out 0.80 ft
```

HARD sits at ~0.7-0.8 ft with SIX of seven keyframes present. That is its
FLOOR, not thinning damage. Thinning to three keyframes costs ~0.15 ft on top
of it, not 0.95 ft of damage. Comparing a clip against an absolute bar without
its own baseline is the same error class as quoting a metric that cannot
return a bad answer — I have now made a version of it twice in this project.

LIKELY CAUSE OF HARD'S FLOOR (not proven here, worth its own test): HARD's
court is CONFIGURED at 84 ft but measures 94 (handoff, 2026-07-27: "HARD
measures 94 too; DJ chose to LEAVE HARD at 84"). TEST1, whose 84 ft config is
correct, has a floor of 0.15-0.29 ft. That is a 3-5x difference in baseline
between the two clips and it tracks exactly with the known-wrong dimension.

### [2026-07-29] THE RULE'S ACTUAL RESULT — restricted to frames DJ already
### clicked, so its choice can be scored by the TEST 27 holdout

```
TEST1   rule chooses [120, 500, 580]   3 of 6   48% fewer clicks
        probes: 120->580 ratio 0.367 REFUSED, 120->500 0.645 KEPT, 500->580 0.861 KEPT
        held-out: 0.22 / 0.19 / 0.33 ft   worst 0.33 ft   PASS
        (baseline 0.15-0.29 ft -> essentially free)

HARD    rule chooses [600, 1100, 1200]  3 of 7   58% fewer clicks
        probes: 600->1200 ratio 0.068 REFUSED, 600->1100 0.686 KEPT, 1100->1200 0.825 KEPT
        held-out: 0.95 / 0.80 / 0.75 / 0.86 ft   worst 0.95 ft
        (baseline 0.69-0.80 ft -> ~+0.15 ft for 58% fewer clicks)
```

**AND THE RULE CAUGHT THE REAL FAILURE.** TEST 27's catastrophic case was
HARD's two-keyframe fit at 39.14 ft. That is exactly the 600->1200 pair, and
the rule REFUSED it at ratio 0.068. The guardrail fires where it matters.

### VERDICT
~50% fewer clicks on BOTH clips, self-tuning per clip, no new threshold
invented, and it declines the jumps that would break the court. Not adopted
into the workflow yet — DJ has not eyeballed a court built from a rule-chosen
subset, and this project adopts nothing on numbers alone.

### THE HONEST HEADLINE FOR A QUARTER OF A GAME (~8 min, ~14,400 frames)
```
today's every-100-frames convention:  ~144 marked frames  ~1,440 clicks
at TEST1/HARD's measured rule spacing: ~48 marked frames    ~480 clicks
```
Halving is real, but ~480 clicks is still ~40 minutes of solid clicking. The
tool's larger value is that it reports the price BEFORE DJ starts.

---

## TEST 30 — AUTOMATIC COURT DETECTION (KaliCalib), tested on all three gyms

DJ, 2026-07-29: "40min of clicking is absolutely unacceptable. Lets try an
automatic approach." KaliCalib (CEA-LIST, MMSports 2022 winner, basketball-
specific, pretrained weights included) cloned to the SCRATCHPAD ONLY — its
CeCILL copyleft must not touch the product before a licence decision.

Harness: scratchpad/kali_test.py. Runs the pretrained model on DJ's own
keyframes, builds a court, then pushes DJ's CLICKED landmark pixels through it
and compares against each landmark's true position (COURT_MODEL). Metric is
FEET, directly comparable to everything else here.

BAR DECLARED BEFORE RUNNING: <=0.5 ft = clicking solved. 0.5-1.0 = usable as a
pre-fill. >1 ft = does not transfer.

### TWO BUGS IN MY OWN HARNESS, both found before reporting a verdict

1. **COURT SIZE.** Upstream hardcodes FIELD_LENGTH=2800/WIDTH=1500 cm — a FIBA
   floor, 91.86 x 49.21 ft. TEST1/HARD are 84x50, TEST2 is 94x50. Feeding a
   91.9 ft template to an 84 ft floor is the exact bug that made TEST2 read
   0.94 ft. Its keypoint grid is PROPORTIONAL, so the clip's real dimensions
   are substituted instead.
2. **WIDTH AXIS INVERTED.** DJ's COURT_MODEL puts y=0 on the NEAR sideline;
   KaliCalib counts width down from y=W. Measured on TEST1 kf320:
   `as-is 19.40 ft | flip length 47.25 | FLIP WIDTH 2.63 | flip both 37.67`.
   My first run reported 17-23 ft and "does not transfer" — that was MY bug.
   NOTE WHY IT COULD NOT BE AUTO-SELECTED: a flip is AFFINE, so the homography
   absorbs it and RANSAC inlier counts were IDENTICAL across all four
   orientations (27/41, then 31/41). Inliers can never distinguish a flip.
   Also: a 1.0 ft RANSAC threshold starved the fit; 5.0 ft is realistic for
   keypoint noise on this footage.

### [2026-07-29] MEASURED, cold pretrained model, three gyms

```
TEST1 (84 ft)   6 keyframes   mean 2.11 ft   best 1.24   worst 3.45
HARD  (84 ft)   7 keyframes   mean 35.0 ft   best 2.80   worst 104.19
TEST2 (94 ft)   8 keyframes   mean 3.12 ft   best 1.53   worst 10.42
                              (7 of 8 keyframes land 1.5-2.5 ft)
```

**VERDICT AGAINST THE DECLARED BAR: FAILS. Not a drop-in replacement for
clicking.** Best single keyframe anywhere is 1.24 ft; the target was 0.5.

### BUT THE FAILURE IS INFORMATIVE, NOT FLAT

- **The keypoint detection genuinely transfers.** Rendered
  scratchpad/kali_kpts_TEST1_kf320.png: a clean, correctly-ordered 7x13 grid
  sitting ON the floor, rows banding away from the camera in perspective,
  columns monotonic left-to-right, nothing on the crowd. Column 6 lands at half
  court, which is geometrically right. The hard part works.
- **Error is SYSTEMATIC, not chaotic.** TEST2 holds 1.5-2.5 ft across 7 of 8
  keyframes. That is an offset to be corrected, not noise.
- **KEYPOINT COUNT PREDICTS ACCURACY.** TEST2 detects 37-52 points and scores
  1.5-2.5 ft; HARD detects 18-29 and blows up to 95-104 ft on two keyframes.
  The failure mode is TOO FEW POINTS, not wrong ones.
- **TEST2 — the 94 ft floor — does best, and that is a real signal.** It is the
  closest to KaliCalib's built-in 91.9 ft assumption. The 84 ft gyms are 8 ft
  from what the model was trained to expect.

### WHAT THIS JUSTIFIES (no longer speculation)
FINE-TUNING on DJ's own footage is now a well-founded next step rather than a
guess: the model already FINDS the court, it just has not seen a high-school
gym. DJ has 187 clicked landmarks across 3 gyms as a starting set, and he is
renting a GPU. This is the same shape as the ball model's own history — stock
weights were weak, v2/v3 fine-tuned on DJ's frames were not.
HONEST CAVEAT ON THE PRE-FILL IDEA: at 2 ft off, a proposed mark is a visibly
wrong mark. Nudging from 2 ft may not beat clicking fresh. The pre-fill argument
that worked for PLAYER boxes does not automatically carry over, and should not
be claimed until measured.

---

## TEST 31 — HYBRID (detector + a few clicks): DEAD. But it found the real win.

DJ, 2026-07-29: "Is there a way we can do a hybrid?" The reasoning was sound --
the two methods fail in opposite directions, KaliCalib giving ~40 spread points
at ~2 ft and DJ's clicks giving pinpoint accuracy with poor spread. Use the
detector for shape, a few clicks for precision.

TEST DESIGN (true holdout, no new labelling): take k of DJ's clicks on a
keyframe, fit, score against the clicks HELD BACK. Three arms so the hybrid has
to earn its place -- A all clicks (ceiling), B k clicks alone, C k clicks +
KaliCalib points with DJ's replicated to outweigh the detector. The k clicks are
chosen SPREAD OUT by mutual court distance, since clustered points cannot
constrain a homography and a coach told "click 3 spots" would pick corners.

### [2026-07-29] MEASURED, all three gyms, error in FEET

```
                  all ~10 clicks   5 clicks   4 clicks   HYBRID (5 + kali, w=20)
TEST1  (84 ft)         0.16          0.29       0.48            1.67
TEST2  (94 ft)         0.19          0.32       0.57            1.00
HARD   (84 ft)         0.64          0.93       1.52            2.92
```

### THE HYBRID IS WORSE THAN CLICKS ALONE ON EVERY CLIP. IDEA DEAD.
KaliCalib's ~2 ft points actively POISON the fit. Swept the weighting 1 / 5 / 20
(higher = DJ's clicks count for more) and the hybrid never caught plain
clicking: at w=1 it is 5-8 ft, and even at w=20 -- where the detector is nearly
drowned out -- it still loses. There is no weight at which the detector helps.
That is a clean negative and it closes the approach.

### THE ACTUAL FINDING, and it needs no model, no GPU, and no licence
**DJ HAS BEEN CLICKING ROUGHLY TWICE AS MANY POINTS AS A HOMOGRAPHY NEEDS.**
FIVE well-spread clicks land within 0.13-0.29 ft of all ten, on all three gyms.
A homography has 8 degrees of freedom = 4 point minimum, so 5 gives one point of
margin; 4 is measurably worse (0.48-1.52 ft) and has no slack for a bad click.

So the answer to "how do we reduce clicking" was never the AI detector. It was
that the workflow asks for about double what the maths requires -- the same
shape as TEST 29's finding that the ~100-frame keyframe spacing was a
convention, not a requirement. Two conventions, both never measured, both
roughly 2x too expensive.

### CAVEAT THAT MUST BE TESTED BEFORE ANYONE CLICKS A QUARTER
The two savings are measured SEPARATELY:
    TEST 29  about half as many marked FRAMES  (inlier-ratio spacing)
    TEST 31  about half as many CLICKS per frame (5 instead of 10)
Naively multiplied that is ~75-80% fewer clicks -- a quarter of a game falling
from ~1,440 to ~240. **THAT COMBINATION IS UNTESTED.** Fewer frames AND fewer
clicks per frame shrinks the shared court-fit's constraint pool from both ends
at once, and this project has been burned before by assuming things compose.
The combined holdout is cheap (both harnesses exist) and must run first.

### ALSO NOTE
HARD is worst in every column (0.64 ft even with all clicks). Consistent with
its court being CONFIGURED 84 ft while measuring 94 -- see TEST 29's note.

---

## TEST 32 — IS THE CAMERA A TRIPOD ROTATION? **NO.** The elegant route is dead.

The premise behind methods M4 (PTZ reparameterisation, claimed 20-40 clicks/game)
and M9 (whole-game mosaic, claimed 5-10) is that a tripod camera relates its
views by a PURE-ROTATION homography -- 4 DOF instead of 8, with those 4 free from
SIFT. DJ's 5-10 minute target rests on it. Tested BEFORE building, in an hour,
using homographies already computed by the working 8-DOF solution.

METHOD: for a pure rotation, H = K_ref . R . K_i^-1, so K_ref^-1 . H . K_i must
be a rotation matrix. Form that product, project it to the nearest rotation by
SVD (R = U V^T), and measure in PIXELS how far it had to move. Focals from a
deterministic scan; per-keyframe focal allowed so zoom is not penalised.
BAR STATED FIRST: <=2 px premise holds / 2-10 px partial / >10 px fails.

### THREE ATTEMPTS -- the first two were MY bugs, and both would have lied
1. Nonlinear fit, principal point free, reference unpinned -> diverged to focal
   7,377,615 px and a principal point 344,500 px off frame. Reported "PREMISE
   FAILS" for entirely spurious reasons.
2. Pinned reference, fixed principal point, bounded focal -> `converged=False`
   on all three clips (hit the 4000-nfev cap). Numbers unusable. A blind start
   is hopeless when keyframes 460 frames apart sit nowhere near identity.
3. CLOSED FORM (above). Deterministic, nothing to converge, reproducible.
Recording this because a "premise fails" verdict from a broken optimiser is
indistinguishable from a real one unless the optimiser is checked.

### [2026-07-29] MEASURED (closed form), with the centre/edge diagnostic

```
TEST1   kf   zoom   rot deg   CENTRE px   full px   max px
       120   1.385   15.57      11.23      11.35     17.15
       220   1.385   15.59      11.97      11.75     17.77
       320   1.277   10.09       7.99       7.65      9.98
       420   1.000    0.00       0.00       0.00      0.00   (reference)
       500   1.000   12.79      15.09      30.97     51.36
       580   1.000   29.79      25.75      26.67     42.09
   centre-only worst 25.75 px | full-frame worst 30.97 px | ratio 1.2x

HARD   centre-only worst 19.33 px | full-frame worst 157.61 px | ratio 8.2x
TEST2  full-frame worst 32.01 px
```

### VERDICT: THE PREMISE FAILS, AND NOT BECAUSE OF LENS DISTORTION
The centre/edge diagnostic exists to separate two explanations. Distortion would
show a SMALL centre error and a large edge error -- the premise would survive and
just need a radial term. Instead the CENTRE error alone is **19-26 px** on both
clips, against a bar of 2 px and against the 8-DOF solution's own 0.6 px
keyframe mutual consistency. The rotation model is wrong in the middle of the
frame, where distortion is negligible. HARD's 8.2x ratio says distortion is
present TOO, but it is not the primary cause.

Likely physical reasons (not distinguished here): the tripod head does not
rotate about the optical centre, so real parallax exists; or the "pan" is partly
a DIGITAL CROP from a wide sensor (Veo-style), which is not a rotation at all.

### WHAT THIS COSTS AND WHAT SURVIVES
DEAD, or at least unfounded: M4's 20-40 clicks and M9's 5-10 clicks. Both were
projections resting on this premise; neither was ever measured. An hour of
testing saved a day of building on sand -- which is the entire argument for
testing premises before implementations.
STILL STANDING, because it never depended on the rotation model: **M1, keyframe
by VIEW.** That is empirical -- 13 distinct views measured in a 5-minute clip,
151/151 frames matching one reference. ~40 views for a full game x 5 clicks
= ~200 clicks. Not DJ's 5-10 minutes, but the honest route that is still open.
Also untouched: M2 (dead-ball skip, ~2x, already built), M3 (venue reuse), M5
(line snapping), M6 (fine-tuned detector).

---

## TEST 33 — A REAL FULL GAME: **8 MARKS, ~40 CLICKS, ~3 MINUTES**

DJ, 2026-07-29: "you keep testing on a five minute clip. What if I just gave you
a full game?" He was right — every full-game figure quoted before this was
arithmetic from 1-5 minute clips. He supplied `Full_Game.mp4`.

```
171,120 frames | 30 fps | 1920x1080 | 95.1 minutes | 3.6 GB
```
FIRST CORRECTION: 95 min, not the 60 I had assumed, so every earlier full-game
estimate was understated ~60%. Corrected baselines: old convention ~1,711 marks
(~17,000 clicks); today's measured best ~570 marks (~2,850 clicks, ~3.2 hours).

### METHOD — the operational question, asked the way the pipeline works
A marked frame serves every frame SIFT can bridge back to it at inlier ratio
>= 0.6 (the project's own weak-pair bar). So: walk the game, and whenever a
frame cannot reach any existing mark, open a new one. The count of marks IS the
answer. `spikes/full_game_views.py`, greedy incremental cover, signature
shortlist + most-recently-used ordering for speed.

### [2026-07-29] MEASURED (stride 600, 286 samples, 232 s)

```
MARKS NEEDED: 8    -> ~40 clicks at 5 each  ->  ~3 minutes

new marks opened per 10-min block:
   0-10 min: ### 3
  10-20 min: #### 4
  30-40 min: # 1
  40-95 min: (none)
busiest single view covers 165 of 286 samples (58% of the game)
```

**IT FLATTENS COMPLETELY.** After 40 minutes the camera never presented a
framing that could not reach an existing mark. That is the good case, and it is
the answer to whether extrapolating from 5 minutes was ever valid — it was not,
but it was PESSIMISTIC, not optimistic.

### THE CALIBRATION THAT MAKES THE 8 CREDIBLE
"Reachable" is not "accurate" — TEST 29 proved they diverge (HARD's 600->1100
pair cleared ratio 0.686 yet produced 0.95 ft). So the same reachability rule was
run on the two clips where an ACCURACY-verified answer already exists:

```
              reachability picks    TEST 29 accuracy holdout
TEST1               2 marks         2 marks -> 0.33 ft, PASS
HARD                3 marks         3 -> ~baseline; 2 -> 39 ft COLLAPSE
```

It matched on both, **including correctly refusing to take HARD down to 2**,
which is exactly where that clip breaks. The rule is not optimistic on the data
where it can be checked.

### WHAT IS STILL NOT PROVEN — do not quote 40 clicks as final
1. **Accuracy on THIS game is unmeasured and unmeasurable today.** Full_Game.mp4
   has no clicked landmarks, so only reachability could be tested. The
   calibration above comes from 15-20 second spans; applying it to a 95-minute
   span is a real extrapolation.
2. **Stride 600 = samples 20 seconds apart.** A brief view appearing between
   samples would be missed. A stride-200 confirmation run is in flight.
3. **The "191 hard cuts" figure in the output is JUNK.** A cut threshold was
   applied to samples 20 seconds apart, where large frame-to-frame change is
   normal play, not a cut. Ignore it; the detector needs consecutive frames.

### NEXT STEP THAT WOULD SETTLE IT
DJ clicks the 8 frames this run selected, then the existing holdout harness
(`spikes/keyframe_thinning_test.py`) scores the court they produce. That turns
"8 marks are reachable" into "8 marks are accurate" — and it costs him ~3
minutes of clicking, which is the whole claim being tested.

---

## TEST 34 — the full-game mark count, resolved: **~15 marks / ~75 clicks / ~5 min**

TEST 33 reported 8 marks at stride 600. That was wrong twice over, and both
errors are recorded here because each one alone would have produced a confident
false answer.

### ERROR 1 — the count depends on how finely you sample
```
stride 600 (286 samples, every 20.0s)   ->   8 marks
stride 200 (856 samples, every  6.7s)   ->  18 marks
stride  60 (2852 samples, every 2.0s)   ->  42 marks
```
A power-law fit to the first two points predicted 44 at stride 60. It came in at
42. The prediction and its reading were both stated BEFORE the run, and the
optimistic case lost. Extrapolated to every frame this metric implies ~900
marks — worse than the convention it was meant to replace.

### ERROR 2 — part of that growth was a bug in my own algorithm
Candidates were shortlisted to the TRY_TOP=3 most similar marks before spending
SIFT. With 42 accumulated marks a frame was offered 3 of 42, so one that WOULD
have matched mark #20 never got the chance and opened a spurious new mark —
shrinking the shortlist's coverage further. A feedback loop that reproduces the
observed curve exactly.
I predicted this was the dominant cause. **It was not.** Exhaustive re-run:
```
stride 200, TRY_TOP=3        ->  18 marks
stride 200, exhaustive       ->  15 marks     (the shortlist caused 17%, not 83%)
```

### [2026-07-29] THE CLEAN MEASUREMENT
```
Full_Game.mp4, 171,120 frames / 95.1 min, exhaustive, stride 200
  MARKS NEEDED: 15   -> ~75 clicks   -> ~5 minutes at 4s/click
  new marks by 10-min block: 3, 9, -, 1, 1, -, -, -, -, 1   (FLATTENS hard)
  busiest 5 views cover 418+254+116+38+7 = 833 of 856 samples = 97%
```

### THE READING THAT MATTERS: THE TAIL IS BAD FRAMES, NOT NEW VIEWS
The same handful of views dominates at EVERY sampling density (frame 200 covers
418, frame 16000 covers 254, frame 65800 covers 116 — stable across runs). What
grows with finer sampling is a tail of near-singleton marks. A genuinely new
camera angle would RECUR; a motion-blurred or occluded frame during a fast pan
appears once and never again.
So the growth is the sampler finding more UNPLACEABLE frames, not more views.
A frame whose court cannot be placed should be SKIPPED, not clicked for — which
is exactly this project's standing abstention rule. The operational answer is
therefore "mark the dominant views, abstain on the tail", and that is ~5-15
marks, not 900.

### WHAT IS STILL NOT PROVEN — the ladder
```
1. can a frame REACH a mark              <- measured above (optimistic floor)
2. does the COURT LAND ACCURATELY there  <- NOT measured, and cannot be on this
                                            game: Full_Game.mp4 has no clicks
```
TEST 29 already proved these two diverge (HARD cleared ratio 0.686 and still put
the court 0.95 ft off). So ~75 clicks is a well-founded ESTIMATE, not a result.
Converting it needs one ~5-minute clicking session from DJ on the 15 frames this
run selected, after which `spikes/keyframe_thinning_test.py` scores the court.

### ALSO JUNK, twice reported, ignore it
"hard cuts detected: 401/610" — a cut threshold applied to samples 6.7-20s
apart, where large change is ordinary play. The detector needs consecutive
frames; the number is meaningless as printed.

---

## TEST 35 — RESOLVED: **5 marks cover 99% of a 95-minute game (~25 clicks)**

TEST 34 landed on 15 marks and called the extra ten a "long tail of unplaceable
frames". Exporting the actual frame list and LOOKING at the stills identified
what that tail really is.

### [2026-07-29] THE FULL RANKED LIST (exhaustive, stride 200)
```
rank   frame     when     covers   cumulative
   1     200    0m06s      418       49.7%
   2   16000    8m53s      254       79.9%
   3   65800   36m33s      116       93.7%
   4   79200   44m00s       38       98.2%
   5  169000   93m53s        7       99.0%
   6   23400   13m00s        3       99.4%
   7   21800   12m06s        2       99.6%
   8   22200   12m20s        2       99.9%
   9   23800   13m13s        1      100.0%
  10-15  frames 0, 21400, 21600, 22400, 22800, 23000   0 each
```

### THE TAIL IS THE PRE-GAME CEREMONY, WITH THE LIGHTS OFF
Marks 6-15 sit between 11m53s and 13m13s — a single 90-second window. The still
for mark 8 (`spikes/out/FULLGAME_mark_frames_stride200/08_f22200_covers2.jpg`)
shows the house lights DOWN, red/purple stage lighting, an EMPTY floor, and
officials standing about waiting. It is the player introductions, not basketball.
SIFT fails there because the floor is unlit and textureless, not because the
camera found a new angle. Ten marks, 1% of coverage, zero basketball.

Mark 1 (`01_f200_covers418.jpg`) by contrast is a wide view of the ENTIRE floor
during warmups — centre logo, both baskets, whole court visible. The single best
possible calibration reference, and it alone serves half the game.

### THE ANSWER
```
Old convention            ~1,711 marks   ~17,000 clicks   ~12 hours
Best before this session    ~570 marks    ~2,850 clicks   ~3.2 hours
MEASURED, 99% coverage           5 marks       ~25 clicks   under 2 minutes
```
Frames that cannot be placed (lights out, ceremony, blur) are SKIPPED, not
clicked for -- the standing abstention rule, applied to calibration. That single
policy removes 10 of the 15 marks at a cost of 1% coverage, none of it live play.

### WHAT REMAINS UNPROVEN -- unchanged from TEST 34, and it is the whole gap
This measures whether a frame can REACH a mark, not whether the court then LANDS
ACCURATELY (TEST 29: HARD cleared ratio 0.686 and was still 0.95 ft off).
Full_Game.mp4 carries no clicked landmarks, so accuracy cannot be scored on it
at all. ~25 clicks is a well-founded ESTIMATE.
TO CONVERT IT: DJ clicks the 5 top-ranked frames (listed above, stills exported),
then `spikes/keyframe_thinning_test.py` scores the resulting court against the
marks. Cost to test the claim is the claim itself -- about two minutes.

---

## TEST 36 — THE 5 CHOSEN FRAMES DO NOT CALIBRATE. Coverage != chainability.

DJ clicked 63 landmarks across the 5 frames TEST 35 selected. Court identification
succeeded; the full calibration solve FAILED.

### [2026-07-30] RESULT
```
court identification (court_detect):  0.23 ft over 63 marks   -- 84 ft floor,
                                      runner-up 94 ft is 3.4x worse. CLEAN.
full calibration (refit_keyframes):  15.45 ft mean / 50.52 max. BROKEN.
                                      (0.94 ft is what DJ judged broken by eye)
```

### THE CAUSE, in one line of the log
```
kf   200 <-> kf 16000    352 inliers
kf 16000 <-> kf 65800      9 inliers   <-- the chain is severed here
kf 65800 <-> kf 79200    116 inliers
kf 79200 <-> kf169000    199 inliers
```
Frames 16000 and 65800 are 49,800 frames (~28 min) apart -- different lighting,
different crowd, different pan. NINE inliers is no match at all.

refit_keyframes ties every keyframe into ONE reference frame using dense
adjacent-keyframe SIFT correspondences. One severed link breaks the chain, and
the global solve then degrades the HEALTHY pairs trying to accommodate the broken
one -- visible in the numbers: kf200<->16000 went 3.8 -> 22.5 px, kf65800<->79200
went 30.6 -> 65.6 px. Overall consistency 934.7 -> 70.0 px, against TEST1's 0.6 px.

### THE METHODOLOGICAL ERROR -- MINE, and it invalidates TEST 33-35's conclusion
TESTs 33-35 measured whether **each frame can reach a mark**. Calibration requires
that **the marks can reach each other**. Those are different questions and I
conflated them.
The greedy cover in full_game_views.py optimises COVERAGE: it picks frames that
are maximally DIFFERENT from the existing marks, since a frame only opens a new
mark when it matches none of them. That is precisely the opposite of what a
keyframe CHAIN needs, which is consecutive frames that match each other WELL.
So the algorithm was, by construction, selecting an unchainable set.
"5 marks cover 99% of the game" remains true and is now known to be the wrong
metric. ~25-63 clicks is NOT established as sufficient.

### WORSE: THE PROJECT ALREADY HAD THE GUARDRAIL AND I DID NOT RUN IT
spikes/stage2_multikeyframe.py flags any adjacent pair below 0.6 inlier ratio
("WEAK PAIR FLAG"). It fired correctly in TEST 27 and TEST 29. Nine inliers would
have screamed. I never ran it over the chosen frame set before asking DJ to spend
his time clicking. That check costs seconds.

### WHAT SURVIVES
- DJ's 63 clicks are GOOD -- the court identification is clean and unambiguous.
- The court IS an 84 ft high-school floor, 0.23 ft, a 3.4x clear call.
- The frame set needs intermediate frames so consecutive keyframes MATCH. DJ's
  stated budget is 5-8 frames, so 7-8 chainable frames remains inside it.

### THE FIX, and the rule to carry forward
SELECT KEYFRAMES BY CHAINABILITY, NOT COVERAGE: walk the game and place the next
keyframe at the FURTHEST frame that still holds inlier ratio >= 0.6 against the
previous one -- which is exactly what spikes/plan_keyframes.py already does
(TEST 29). It was built, validated on TEST1/HARD, and then not used here.
VERIFY THE CHAIN BEFORE ASKING FOR CLICKS. Never again request a clicking session
for a frame set whose adjacent-pair inlier ratios have not been printed.

---

## TEST 37 — TWO FINDINGS THAT CHANGE THE PICTURE (and correct TEST 36's diagnosis)

### FINDING 1 — TIME IS NOT THE PROBLEM. My TEST 36 explanation was WRONG.
TEST 36 blamed the broken link (kf16000<->kf65800, 9 inliers) on the frames being
"28 minutes apart -- different lighting, different crowd". The chain planner
(spikes/plan_fullgame_chain.py) disproves that outright:

```
frame    600 -> 127200    ratio 0.712     <- 70 MINUTES apart, matches fine
frame  16000 ->  65800    9 inliers       <- 28 minutes apart, failed
```

A 70-minute gap matches comfortably; a 28-minute gap failed. **What matters is
whether the camera happened to be POINTING THE SAME WAY, not elapsed time.**
Lighting and crowd drift are not the mechanism. Correct the mental model: a
keyframe pair is chainable when the VIEWS overlap, full stop.

### FINDING 2 — DJ WAS RIGHT: SAME CAMERA, SAME GYM AS TEST1, AND IT REACHES.
DJ: "I used the same camera that I used for test one... same gym." Measured
against TEST1's six clicked keyframes:

```
FG frame     best match to a TEST1 keyframe
   200            0.728   OK
 16000            0.843   OK
 40000            0.765   OK
 65800            0.710   OK
 79200            0.701   OK
110000            0.895   OK
140000            0.863   OK
169000            0.705   OK
  -> 8 of 8 across the full 95 minutes, none marginal
```

**TEST1's ALREADY-VALIDATED calibration (58 clicked marks, 0.15 ft) is reachable
from every part of this game.** That is method M3 (calibrate the VENUE, not the
game) with real evidence behind it, and DJ found it by noticing the room, while
I was optimising frame selection.

PRODUCT CONSEQUENCE: you do not calibrate a GAME, you calibrate a GYM + CAMERA
SETUP, once. Every future game filmed from the same mount is already done.
HONEST BOUND: it works because the tripod was in the same place. Move it, bump
it, change the height, and it stops. So the claim is "same gym AND same setup",
never just "same gym" -- and a cheap has-the-camera-moved check is required
before reusing anything.

### THE CHAIN PLAN FROM SCRATCH (spikes/plan_fullgame_chain.py, no clicking)
```
MARKED FRAMES NEEDED: 5   (4 usable -- frame 0 is dead, ratio 0.000)
   0        0m00s   <- dead frame, black/intro
   600      0m19s
   127200  70m39s   <- one 126,600-frame jump at ratio 0.712
   151200  83m59s
   171000  94m59s
```
So a 95-minute game needs about FOUR chainable marked frames. But note these have
ZERO overlap with the five DJ already clicked -- a naive switch would waste his 63
marks. A bridge search is running to find the ONE frame that repairs his existing
set instead.

### STILL NOT PROVEN -- the same trap as TEST 36
Everything above is REACHABILITY again. TEST 36 failed at exactly this step. The
difference now is that the metric matches what calibration actually needs
(adjacent-pair matching), rather than coverage. But no calibration has been RUN
on these frames yet, and no overlay has been watched. No verdict until both.
