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
