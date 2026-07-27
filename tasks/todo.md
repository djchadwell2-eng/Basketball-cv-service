# CV QUALITY -- STATUS BOARD + TEST 15-20 PLAN (current task, 2026-07-25)

This section belongs to the CV-QUALITY chat (ball model / tracker safety /
scoreboard). Shipping and web-app wiring are a DIFFERENT chat's job -- do not
mix them here. The calibration section below (same date) is that other track.

## PART A -- STATUS BOARD (what is done vs still open, as of 2026-07-25)

Fourteen gated tests, 2026-07-13 -> 2026-07-23. Full raw evidence in
TEST_LOG.md; this is the index, not a replacement for it.

### DONE / SOLVED (on the two test clips)

- [x] **Robust arc fitting** (TEST 1) -- outlier-tolerant parabola fitter;
      recovered TEST1 shot B, which the plain fitter threw away. 6 new tests.
- [x] **Resolution lever, closed** (DECISIONS 20) -- 1024/1280/1920 bracketed
      on BOTH sides. 1280 is the proven optimum; 1920 raised raw coverage and
      LOST BOTH known shots. Tiling/SAHI abandoned unmeasured for the same
      reason (it is more effective resolution = the direction that hurt).
      DO NOT RE-RUN.
- [x] **Model-capacity lever, closed** (DECISIONS 21) -- yolov8x vs yolov8m
      on stock: a wash.
- [x] **Own ball model, 3 generations** (TESTS 8/10/11) -- v1 (HARD gate
      pass, TEST1 2/5) -> v2 (recipe: yolov8l/cos_lr/scale=0.7, TEST1 4/5 but
      lost shot B) -> **v3** (v2 + DJ's 230 own-footage labels, warm-started
      from v2): **TEST1 5/5, HARD 2/2 + all 4 known non-shots rejected**,
      coverage 79.4% vs hosted 86.2%. Zero API calls. v1/v2's complementary
      blind spots are gone -- one model does it all.
- [x] **Off-the-shelf tracker survey, closed** (TESTS 5/6/7 + DECISIONS 11) --
      BoT-SORT+GMC (117/116.2), OC-SORT (107/114.6), buf=60 (120/107.6),
      appearance re-ID (131/~106, WORSE -- teammates look identical),
      mt=0.7 (256/46.5, catastrophic), vs baseline 122/105.8. StrongSORT
      unrunnable (upstream packaging bug). Only mt=0.9 (93/143.1) is a real
      outlier -- see OPEN.
- [x] **Ref/player semantic split, measured** (TEST 4) -- fine-tuned Player
      class = clean substitute for the per-clip ROI mask (43 on-court ids,
      identical to baseline's own on-court 43; 10 players + 3 refs/frame at
      conf>=0.4). 6 of 43 on-court ids are REFEREES = ~14% review-queue
      reduction available. Not wired in.
- [x] **Scoreboard is readable** (TEST 14) -- fixed screen position all game;
      sliding-window majority vote + monotonicity guard. HARD "15-12", zero
      changes (independently confirms both known shots were misses); TEST1
      reads a sane monotonic progression. Two real bugs found + fixed.
- [x] **ID-switch risk of mt=0.9, proven with pictures** (TEST 12) -- id=17
      carries through an 0.83s gap and reattaches to an OPPOSING-TEAM player
      (white Milford f=196 -> green Little Miami f=221). Also established the
      key nuance: the switch is NOT new, the committed system mishandles the
      same moment -- it just fails SAFER (ends the track vs silently
      continuing it onto the wrong body).
- [x] **Ball-physics-only false-positive gates, RULED OUT** (TEST 11
      follow-ups + TEST 12 follow-up). accel_y looked like a clean separator
      at n=9 (real >=0.848, suspect <=0.655) and DJ's ground truth DESTROYED
      it: the 1352-1378 cross-court pass has accel_y=1.388, squarely inside
      the real cluster. A pass thrown by a human arm and a shot thrown by a
      human arm ARE the same physics. **No further ball-physics-only gates
      should be attempted on this data.**

### OPEN (real gaps, with the diagnosis already done)

- [ ] **FALSE POSITIVES -- the #1 correctness gap.** v3 claims 3 DJ-refuted
      non-shots (all now permanent ground truth in local_weights_check.py):
        (a) 403-415 rebound caught -> dished out (HARD ~13.4s)
        (b) 2234-2250 player just HOLDING the ball on an inbounds (~74.5s)
        (c) 1352-1378 cross-court PASS (~45.1s)
      TWO DISTINCT CATEGORIES, proven: category A (rebound/held ball -- low
      or bogus accel_y) and category B (legitimate pass that geometrically
      arcs near the hoop pixel -- normal shot-like accel_y). Ball position
      cannot see either. **This is why v3 is NOT adopted** despite being the
      best model we have -- it sits on disk as a one-line config change.
      FIX = a player signal answering "does this look like a person shooting
      AT the hoop." Both categories fail that question; neither fails a
      trajectory test. -> TEST 16.
- [ ] **Tracker: the 35% win is unsafe, and the proposed safeguard is dead.**
      TEST 15 (2026-07-25) settled it: with junk detections correctly removed
      the jersey-colour check flags 0 of 2 confirmed switches. Appearance is
      anti-correlated with switch difficulty (switches happen under
      occlusion, which is exactly when colour is unreadable). mt=0.9 remains
      UNADOPTED with no working safeguard. The honest next lever is ID-switch
      GROUND TRUTH (player-tracker item 2), not another heuristic.
- [x] **Junk detections -- FIXED and measured (TEST 15).** The person
      detector fires on the ANIMATED SCOREBOARD GRAPHIC (842 detections on
      TEST1 alone) and tracks REFEREES as players (2 whole tracks). Both now
      filterable from data the project already had: the per-clip
      `exclude_regions` rectangle and v3's Ref class. Not yet wired into the
      real tracker -- it lives in the probe. Cheap, isolated, worth doing.
- [ ] **SCOREBOARD RULE (DJ, 2026-07-26): CONFIRM, NEVER DENY.** DJ observed
      the broadcast scorebug FADES IN AND OUT on at least two clips. The rule
      that makes this harmless instead of dangerous:
        graphic visible + readable + score went up  -> a MAKE, trust it
        graphic absent / faded / unreadable         -> UNKNOWN, abstain
        "no score change seen"                      -> NEVER a miss
      Absence must produce "I don't know", never a value. This is not a new
      risk: TEST 14 already recorded that the matcher would have reported 4 of
      TEST1's 5 shots as MISSES purely from having no data. The fade is a
      second, more frequent cause of the same confident-wrong failure.
      CONSEQUENCE FOR DEAD-BALL DETECTION (different from make/miss): make/
      miss only needs the graphic at isolated moments, so a fade costs
      coverage only. Dead-ball detection needs CONTINUOUS state, where
      "frozen" and "absent" become indistinguishable -- so for the timeout
      work, PLAYER CLUSTERING is primary and the game clock corroborates,
      the reverse of the 2026-07-25 proposal. Fade rate being measured on
      TEST4 via spikes/scoreboard_presence.py.
- [ ] **Scoreboard -> which shot scored.** Coarse pass works; per-shot
      attribution does not. Dense sampling after each attempt is fooled by
      real player bodies walking through that screen corner for several
      frames running (proven with a still -- not single-frame noise, so
      spreading the vote does not fix it). PAUSED, not failed. -> TEST 20.
- [ ] **We are steering on a val set that has never worked.** The only
      validation is 32 public images / 18 ball instances containing NONE of
      DJ's footage. It failed to predict the clip result in EITHER direction
      for v1 vs v2 (TEST 10 says so explicitly). Every model version
      therefore costs a full clip-gate run to judge. -> TEST 18.
- [ ] **Everything we know comes from 2 clips.** Every threshold, gate and
      verified shot is HARD + TEST1. Generalization is UNMEASURED. Given the
      accel_y lesson (a clean-looking rule at n=9 destroyed by new ground
      truth), this is the single biggest unknown in the project. -> TEST 19.
- [ ] **Player labeling queue is 25 days at current pace.** 30 of 280 done at
      ~10/day, started 2026-07-22. The #1 fix is gated behind a month of
      manual clicking and does not need to be. -> PRE-FILL task below.

### STANDING CONSTRAINTS (do not relitigate)

- Camera is fixed (Hudl/Veo, 2026-07-14). Three separate investigations
  (resolution, layups, fragmentation) all bottomed out at the same root
  cause: 24px ball, small distant players. It is the ceiling on all of this.
- Raw detection coverage is a LIAR -- it pointed the wrong way in
  DECISIONS 13, 18 and 20. The only metric is whether real shots survive
  the physics gate and non-shots do not.
- Fragmentation metrics CANNOT see wrong-merges. A confidently-wrong track
  is worse than a fragmented one. Nothing adopts on proxy metrics alone.
- Nothing is adopted without a gate + DJ eyeball. Stock yolov8m is still
  the committed ball detector in run_clip.

## DECIDED 2026-07-26 -- CV FLAGS, GEMINI ADJUDICATES (DJ's call, agreed)

DO NOT LOSE THIS. DJ's proposal, from a parallel chat about the hybrid CV +
Gemini vision architecture: CV emits a cheap "indicator" that SOMETHING
changed, and Gemini looks at that moment and says WHAT it was. Applies first
to dead-ball/timeout detection, and the same shape generalises.

**Why it is the right split (agreed, with reasons, not just deference):**
- Enumerating basketball's stoppage types in code is brittle -- timeout,
  quarter break, injury, substitution, review, jump ball, technical. TEST 20
  already hit this: the ONE error in an otherwise 151/155-correct timeline was
  a scoreboard RESET being indistinguishable from a running clock. Naming that
  case in code means naming all of them, forever, per venue.
- A vision model can read scoreboards this project's detector cannot. THREE
  styles seen so far (HARD broadcast bug, TEST4 LED gym board, Time_out.mp4
  broadcast overlay) and v3's scoreboard classes are 100% blind to one of
  them. Style-robustness is exactly a VLM strength and exactly our weakness.
- Cost works ONLY because the CV filter is cheap and sparse: TEST 20's clock-
  rhythm detector produced 3 transitions in 155s. A full game is perhaps
  50-100 stoppages = 50-100 VLM calls, not 160,000 frames.

**The indicator already exists and is style-independent.** TEST 20's clock-
rhythm method needs no OCR and no per-gym model -- every basketball clock
ticks once per second, so "is this patch changing at 1 Hz?" works on any
scoreboard. It self-locates. That is the flag; Gemini is the adjudicator.

**THE LINE THAT MUST NOT MOVE:** CV keeps the HARD NUMBERS -- ball position,
trajectory, shot location, who was where. Those must be repeatable and
gate-able, and this project's whole discipline (verified shots, ground truth,
abstention) depends on them being measurable. Gemini gets SPARSE SEMANTIC
JUDGEMENT: "what is happening at this moment", "is this a timeout", "is that
a shot or a pass". A non-deterministic answer is fine for a handful of event
labels; it is not fine for a shot chart.

**The one real risk, stated up front:** Gemini's confidence is unearned until
measured. Swapping a measurable CV error for an unmeasurable VLM error is not
progress. Anything Gemini adjudicates needs the same treatment everything else
here got -- DJ ground truth on real clips, a scored result, and abstention
when unsure. Do not skip that because the answers sound fluent.

**Natural extension, not yet decided:** the same flag-then-adjudicate shape
fits the SHOT false positives. CV claims an attempt, the pose rule (TEST 16,
9/9 on the holdout) scores it, and genuinely marginal cases go to Gemini
rather than being guessed. Worth considering AFTER the pose rule is wired.

## PART B -- TEST 15-20 PLAN (DJ approved the direction 2026-07-25)

Same protocol as TESTS 1-14: suite green before and after, raw output only,
nothing adopted, every result logged to TEST_LOG.md as
"MEASURED -- pending DJ review".

### Phase 1 -- no GPU, no new data, can start immediately

- [x] **TEST 15 -- junk-detection filter (scoreboard region + referees).**
      DONE 2026-07-25, see TEST_LOG. TWO results, opposite directions:
      (1) THE FILTER IS A CLEAN WIN and stands alone -- 842 detections on the
      animated scoreboard graphic + 2 whole referee tracks (id 7, 14, both in
      TEST 4's independently-measured ref list) were being carried silently
      through the tracker into everything downstream. Real detector bug,
      cheaply fixed. A bug in my FIRST implementation (per-detection ref
      removal fabricated events, 104 -> 126) was caught by the numbers and
      fixed properly -- refs now excluded whole-track.
      (2) IT KILLS TEST 13's COLOUR SAFEGUARD. With the junk gone, the check
      flags NEITHER confirmed real switch (0/2, both abstain) while still
      flagging the case a human could not call. TEST 13's "100% recall" was
      riding on a contaminated colour scale (scoreboard + referee stripes are
      extreme colour outliers polluting the 2-cluster fit). Structural, not
      fixable by tuning: a switch happens BECAUSE players occluded each
      other, so the reattaching box sits on a half-covered body and its
      colour is a blend -- confident readings (2.9, 2.6) come exactly where
      the check is not needed, mush (1.34, 1.39 vs a 1.40 line) exactly where
      it is. RECOMMENDATION: stop pursuing appearance-based switch detection
      (this is the 2nd independent finding in that direction after §11).
      mt=0.9 stays UNSAFE and unadopted; it needs ID-switch ground truth
      (player-tracker plan item 2), not another heuristic.
      Cheapest open item, and it unblocks the tracker safeguard.
      - Reuse the per-clip `exclude_regions` ALREADY in spikes/clips_config.py
        (TEST1: 0,810,415,1080) -- no new constant, no new hand-tuning.
      - Drop person detections whose box centre falls in that region, and
        drop boxes the fine-tuned Ref class claims (TEST 4 measured Ref at
        3/frame at conf>=0.4 -- sane).
      - Re-run spikes/tracker_color_reattach_check.py on the same mt=0.9
        tracks and re-score.
      - SUCCESS: the 4 scoreboard flags + the 1 ref flag disappear, and BOTH
        confirmed real switches (id=17, id=45) are STILL flagged. That is
        precision 2/8 -> 2/3 with recall intact.
      - FAILURE MODE TO WATCH: if a real switch stops being flagged, the
        filter is too aggressive -- report it, do not tune until it passes.

- [x] **TEST 16 -- pose estimation vs the 3 confirmed false positives.**
      DONE 2026-07-25, see TEST_LOG. My stated hypothesis (shooting posture
      at RELEASE) was WRONG -- release-hand distance and arms-raised do not
      separate at all. What separates is the DESTINATION: 3/3 fakes end at a
      hand, 6/7 real shots end at the rim, 9/10 with no fitted threshold.
      Rim distance alone does NOT separate (2 fakes end closer to the rim
      than 3 real shots), so the wrist is genuinely new information. One
      miss, in the dangerous direction: TEST1 shot B, a REAL shot, ends at a
      hand because its arc stops short of the rim. Two confounds checked and
      CLEARED (clip imbalance -- separation holds within HARD alone; event
      duration -- fakes sit inside the real range). Pose worked fine on small
      distant players, the stated risk did not materialise.
      BUT THE KEY CAVEAT: shifting the arc endpoint by 0.1s flips 3 of 10
      verdicts. The signal is real but FRAGILE frame-to-frame, so a
      single-frame reading is not a safe gate. Untested candidate fix
      (specify BEFORE TEST 19, never tune after): read a WINDOW after the arc
      ends -- a caught ball stays in hands, a ball that reaches the rim does
      not linger in anyone's grip. FROZEN as a hypothesis pending TEST 19.
      THE headline test. Off-the-shelf pretrained pose (yolo11x-pose,
      COCO-trained) gives shoulder/elbow/wrist keypoints with ZERO labeling.
      - Run pose over the exact frames of all 10 events that now have DJ
        ground truth: 7 confirmed real shots + 3 confirmed non-shots.
      - At each arc's START frame, measure raw, uninterpreted quantities:
        distance from the ball to the nearest wrist; whether that person's
        wrists are above their shoulders; the arm-extension angle; and where
        the ball ENDS (near the rim, or near another player's wrists).
      - REPORT THE RAW TABLE FIRST, before proposing any rule. This is the
        explicit accel_y guard: a rule fitted to n=10 after seeing the data
        is a HYPOTHESIS, not a gate, and must be labelled as such.
      - SUCCESS: a visible separation between the 7 and the 3. Any rule found
        is then FROZEN and must survive TEST 19's holdout clip before anyone
        calls it a fix.
      - HONEST RISK, stated up front: pose models degrade on small distant
        figures, which is exactly this footage's known weakness. A clean
        negative here is a real result -- it says the player-labeling grind
        is the necessary path, and it costs an afternoon to find out.
      - Runs on CPU for a few hundred frames; GPU makes full-clip feasible.

- [x] **PRE-FILL the remaining player labels (unblocks DJ's 25-day queue).**
      DONE 2026-07-25. spikes/prefill_player_labels.py -> 3110 proposed boxes
      over all 280 frames (11.1/frame: 2658 player + 452 ref), Roboflow-ready
      YOLO folder at spikes/out/label_prefill_players (images + labels +
      data.yaml, classes [player, ref]). Previews with boxes drawn:
      spikes/out/label_prefill_preview.
      HONEST QUALITY, eyeballed not assumed: on-court players and referees
      are boxed tightly and correctly -- the hard part. BUT wide shots also
      box SPECTATORS in the bleachers (~4-6 per frame), and one referee was
      missed. Raising the confidence floor does NOT fix it (measured 0.25 ->
      0.60: crowd boxes are confidently people, because they ARE people; it
      only costs referees). Same root cause the ROI mask exists for (§9):
      "person" != "player on court".
      NET: still a clear win -- DJ DELETES ~5 boxes rather than DRAWING ~10,
      and deleting is far cheaper than drawing. AVAILABLE IMPROVEMENT, not
      built: phase1/stage1_court_roi.on_court() already solves exactly this
      (incl. a horizon guard for bleacher bodies) but needs a per-frame
      homography for each harvest frame, which reaches into the calibration
      track's active area -- deliberately not done unilaterally. Worth it
      only if the deleting turns out to be annoying in practice.
      OPEN QUESTION FOR DJ (a labelling convention, not a bug): bench players
      and coaches are currently NOT boxed. Decide in or out before labelling,
      so the 280 frames are consistent.
- [ ] (superseded description of the same task, kept for the reasoning)
      Not a test -- a chore that should have been offered before DJ started.
      - Run v3 (whose Player class scores mAP50 0.903) over the ~250 unlabeled
        frames in spikes/out/label_harvest_players, export boxes in Roboflow's
        import format so DJ CORRECTS pre-drawn boxes instead of drawing from
        scratch. Typically 3-5x faster on crowded frames.
      - Also recommend stopping at ~100-120 total rather than 280: the ball
        model needed 230 for a much harder target, and 5-10 min of extra
        labeling is worth less than getting the signal tested sooner.

### Phase 2 -- needs the GPU (DJ opens it; runs unattended for hours)

- [~] **TEST 17 -- newer-architecture bracket on the SAME data.** RUNNING
      since 2026-07-25 20:54 on a fresh RTX 4090 pod (203.57.40.89:10157).
      /workspace/bracket.py, sequential queue, ~70s/epoch. Volume survived
      intact (1370 train imgs incl. DJ's 230, 32 valid). Queue order:
      yolov8l-from-stock (CONTROL) -> yolo11l -> yolo12l -> yolo26l. Logs at
      /workspace/bracket_queue.log + /workspace/bracket_<name>.log. A failing
      candidate is logged and skipped, never kills the queue.
      Installed ultralytics 8.4.75 supports yolo11, yolo12, yolo26 and
      rt-detr; every model to date is yolov8. One word changes in the train
      command; the dataset (1370 train imgs incl. DJ's 230) is already on the
      network volume.
      - Candidates: yolo11l, yolo12l, rt-detr-l, and yolo26 if it trains
        cleanly. Same recipe as v3 (imgsz=1280, cos_lr, scale=0.7,
        epochs 150 / patience 60).
      - CONTROL RUN REQUIRED: yolov8l from STOCK on the same merged dataset.
        v3 was warm-started from v2, so without this control we cannot tell
        "new architecture" from "warm start" -- the comparison would be
        meaningless. This is the test's load-bearing detail.
      - Gate for every candidate: the SAME gate v3 passed -- HARD 2 verified
        shots reproduced, all 4 known non-shots rejected, TEST1 5/5.
      - EXPECTATION, on the record before running: probably incremental.
        Bigger/newer models were already a wash once (DECISIONS 21), and this
        footage is information-limited, not architecture-limited. It is worth
        doing because it is ~2h unattended per run and a few dollars, not
        because it is likely to be the breakthrough.
      - Cost: ~2h/run on a 4090, 4-5 runs, roughly $3-6 total.

### Phase 3 -- needs DJ (ground truth / labels), the most valuable phase

- [ ] **TEST 19 -- THIRD AND FOURTH CLIP AS A TRUE HOLDOUT.**
      The single most valuable test in this plan. Everything we believe is
      tuned on 2 clips; this is the first honest generalization measurement.
      - Clips: TEST2 (Fairfield, 48s, rims already recorded in clips_config)
        + the longer ~5-min clip DJ offered. The 5-min clip matters more --
        48s is too few shots to conclude anything.
      - REQUIRES FROM DJ: a watched-once list of shot timestamps and what
        each one was (shot / layup / pass / rebound / inbounds hold). NOT the
        roster and NOT calibration -- shot detection needs neither. Roster
        and court dims only matter for identity and shot LOCATION, which is
        the other chat's track.
      - Run v3 (and TEST 17's winner) cold against that list. Report caught,
        missed, and falsely claimed.
      - This is also where any TEST 16 pose rule gets its real verdict.
        A rule that survives a clip it was not built on is a finding; a rule
        that only fits HARD+TEST1 is accel_y all over again.

- [ ] **TEST 18 -- own-footage validation set (retire the useless 32-image
      val).** Harvest ~50 ball frames from a clip NOT in training, pre-fill
      boxes with v3, DJ corrects. Held back from training, permanently.
      Turns "one hour of clip gates" into "30 seconds" for every future
      model version -- it compounds across every run in TEST 17 and beyond.

- [ ] **TEST 20 -- scoreboard occlusion-skip (stretch, lowest priority).**
      Detect when a player box overlaps the scoreboard region and SKIP those
      frames entirely rather than voting through them. Optionally read the
      game CLOCK too: it ticks down one second at a time, so it validates
      itself (a read that is not previous-minus-one is wrong, no vote
      needed) and marks whistles, which is when baskets get counted.
      Deliberately last: make/miss is a nice-to-have, shot detection is not.

### Ordering rationale

15 and 16 need nothing and answer the two biggest open questions (is the
junk-detection bug the tracker blocker? does pose separate the false
positives?). 17 runs unattended in the background while 15/16 proceed. 19 is
the highest-value test but is gated on DJ's ground-truth list, so it is
staged last -- and it is the ONLY thing that can validate whatever 16 finds.

### What each phase needs from DJ

1. **GPU opened** -- for TEST 17 only. Phase 1 does not need it.
2. **Shot list for TEST2 + the longer clip** -- timestamps + what each play
   actually was. This is the gating input for TEST 19.
3. **The longer ~5-min clip** -- yes please, more valuable than TEST2's 48s.
4. **Roster/jersey colors** -- NOT needed for any test here. That belongs to
   the calibration/identity track, not CV quality.

---

# ============================================================================
# HANDOFF -- 2026-07-25 to 07-27. READ THIS FIRST.
# ============================================================================

## WHERE THINGS STAND
TEST2 (Fairfield, a brand-new gym) now runs end-to-end and produced its first
real box score. HARD and TEST1 both had real correctness bugs fixed. Nothing is
blocked on code right now -- the next move is DJ marking two tracks as spliced
and a re-run.

Coverage, against DJ's REVISED metric ("80% of what is actually READABLE" --
on court, referees excluded -- NOT 80% of the raw clip):
    HARD   36.3%      TEST1  64.9%      TEST2  first run, see below
DJ's target is 80%. 90% is ABOVE the physical ceiling (~91% generous) because
players leave frame and substitute -- do not chase it.

## WHAT WAS FIXED (all committed, suite 230 green)
1. COURT: Fairfield's floor is 94 ft, not the assumed 84. The config had
   dict(HS_COURT) copied from TEST1, so the engine squeezed a 94-ft court into
   84 and dragged every mark ~10 ft out. TEST2 0.94 -> 0.29 ft, = TEST1's
   glued 0.29. HARD measures 94 too; DJ chose to LEAVE HARD at 84 (test clip,
   not worth re-verifying its results).
2. spikes/court_detect.py -- the court is now MEASURED from the marks and
   snapped to a real court (84/94 x 12/16 ft key), using the rulebook's exact
   numbers rather than the fitted ones (free-fitting returns 82.6 for TEST1's
   real 84.0). REFUSES when two courts are within 1.35x or nothing fits.
   TEST2 uses "court": "auto".
3. RENDERER: a homography is defined up to sign, and to_px read a negative
   depth as "behind the camera", silently deleting the ENTIRE court from a
   view. That was DJ's "missing lines". stage5 had a signfix; stage4 -- the
   renderer every overlay actually goes through -- never got it. Fixed.
4. IDENTITY (the big one): continuity may now only relink the SAME track id.
   Different-id relinks overruled ByteTrack on ~150px of proximity and were
   ~70% wrong, merging up to three girls into one identity. Wrong-player time
   47.2s -> 0 on HARD, 14.5s -> 0 on TEST1. Coverage went UP too, because a bad
   relink also BLOCKED DJ's own click from landing.
5. Referees excluded from seeding (was landing in only 1 of 3 stages at first).
6. Queue/OCR eligibility keyed on ever_unresolved, not final state (an identity
   that died was invisible to both -- ~40% of HARD's player pool).
7. QUEUE CLICKS RE-KEYED TO track_id. They were filed under identity_id, a
   creation counter, so fix 4 silently re-pointed 11/15 HARD and 7/10 TEST1
   clicks at DIFFERENT bodies. Recovered 8+5 from the backup; the rest were
   made on merged chains and are unrecoverable by design (marked needs_review).
8. Coverage denominator unified -- the app showed 53.6% for HARD where the
   honest figure is 36.3%.
9. REVIEW BUNDLE rebuilt: whole player + blown-up number, 6 cells in TIME
   ORDER, frame numbers. Plus a BENCH button and a TWO PLAYERS (splice) button.
10. Web app: 5 fixes (unnamed bucket surfaced, court from data, ambiguous rows,
    AI moved below the numbers, live-vs-recovered shown).

## MISTAKES I MADE -- calibrate trust accordingly
- "Name the unnamed players -> +30 points": WRONG. Two thirds of that bucket is
  referees. Would have credited officials as players.
- "wrong-player time 0.0s": a TAUTOLOGY, not a measurement. After fix 4 every
  identity sits on one track so the check cannot fire. identity_report.py now
  declares this itself. WE CURRENTLY HAVE NO WORKING CORRECTNESS METRIC.
- "same-id relinks 52/52 correct": vacuous, it compared a track to itself.
- Warned DJ the queue clicks would break, then ran without a guard.
- Patched Part 2 of the bundle by string replace without asserting; it silently
  did not apply and I reported it done.
LESSON: assert every patch matched, and never quote a metric without asking
whether it CAN return a bad answer.

## TEST2 STATE
Config in clip_config.TEST2_CLIP. Roster user-confirmed: Fairfield (white/red)
1/3/4/13/25 vs Milford (black/red) 1/4/13/23/44 -- THREE shared numbers
(1, 4, 13), the hardest dual-roster case yet, kept deliberately because it will
happen in production. Span 40..400 (12s of a 48s clip; keyframes 40 and 140
were recovered from git, worth 0.10/0.19 ft). Caches built.
FIRST BOX SCORE: 8 players named, unnamed bucket 0.0s, and the COLOUR TIEBREAK
RESOLVED 6/6 ambiguous cases -- it correctly split the two different girls both
wearing #1 onto separate lines. That was the main risk and it worked.
OPEN: 3 clicks were REFUSED ("same number in two places") and #25/#23 show
disputed seconds. Cause is almost certainly t8 and t137, which DJ confirmed BY
EYE each follow two different girls. purity.py scores both "consistent".
NEXT ACTION: DJ marks t8 + t137 with the new TWO PLAYERS button -> re-run.

## THE BIGGEST GAP (DJ's own #1 want)
DJ: "what they do with the ball in their hand is the biggest one." NOTHING IN
THE SYSTEM LINKS THE BALL TO A PLAYER. Ball detection works, player tracking
works, they are never joined. phase2/possessions.py is about which HALF of the
court the bodies are on, not who holds the ball. So "drove left 70%" is NOT
honestly producible -- position data can only say a player MOVED left.
Building it = "which tracked player is nearest the ball each frame". No new
model needed. This is the highest-value item left.

## NEXT, IN ORDER (my recommendation)
1. DJ marks t8/t137 spliced -> re-run TEST2 clean.
2. BALL-TO-PLAYER POSSESSION. Unlocks the sentences the product sells.
3. Demotion fix: 53.6s (27.6% of HARD's readable) is thrown away because a
   CONFIRMED identity is demoted when ByteTrack drops its box for 1-2 frames.
   Safe version (gap <= 2) measured at the noise floor, +2.6/+1.6. Bigger
   version (gap <= 10 AND jump <= 3 ft) is +13.4 HARD / +16.1 TEST1 but wants
   an eyeball pass over ~40 cases first.
4. Sort the review queue by seconds-unlocked: the best 10 clicks are worth 54s
   instead of 8s. One sort key.
5. Hold-out label test -- withhold k labels, re-run, see if that player's time
   gets credited to someone else. This is how we get a correctness metric that
   CAN be non-zero again.
6. Browser demo loop (upload -> click court -> click players -> stats) on a
   SHORT clip.

## FULL-GAME DEMO: NOT CLOSE, and DJ has been told
Three things scale badly: CALIBRATION (8 marked frames bought 12s; a game is
160x), COMPUTE (per-frame SIFT: 361 frames -> ~57,600), CLICKING (14 clicks per
12s). Needs auto-calibration + a compute story, both real projects. The
achievable near-term demo is the full loop in the browser on a short clip.

## CONSTRAINTS / GOTCHAS
- ONE CLIP PER PROCESS. clip_config.ACTIVE_CLIP binds at import.
- Always .venv/Scripts/python.exe.
- DJ is NOT technical. Plain English, no jargon dumps; keep detail in this file.
- CORRECTNESS OUTRANKS COVERAGE. A number going DOWN after a fix is normal
  here -- it usually means fiction was removed.
- DJ's own labels are the ground truth everything is measured against.
- Backups: phase2/out/_baseline_20260725/ (pre-identity-fix artifacts) and
  *_decisions.backup-20260726-pre-rekey.json. Do not delete.
- Run phase2/identity_report.py BEFORE and AFTER any identity change.

# INDIVIDUAL TRACKER -- coverage + a CORRECTNESS bug (2026-07-25)

DJ: "the individual tracker is by far and away the most valuable piece... I
don't want a lost player as long as it's correct. The metric is 90%; for now
I'm fine with 70%." Two review subagents were run at DJ's request; every
load-bearing claim below was re-verified by me against shipped artifacts.

## BASELINE (measured, not assumed)
Named coverage = player-seconds carrying a jersey number / (frames x 10):
    HARD  29.8%      TEST1 73.3%
Better denominator (on-court, referees excluded -- refs are ~2.98 track-frames
per frame on BOTH clips, a ~30% tax, and are the whole reason TEST1 reported
>100%):
    HARD  31.2%      TEST1 75.3%
PHYSICAL CEILING: non-ref bodies actually on court per frame is mean 9.7,
median 9, min 5 on HARD (55% of frames have FEWER than 10). Ceiling for a
"x10 players" metric is 91.2% HARD / 94.8% TEST1 -- and that is generous,
since unlabelled refs/coaches still count. **90% is at or above the ceiling;
70% is reachable.** Do not chase 90% -- anything reporting it is contamination.

## THE CORRECTNESS BUG (verify: scratchpad/verify_relink.py, shipped data only)
When ByteTrack loses a player, identity.py `_handle_reappearance` relinks the
lost identity onto a DIFFERENT track id purely on motion proximity
(MAX_MATCH_DIST_PX=150 ~ 6 ft, MAX_GAP_FRAMES=30). Those guesses are wrong
most of the time, and the identity keeps its old roster_number the whole way:
    HARD  7 of 18 judgeable chains (39%) span >=2 DIFFERENT human-labelled
          numbers -- 1415 track-frames / 47.2s
    TEST1 2 of 10 (20%) -- 436 frames
    worst: window 0 identity 19 -- ONE identity, 278 frames, covering tracks
           the human labelled #20, #23 AND #44.
So floor time and zone data are being credited to the WRONG PLAYER. This is a
correctness failure, not just missing coverage -- it outranks everything else.
Agent measurement (replay reproduced shipped artifacts exactly): different-id
relinks 10/10 wrong on HARD, 2/2 on TEST1; same-id relinks 52/52 CORRECT. The
discriminator is same-id vs different-id, NOT distance (right and wrong relinks
overlap fully in px, so tightening the gate is a measured dead end).

## WHY HARD (30%) TRAILS TEST1 (73%)
Not OCR, not the roster, not crop size. HARD fragments more (48 on-court
non-ref fragments in w0 vs TEST1's 30), so the 70%-wrong relinker runs 209
times vs 54 -- and EACH wrong relink also BLOCKS the human's own label:
windows.py:76-78 refuses to late-seed a track whose identity is already a
relinked CANDIDATE. Result: 53% of the frames on tracks DJ personally labelled
(1675 frames / 55.8s on HARD) never reach the box score; 92% of them sit in
CANDIDATE. TEST1 loses only 10%.
CORRECTION to DECISIONS 4b ("the gap is DISTANCE / crop-size"): not supported
by the cached data. On-court bbox height median HARD 172px vs TEST1 185px, and
99%+ of BOTH clear MIN_OCR_HEIGHT=90. What separates them is CAMERA MOTION --
HARD pans 3.6 px/frame median vs TEST1's 0.8 (4.5x) -- and crop sharpness
(Laplacian median 213 vs 346). The live hypothesis is MOTION BLUR.

## THE UNNAMED BUCKET IS A DEAD END (my earlier hypothesis was WRONG)
I claimed naming the 30.8% confirmed-but-unnamed would take HARD to ~60%.
Verified false: 67% of it (41.5s) is REFEREES the human already labelled
"ref", and the other 33% is ten identities the human looked at and marked
null = "can't tell". TEST1 is 93.5% refs. Naming them would credit officials
as players -- a ref stands in the paint all possession, so it would fabricate
exactly the positional tendency the product sells. Treat as ~0 opportunity.
Also: roster.py:37-39 DISCARDS the "ref" labels, so nothing stops
stage4_seed_queue.py:104 seeding officials as CONFIRMED.

## RESULT (items 1-3 built + measured 2026-07-25; DJ approved "do the first 3")
DJ also RESET THE TARGET: 80% of what is actually READABLE (on court, refs
excluded), not 80% of the raw clip -- "good enough to be majority correct...
it doesn't have to be 100% to get the idea of the player." identity_report.py
measures exactly that.

                       wrong-player time    coverage     your clicks used
    HARD   before          47.2s             30.7%           46.9%
    HARD   after            0.0s             36.7%           63.5%
    TEST1  before          14.5s             74.8%           89.9%
    TEST1  after            0.0s             70.6%           90.6%

CORRECTNESS IS CLEAN ON BOTH CLIPS: zero identity chains span two differently-
labelled players (was 7 on HARD, 2 on TEST1). The #20/#23/#44 chain is gone.
TEST1's coverage FALLING is the fix working -- the old 74.8% counted seconds
credited to the wrong player. A clean 70.6% beats a dirty 74.8%.
NOT YET AT TARGET: HARD 36.7%, TEST1 70.6% vs 80%.
COST TO WATCH: the review queue grew (HARD 23->82, TEST1 14->46) because item 3
surfaces identities that were previously invisible. Those are clicks that are
now POSSIBLE, not clicks that are now required -- but the queue must be sorted
by payoff (seconds unlocked) before it is put in front of a coach, or it reads
as more work rather than more reach. That is the next thing to build.
Safety tests: 9 failed because they asserted the removed MECHANISM (relink onto
a different track id -> CANDIDATE), not the property. A different-id
reappearance now abstains to UNKNOWN, which attributes strictly LESS than
CANDIDATE. Fixture moved to same-id, every safety assertion kept, 4 new tests
added. Suite 223 -> 226.

## FEWER CLICKS -- options for DJ, ranked (asked 2026-07-25, none built yet)
- SORT THE QUEUE BY PAYOFF (seconds unlocked). Cheapest, biggest felt win:
  click the top 10 of 82 instead of all 82.
- RAISE ByteTrack's track_buffer (currently 30 = ~1s). Its OWN re-acquisitions
  measured 52/52 correct while our guesses were ~30%, so pushing the work INTO
  the tracker converts unreliable relinks into reliable ones and cuts breaks.
  A buf60 sweep already exists (spikes/out/TEST1_tracks_sweep_buf60.json).
- JERSEY-COLOUR VETO: color_tiebreak already builds team colour centroids. It
  cannot separate teammates (DECISIONS 11) but it CAN prove two bodies in
  different team colours are not the same person. Cheap negative filter.
- COURT-FEET GATE instead of 150 PIXELS. Unlocked by today's calibration fix:
  a pixel gate means different real distances at different zooms. Measure the
  relink in feet and apply a real speed limit (a player cannot cross ~15 ft in
  a third of a second).
- LEGALITY CHECK: >5 players of one team on court is impossible. HARD credits
  6 Milford players in a 20s clip today and nothing flags it.

## PLAN (items 1-3 DONE, see RESULT above)
- [ ] 1. Stop discarding "ref" labels; exclude ref-labelled tracks from
      seeding. Cheap, safe, kills the >100% artifact, shrinks the queue by
      removing non-players (the ONE safe queue reduction -- it ADDS
      information rather than lowering a bar).
- [ ] 2. Restrict continuity relinking to the SAME track id (identity.py:213,
      ~3 lines). Removes 10/10 measured wrong relinks; coverage goes UP
      (HARD 24.0->30.2%, TEST1 44.0->53.6% live-named) because refusing a bad
      guess leaves the player a fresh UNKNOWN that DJ's label can then seed.
      Safer AND more coverage -- rare, so verify hard. TESTS FIRST.
      !! MIGRATION: identity ids shift, so DJ's saved queue_resolutions in
      decisions.json land on DIFFERENT identities and stage7_merge.py:156-158
      will NOT catch it. Needs a generation stamp + a re-run of the queue
      session. Track labels (keyed by track_id) survive fine.
- [ ] 3. Queue/OCR eligibility currently keys on the identity's FINAL state
      (stage4:127-128, stage6:104), so an identity that was CANDIDATE and then
      died is LOST at the end and invisible to BOTH. Hides 2290 track-frames
      = 39.9% of HARD's player pool. Change to "was EVER candidate/unknown".
      MUST come AFTER 2 -- done first it would surface the wrong chains and one
      click would credit ~9s to the wrong player.
- [ ] 4. Honest metric: key on (team, number) not number alone (HARD's #3 and
      #23 are on BOTH rosters), denominator = on-court non-ref bodies, report
      the ceiling, and add a "<=5 players per team on court" legality check --
      HARD currently credits 6 Milford players in a 20s clip, which nothing
      flags today.
- [ ] 5. DJ'S CALL, NOT MINE: let a SAME-ID reappearance keep its prior state
      instead of demoting to CANDIDATE. Would give HARD 54.5% / TEST1 76.9%
      live-named -- clears 70% on TEST1 with no retro credit. Evidence 52/52
      same-id relinks correct, but n is small and it loosens DECISIONS 1.
      Argument FOR: the demotion does not defend against the actual observed
      threat -- the t49-class splice (DECISIONS 7) happens INSIDE one track id
      with no gap, and identity.py:206-209 already keeps that CONFIRMED. The
      demotion only fires on gaps, which is not where splices were found.
      Do NOT slip this in; it needs an explicit decision + an eyeball pass.
- [ ] 6. Cross-channel contradiction check (existing KNOWN DEBT): a chain
      holding >=2 different human-labelled numbers should be quarantined like
      a spliced track (roster.py:74-84). Pure detection, coverage-neutral,
      catches the exact class of error item 2 removes. w0 id19 is sitting in
      the current data unflagged.

NOT worth doing (measured dead ends): tightening the relink distance gate;
investing in OCR for coverage on HARD (2.0% confident reads per crop -- the
blocker is motion blur, a capture lever not a code lever); appearance re-ID
for teammates (DECISIONS 11).

# COURT HOMOGRAPHY -- REAL ROOT CAUSE FOUND (2026-07-25)

DJ: "the court homography is completely wrong... I have clicked right, and we
built this before on a different clip and it was glued. We are not moving on
until we perfect the coding and the math of the core homography."

## THE ROOT CAUSE: Fairfield's court is 94 ft long. We told the code 84.

TEST2 (Fairfield) is a 94-foot floor. `clips_config` gives it
`dict(HS_COURT)` = 84 x 50, copy-pasted from HARD/TEST1 without ever checking.
So the engine was told to squeeze a 94-ft court into 84 ft. Every mark DJ
clicked was then forced ~10 ft out of place along the court's length -- which
is exactly the symptom: arcs off, lines not glued, the far end wrong.

DJ's clicks were never the problem. Neither was the camera, the pan, the
glare, the worn lines, or lens distortion (all four investigated and all four
wrong). This was measured, not guessed:

- SOLVED the court from DJ's clicks instead of assuming it (unknowns = one
  homography per keyframe + court length/lane width/FT distance/circle radius/
  3pt apex; 56 marks, held width at 50 ft as the scale gauge):
      TEST1 -> length 82.6, lane 11.9, FT 18.4, circle 5.96, apex 25.0
               = a standard HS court. Which is why TEST1 came out glued.
      TEST2 -> length 93.86, lane 11.75, FT 18.8, circle 5.97, apex 24.82
               = a 94-ft floor with ordinary HS markings (12-ft lane,
                 19.75 3pt). Error 0.62 -> 0.19 ft just by freeing length.
- SWEPT the length 80->100 ft, everything else held at HS values. One clean
  minimum each: TEST1 at 85.0 ft, TEST2 at 94.5 ft. At 84 ft TEST2 is 3x
  worse (0.618 vs 0.199 ft).
- Cross-ratio check (no homography, no assumed dims -- 5 marks on the
  half-court line, 3 of which cannot be wrong): centre circle measures
  ~6 ft radius on BOTH clips. So the circle is fine and the width is fine;
  only the LENGTH was wrong.

RESULT with length = 94 and the ORIGINAL shared model:
      TEST2  mean 0.29 ft / max 0.72 ft   <-- was 0.94 / 2.44
      TEST1  mean 0.29 ft / max 0.56 ft   (the glued benchmark)
  Identical. And circle_left/right, the two marks that were 1.3-1.5 ft off,
  land at 0.03 / 0.02 ft. Eyeball-confirmed on kf 240/340/400: centre circle
  on the painted circle, half-court line on the painted line, both 3pt arcs
  on the paint, sidelines on the border.

## SECOND BUG: last session's "fix" was overfitting, and must come out

`direct_keyframe_homography` (fit each keyframe to ONLY its own marks) made
the reported error fall 1.16 -> 0.60 ft, which is why it looked like a fix.
It was not. A homography has 8 degrees of freedom and those keyframes have
6-7 marks, so the fit just ate its own data. Leave-one-out proves it -- hold
back one mark, fit on the rest, predict it:
      TEST1 per-keyframe:  in-sample 0.16 ft -> held-out 0.37 ft
      TEST2 per-keyframe:  in-sample 0.62 ft -> held-out 1.62 ft  (2.6x)
      HARD  per-keyframe:  in-sample 0.56 ft -> held-out 26.3 ft, max 538 ft
HARD explodes because two of its keyframes carry only 5 marks. That is
exactly what the shared model exists to prevent: it pools all ~56-66 marks
across every keyframe into one court fit, so no single frame has to be
self-sufficient. The per-keyframe path also throws away the whole near half
of the court on frames 240/275 (no mark below y=19), so the overlay was
extrapolating blind -- hence "missing lines".

## Plan  (DJ approved 2026-07-25: "if you believe that's the best route,
## I 100% back it" -- the court comes from the clicks, not a hard-coded dim)
- [x] 1. TEST2 court -> measured, not assumed. It is now `"court": "auto"`.
- [x] 2. Removed `direct_keyframe_homography` -- the flag, stage4's
      `compute_H_court_per_keyframe()`, and the branch in
      stage1_court_roi.build_court_anchor(). One shared validated path again.
- [x] 3. Un-hard-coded the court length downstream:
        spikes/shot_location.py  -> reads the active clip's court (like stage4)
        measured_stats.py        -> classify_zone(x, y, court_len) +
                                    court_length_for(clip); default kept so
                                    existing callers/tests are untouched
- [x] 4. spikes/court_detect.py -- the thing that stops this recurring.
      DESIGN NOTE (why it is not "fit the court freely"): free-fitting
      returns 82.61 ft for TEST1, whose floor is really 84.0 -- it buys a
      hair of reprojection error and pays with a 1.4 ft error baked into
      every shot location. A court is not an arbitrary shape. So the module
      SCORES the four courts that actually exist (84/94 ft x 12/16 ft key)
      and uses the winner's EXACT published dimensions: the measurement
      picks the court, the rulebook supplies the numbers. It refuses in two
      cases rather than guess -- when the runner-up is within 1.35x (the
      marks can't tell the floors apart) and when even the best is over
      1.0 ft off (a mark is wrong, not the court). `"court": "auto"` in
      clips_config resolves through it and RAISES if it can't tell.
      court_model() is now the single source of truth for tag -> court feet;
      stage4 builds COURT_MODEL from it so the two can never drift.
- [x] 5. Suite 204 -> 218 green. TEST2 re-rendered through the SHIPPED path
      (nothing patched): 0.29 ft mean / 0.72 max, and eyeball-confirmed on
      keyframe 300, which was never used for any of the tuning.

NOT in scope: auto-calibration, the marking tool, roster, shot-chart run.

## Review (2026-07-25)
- TEST2 is glued: 0.29 ft mean / 0.72 ft max, identical to TEST1's 0.29.
  Both keyframe 300 and 340 confirm by eye -- centre circle on the painted
  circle, half-court line on the painted line, both arcs on the paint.
- The court is no longer declared anywhere for TEST2. It is measured from
  DJ's clicks (0.20 ft for the 94-ft court vs 0.62 for the 84, a 3.1x call),
  which is the actual feature: the next new clip needs no court knowledge.
- TEST1 re-verified UNCHANGED at 0.29 ft / 0.56 max. HARD unchanged. Both
  keep explicit dims; only TEST2 is on "auto".
- Blast-radius check on the hard-coded 84s, since it is easy to under-rate:
  the same court position reads as a 20.0 ft three on an 84-ft model and
  26.5 ft on a 94-ft one. Zone classification and the chart shape both move.

## SECOND BUG FIXED: the renderer silently deleted the whole court
DJ's other complaint -- "there's missing lines" -- was a real and separate
bug, not a symptom of the court length. A homography is defined only up to
sign, so the same correct transform can arrive with every depth negated.
`stage4.to_px` reads a negative depth as "behind the camera" and returns None,
and `draw_court` only draws a segment when BOTH ends survive -- so on a view
where the sign came out negative, EVERY line vanished. Measured on HARD
keyframe 1200: 205 of 207 court points dropped, including the right lane and
rim sitting in plain view, while that keyframe's own marks reprojected fine at
0.59 ft. The calibration was right; the picture was empty.
Already known and already solved ONCE: spikes/stage5_courtmap.py exists
because of this exact bug ("the Stage-4 blank right half was NOT a
representation limit, it was a homography SIGN-convention bug") and carries
its own `signfix`. stage4 -- the renderer every calibration overlay actually
goes through, including stage1's eyeball video and every image DJ has been
judging -- never got it. Fixed at the source: `stage4.signfix()` + one line in
`draw_court`, so every caller is covered. Uses the real court centre (CX, CY)
rather than stage5's hard-coded (42, 25), which is itself wrong on a 94-ft
floor. No effect where the sign was already positive.
The MEASUREMENT pipeline was never affected -- phase1's `on_court()` compares
depth against a reference point's sign, so it is sign-agnostic. This was an
eyeball-only bug, which is exactly why it survived: the numbers looked fine.

## RESOLVED: HARD is a 94-ft floor too
Found while testing the detector on all three clips, then confirmed through
the REAL pipeline path (refit_keyframes, not stage4.run_optimization):
    HARD court = 84 ft  ->  mean 0.76 ft / max 1.97   (its configured value)
    HARD court = 94 ft  ->  mean 0.32 ft / max 0.74
    length sweep        sharp minimum at 94.0 ft (0.202) vs 84 ft (0.555)
    free solve          93.90 ft; lane 11.70, FT 18.97, circle 5.98
    detector            94 ft, 2.7x clear of the runner-up
    eyeball             at 94 ft the arc lands on the painted arc at kf1200;
                        at 84 ft it is visibly inside the paint
All three clips now sit together: TEST1 0.29, TEST2 0.27, HARD 0.32 ft.
CORRECTION to an earlier note here: the free solve's "apex 25.00" was cited as
HARD evidence and should not have been -- HARD has no arc marks at all, so that
parameter was unconstrained and simply stayed at its starting value. The other
four lines above stand on their own.
This also explains something never questioned: HARD's calibration always sat
near 0.95 ft while TEST1 sat at 0.25, and the gap was written off as "on par".
It was the wrong court the whole time.
WHY IT IS STILL NOT SWITCHED: HARD is the validated baseline the whole shot
layer was gated on (TEST 10). Switching it moves every HARD court_feet
position, its shot chart, its zones, and the numbers currently rendering in
the web app. The suite does NOT lock it (the shot-layer regressions are all
pixel-space; test_possessions passes its own synthetic LEN), so nothing would
fail loudly -- which is precisely why it needs a deliberate call rather than a
quiet edit. Recommended: switch HARD to "auto" and re-verify its shot chart.

## Notes for the marking tool (later, not now)
The tool's mini court diagram is drawn to HS proportions, and the config
palette assumes them too. When court length becomes per-clip, the tool
should ask which floor it is (or infer it after the first fit) rather than
silently drawing an 84-ft court over a 94-ft gym.

## Notes for the marking tool (later, not now)
The tool's mini court diagram is drawn to HS proportions, and the config
palette assumes them too. When court length becomes per-clip, the tool
should ask which floor it is (or infer it after the first fit) rather than
silently drawing an 84-ft court over a 94-ft gym.

# PHASE 7 -- connect the pipeline to DJ's real web app (2026-07-22)

Ship-handoff item 3. Goal (ROADMAP Phase 7): a coach works entirely in
the web app -> a job runs -> box score + shot chart appear there.

## DJ STEER 2026-07-22 (reshaped the plan -- DO NOT revert to local-first)
- Manual per-game setup STAYS (calibration clicks, hoop anchors, roster).
  DJ does NOT want auto-calibration yet ("we dont need full autonomy
  yet"). The ask is to make that manual setup "slightly easier and
  faster" by moving it INTO the web app, not to eliminate it. So Phase 4
  auto-cal stays deferred; the setup-clicking becomes a web-app workflow
  over time, not a code/terminal chore.
- Wire into the REAL web app NOW (rejected local-first stand-in),
  because DJ sees the sequence as: web-app connection THEN the AI
  analyzer (Phase 8) as the final step.

## WEB APP DISCOVERED 2026-07-22 (major strategic finding)
Location: c:\Users\djcha\New folder\basketball analysis app (SIBLING
folder). It is a COMPLETE, working, deploy-ready product -- NOT a shell:
- Stack: Next.js 15 / React 19 / TypeScript, Supabase (auth + Postgres +
  storage), Tailwind. Hosted-ready, invite-only auth, per-account quota.
- What it DOES today ("Basketball Film Analyzer"): upload video ->
  a 3-pass GEMINI VIDEO CASCADE (motion scan -> wide pass -> deep pass
  -> synthesis) -> a qualitative SCOUTING REPORT: possessions, play
  types, tendencies, game patterns, per-player reports (jersey #/color
  as GEMINI READS them), game plan, coaching narrative. Streams live,
  saves to Supabase, has a History page.
- Supabase tables: videos, sequences, possessions, analyses,
  game_patterns, player_reports, folders. NO jobs table, NO measured-
  stats tables yet. Processing runs server-side in a Next.js API route
  (app/api/analyze/route.ts), Node/TS -- NOT Python.

THE CORE TENSION (this is why "connect them" is a strategic decision,
not plumbing): the web app ALREADY does "AI analysis" -- but in exactly
the way THIS project's core philosophy (ROADMAP Principle 3) warns
against: an LLM WATCHING VIDEO and computing the stats/possessions
itself. The CV pipeline exists precisely because Gemini-on-video is
confident-wrong on the hard numbers (jersey-level box score, per-player
tracking through occlusion, precise shot locations). So:
  - web app = fast, automatic, qualitative "story" (LLM's read of video)
  - CV pipeline = slow + manual-setup, but MEASURED hard numbers
    (deterministic box score, real shot chart, tracked identities)
They are complementary (story vs measured numbers), and how they fit is
DJ's call. Also a practical mismatch: the app produces results in minutes
on upload with NO setup; the CV pipeline needs manual calibration + slow
processing, so its stats can't appear "instantly on upload" the same way.
DJ DIRECTION 2026-07-22: the HYBRID. CV pipeline owns the hard numbers
(replacing the AI's shaky guessed numbers); the AI KEEPS watching the
video for the story (plays/tendencies/coaching read) but narrates
GROUNDED in the real measured numbers instead of inventing them.

## THE NORTH-STAR GOAL (DJ: "this is the whole goal") -- build toward it
The AI's advice must be ACTIONABLE + SPECIFIC, not vague fluff. DJ's
target examples: "drove left 70%, made 30% of those"; "60% of shots from
behind the arc but struggled inside it." NOT "player is fast-paced /
tends to pass." Full honest capability map lives in memory
project_actionable_stats_goal.md. Short version for planning:
  - STRONG now (CV measures): shot DISTRIBUTION by court zone (3pt vs
    mid vs paint, left/right/corner/wing), shot volume per player,
    floor-time + operating zones. -> real actionable spatial lines.
  - NOT reliable yet: make/miss % ("made 30%") -- Gate 4 unpassed;
    points/reb/ast/stl unmeasured. Don't promise shooting percentages.
  - AI-watching only (not CV numbers): drives + direction, P&R, post-ups
    -- AI describes from video; numbers can't quantify "drove left 70%".
  - Sample size: per-player % tendencies need FULL + MULTIPLE games
    (Phase 6 scale + multi-game aggregation), not one short clip.
The wiring plan (below, still provisional) must feed the CV spatial
stats into the AI narrative step so the scouting report cites real
measured distributions -- that is the concrete path to "actionable."

## WEB APP UI ARCHITECTURE (read 2026-07-22, grounds the demo plan)
- components/AnalysisTabs.tsx = the tab container (Film Room, Scouting
  Report, Game Plan, Game Flow, Events, Analytics, Stats, Player). Adding
  a MEASURED tab = one ALL_TABS entry + one new component + wire its data
  as a prop. Clean, ADDITIVE insertion point -- touches none of the
  existing Gemini tabs.
- Existing "Stats" tab (TeamStatSheet, tab 6) is derived from GEMINI
  possessions -> the measured stats are a SEPARATE new tab, clearly
  labeled MEASURED, so the two signals never blur.
- Data reaches AnalysisTabs as props from the page (home after analysis +
  history/[id] detail). app/api/analyze/route.ts already shells out to
  ffmpeg via execFile -- precedent for the app calling external tools.

## Plan (concrete, DEMO-FIRST; slice A uses ALREADY-set-up HARD/TEST1)
- [x] 0. Discover web app + confirm HYBRID direction + north-star goal. DONE.
- [ ] A. DEMO SLICE -- measured stats visible IN the web app for an
      already-calibrated game (HARD/TEST1). No calibration UI needed here
      (those clips are already set up), so this is the fast path to
      something DJ can show.
      - [ ] A1. CV side (my home turf): measured_stats.py ->
            one clean {clip}_measured_stats.json combining box_score +
            shot_locations/attempts into a WEB-READY CONTRACT:
              box_score rows (jersey #, team, floor-time seconds --
                ATTEMPTS only, NO make% since Gate 4 unpassed);
              shots [{court_x, court_y, zone: three|midrange|paint,
                shot_type, shooter_status, hoop}];
              shot_distribution summary (% of attempts by zone) -- the
                first ACTIONABLE spatial stat, directly proving the goal.
            This IS the Phase-7 "freeze the contract" step. Tests first
            for zone classification (3pt-arc geometry, reuse
            shot_location.py's R3/HOOP_DX) + distribution math.
      - [x] A2. Web side DONE (ADDITIVE, 3 NEW files, touched NOTHING
            existing -- the Gemini path is byte-unchanged). In the web app
            (sibling "basketball analysis app"):
              app/api/measured/[clip]/route.ts -- read-only bridge that
                serves {clip}_measured_stats.json from the CV project
                (CV_OUTPUT_DIR, default the sibling spikes/out). Rejects
                non-identifier clip names (path-traversal -> 400), 404 on
                missing. Later this is the Supabase-read swap point.
              components/MeasuredStats.tsx -- box score table ("Where They
                Operated" = per-player zone %), SVG half/full-court shot
                chart (mirrors shot_location.py geometry), and the
                behind-vs-inside-the-arc distribution headline. Honesty
                banner; NO make% shown.
              app/measured/[clip]/page.tsx -- standalone /measured/HARD
                view (client fetch of the bridge). NOT gated by middleware
                (fine for local demo). Later: fold into AnalysisTabs.
            Decision: standalone page (not an AnalysisTabs tab yet) --
            lower risk, zero edits to existing pages; tab integration
            waits until measured data is associated with a game record.
      - [x] A3. VERIFIED by screenshot (verify skill discipline):
            typecheck clean (npx tsc --noEmit, 0 errors); dev server up;
            /api/measured/HARD returns real data; path-traversal ->400,
            unknown ->404. FIRST render caught a REAL bug -- page was on a
            white bg (I'd assumed a global dark theme; the app applies
            bg-[#05080f] per-page) so light text was invisible. Fixed
            (wrapped the page in the app's min-h-screen bg-[#05080f]
            text-white). Re-screenshot: HARD renders box score (11
            players w/ zone %), the court + 1 located shot at the 3pt arc
            (right side), "100% threes" distribution. TEST1 renders the
            HONEST empty state ("No shots placed yet, 4 detected, shooter
            position not yet confident") + its box score. Screenshots:
            /tmp/measured_HARD_full.png, measured_TEST1_dark.png.
            Web app git: 3 new files on main, NOT committed (awaiting DJ
            -- don't push to their app's main without asking).
- [ ] B. (BIGGER, AFTER the demo) Calibration INSIDE the web app: a
      browser workflow to set up a NEW game -- upload -> pick keyframes ->
      click court landmarks -> mark hoops -> enter roster -> stored + fed
      to the CV pipeline. Replaces the code/terminal setup ("easier +
      faster", DJ's ask). LARGE front-end build (canvas clicking on
      frames); its own plan + check-in when we reach it.
- [ ] C. (CV priority right AFTER connect, per DJ) Strengthen MAKE/MISS
      so shooting PERCENTAGES become trustworthy -> unlocks "made 30%"
      style stats. Gate-4 work / scoreboard OCR. Its own plan.
- [ ] D. (real plumbing, when off the local-file demo) CV worker pushes
      measured_stats into Supabase per game; the Measured tab reads from
      Supabase like the rest of the app; feed measured distributions into
      the Gemini synthesis prompt so the scouting narrative CITES real
      numbers (the grounded-narrative half of the hybrid).

NOT in scope for the DEMO (slice A): calibration UI (B), make/miss (C),
Supabase plumbing (D), auto-calibration (Phase 4). Slice A = already-
computed HARD/TEST1 outputs + a local read-only route + one new tab.

## Review (Phase 7 slice A -- the measured-stats demo, 2026-07-22)
- DONE + verified by eye: the CV pipeline's trustworthy numbers now
  render IN DJ's real web app at /measured/{clip}, in the app's own
  style, WITHOUT touching the existing Gemini analysis (3 new files,
  zero edits to existing app code). This is the hybrid's first half made
  visible: measured hard numbers beside the AI's watched story.
- The north-star goal shows up already: the box score's "Where They
  Operated" column is real per-player spatial data (e.g. HARD #23 Milford
  84% Left Wing; #20 Winton Woods 74% Perimeter) -- exactly the
  actionable "where" the vague old advice lacked. The shot-distribution
  headline (behind vs inside the arc) is the shot half; today it's 1
  located HARD shot (a 20ft three), honestly labeled, with the other 3
  shown as "detected but not yet placed".
- Honesty held throughout: NO shooting % (make/miss unverified),
  presence-seconds caveat on floor time, TEST1's all-unlocated shots
  render as an explicit empty state (no fake dots).
- CV side is one new file + tests (measured_stats.py, 8 tests, suite
  196->204). Contract doubles as the Phase-7 "freeze the contract" step.
- Bug caught by the eyeball gate (not tests): white-bg render made light
  text invisible -- the app themes per-page, not globally. Fixed. Same
  lesson as the shot-chart mirror bug (DECISIONS 16): typecheck/tests
  green != looks right; a human/screenshot look is mandatory.
- NEXT (DJ's stated order): (C) make/miss so shooting % becomes real ->
  (B) calibration inside the web app for NEW games -> (D) Supabase
  plumbing + feed measured distributions into the Gemini narrative so
  the scouting report cites real numbers. Also open: commit the 3 web
  files (DJ's call), and later fold /measured into AnalysisTabs as a tab.

## DEMO RESCOPE + NEW WANTS (DJ 2026-07-22, after seeing slice A)
DJ: "this isnt the demo I wanted -- I want to LOCALLY HOST the web app,
input a game clip, and get one out." Slice A was a static VIEWER of
pre-computed HARD/TEST1 stats; DJ wants the real IN->OUT LOOP running
locally. The results-display half (slice A's component + bridge) is
REUSABLE -- what's missing is the TRIGGER (press Analyze -> run_clip
actually runs) and, for NEW clips, the browser SETUP (calibration).

HARD TRUTH restated: a NEW arbitrary clip cannot run without setup
(court landmarks + hoop anchors + roster) -- that's manual today and
must exist in the browser before "upload any clip -> stats" works. So
the loop comes in two honest steps:
  E1. WORKING LOOP on an ALREADY-SET-UP game (HARD/TEST1): web app
      "Analyze" button -> triggers the CV pipeline locally (worker /
      run_batch subprocess, the seam already ~built) -> processing state
      in the UI -> measured stats render (slice A component). Proves the
      real in->out loop without the huge calibration UI. This is the
      demo DJ actually wants, scoped to what's runnable now.
  E2. Then B (calibration IN the browser) so DJ can feed NEW clips.
Ordering vs DJ's earlier (C make/miss first): DJ's NEW demo ask (E1)
likely takes priority over C now -- CONFIRM with DJ. E1 is the loop; C
strengthens a number inside it.

AI GROUNDED NARRATIVE DONE 2026-07-22 (the dad demo DJ asked for --
"get gemini to analyze the box scores/stats we have"): the web app's
Measured view now has an "AI Scouting Read" button that sends the
measured numbers to Gemini (text-only, gemini-3.5-flash, reuses the
app's GEMINI_API_KEY) and shows a scouting write-up GROUNDED in those
numbers. New web files (all additive, Gemini video flow untouched):
lib/measuredStats.ts (shared reader; bridge route refactored onto it),
lib/measuredNarrative.ts (honest prompt + 3x 503-retry), app/api/measured/
[clip]/narrative/route.ts (POST), + MeasuredStats.tsx button/section.
GUARDRAILS held (verified live via curl x2): only uses given numbers,
cites a specific number per claim, flags thin-sample players as
low-confidence, says make/miss "not measured" (no invented %), uses
they/them (dad's team is girls -- caught + fixed a he/his default).
IDENTITY/FLOOR-TIME honesty: DJ noticed 3s floor times look wrong. They
are REAL, not a bug: (a) HARD span is only ~20s; (b) tracker fragments +
NO reliable auto re-ID (teammates identical -> appearance re-ID failed,
DECISIONS 11), so only confidently-identified stretches are credited
(under-credit by design). The prompt tells Gemini to treat small
floor-times as "limited confident tracking, not barely played." DJ's
parallel player-detector work directly attacks this (fewer fragments).
Web files still UNCOMMITTED on the app's main (DJ's call).

NEW WANT (DJ, record it -- "I still want it"): PER-PLAYER HEATMAPS.
  - shot-location heatmap per player = LATER (needs dense per-player
    shots -> rides on make/miss + full/multiple games; HARD has 1
    located shot today). DJ acknowledged "probably a later thing."
  - position/roam heatmap per player = ACHIEVABLE SOONER: court_feet is
    already stored per track per frame (oncourt cache) + zone_seconds
    already derived + Phase-1/2 heatmap infra exists (stage3_heatmap).
    Good candidate to add to the measured view relatively soon.
  See memory project_actionable_stats_goal.md.

# PHASE 7 CV-PRIMARY BUILD PLAN (DJ-approved architecture 2026-07-22)

DJ confirmed ("yes"): CV owns ALL the numbers; the AI is demoted to the
grounded STORY on top (runs AFTER CV, never overrides facts). Honest
limit DJ holds: CV = facts (who/where/shots/positions); the tactical
story (plays/tendencies/game plan) still needs the AI WATCHING, grounded
in CV. App's current AI-guessed Stat/Player tabs -> DEFAULT keep-but-mark
"AI estimate" beside CV truth (reversible; remove later). Full direction:
memory project_phase7_webapp_direction.md.

THE CRUX: "CV connected to everything" + "brand-new-clip path" are the
SAME build -- the PROCESSING LOOP: app runs CV on a clip -> authoritative
facts -> AI runs second grounded in them -> app leads with CV. Everything
DJ wants flows from "after the CV analyzes the clip."

Key architectural find: app/api/analyze/route.ts already shells out to
ffmpeg via execFile -- so the Next app can invoke the PYTHON CV pipeline
the same way (execFile `.venv/Scripts/python run_clip.py CLIP` /
run_batch), no separate worker service needed for a LOCAL host. Slow
(minutes) + needs the clip SET UP -- so first prove it on HARD (already
set up), add browser setup for NEW clips after.

## Build order (each its own check-in before touching the working app)
- [x] L1. CV-RUN BRIDGE DONE + VERIFIED 2026-07-22. CV side: analyze_clip.py
      (one entry point = run_clip + measured_stats.generate; prints STAGE
      markers; exits nonzero if the clip isn't a set-up ClipConfig) +
      measured_stats.generate() extracted. App side: lib/cvRunner.ts spawns
      the CV venv python on analyze_clip.py, captures stdout, tracks a
      status file (running->done/failed, surfaces STAGE as progress);
      app/api/cv-run/[clip] POST=start(bg)/GET=poll; /measured page gained
      a "Run CV analysis" button -> progress state -> auto-reload of the
      numbers. VERIFIED: app->python plumbing smoke-tested via cvRunner
      (fast script, done/exit0/facts produced); full analyze_clip.py HARD
      ran end-to-end 277s exit 0 (11 players, 1 shot). Button renders.
      Suite 204 green. Web files uncommitted on app's main. CONSTRAINT
      holds: only set-up clips (HARD/TEST1) run; NEW clips need L4 setup.
- [x] L2. DONE 2026-07-22 (Option A, DJ-chosen -- app analyses & CV clips
      aren't linked until CV runs on an uploaded game = L4, so full merge
      into per-game AnalysisTabs waits; instead make the CV view a
      first-class navigable PRIMARY surface now). Built:
      (a) AnalysisTabs Stat/Player tabs get an "AI estimate" banner
          (AiEstimateBanner) -- AI numbers demoted app-wide. Typechecked
          (can't screenshot -- needs a logged-in saved analysis).
      (b) /measured index page ("Precise Analysis / Measured games")
          listing clips w/ measured_stats.json via new GET /api/measured;
          tagline states the hierarchy ("AI read sits on top of these
          numbers, never the other way around"). Screenshot-verified.
      (c) "Measured (CV)" nav link added to analyze + history headers.
      Detail page (/measured/[clip]) is the CV-led analysis: CV facts
      lead, AI scouting read secondary, Run-CV button. Typecheck clean.
      Web files uncommitted on app main (DJ's call -- now ~11 files).
- [~] L3. GROUND THE AI. Split into two:
      (a) TEXT grounding = DONE already (the AI Scouting Read narrates the
          CV numbers under honesty guardrails -- lib/measuredNarrative.ts).
      (b) VISION grounding (Gemini WATCHES the clip while holding the CV
          box score -> tactical story matched to the numbers) = DEFERRED
          by DJ 2026-07-22 into the brand-new-clip demo (L4), since it's a
          paid Gemini VIDEO call and DJ wants to spend it on the real
          demo, not a HARD preview. Build plan is ready (reuse the app's
          uploadAndPoll + fileManager video path; CV exports the analyzed
          span mp4 so Gemini watches the SAME window the box score covers).
          VISION CODE BUILT 2026-07-22 (ready to fire on Test2, DJ said
          "continue to the last part"): CV side export_span.py writes
          {clip}_span.mp4 (the tracking-span window), wired into
          analyze_clip; verified (HARD_span.mp4, 601f/20s). App side
          (branch cv-integration): lib/measuredVision.ts (Gemini VIDEO
          call = GoogleAIFileManager upload+poll then a fileData part,
          grounded prompt: watch the clip but every number traces to the
          CV data, no invented %, they/them), app/api/measured/[clip]/
          vision POST, + a "Watch the clip (deeper)" button beside "Read
          the numbers" in MeasuredStats. Typecheck clean; UI renders
          (screenshot); CV suite 204. NOT RUN yet -- the paid Gemini VIDEO
          generateContent is saved for the Test2 demo (DJ cost pref);
          mirrors the app's own analyzer so low-risk, but that one call
          is the single unverified piece.
- [~] L4. BRAND-NEW-CLIP DEMO (prove-the-flow-first, DJ 2026-07-22).
      CLIP: "C:\Users\djcha\New folder\Throw away repos\Basketball
      Analyer CV System Test\clips\Test2.mp4" -- Fairfield Indians girls
      game, DIFFERENT court/camera/teams. 1920x1080, 30fps, 1442 frames
      (48.1s). Panning follow-cam; standard HS court; big red "F" at
      center + GMC center-circle logo; scorebug bottom-left (mask it).
      DJ DECISION: DJ does the MANUAL calibration himself (NO auto-cal
      until post-ship). So the browser-setup UI is NOT this task.
      HANDOFF: DJ sets up Test2 like HARD/TEST1 (clips_config entry w/
      keyframes+landmarks+scorebug mask+hoop anchors; ClipConfig w/ span
      +roster+hoop_anchors) + runs cache_tracks + cache_oncourt. THEN DJ
      says "Test2 is calibrated" and I: analyze_clip.py TEST2 -> build+run
      the grounded Gemini VISION pass (deferred L3b) -> show in the app.
      (DJ may do calibration in his separate CV chat.)
      Sample frames dumped to scratchpad (test2_f*.jpg) for reference.
      UPDATE 2026-07-22: DJ said "idk how to set it up can you do it for
      me?" -> I'm now doing the Test2 calibration myself (by reading
      frames -- the hard, deferred new-gym problem; expect rough +
      iterative). PROGRESS: pan mapped via a 12-frame montage -- camera
      pans left<->right following the ball; clean LEFT->RIGHT sweep early
      (frames ~30-400). Court = HS (84x50). Scorebug bottom-left (mask
      ~x0-440,y880-1080). Chosen calibration keyframes = [40,140,240,340,
      400] (left->right), extracted full-res to scratchpad (test2_kf*.jpg).
      NEXT (multi-turn): mark court landmarks on each keyframe -> create
      clips_config TEST2 entry + ClipConfig (HS court, span ~40-400,
      hoop anchors, roster) -> run refit + render court overlay -> eyeball
      w/ DJ -> iterate. Then cache_tracks + cache_oncourt -> analyze_clip
      TEST2 -> vision pass -> app. HONEST RISK: JPEG-read landmark
      precision may be poor; if calibration won't converge, that's the
      finding that justifies the browser-setup tool.
      *** CALIBRATION SUCCEEDED 2026-07-23 (DJ marked via the tool). ***
      TEST2 clips_config entry built from DJ's marks (5 kf, left->center
      ->right; 40/140/240 = left basket held, 340 center, 400 right;
      weak-pair 240->340 fast pan but fit held). Added L_arc_top/
      R_arc_top to stage4 COURT_MODEL (apex=HOOP_DX+R3=25ft) + arcs in
      overlay. FIT mean 1.04 ft / max 2.32 (on par w/ HARD ~0.95),
      eyeball-confirmed all 3 views (center circle dead-on the F logo).
      Overlay viewer: https://claude.ai/code/artifact/f7635aa2-2166-4c66-
      a9b6-cae4fb9a21fb . REMAINING for the demo: (1) ClipConfig TEST2 =
      needs ROSTER from DJ + tracking span + HOOP anchors (DJ hasn't
      marked hoops; add to the tool OR skip shots first via
      ball_span_len=0); (2) cache_tracks + cache_oncourt TEST2;
      (3) analyze_clip TEST2; (4) vision pass; (5) app. Suite 204;committed.
      REFINEMENT ROUND 2026-07-23: DJ (rightly) wants the calibration
      RIGHT before proceeding -- overlay showed the 3pt arcs ~1-2ft
      inside (2 issues: (a) MY overlay arc was a stub half-circle, FIXED
      to sweep to baseline in stage4.court_polylines; (b) real residual
      from the fast 240->340 pan / weak SIFT pair). FIX: marking tool v2
      -- dropped redundant left frames 40/140, added 3 BRIDGE frames
      275/300/325 across the pan, PRE-FILLED DJ's existing 240/340/400
      marks, added a Hoops/rims group (L_rim/R_rim). DJ also caught a
      COSMETIC bug: the mini-map 3pt arc was drawn as a stub (apex dot
      floated off it) -- FIXED (basket-centered polyline to baseline).
      DJ re-marking F2-F4 + rims now; then re-run calib + eyeball; only
      THEN roster. Tool: same URL 4bc9fb91-....
      *** ROOT CAUSE FOUND + FIXED 2026-07-24 (DJ pushed back: arc STILL
      wrong after bridge frames). Diagnosed: the arc distortion is NOT the
      marks and NOT a short line (DJ confirmed court is standard). It's the
      calibration METHOD -- the shared H_court AVERAGES all 6 keyframes into
      one court fit + CHAINS them to a reference; on this fast follow-cam
      that drags each frame ~0.6ft off its OWN marks, worst in the arc/FT
      region (19.75 arc bulged ~2ft). Proof: single-frame fit halved the
      error (1.16->0.50ft) and lens-distortion search found nothing.
      FIX (3 files, opt-in, HARD/TEST1 untouched): (1) clips_config TEST2
      flag "direct_keyframe_homography": True + arc marks RESTORED (240 L,
      340 L+R, 400 R -- they were GOOD data; I wrongly removed them);
      (2) stage4.compute_H_court_per_keyframe() fits each kf directly, re-
      expressed in the (H_court,Hs) contract so H_court@Hs[pos]==direct fit
      EXACTLY (0.000000ft), zero downstream change; (3) branch in phase1/
      stage1_court_roi.build_court_anchor on the flag. RESULT via REAL
      pipeline math: mean 0.60ft / max 0.83 (was 1.16/1.89), arcs on the
      paint in all 3 views. Suite 204 pass. Viewer updated (same URL
      f7635aa2-...). Waiting on DJ's eyeball before roster.
      ALSO DONE 2026-07-22: AI reads now AUTOMATIC after a CV run +
      cached ({clip}_ai_read.json), so no button + no re-cost on reload
      (branch cv-integration). DJ asked for auto; done.
      COURT-MARKING TOOL BUILT + PUBLISHED 2026-07-22 (DJ loved it): a
      guided point-and-click Artifact (scratchpad/make_marker.py ->
      court_marker.html) -- 5 Test2 keyframes embedded, magnifier loupe,
      mini court-diagram guide, grouped landmark list incl. TOP-OF-3PT-
      ARC (DJ asked -- HARD's arc was off), outputs pixel coords per
      (frame, tag). URL https://claude.ai/code/artifact/4bc9fb91-6264-
      4b85-83c4-5d54c8064988. DJ GREENLIT this as the REAL clip-config /
      court-homography interface (= the L4 browser-setup, now happening):
      real version = in the web app, frames auto-extracted from any
      uploaded video, a SAVE/Enter button that writes the config (no
      copy-paste), + hoops + roster. SEQUENCING (agreed): FIRST DJ marks
      Test2 in the current tool + pastes back -> I build the clips_config
      TEST2 entry from the tag->pixel coords + add L_arc_top/R_arc_top
      to the engine's landmark palette (court-feet: arc apex =
      (HOOP_DX+R3, 25) each end) -> run refit + overlay -> prove the flow
      end-to-end. THEN build the real integrated Save tool. Prove-then-
      productize (don't build the polished tool around an unproven fit).
- [ ] L5. (parallel CV priority, DJ) MAKE/MISS -> shooting %; and the
      per-player POSITION heatmap (data exists) as a nearer-term add.

NOT yet: Supabase persistence of CV facts (local-file bridge is fine for
local host); auto-calibration (Phase 4). DJ is improving detectors
(ball done, players now, rim next) in a separate chat -- rim helps
auto-hoop (setup friction); a court-line detector later = auto-cal.

# PHASE 6 MINIMAL -- full-game scale, demo-first slice (DONE 2026-07-22 -- both clips verified, see review below)

Ship-handoff item 2. G5 (compute) is ANSWERED by practice: rented GPU per
game (RunPod, memory infra_runpod_gpu.md). The demo can be ONE clip, so
this is the MINIMAL slice per ROADMAP Phase 6: kill the load-span-into-RAM
patterns (the ~6MB/frame bombs that make a full half arithmetic-impossible
in memory) + the batch runner/manifest seam that Phase 7's worker will
call. Nothing else.

MEASURED RAM BOMBS (recon 2026-07-19; everything else already streams):
- phase2/oncourt.py build(): extract_frames() of EVERY span frame into a
  dict -- HARD 601 frames ~3.6GB already; a 57k-frame half = impossible.
- phase2/stage6_ocr_confirm.py read_span_frames(): whole tracking span
  into a dict -- same math.
- phase1/stage2_generate_events.py main(): all sampled event frames at
  once -- fine at today's 31-47 samples, ~17GB at full-game sampling.
Known full-game COST (not RAM, flagged not fixed): hoop_anchor per-frame
SIFT ~0.5-1s/frame -> hours/half on CPU; revisit when a real full game
exists (GPU/keyframe-cache options), not now.

## Plan
- [x] 0. Check in with DJ before building. APPROVED ("yes 100% yes").
- [x] 1. Streaming frame access (one shared helper, three call sites):
      iter_frames(video_path, indices) generator in
      spikes/stage2_multikeyframe.py beside extract_frames() -- yields
      (idx, frame) in index order, single pass, never materializes the
      span. Port oncourt.build(), stage6's read_span_frames() consumers,
      and stage2_generate_events.main() to iterate instead of dict-load.
      PRE-CHECK each consumer's access pattern is monotonic-in-frame
      before porting (stage6 windows are time-ordered; verify in code,
      adapt the loop shape if any consumer needs lookback, never cache
      whole frames). DONE: iter_frames() added beside extract_frames()
      in spikes/stage2_multikeyframe.py. oncourt.build() and
      stage2_generate_events.py port directly (true monotonic streams).
      stage6_ocr_confirm.py needed a real restructure, not a drop-in --
      its imgs[f] access follows a LARGEST-BOX-FIRST pick, not frame
      order, so it now runs two-pass: pick frames from track bboxes
      ONLY (no images), union the picks, THEN targeted-iter_frames just
      those. Net win beyond "streaming": this dict now scales with
      (candidates x MAX_ATTEMPTS), not clip length -- exactly REVIEW.md's
      own suggestion ("fetch on demand per OCR attempt"). NOT ported
      (found, flagged, out of scope): phase2/stage2_recovery.py's
      read_span_frames -- a standalone eyeball/diagnostic overlay tool,
      NOT in run_clip's live path, doesn't block the demo.
- [ ] 2. Regression = BYTE-IDENTICAL outputs: snapshot current artifacts
      (team_events, oncourt cache, ocr_confirms, player_events_merged,
      box_score) for BOTH clips; rerun the ported stages; diff must be
      empty. Any diff = stop and report. Suite (183) green throughout;
      new unit test for iter_frames ordering/completeness vs
      extract_frames on a tiny synthetic video. DONE: baseline snapshot
      taken (md5s recorded) before any edit; 5 new iter_frames tests +
      8 new run_batch tests; suite 183 -> 196 green (2.4s).
      TEST1 VERIFIED 2026-07-22: oncourt.json rebuild showed a REAL
      diff (447/461 frames' diagnostic `inliers` count shifted, e.g.
      6307->6303) -- investigated, not hand-waved. ROOT CAUSE FOUND:
      isolated single-frame re-check (same frame, same code, fresh
      process) reproduced a DIFFERENT inlier count than the full-loop
      run got for that same frame -- proves the wobble is OpenCV
      SIFT's own cross-process nondeterminism (no cv2.setNumThreads(1)
      anywhere in the repo; setRNGSeed(0) fixes RANSAC's sampling but
      not multi-threaded keypoint detection), NOT something iter_frames
      introduced. Confirmed harmless: EVERY frame's actual decision
      (on/off-court bool + court_feet position) was byte-identical
      across all 461 frames; the confidence-state classification
      (ok/low_confidence, gated at 150 inliers/2.0px) never flipped
      either -- worst-case inliers stayed >=965, nowhere near the
      threshold. Full run_clip TEST1 rerun: team_events.json,
      ocr_confirms.json, player_events_merged.json, box_score.json+csv
      ALL byte-for-byte identical to baseline (these stages' diagnostic
      SIFT calls happened not to wobble this run -- consistent with the
      nondeterminism being probabilistic/thread-timing-dependent, not
      code-path-dependent).
      HARD VERIFIED 2026-07-22: SAME story -- oncourt.json rebuild
      (601 tracking-span frames, not the 2746 ball-layer span) showed
      the identical diagnostic-only inliers wobble (0/601 real
      on/court_feet differences, confidence-state never flipped, worst
      inliers 1130 new vs 1163 baseline, both far above the 150
      threshold). Full run_clip HARD rerun: team_events.json,
      ocr_confirms.json, player_events_merged.json, box_score.json+csv
      ALL byte-for-byte identical to baseline. Both clips fully
      verified, zero behavior change from the streaming port.
- [x] 3. run_batch.py (new, top level): takes clip names, runs each as
      .venv/Scripts/python -c "run_clip.run(...)" SUBPROCESS (the
      one-clip-per-process invariant), captures per-clip log + exit
      code, writes batch_manifest.json (clip, git commit, config
      fingerprint summary, stage reuse notes from the log, duration,
      artifact paths, pass/fail). No queue framework -- a loop. This is
      the exact seam Phase 7's worker calls later.
- [x] 4. Verify: run_batch over TEST1+HARD end-to-end DONE 2026-07-22 --
      2/2 PASS, TEST1 331.2s / HARD 318.1s (Phase 5 caches reused via
      fingerprint, so this is the true whole-pipeline cost). manifest
      (batch_manifest.json) recorded git commit + fingerprints +
      artifact paths for both.
- [x] 5. Review section below + commits per CLAUDE.md.

NOT in scope: resumable-stages tier 2 (fingerprinted caches already
give the cheap 80%), distributed anything, hoop-anchor SIFT scaling,
GPU ops automation, Phase 7 worker itself, findings (a)/(b) from the
shot-layer review (empty TEST1 chart backlog item stays visible there).

## Review (Phase 6 minimal, 2026-07-22)
- The 3 measured RAM bombs are fixed: iter_frames() (spikes/
  stage2_multikeyframe.py) streams frames one at a time instead of
  materializing a whole span/sample into one dict. oncourt.build() and
  stage2_generate_events.py port directly (true monotonic streams).
  stage6_ocr_confirm.py needed an actual two-pass restructure (pick
  frames from track bboxes first, THEN targeted-read only those) since
  its access pattern is largest-box-first, not frame order -- this is
  a bigger win than plain streaming: that dict now scales with
  (candidates x MAX_ATTEMPTS), never with clip length, matching
  REVIEW.md's own recommendation. One RAM bomb found or NOT ported
  (out of scope, flagged): phase2/stage2_recovery.py -- a standalone
  eyeball/diagnostic overlay tool, not in run_clip's live path.
- run_batch.py is the new seam: one clip per subprocess (the existing
  invariant), a manifest recording git commit, config fingerprint,
  duration, and artifact paths per clip. Exit code alone doesn't mark
  a pass -- it also requires finding run_clip's own completion line in
  the log (abstention-first: a caught exception that still exits 0
  must not read as success). This is the exact shape Phase 7's worker
  will call per job.
- REAL finding during verification, investigated to root cause (not
  hand-waved): rebuilding the on-court cache with the new streaming
  code produced a DIFFERENT file (different checksum) than the
  pre-change baseline on BOTH clips. Traced by isolating a single frame
  and re-running its SIFT+RANSAC match in a fresh process: the exact
  same frame, same code, gave a different keypoint-match inlier count
  than it got inside the full sequential rebuild. Root cause: OpenCV's
  SIFT keypoint detection has cross-process nondeterminism (no
  cv2.setNumThreads(1) anywhere in the repo; cv2.setRNGSeed(0) fixes
  RANSAC's own sampling but not multi-threaded feature detection
  upstream of it) -- a pre-existing property of the library, unrelated
  to iter_frames. CONFIRMED harmless on both clips: every single
  frame's actual decision (on/off-court bool + court_feet position)
  was byte-identical, and the confidence-state classification
  (ok/low_confidence, gated at 150 inliers) never flipped -- worst-case
  inliers stayed 6-8x above the threshold. The original "byte-identical"
  regression bar was the wrong bar for any stage touching SIFT/RANSAC;
  the real bar (decisions unchanged) was met everywhere it could be
  checked. Downstream of oncourt, every artifact that doesn't carry
  raw SIFT diagnostics (team_events, ocr_confirms,
  player_events_merged, box_score json+csv) came back byte-for-byte
  identical on BOTH clips, both times.
- Suite 164 -> 196 (13 iter_frames/run_batch tests + earlier ball_stages
  tests). run_batch verified end-to-end: 2/2 pass, ~5.3-5.5 min/clip
  with Phase 5 caches reused via fingerprint -- the real per-clip cost
  once a clip's slow stages are already cached.
- NEXT per SHIP HANDOFF: item 3, the Phase 7 worker (jobs table +
  subprocess run_clip + artifacts to storage) -- run_batch.py is
  already most of that seam's shape.

# SHOT-LAYER INTEGRATION INTO run_clip (DONE 2026-07-19 -- gate passed, see review below)

Ship-handoff item 1: the proven spike chain (ball detect -> trajectory ->
hoop anchor -> shot attempts -> shot location -> shot outcome) becomes
real pipeline stages run by run_clip.py, config-driven from ClipConfig.
Plumbing only: REUSE the spike functions (import, don't rewrite), no
threshold tuning, no Phase 6/7 work, nothing writes into team_events
(ROADMAP Principle 4). The verified configuration being integrated is
TEST 10's exact chain: v2 weights detection (conf=0.05, imgsz=1280) ->
conf>=0.10 analysis filter (local_weights_check.CONF_FLOOR, the
TEST 2/8/10 protocol) -> build_chains/classify_chain -> classify_shot
both hoops.

## Plan
- [x] 0. Check in with DJ on this plan before writing code. APPROVED
      ("yes build"), incl. the two flagged judgment calls (new
      {clip}_ball_detections.json artifact name; fingerprint-gated
      reuse for the two slow stages).
- [x] 1. ClipConfig gains ball-layer fields (clip_config.py):
      ball_weights_path (default models/ball_finetuned_v2.pt -- swap to
      v3 later = one line), ball_span_start/ball_span_len (the spans the
      TEST-10 gate ran: HARD 0/2746, TEST1 0/605), hoop_anchors
      {"far": (keyframe, (x,y)), "near": ...} moved from
      spikes/hoop_anchor.RIM_ANCHORS constants -- TEST1 far kf-120
      (582,143) / near kf-580 (1377,233); HARD far kf-1100 (1855,228) /
      near kf-600 (633,190). validate() checks: weights file exists,
      spans sane, both anchor keys present. Tests for the new
      validations first.
- [x] 2. Minimal spike edits so the modules are cleanly importable:
      DONE -- both import-time clobber traps guarded (hoop_anchor
      clips_config.ACTIVE, ball_spike clip_config.ACTIVE_CLIP; verified
      by an import test under ACTIVE='TEST1'); ball_spike.detect()
      extracted, main() calls it with the exact argv values.
      (a) hoop_anchor.py: `_cc.ACTIVE = CLIP_NAME_ARG` (line 51) runs
      unconditionally at IMPORT time and would clobber run_clip's
      clip sync with "HARD" on a TEST1 run -- guard it under __main__
      (standalone CLI behavior unchanged; tests import only pure
      functions, verified). main() reads anchors from ClipConfig
      (RIM_ANCHORS constants retired -- one source of truth).
      (b) ball_spike.py: extract the detection loop into a callable
      detect(...) taking explicit clip/span/model/output paths; the
      CLI main() calls it with today's exact argv behavior (byte-same
      spike usage, now also importable by the pipeline).
      (c) ball_trajectory / shot_attempts / shot_location /
      shot_outcome: NO edits -- their pure functions are already
      importable; the literal-data regression tests in
      tests/test_ball_trajectory.py stay untouched.
- [x] 3. New glue module ball_stages.py (one file, top level beside
      run_clip.py): six stage functions taking a ClipConfig, passing
      EXPLICIT paths between stages (no argv, no module-level state):
        s1 ball detection  -> spikes/out/{clip}_ball_detections.json
                              (+ overlay mp4), via ball_spike.detect()
                              with config.ball_weights_path
        s2 hoop anchor     -> {clip}_hoop_track.json via
                              hoop_anchor.build_hoop_track(config anchors)
        s3 trajectory      -> {clip}_ball_arcs.json: conf>=0.10 filter
                              (CONF_FLOOR imported from
                              local_weights_check) then
                              build_chains/classify_chain -- the exact
                              TEST-10 chain
        s4 shot attempts   -> {clip}_shot_attempts.json: classify_shot
                              both hoops + find_release/identity join
                              (same candidate-pick logic as the spike
                              main; attempts outside the tracks span
                              stay honest no_identity_data)
        s5 shot location   -> {clip}_shot_locations.json + shot chart
                              png via find_shot_location/render_shot_chart
        s6 shot outcome    -> {clip}_shot_outcomes.json via
                              below_rim_fall_evidence/deflection_evidence
      NOTE detection log filename: s1 writes {clip}_ball_detections.json
      (new pipeline artifact) rather than overwriting the spike-era
      {clip}_ball_spike_log.json (stock-model canonical) or the
      suffixed TEST-10 measurement logs -- everything sits beside the
      existing artifacts, nothing verified gets clobbered.
      SLOW-STAGE REUSE (mirrors the tracks-cache pattern): s1 and s2
      (the two multi-hour/SIFT stages) reuse an existing output ONLY on
      an exact fingerprint match (clip, span, model basename, imgsz,
      conf / anchors), loudly printed; any mismatch = rerun and
      overwrite. Fingerprint fields already live in both docs. This is
      reuse-or-rerun, never reuse-approximate.
- [x] 4. run_clip.py: append the six numbered PHASE 5 sections after
      the box score, calling ball_stages in order; integrity report
      gains one line (attempts / located / outcomes counts).
- [x] 5. Tests DONE (19 new in tests/test_ball_stages.py; suite
      164 -> 183 green, 2.4s). The literal-data regression PASSES:
      the integrated chain reproduces TEST 10's exact TEST1 result
      (4 attempts incl. both target layups + shot B missed + both
      rejections) from the saved v2 log. Original item text: new
      ClipConfig validations; conf-floor filtering (literal check that
      a 0.09-conf det is excluded, 0.10 kept); anchors-from-config;
      fingerprint reuse-vs-rerun. Plus one literal-data regression:
      feed the SAVED TEST1 v2 log (spikes/out/
      TEST1_ball_spike_log_ball_finetuned_v2.json) through the s3+s4
      functions and assert TEST 10's exact 4 attempts incl. the 3
      layups -- fast, no video, locks the integrated chain to the
      measured result. Full suite (164) green throughout; the
      test_ball_trajectory literal-data tests are never weakened.
- [x] 6. VERIFICATION GATE PASSED 2026-07-19 -- both clips reproduce
      TEST 10 EXACTLY through the integrated run_clip (evidence:
      spikes/out/verify_TEST1_runclip.log, verify_HARD_runclip.log).
      TEST1: coverage 309/605 (51.1%), 13 arcs, all 4 attempts w/
      identical min_dists, shot B missed, rejections (103,110)/(188,202).
      HARD: coverage 1315/2746 (47.9%), 48 arcs, all 4 attempts w/
      identical min_dists (incl. the known FP 401-415 and the
      unverified 1352-1381 -- correct reproduction), both deflections
      rejected w/ identical distances. CPU-vs-GPU risk RETIRED for v2:
      fresh CPU detection matched the GPU-made TEST-10 log on every
      gate number. Original gate text follows:
      full run_clip.py on HARD and TEST1 must reproduce TEST 10's
      EXACT results from the integrated pipeline:
        HARD: attempts (351-375 near) + (1188-1213 far) present;
        deflections 416-438 and 1216-1250 REJECTED. The known v2
        artifacts must also reproduce exactly -- the DJ-refuted FP
        (401-415 near "layup") and the unverified 1352-1381 candidate
        WILL appear; that is correct reproduction, not something to
        tune away (they are on the Milestone-2/v3 list).
        TEST1: 4/5 verified attempts -- (58-70 extrapolated),
        layups (164-184), (236-250), (581-589); shot B 315-327
        missed (v2's known blind spot, expected).
      ANY deviation = STOP and report; thresholds are never adapted to
      pass. Known risk on the record: TEST 10's HARD log was GPU-made;
      this rerun is CPU. TEST 8 measured GPU-vs-CPU identical for v1;
      if v2 differs on CPU, that is a finding to report, not to patch.
- [x] 7. Review section below + commits per CLAUDE.md.

NOT in scope: Phase 6 scaling, web worker, tracker changes, threshold
tuning, v3 training, make/miss improvements.

## Review (shot-layer integration, 2026-07-19)
- `python run_clip.py` now produces box score + shot attempts + shot
  chart + candidate outcomes for a clip in ONE command -- ship-handoff
  item 1 done. New code is one glue module (ball_stages.py, explicit
  paths, no argv/module state) + config fields + ~20 lines in run_clip;
  every algorithm line is IMPORTED from the already-verified spikes.
- The verified TEST-10 protocol is locked in three ways: CONF_FLOOR
  imported (not copied) from local_weights_check; a literal-data
  regression test reproducing TEST 10's TEST1 numbers from the saved
  v2 log; and the full-pipeline gate runs on both clips (exact match,
  see item 6). Suite 164 -> 183, ~2.5s.
- Two import-time clip-clobber traps found and fixed (ball_spike and
  hoop_anchor each reset the active clip to "HARD" when merely
  imported) -- the same §9b/§13 trap class, now guarded under __main__
  with an import test.
- Slow stages reuse on exact fingerprint only (detections: clip/span/
  model/imgsz/conf; hoop track: anchors + covering span), mirroring
  the tracks-cache refuse-loud pattern. HARD's full-clip hoop track
  was legitimately reused; detection ran fresh on both clips.
- FINDINGS surfaced, deliberately not chased (out of scope):
  (a) TEST1 shot chart is EMPTY -- all 3 release-bearing attempts hit
      the oncourt classifier's off-court abstention at the release
      frame (layup drives near the baseline). Honest, but demo-visible;
      DJ's call on whether to revisit the abstention join later.
  (b) TEST1 236-250 (the MADE putback) got candidate_miss -- a wrong
      candidate label that review would catch; Gate-4 review-first
      stance unchanged.
  (c) HARD 1188-1213 located at the exact DECISIONS-§16 verified spot
      (68.7, 42.3) and outcome candidate_miss matches the verified
      rim-out -- full-chain continuity through the integrated path.
- Next per SHIP HANDOFF: item 2 (minimal Phase 6 scale) or item 3
  (Phase 7 worker); v3 swap-in stays a one-line config change.

# FINE-TUNED MODEL + PLAYER-TRACKER PROBE (current task, 2026-07-14)

HARD CONSTRAINT (user, 2026-07-14): the system is built around EXISTING
infrastructure (Hudl / Veo). The CAMERA IS NOT CHANGING. -> my "better
footage" recommendation (§20/21/22) is OFF THE TABLE for this product;
everything must be solved in SOFTWARE on Hudl/Veo-quality footage. GPU is
purely for SPEED unless a lever 10x's value. Directives: (a) get a
FINE-TUNED ball model in; (b) answer definitively whether a bigger
detector (yolov8x) improves PLAYER tracking, not just the ball.

## Player-tracker question (user asked twice) — MEASURING
- [x] player_detector_probe.py DONE (DECISIONS 23): yolov8x = MODEST win.
      122->106 distinct tracks (~13% fewer fragments = ~13% fewer clicks)
      but IDENTICAL mean lifespan (105.8) -> cleaner detections, NOT
      better occlusion-association. Tracker is the bottleneck, not the
      detector (confirms §11). Worth it once GPU makes it free; not huge.

## Fine-tuned ball model — VALIDATED, BREAKTHROUGH (DECISIONS 24)
- [x] User provided Roboflow key. Model basketball-players-fy4c2 v25
      (classes Ball/Hoop/Player/Ref/scoreboard). roboflow_ball_probe.py
      (key from env, never committed). TEST1 0-605.
- [x] DRAMATIC: shot A traced on ALL 20 frames @0.79-0.89 (stock: ~10
      sparse). Arcs 0-450: stock 6 -> fine-tuned 14; shots LONGER.
- [x] LAYUPS REOPENED: ball now SEEN reaching the rim during layup-1
      (arcs arriving 13/25/30px) -- §22's ball-invisible blocker is GONE.
- [x] Diagnosed the remaining gap: the ORIGIN GATE (§18) rejects layups
      (they release <125px from rim, like deflections). Clean separator
      measured: layups ARRIVE (start 83->end 13; 45->25), deflections
      LEAVE (30->131; 67->167).
- [x] DONE (DECISIONS 25): arriving-vs-leaving gate replaces the §18 hard
      origin gate. A near-rim arc is a shot (layup) only if it ENDS at the
      rim; a deflection ends far. HARD byte-stable (2 shots, deflections
      rejected), TEST1 stock 2. On fine-tuned ball data: 5 shots = 2 jump
      + 3 layups, catching BOTH user-confirmed layup sequences (5.5/8.1s
      + 19.4s). shot_type layup|jumpshot added. Suite 156->158.
- [ ] ADOPTION diligence: verify fine-tuned model on HARD (no ground-
      truth break) before making it the ball-layer default. Cost: hosted
      = ~72k API calls/game -> free tier won't scale; production = paid
      plan OR local weights on GPU (= the concrete GPU justification).
- [ ] BONUS levers this model unlocks (later): Hoop class -> auto rim
      anchor (retire manual clicks); scoreboard -> make/miss + real clock.

## Prior asks this session (context)
Ball detection R2: §20 resolution swept (1280 optimal), §21 model
capacity yolov8x = wash. Layups: §22 measured NEGATIVE (3 signal
failures) -- not built. All pointed at footage, which is now fixed by
constraint -> fine-tuned model + tracking are the remaining SOFTWARE
levers.

## Ask 1 — ball detection, remaining levers (ranked cheap->dear)
- [~] MODEL CAPACITY (running): yolov8x.pt vs yolov8m.pt, held at the
      proven-best imgsz 1280, TEST1 0-450. Bigger model = more capacity
      for BOTH failure modes (small size + motion blur). Compare ARCS
      (not raw coverage). ~30-40 min. DECISIONS 21.
- [ ] If x helps: adopt for the ball layer (measure cost; the identity
      tracker keeps yolov8m -- this is ball-only). If not: next lever.
- [ ] MULTI-RES ENSEMBLE (only if single-model insufficient): detect at
      1024 AND 1280, merge -- §20 showed the two shots have OPPOSITE
      optimal resolutions, so a merge could catch both. 2x compute;
      measure before adopting.
- [ ] Motion-based gap-fill / custom fine-tune: deferred, heavier, only
      if the above stall.

## Ask 2 — layup detection (NEW separate system, measure-first)
Why layups fail the arc tracker: short + occluded ball flight -> no
parabola -> no arc. So key on a DIFFERENT signal than the ball arc.
PRIMARY HYPOTHESIS (to validate, not assume): a layup = a tracked player
drives into the hoop/paint region and their court-path TERMINATES at the
rim (jump/gather), optionally corroborated by raw ball detections near
the rim pixel in that window (which exist even when no arc forms). Reuses
the validated foundation (oncourt court_feet + tracking + carried hoop
pixel) -- no new perception model.
THE HARD PART (the reason to measure first): the paint is crowded --
rebounds, post-ups, cuts, kick-out drives, defense all put players near
the rim. Is a layup SEPARABLE from that traffic, or not? Unknown until
we look at real data.

- [x] User gave 2 TEST1 layups: 3.5-9s (attempt->miss->rebound->putback
      made), 18-20s (made layup).
- [x] MEASUREMENT SPIKE done (read-only, existing data). NEGATIVE on all
      three candidate signals: (1) ball at rim essentially invisible (1
      weak det across the whole miss/rebound/putback; 0 for the made
      layup) -- occlusion+blur+size; (2) player proximity too crowded (10
      tracks within 4ft of the rim); (3) player trajectory fragmented (10
      short 0.2-2.6s track fragments -- ByteTrack shatters in the paint,
      the §10/§11 re-ID limit). Root cause = footage (small, distant,
      crowded, occluded paint at 30fps), same theme as §20/§21.
- [x] VERDICT (DECISIONS 22): do NOT build a layup detector on this
      footage -- it would catch a fraction (0 of the MADE layups) while a
      coach believed it tracked layups = confident-wrong, not built. Real
      paths: FOOTAGE (closer + higher fps fixes ball-at-rim AND
      fragmentation -- the common denominator behind §20/21/22), pose
      estimation (heavy, unmeasured, needs no ball), scoreboard OCR
      (partial make-confirmation, ROADMAP Gate-4 second signal).

## Review (ball R2 + layup measurement, 2026-07-14)
- Both of the user's product-critical asks came back the same way, and
  it's a COHERENT signal not a coincidence: bigger model (§21) and layup
  signals (§22) both failed for lack of information in the footage, not
  lack of algorithm. Every ball-seeing lever this session (resolution,
  model capacity, layup signals) points at the same root: capture quality.
- Nothing broken was shipped: no layup detector built (would abstain/
  false-fire), yolov8x not adopted (wash). The honest deliverable is the
  measurements + a clear, evidence-backed spending recommendation.
- The strongest, cheapest product lever is now unambiguous from data:
  a closer, higher-frame-rate camera. It fixes ball detection, layups,
  AND track fragmentation simultaneously -- more than any GPU or model.

Constraints (unchanged): tests-first, eyeball-gate before trusting,
never write into team_events, ball/layup layers sit BESIDE the spine.

# BALL-SEEING FIX — resolution sweep (DONE 2026-07-14 — DECISIONS §20, 1280 optimal, resolution lever exhausted)
# BALL-SEEING FIX — resolution bump measurement (current task, 2026-07-14)

Root cause CONFIRMED with data, not assumed: ball detection rate tracks
apparent ball SIZE. HARD median ball width 39px -> 66% frames covered;
TEST1 median 24px -> 32% covered. Same root cause as jersey OCR (DECISIONS
4c: camera distance, not contrast). Stock detection also DOWNSCALES the
1920px frame to imgsz=1280 before inference, shrinking the already-small
ball further.

Fix ladder (cheap -> expensive, MEASURE at each rung before climbing):
1. imgsz bump (native 1920+) -- stop throwing away resolution. One
   parameter, ~2x slower. FIRST, being measured now.
2. Tiled/sliced inference (SAHI-style) -- only if #1 falls short.
3. Custom-trained ball detector -- ROADMAP-gated ("only if stock +
   trajectory measurably fails"); DEFERRED until 1+2 exhausted.
NOT on the ladder: lower conf (already 0.05, physics is the gate);
widening trajectory gap tolerance (risks fake arcs in validated code);
zoom/4K (real root fix but only helps FUTURE footage). Caveat: even a
perfect detector won't recover TEST1's layups (too brief at the rim).

- [x] Made imgsz configurable in ball_spike.py (4th CLI arg; non-default
      writes to a suffixed file, never clobbers the validated 1280 log).
      Suite green (156).
- [~] MEASURING (background): `ball_spike.py TEST1 0 450 1920` -- native
      resolution on a span covering both known TEST1 shots (59-71,
      315-327) + general play. ~20-25 min.
- [x] imgsz 1920 measured, frames 0-450: NEGATIVE RESULT. Raw coverage
      rose 34%->44% BUT physics-gated arcs got WORSE -- 1280 formed 6
      arcs incl. both known shots; 1920 formed 5 and LOST BOTH SHOTS.
      Detection detail: on the flights, 1920 fires on FEWER frames (shot
      A 10->3) at HIGHER conf -- motion-blurred ball looks less like a
      compact "sports ball" at high res, so it's rejected; downscaling
      compacts the blur into a ball-like blob. The +10pp coverage is junk
      elsewhere, not better ball tracking. DECISIONS 13 lesson repeats:
      raw coverage is misleading; trusting it would have shipped a
      REGRESSION. Higher = worse; tiling (option 2) now downweighted
      (it's essentially more resolution).
- [x] imgsz 1024 measured. THREE-WAY BRACKET COMPLETE (arcs on 0-450):
      1024 -> 3 arcs (loses shot B), 1280 -> 6 arcs (ONLY one with BOTH
      shots), 1920 -> 5 arcs (loses both). 1280 is the proven optimum,
      bracketed both sides. Bonus finding: the two shots have OPPOSITE
      optimal resolutions (A best at 1024, B needs 1280+), so no single
      setting wins everything -- 1280 is the robust choice.
- [x] VERDICT (DECISIONS 20): 1280 STAYS, resolution lever EXHAUSTED
      (do not re-run, like the reid probe). Tiling abandoned unmeasured
      (it's more effective-resolution = the direction that hurt).
      Ball-seeing is FOOTAGE-limited (ball size + motion blur), not
      tuning-limited. Real lever = zoom/4K for future recordings.
      Layups unrecoverable by any resolution (too brief/occluded).

## Review (ball-seeing sweep, 2026-07-14)
- Clean negative result, honestly measured: the intuitive fix (higher
  resolution) BACKFIRED, and only bracketing both directions on the
  RIGHT metric (physics-gated arcs, not raw coverage) revealed 1280 as
  the genuine optimum rather than the untested middle.
- Raw coverage was actively misleading a THIRD time (§13, §18, now §20):
  1920 had the most coverage and the worst arcs. The discipline of never
  trusting raw detection count is the load-bearing habit here.
- No code adopted (1280 was already the default); the only lasting code
  change is the now-configurable imgsz, useful for any future sweep.
  Measurement logs kept as evidence; throwaway overlay mp4s removed.

# PHASE 5 ON TEST1 — second clip, real Gate-4 sample (DONE 2026-07-14 — DECISIONS §19)

Goal: run the same Phase 5 pipeline (already built + validated on HARD)
on TEST1.mp4 -- a second real game clip, already calibrated, tracked,
and identity-resolved, that Phase 5 has never touched. This is the
cheapest path to an actual Gate-4 sample: no new footage needed, just
background compute, reusing code already proven on HARD.

User feedback logged (memory: feedback_ship_speed_vs_working_product.md):
"ship ASAP" defers polish/scaling, never correctness -- a broken
pipeline isn't a faster MVP, it's no MVP. Keep applying every eyeball
gate and regression-test habit from the HARD sessions; don't skip them
to move faster.

## Plan
- [x] Generalized ball_spike.py / hoop_anchor.py / shot_attempts.py /
      shot_outcome.py / shot_location.py to accept an optional clip name
      (default HARD, zero behavior change). Fixed a real bug in the same
      pass: reading sys.argv at hoop_anchor.py's module level broke
      pytest's own import of it (pytest's argv misread as a clip name) --
      guarded by __main__. Suite green (151) throughout.
- [x] Marked TEST1's two hoops, user-confirmed after fine-tuning: far at
      keyframe-120 px (582,143), near at keyframe-580 px (1377,233).
- [x] Both full-clip runs landed clean: 472 raw detections (31.9% of
      frames — HALF of HARD's rate, harder footage for the ball
      detector), hoop coverage complementary near-full-clip, zero
      out-of-bounds, 13 candidate arcs.
- [x] FIRST RESULT: 0 shot attempts vs user's ~4 real ones. User asked
      me to re-examine my restrictions. DIAGNOSIS: the origin gate
      (their suspect) rejected ZERO TEST1 arcs; the REAL bug was the
      at/after-apex rule breaking on truncated ascent-only arcs (315-327
      approached 101px but the point was "pre-apex" only because
      truncation made the last point the apex).
- [x] REDESIGN (DECISIONS §19), measured on all 43 arcs both clips
      (shots <=110px vs non-shots >=163px, clean gap): all observed
      points count; HOOP_RADIUS_PX 100->125 (origin gate gets STRICTER);
      bounded forward extension of the fit (<=15 frames, descending
      only, stamped observed|extrapolated). 5 new tests, suite 156.
      HARD ground truth byte-stable (same 2 shots).
- [x] TEST1: 2 shots. 315-327 = FIRST FULL-CHAIN ATTRIBUTION: jersey
      #14 (user-confirmed identity), release extrapolation 0.0px on her
      bbox, located (-0.6, 21.0) ct-ft (baseline ~7ft; x within ~1ft
      calibration error, flagged). Layups honestly unrecoverable at
      detector level (raw at-rim dets exist, flights too sparse to ever
      chain — footage-quality lever, not a gate problem).
- [~] User eyeball pass on TEST1's 2 claimed shots (overlay/chart)
      pending.
- [x] DECISIONS §19 written; Gate-4 tally = 4 genuine attempts / 2 clips
      (1 verified outcome, 2 unknown, 1 unverified) — still not a rate.

## Review (TEST1 + apex-rule fix, 2026-07-14)
- The user's challenge was RIGHT that the pipeline was over-restricting
  and WRONG about which restriction — and the difference matters: the
  origin gate they suspected is regression-tested against real
  deflections and rejected nothing here; the apex rule I'd designed in
  §15 was the actual defect, invisible until sparser footage produced
  truncated arcs. Diagnosis-before-change prevented removing the wrong
  gate.
- Every change is justified by the measured 110/163 separation across
  all 43 arcs on both clips, not by chasing a marginal case; HARD's
  verified ground truth stayed byte-stable as the regression bar.
- The first full-chain shot attribution (#14) exists because every
  layer underneath (identity confirmation, oncourt court_feet, release
  extrapolation) was already individually validated — the "foundation
  first" bet paying off visibly for the first time.

# PHASE 5 GATE-4 HARVEST — full-clip HARD (DONE 2026-07-14 — DECISIONS §18, n=2 clean genuine shots, still unmeasurable)
# PHASE 5 GATE-4 HARVEST — full-clip ball detection (current task, 2026-07-13)

User chose option 1 (harvest more shots) after step 5's ground truth
passed at n=1. Scoping caveat given + accepted: hoop calibration only
covers frames 600-1200 (20-40s of the 91.5s clip) -- shots outside that
window will get raw ball data + physics arcs, but NO hoop position (no
shot-attempt classification possible there) unless calibration is later
extended. User chose full 91.5s clip anyway (raw ball data has value on
its own, e.g. flicker/false-positive stats per DECISIONS 13's spirit).

- [x] Backed up the verified 12s-span artifacts (HARD_ball_spike_log,
      HARD_hoop_track, HARD_ball_arcs, HARD_shot_attempts,
      HARD_shot_outcomes, HARD_shot_locations) with a
      `.backup-2026-07-13-pre-fullclip.json` suffix before overwriting.
- [x] Added optional CLI span override to ball_spike.py / hoop_anchor.py
      (default unchanged = today's verified 1020/360 span). Suite green.
- [x] hoop_anchor.py 0 2746 FIRST run exposed a real bug (not just the
      expected coverage gap): 1089/2746 frames (matched to keyframes
      600-1000, outside the previously-validated 1100/1200 range)
      produced geometrically absurd hoop positions (e.g. x=42625 in a
      1920px frame) -- a SIFT match clearing MIN_INLIERS on OTHER visible
      features doesn't guarantee the homography is well-conditioned for
      extrapolating all the way to the rim. NOT a sign bug (verified: an
      overall matrix sign flip cannot change a projective-divide output).
      FIXED: in_plausible_bounds() rejects wildly-out-of-frame results,
      treated as an honest no-match. 5 tests, suite 140->145. Re-running
      (bttbbxljy) with the fix.
- [x] Both background runs completed clean. ball_spike: 3102 raw
      detections/2746 frames (65.5% w/ >=1 det). hoop_anchor (fixed):
      1702/2746 plausible (0 remaining out-of-bounds), covering ~33s-91s
      continuously (camera framing stayed close to keyframe 1200's view
      for most of the 2nd half) -- BETTER than the expected 600-1200-only
      window. ball_trajectory: 30 candidate arcs across the full clip
      (up from 2 in the original 12s slice).
- [x] shot_attempts.py + shot_outcome.py rerun over the full-clip data.
      RESULT: still only the SAME 2 shot attempts (1188-1211, 1217-1250)
      -- the harvest did NOT expand the sample. REAL FINDING: several of
      the 28 non-qualifying arcs DO have a hoop position at their check
      frame but sit 295-1043px away (min_dist), far past the 100px gate
      -- the signature of shots at the OTHER basket, which was never
      anchored (hoop_anchor only tracks the one hoop marked in step 3).
      Not a bug -- correct abstention -- but it caps the real sample at
      n=2 regardless of how much more footage gets processed, until a
      second hoop anchor is added.
- [x] User chose: add a second hoop anchor. Marked near hoop at
      keyframe-600 px (633,190), 3 rounds of fine-tuning, user-confirmed.
      hoop_anchor.py generalized to carry BOTH anchors via the same
      per-frame match (no extra SIFT cost). Coverage: near 0-40s, far
      ~33-91s -- complementary, spans nearly the whole clip. Zero
      out-of-bounds for either. Suite unaffected (145 green).
- [x] Re-ran shot_attempts.py/shot_outcome.py: sample DOUBLED to 4 (2 new
      near-hoop candidates, ~12.7s and ~14.6s).
- [x] USER EYEBALL on the 2 new candidates, two real findings:
      (1) 356-381 real shot BUT camera panned mid-arc, distorting the
      fit (~300px hoop-position drift from camera motion alone) -- first
      CONFIRMED instance of DECISIONS 14's hypothetical pan-model gap.
      Impact stayed contained (still classified correctly; outcome
      correctly abstained) -- logged as KNOWN DEBT, not fixed (bigger
      than today's timebox).
      (2) 418-438 user-identified as "a shot falling down after a shot"
      -- the SAME double-count pattern from section 15, now a second
      independent instance. FIXED: classify_shot gained an ORIGIN GATE
      (arc must start >HOOP_RADIUS_PX from the hoop, else rejected as a
      continuation) -- validated as a clean, consistent split across all
      4 real arcs found this session. 6 tests incl. 4 literal-data
      regressions; suite 145->151.
- [x] Re-ran the classifier with the origin-gate fix: sample settles at
      2 -- but a CLEANER 2 than before (two independent genuine shots at
      two different baskets, not one shot plus its own duplicate).
      DECISIONS §18 written with the full story.

## Review (Gate-4 harvest, 2026-07-14)
- Two real bugs found and root-fixed (implausible hoop extrapolation;
  arc double-counting), one real limitation found and logged, not fixed
  (camera pan distorting the trajectory fit).
- Every fix is backed by a literal-data regression test built from the
  exact real chains that exposed the bug -- not just synthetic cases.
- Gate 4 stays UNMEASURABLE: n=2 genuine shots is higher QUALITY than
  before (no duplicate) but still not an accuracy rate. The actual
  harvest yield was bugs found and fixed, not sample size grown -- this
  clip has few enough clean, resolvable shots that more processing of
  THIS footage won't produce a real rate. Next honest step is more
  games, not more compute on this one.
- Phase 5 stopping point: attempts + locations + reviewed (never
  auto-trusted) outcomes, exactly as ROADMAP anticipated for the
  low-sample case -- except here it's "not yet measured" rather than
  "measured low."
- [ ] DECISIONS §18 with the full harvest result + the single-hoop-anchor
      scoping finding.
      whether Gate 4 is now measurable.
- [ ] DECISIONS §18 with the harvest result + the plausibility-bug finding.

# PHASE 5 STEP 5 — MAKE/MISS, TIMEBOXED (DONE 2026-07-13 — DECISIONS §17, Gate 4 unmeasurable at n=1)

Goal (ROADMAP step 5 + GATE 4): simple visual outcome discriminators whose
outputs are CANDIDATE labels feeding review — candidate_make /
candidate_miss / unknown — never a bare made/missed stat. Gate 4: if
automatic outcome accuracy on eyeballed samples is honestly <~85%, ship
attempts + locations + reviewed-outcomes and move on. Do NOT chase
make/miss for weeks.

HONESTY UP FRONT: we have exactly ONE verified shot (the ~40s rim-out
MISS). n=1 cannot measure an accuracy rate, so Gate 4 CANNOT be evaluated
this session no matter what gets built. This step therefore: (1) builds
the discriminator small, (2) verifies it gets the one known shot right
(miss), (3) reports the gate as UNMEASURABLE at n=1 and hands the user
the decision on harvesting more shots (e.g. a full-clip HARD ball-
detection run, ~90min background, would surface every shot in the 91s
clip for a real sample).

Design (geometric, abstention-first, both signals from data we already
have — raw detections + claimed arcs + carried hoop positions):
- MAKE evidence: after the shot's hoop-arrival frame, raw detections in
  the BELOW-RIM CORRIDOR (narrow horizontal window around the hoop x,
  y just below the hoop) with y increasing — the ball falling through
  the net. Uses RAW detections (not arcs) because a dropping ball over
  a few frames may not earn a full physics claim.
- MISS evidence: a subsequent chain/arc STARTING near the hoop and
  exiting laterally/upward — a deflection (exactly the section-15
  rim-out pattern, now used as signal instead of noise).
- BOTH or NEITHER -> unknown (abstain; conflict never resolved by guess).
- Output: outcome + evidence counts stamped on each shot attempt; ALL
  outcomes are review items by design (candidate labels, per ROADMAP).

## Plan
- [x] 0: check in with user on this plan.
- [x] A: tests first (12 synthetic), then spikes/shot_outcome.py. Suite
      128->140.
- [x] B: run on HARD — GROUND TRUTH PASSED: the known shot (1188-1211)
      classified candidate_miss, matching the user-verified rim-out.
      Evidence traced to the EXACT deflection chain from DECISIONS §15
      (starts 26.2px from the hoop at f=1217, ends 344.2px away at
      f=1257) -- the same signal that was a false-positive "shot attempt"
      in step 3 is a correct outcome signal here. Zero make evidence
      found (correct -- ball never fell through the below-rim corridor).
- [ ] C: DECISIONS §17: result + explicit Gate-4 statement (n=1-2,
      unmeasurable) + harvest-more-shots decision handed to user.
- Commit after tests+code green (done), again after DECISIONS write-up.

NOT in scope: scoreboard-OCR outcome second-signal (later project, per
ROADMAP); rebounds/assists/steals; possessions (step 6); any non-review
outcome stat; writing into team_events or any existing artifact.

# PHASE 5 STEP 4 — SHOT LOCATION (DONE 2026-07-13 — DECISIONS §16, gate to step 5 PASSED)

Goal: put the user-verified shot on a court diagram — shooter position in
court feet at release, rendered as a shot-chart dot, with review status
propagated (an unconfirmed shooter's dot is a REVIEW dot, never presented
as attributed).

OPENING FINDING (diagnosed before building, spikes/out/
HARD_shooter_diag_1188.jpg): step 3's shooter HINT is wrong on the real
shot. Track 3317 (recorded hint) is a bystander on the far baseline;
the real shooter is the white/red player in follow-through at the left
of the paint. Cause: §14 release-blindness compounding — the arc claim
starts near APEX, so "nearest body to arc start" picks whoever stands
under the apex, not who released 5-8 frames earlier. Safety held (it was
a review_item, never auto-credited) but the hint would misdirect the
human reviewer. Location depends on the shooter, so this gets fixed
FIRST, with a gate, as part of this step.

Design:
- RELEASE BACK-EXTRAPOLATION (bounded + gated, tests first): the claimed
  arc carries its fitted quadratic. Extend it BACKWARD a bounded number
  of frames (<= ~10); at each backward frame, measure distance from the
  extrapolated ball position to each tracked body (distance to the bbox,
  since release happens at hands, not feet). Best (frame, track) under a
  distance gate = shooter hint; nothing under the gate = honest
  no_confident_shooter review item. Extrapolation is a CLAIM EXTENSION:
  it only ever produces a review HINT + a release-frame estimate, never
  an auto-attribution (unconfirmed shooter stays a review item, same as
  today).
- SHOT LOCATION = the hinted shooter track's FEET court position at the
  estimated release frame, read from the oncourt cache (court_feet is
  already stored per track per frame — free join, zero new geometry).
  NOTE, deviation from ROADMAP wording: "arc origin -> court feet" would
  floor-project an ELEVATED point through the floor homography (wrong —
  same reason the hoop needed its own anchor); the shooter's feet are
  the honest ground point at release. No shooter hint -> no location ->
  the attempt surfaces as a location-unknown review item.
- SHOT CHART: court diagram (reuse stage4_courtmap.court_polylines +
  stage6's 3pt geometry, drawn in court-feet space, no homography needed
  for a flat diagram) with the dot + status label. User eyeballs the dot
  against where the shooter actually stood on film.

## Plan
- [x] 0: check in with user on this plan.
- [x] A: tests first (6 synthetic), then extended spikes/shot_attempts.py
      with find_release() (bounded backward extrapolation of the arc's own
      fit, gated by bbox distance). Suite 115->121.
- [x] B: rerun on HARD; NEW HINT: track 1502, release_frame=1178 (39.27s),
      still review_item/candidate (correctly not auto-attributed). USER
      CONFIRMED correct ("Yes it is", noted she's slightly airborne but
      shouldn't matter much). Fix verified: wrong bystander hint (3317)
      -> correct shooter (1502).
- [x] C: spikes/shot_location.py — oncourt join (court_feet at release
      frame 1178, track 1502) + flat court-diagram shot chart. Located:
      (68.7, 42.3) ft -- ~20.0ft from the right hoop center (78.75,25),
      right at the 3pt line. Second arc (rim deflection) correctly
      location_unknown (no shooter -> no location, not guessed). 6 tests,
      suite 121->127.
- [x] D: FIRST render was MIRRORED top-to-bottom (user eyeball caught it
      immediately: "it is mirrored, should be on the other side of the
      arc"). Root cause: phase1/stage3_heatmap.py (already validated)
      draws with matplotlib origin='lower' (near-sideline y=0 at BOTTOM);
      my new cv2 chart plotted court_feet y straight into image rows
      (y=0 at TOP) — a render-only bug, the underlying court_feet data
      was never wrong. Fixed with one flip helper + a regression test
      (near-sideline point must render in the bottom half). Suite 127->128.
      Re-rendered, USER CONFIRMED correct. DECISIONS §16 next.
- Commit after tests+code green (done x2), again after DECISIONS write-up.

NOT in scope: make/miss (step 5); possessions (step 6); auto-attribution
of any shooter (review-only until identity is CONFIRMED); writing into
team_events or any existing artifact.

# PHASE 5 STEP 3 — SHOT ATTEMPTS (DONE 2026-07-13 — DECISIONS §15, gate to step 4 PASSED)

Goal: pick the SHOTS out of the step-2 arc claims. ROADMAP rule: a shot
attempt = an arc terminating at the HOOP REGION; shooter = nearest identity
at release, stamped with identity_state (unconfirmed shooter = review item,
never guessed). Dribbles/passes are correctly-claimed flight that must NOT
become shots.

Hoop pixel problem + chosen design: every calibration landmark is a FLOOR
point — a floor homography cannot give the ELEVATED rim pixel directly.
Chosen: ONE-CLICK RIM ANCHOR — mark the rim once in a keyframe still (user
confirms a marked image; click-seeding philosophy, one trusted human input
per clip per visible basket), then carry it to every frame through the
EXISTING frame->keyframe SIFT homographies. Valid because the camera pans
in place (rotation-only => the homography holds for the whole scene incl.
elevated points — the same assumption the calibration spine already makes).
Eyeball gate before use: render the carried hoop region on the arc overlay.

Data reality (stated up front, not discovered later): HARD tracks cache
covers frames 600-1200 only; the ball span runs to 1380. Any arc after
frame 1200 gets an honest "no identity data" review item for its shooter —
abstention, not a guess. Also §14's release-point blindness: claims start
near apex, so "release" = first claimed point, an approximation logged in
the output, not hidden.

## Plan
- [x] 0: check in with user on this plan.
- [x] A: hoop anchor DONE + user eyeball PASSED ("glued to the hoop").
      Marked rim at keyframe-1100 px (1855,228), user-confirmed still.
      spikes/hoop_anchor.py carries it via Hs_opt @ Hfk (rotation-only
      camera => valid for elevated points). Hit + fixed the SAME
      clip-selector trap as ball_spike.py (spikes/clips_config.ACTIVE
      binds stage1/2/4/5 AT IMPORT — must set BEFORE importing, not just
      clip_config.ACTIVE_CLIP). Carrying run: 360/360 frames matched (100%,
      zero abstentions); user verified against stills incl. frames matched
      to a DIFFERENT keyframe (1200) than the anchor (1100) — confirms the
      carry math, not just the anchor point. 6 math tests, suite 100->106.
- [x] B: tests first (9 synthetic), then spikes/shot_attempts.py — arc is
      a SHOT ATTEMPT iff it passes within HOOP_RADIUS_PX (100) of the
      carried hoop position, at or after its apex. Suite 106->115.
- [x] C: shooter at release — nearest tracked body (feet pixel) to the
      arc's first claimed point, joined to identity_state from merged
      player events; no data (outside tracks span / untracked) = honest
      review item. Writes {clip}_shot_attempts.json + annotated overlay
      (hoop circle + red=shot/green=arc/gray=no-claim curves).
- [x] D: run on HARD — GROUND TRUTH PASSED: the user-verified ~40s shot
      claimed correctly (1188-1211, min_dist 54.6px, shooter=review_item
      track 16/candidate — correctly NOT auto-attributed, an unconfirmed
      identity stays a review item exactly as designed). Dribbles/passes
      correctly NOT shots (6/8 arcs). REAL FINDING, not a bug: a SECOND
      arc (1217-1250) also passed the hoop-proximity gate — the rim-out
      deflection of the SAME shot (ball clips the rim, changes velocity,
      the trajectory layer correctly starts a new physics segment per
      DECISIONS 14, but "new arc" != "new shot attempt"). User eyeballed
      the overlay and confirmed: one shot, rim-out miss, ball falls to
      the floor by 41.7s — NOT a second attempt. Logged as a real,
      un-fixed limitation in DECISIONS 15 (arc identity vs shot identity),
      not silently patched. Verdict: PASS with a known false-positive
      class carried forward, same honesty as every prior step.
- Commit after tests+module green (done), again after the measured run
  + DECISIONS write-up (next).

NOT in scope: make/miss (step 5, timeboxed later); shot location / court
feet (step 4); possessions feedback (step 6); writing into team_events or
ANY existing artifact (ball layer stays beside the spine, forever).

# PHASE 5 STEP 2 — TRAJECTORY LAYER (DONE 2026-07-13 — DECISIONS §14, gate to step 3 PASSED)

Goal: turn the spike's raw detections into honest BALL-IN-FLIGHT claims.
Input = the existing spike log (spikes/out/HARD_ball_spike_log.json, ALL
759 low-conf detections — DECISIONS §13: confidence cannot gate, physics
must). Output = arc segments the system is willing to claim, everything
else = no claim (abstention, same as identity).

Design (driven by the spike's two measured facts):
(1) real ball moves smoothly ~12px/frame; glare junk is positionally
    static -> CHAINING by position separates them where confidence can't;
(2) the ball appears in 5-25 frame streaks with small gaps -> SHORT
    parabolic fits with bounded gap tolerance, never long extrapolation.

Pipeline (one new file, spikes/ball_trajectory.py, reads the log JSON,
touches NOTHING else — no video re-detection needed this session):
  a. CHAIN: greedy frame-to-frame association on box centers; gate =
     max displacement/frame (~40px, 3x the measured ~12); tolerate gaps
     <= 3 frames (linear-predicted position must still gate).
  b. DE-JUNK: drop chains whose total travel is tiny (static glare) or
     shorter than 6 frames (too little evidence for any physics claim).
  c. FIT: quadratic cy(frame) + linear cx(frame) per chain (short
     segments, camera pan stays small at this scale — v1 does NOT model
     pan; if residuals prove otherwise, that's a finding, not a hack).
     Physics gate: downward accel (image +y) within a plausible band +
     residual ceiling. Pass = ARC (a ball-flight claim). Fail = chain
     stays visible in output as no-claim (honest, reviewable).
  d. MEASURE against known ground truth from step 1: the user-verified
     shot arc (frames ~1188-1211) MUST come out as an ARC; the glare
     chains MUST NOT; report every claimed arc for user eyeball via an
     overlay video (arcs drawn as curves, no-claim chains dim).

## Plan
- [x] 0: checked in; user approved ("build!").
- [x] 1: tests first — 12 synthetic tests, all passed on first module run.
- [x] 2: spikes/ball_trajectory.py built (one import-path fix for the
      overlay render: phase2 on sys.path for run_tracking).
- [x] 3: run on HARD log + user eyeball, TWO iterations:
      run 1: shot arc claimed exactly (1188-1211) BUT 6 false arcs =
      camera-pan glare drift (horizontal, accel 0.10-0.15 at band edge);
      user confirmed glare -> ACCEL_Y_MIN 0.1->0.3.
      run 2: one 8-frame glare slice squeaked in at accel 0.309 with only
      11px vertical travel -> MIN_Y_RANGE_PX=25 (real arcs span 48-351px)
      + literal-data regression test. Final: 8 arcs / 7 chains, all real,
      zero glare. Suite 100 green.
- [x] 4: logged DECISIONS.md §14. Gate to step 3: PASSED — arcs are
      trustworthy (all claims real, abstention working incl. the claim
      stopping at a floor bounce).
- Commit after tests+module land green (174280f), again after the
  measured run (this commit).

NOT in scope: shot attempts, hoop region, shooter attribution (step 3);
make/miss (step 5); feeding possessions (step 6); custom detectors;
writing anything into team_events (ball layer is a NEW layer, forever).

## Review (trajectory layer, 2026-07-13)
- Two new files (spikes/ball_trajectory.py, tests/test_ball_trajectory.py),
  one import-path line touched in nothing else. Suite 87 -> 100.
- Both false-claim classes were killed by MEASURED gates (accel band floor,
  min vertical travel), each justified by data + user eyeball, each locked
  in by a test — no hand-tuning until it looked right.
- Known gaps carried forward honestly (DECISIONS §14): release-point
  blindness (claims start near apex; matters for shot location in step 4),
  dribbles/passes correctly claimed as flight (step 3 must select shots by
  hoop-terminating arcs), no pan model in v1.
- Next: Phase 5 step 3 — shot attempts (upward arc terminating at hoop
  region; hoop pixel from existing court homography; shooter = nearest
  identity at release stamped with identity_state).

# PHASE 5 STEP 1 — BALL SPIKE (DONE 2026-07-13, verdict GO — DECISIONS §13)

Goal: answer ONE question before any ball-tracking code gets built — can a
stock YOLOv8m (COCO "sports ball" class) see the ball on this footage often
enough, and cleanly enough, to be worth building a trajectory layer on top
of? Per ROADMAP.md Phase 5 step 1: measure BEFORE building. No ball code
exists in the repo today; the old "flickery but arc-detectable" note is a
hypothesis to re-verify, not a result.

Scope: detection only, this session. NO tracking-across-frames, NO
trajectory/arc fitting, NO shot-attempt logic — those are steps 2-3 and only
happen if this spike's numbers justify it (per-ROADMAP decision gate).

## Plan
- [x] 0: CHECK IN WITH USER — user confirmed HARD.mp4, ~35-45s has a shot
      attempt (30fps -> frames 1050-1350; used span 1020-1380 with buffer).
- [x] 1: `spikes/ball_spike.py` — new, throwaway-probe style (same pattern as
      `spikes/reid_fragment_probe.py`, DECISIONS.md §11). Extracts the chosen
      frame span (reuse `run_tracking.extract_subclip`), runs YOLOv8m
      (`yolov8m.pt`, already in repo root — same model already used for
      person detection) filtered to COCO class 32 "sports ball", LOW
      confidence threshold (e.g. conf=0.05, far below the person-detector's
      implicit default) + imgsz=1280 (same as the validated person config).
      RAW per-frame detections only — no tracker, no persistence, per
      ROADMAP step 1 ("plot raw detections" before building anything else).
- [x] 2: draw every raw detection box + confidence on each frame, write an
      overlay .mp4 to `spikes/out/` (matches `reid_fragment_probe.py`'s
      convention), plus a per-frame JSON log (frame_index, N detections,
      confidences, boxes) so flicker can be measured from data, not just eyes.
- [x] 3: ran in background, completed clean (360/360 frames). One hygiene
      fix after the run: ball_spike.py now sets clip_config.ACTIVE_CLIP
      BEFORE importing run_tracking (the temp subclip had been named
      TEST1_span_* — the DATA was correct, video path is passed explicitly;
      only the temp filename was mislabeled — same naming-trap class as
      DECISIONS §9b, fixed at the source).
- [x] 4: MEASURED + user eyeballed frame-by-frame (2026-07-13). User
      CORRECTED my first read: on the shot arc the boxes are glued to the
      ball but conf NEVER crosses 0.5. Log confirms — the arc (frames
      1188-1211, 39.6-40.4s) is a textbook parabola at conf 0.05-0.33.
      Conf tracks apparent size/background, not ball-ness; glare junk
      shares the low band but is positionally static while the real ball
      moves smoothly. 759 raw dets / 360 frames total. Stills:
      spikes/out/HARD_ball_still_*.jpg, overlay: ..._ball_spike_overlay.mp4.
- [x] 5: logged as DECISIONS.md §13 — VERDICT: GO for Phase 5 step 2
      (trajectory layer), with the measured design constraint: confidence
      CANNOT gate ball claims (any threshold that kills glare also kills
      the rising shot arc); physics-consistency must be the confidence.
      Stock detector positions are sufficient; no custom ball detector
      (per ROADMAP's "do NOT build yet" list).

Constraints (per CLAUDE.md + project architecture): smallest possible
change — one new throwaway script, zero edits to any existing file, zero
edits to team_events/box_score/identity code. Commit after the spike script
+ measurement are done (nothing to commit before — it's a new, isolated file).

## Review (ball spike, 2026-07-13)
- One new file: `spikes/ball_spike.py`. Zero existing files edited; the
  identity pipeline, caches, and test suite were never touched.
- Result: POSITIVE with a user-caught correction. The detector's POSITIONS
  are arc-fit-ready (boxes glued to the ball through a full parabolic shot
  arc, near-continuous ~24 frames) but its CONFIDENCES are not a usable
  gate — the entire rising arc sits at 0.05-0.33, below any threshold
  that would remove floor-glare junk. Physics consistency, not detector
  confidence, must make ball claims (ROADMAP step 2's design, now measured
  as a necessity rather than assumed).
- Full numbers + the ACTIVE_CLIP-import-order trap in DECISIONS.md §13.
- Next: Phase 5 step 2 — trajectory layer (parabolic segments over ALL
  low-conf detections, physics-consistency as confidence, abstention in gaps).
# COLOR TIEBREAK (current task, 2026-07-13)

Goal: HARD's two AMBIGUOUS box-score lines (#3, #23 — both numbers appear on
Milford white/red AND Winton Woods black/green) split into correct per-team
lines using jersey color, WITHOUT ever guessing when the color evidence is
unclear. Scope confirmed: TEST1 has no roster overlap, so this is HARD-only.
Credited-identity scope is small: #3 has identities 4/7/137(x3 tracks)/10
(window-local numbering), #23 has 2 identities (one per window) — a handful
of classifications, not a heavy new pipeline stage.

DESIGN (reuses validated pieces, no new human input required):
- TEAM COLOR CENTROIDS built automatically from crops the system ALREADY
  trusts: every CONFIRMED frame whose number is UNAMBIGUOUS (on exactly one
  roster — e.g. HARD's #24, #10, #44, #20, #1, #0, #13) is free labeled
  training data for what that team's jersey looks like on THIS footage, same
  camera/lighting/court. No hardcoded RGB guesses, no new config.
- Per ambiguous-number IDENTITY (not per frame — one real player, less
  noise): sample a handful of its claimed frames, compute a color signature
  per crop reusing ocr_reader's torso-region logic, classify against the two
  centroids with a MARGIN. If the two teams' distances aren't clearly
  separated -> ABSTAIN, stays in the current AMBIGUOUS bucket. Same
  abstention principle as everywhere else in this codebase — never guess.
- Box score gains, for each number that resolves: separate lines per team
  ("#3 Milford", "#3 Winton Woods") instead of one blended AMBIGUOUS line.
  Any identity that can't be classified stays in a residual AMBIGUOUS line —
  visible, never dropped, never silently assigned.
- Color tiebreak does NOT touch the disputed-frames mechanism (two identities
  claiming the SAME frame simultaneously stays a separate contradiction,
  unaffected by this work) — different problem, different existing detector.

TODO:
- [x] C1. phase2/color_tiebreak.py: crop_color_signature(), build_team_
      centroids(), classify_team() with margin-based abstention. Pure
      functions, no I/O.
- [x] C2. 10 unit tests written first (synthetic solid-color crops incl. a
      50/50-blend abstention case + a tie-vote abstention case). All pass.
- [x] C3. Wired into stage8_box_score.py (_identity_occurrences,
      _resolve_ambiguous_teams, optional identity_team override on
      build_box_score). 4 new box-score tests (13 total in that file).
      Suite 73 -> 87, all green.
- [x] C3a. BUG CAUGHT + FIXED before shipping: first cut collapsed one
      identity to one number (last-write-wins), silently skipping a
      claim-group when an identity carried 2 numbers across separately
      merge-stamped spans. Fixed: keyed by (window, identity, number)
      triple. Caught by eyeballing real output, not by unit tests alone --
      it was a data-shape bug in the I/O wiring, not the pure classifier.
- [x] C4. Full HARD run: 6/6 ambiguous claim-groups resolved (0 abstained).
      #3 -> one clean "Milford 6.9s" line; #23 -> "Milford 4.4s" +
      "Winton Woods 2.7s". Disputed accounting unchanged (2.7s still
      surfaces on its own AMBIGUOUS row, never team-attributed).
- [x] C5. Eyeballed 4 of 6 resolved crops against real footage (incl. the
      largest-credit identity): every classification correct, no
      misattributions. See DECISIONS section 12.
- [x] C6. TEST1 regression: box_score.json + .csv BYTE-IDENTICAL before/
      after (confirmed via diff); zero video reads triggered -- zero-cost
      path proven, not just claimed.
- [x] C7. DECISIONS section 12 recorded, incl. TWO findings surfaced (not
      silently fixed): (a) disputed dual-team frames COULD also be color-
      resolved but aren't yet -- real WW #3 credit (2.7s) sitting at zero,
      flagged as the natural next unit; (b) a genuine Part-1-vs-Part-2
      label contradiction on HARD track 2475 (#3 vs #13, both Milford so
      team-safe either way, but the specific number is uncertain) --
      needs the user's eyes on footage, same as the HARD_check24 precedent.
- [x] C8. Full run_clip end-to-end both clips (background): HARD exit 0,
      identical box score to the standalone run; TEST1 exit 0, box_score
      JSON/CSV byte-identical to pre-change snapshot even after a full
      pipeline rerun (not just standalone stage8). Also fixed a stale
      validate() warning string that still said "color tiebreak not built
      yet". Suite 87 green. Committed.

## Review (color tiebreak, 2026-07-13)
HARD's two AMBIGUOUS lines are gone: #3 and #23 now show correct per-team
credit, built entirely from crops the system already trusted (no new human
input, no hardcoded colors) and never guessing where evidence is unclear (0
of 6 claim-groups needed to abstain this run, but the path exists and is
tested). TEST1 is provably unaffected (byte-identical output, zero video
reads). A real bug (identity-to-number collapsing) was caught and fixed
before shipping by eyeballing actual output against footage, not just by
the unit tests passing.
Two findings were surfaced and deliberately NOT silently resolved:
1. Color could also legitimately un-stick some disputed-frame conflicts
   (proven case: HARD id7's real Winton Woods #3 credit, 2.7s, currently
   sits at zero because it's 100%-disputed against id4's simultaneous #3
   claim, even though color cleanly tells them apart). Natural next unit,
   not built this session.
2. A genuine label contradiction on HARD track 2475 (Part-1 labeled #3,
   Part-2 queue-resolved #13 -- same track, different numbers, same team
   either way) needs the user's eyes on footage to resolve, same as the
   HARD_check24 precedent (section 7a).
NEXT: user's call -- resolve the track-2475 contradiction, extend the
tiebreak into disputed frames, or move to Phase 5 (ball/shot detection).

## Follow-up: track 2475 contradiction resolved (2026-07-13)
Root cause confirmed by eyeball: 4 stills across track 2475's full span
(37.7s-39.8s) show one visually continuous player throughout -- NOT a
splice like t49. A 5x-upscaled crop at frame 1155 (38.5s) shows a legible
"3" on her back. User confirmed from the same evidence + a footage check.
FIX: HARD_decisions.json backed up (HARD_decisions.backup-2026-07-13-pre-
num-fix.json); the Part-2 queue-resolution entry {window:1, identity:10}
corrected 13 -> 3 (Part-1's original track label was right all along; the
Part-2 click was the honest mistake).
VERIFIED (full run_clip HARD rerun, exit 0): #3 6.9s -> 7.3s (+0.4s), #13
1.3s -> 0.8s (lost the incorrect 0.4s retro credit), every other line
byte-unchanged. Suite 87 green. DECISIONS section 12 updated with the
resolution + a new KNOWN DEBT item (section 4): Part-1 track-labels and
Part-2 queue-resolutions still don't cross-check the same track against
each other -- this instance was fixed by hand, the underlying gap remains
for a future session. Committed.

---

# SANITY CHECK (dad-demo substitute) (completed 2026-07-12)

Joint validation instead of a formal demo (agreed: no real stat exists yet to
show dad, per DECISIONS §9b context). User independently scrubbed both source
videos against 5 claims produced by stills I generated.
- [x] V1. Extracted + reviewed stills myself before handoff: 2 possession-
      boundary frames per clip, the 2 existing OCR-ground-truth crops (#24
      both clips), 1 new labeled still for a pure-human-click identity
      (HARD #44, identity 103, zero OCR backing)
- [x] V2. USER RESULT — boundaries: ball does cross half-court at both
      flagged moments (confirmed), BUT correctly caught that crossing
      half-court != a possession starting (mid-advancement, not a
      boundary). NOT A BUG — exactly possessions.py's documented scope
      (court-side stand-in, ROADMAP Phase 3 "no ball needed"); the WORD
      "possession" oversells it. Rule recorded: never present window counts
      as "N possessions" to a coach without this caveat (DECISIONS §9b).
- [x] V3. USER RESULT — OCR ground truth: both #24 crops confirmed correct
      number + correct team color (TEST1 green, HARD white/red).
- [x] V4. USER RESULT — HARD #44 (identity 103, 5.3 of 5.8s from a single
      queue-resolution click, ZERO OCR backing): user confirmed correct by
      means other than the jersey number itself. First real proof a pure
      human click held up under independent scrutiny.
- [x] V5. DECISIONS §9b recorded (placed after §9a, chronologically correct);
      commit.

## Review (sanity check, 2026-07-12)
Zero errors found across 5 independent checks. The identity/tracking
foundation (both clips) holds up under real scrutiny, not just eyeballing by
the builder. The one non-obvious finding: "possession" windows are a
court-side stand-in, not real basketball possession-start detection — a
naming trap for future coach-facing material now on record. This reinforces
(doesn't reverse) the decision to defer the formal dad demo until Phase 5
produces an actual stat line.
NEXT: color tiebreak (2 measured dual-team cases) or start Phase 5 (ball/shot
detection) — user's call on ordering; Phase 4 (multi-gym auto-cal) stays
deferred to after Phase 8 per user's ship-ASAP directive (not yet reflected
in ROADMAP.md — pending edit, intentionally not made yet).

---

# HARD QUEUE-RESOLUTION SESSION (completed 2026-07-12)

User resolves HARD's 23-item queue via the review page; resolutions apply
through stage7's shared contradiction-checked merge path (same as TEST1's R4).
- [x] S1. Verified bundle current (boundaries [600,1097] embedded, labels
      pre-populated); backed up HARD_decisions.json first
- [x] S2. USER resolved 15/23: 8 named (#20, #0, #3, #44 in poss-0;
      #1, #23, #13, #20 in poss-1), 7 rejected as non-players, 8 left unsure
- [x] S3. BUG FOUND BY REAL USE: re-download DROPPED 17 prior labels (tracks
      not shown on the regenerated page never enter the download's dec/qdec).
      ROOT FIX in make_review_bundle.py: dec/qdec initialize from presets so
      unshown labels carry through; queue presets only when boundaries match
      (ids shift otherwise). DATA FIX: merged the 17 back from the backup
      (10 player labels incl. late-seed vouchers + 7 refs). Suite 73 green.
- [x] S4. Full run_clip HARD rerun DONE (exit 0, end-to-end)
- [x] S5. Verified: all 8 named resolutions merged [human] (642 frames =
      21.4s recovered); 7 rejects recorded, never credited; 1 contradiction =
      the known #24 OCR splice (correctly stuck); 0 continuity confirms;
      0 disagreements; max line 15.1s in a 20s span (no impossible credit).
      BOARD: 9/9 distinct roster numbers named (was 7/9); #3/#23 AMBIGUOUS
      (color tiebreak); disputed surfaced not counted (#24 1.1s, #3 2.7s);
      unnamed 15 ids/61.7s; 7 queue items still open + 2 dual-team cases.
- [x] S6. Bundle regenerated (15 resolutions verified embedded in presetQ);
      DECISIONS §10a recorded; committed.

## Review (HARD queue session, 2026-07-12)
Both clips now have fully-named boards (TEST1 10/10, HARD 9/9) — the dad-demo
prerequisite is met. The session also caught bug #3-found-by-real-use: the
review page's re-download silently dropped labels for tracks not shown on the
regenerated page (incl. late-seed vouchers); root-fixed in make_review_bundle
(dec/qdec initialize from presets; queue presets gated on boundary match) and
the 17 dropped labels were restored from the pre-session backup. Practice
locked in: back up {clip}_decisions.json before every review session.
NEXT: DAD DEMO (everything ready), then color tiebreak (two measured
dual-team cases blocking #3/#23 team attribution).

---

# HOUSEKEEPING COMMITS + RE-ID TRACKER PROBE (completed 2026-07-12)

## Task 1 — commit the uncommitted work in logical chunks
The working tree holds everything from Phase 0-lite through queue-resolution v2
(multiple completed units interleaved across the same files, so per-unit commits
are no longer reconstructable; chunks are thematic, by whole file).
- [x] H1. Commit REVIEW.md + ROADMAP.md (the 2026-07-02 read-only review deliverables)
- [x] H2. Commit the pipeline work in one honest chunk: new modules
      (phase2/oncourt.py, purity.py, possessions.py, stage7_merge.py,
      stage8_box_score.py, make_review_bundle.py, cache_oncourt.py,
      cache_purity.py) + modified core (identity.py, roster.py, stage4/5/6,
      windows.py, run_clip.py, cache_tracks.py, clip_config.py, .gitignore)
      + tests/ (73-test suite) — message names the units it contains
      (Phase 0-lite, ROI seeding, fair remeasure, merge, box score,
      click-seeding, purity, possessions, queue-resolution v2)
- [x] H3. Commit phase2/DECISIONS.md + tasks/todo.md (the record)
- [x] H4. Commit diagnostics/test1_probe/ + spikes/out stills/summary (artifacts)
- [x] H5. Suite green before and after (73 passed, 0.35s, both times)

## Task 2 — Re-ID tracker probe (BoT-SORT), READ-ONLY experiment
Question it answers: does an appearance-embedding tracker cut fragmentation
(TEST1 baseline: 122 distinct track_ids / 461 frames)? Clicks scale with
fragmentation, so ~4x fewer fragments = ~4x less human work (DECISIONS §10).
SAFETY RAIL: the existing TEST1 tracks cache, oncourt/purity caches, user
labels, and queue resolutions are NOT touched — BoT-SORT output goes to a
separate probe JSON. Adopting the tracker (= rebuild caches, re-label) is a
SEPARATE decision made on the measured number.
- [x] T1. phase2/botsort_reid.yaml — copy of ultralytics' botsort.yaml with
      with_reid: True (model: auto, gmc sparseOptFlow — good for a panning cam)
- [x] T2. tracking.iter_tracks gains an optional tracker_config param
      (default "bytetrack.yaml" — zero behavior change; suite 73 green)
- [x] T3. spikes/reid_fragment_probe.py — extracts TEST1's exact span
      (reuses run_tracking.extract_subclip), tracks with botsort_reid.yaml,
      writes spikes/out/TEST1_tracks_botsort.json + prints distinct-id count
      and per-frame track-count stats vs the cached ByteTrack baseline
- [x] T4. Probe run TWICE (one variable apart): v1 stock reID 122 -> 131 ids;
      v2 track_buffer 120 (4s relink window) 122 -> 128. Mean lifespan flat
      (~106 frames) both runs. NO fragmentation gain.
- [x] T5. DECISIONS §11 recorded: NEGATIVE RESULT, no adoption — bytetrack.yaml
      stays. Likely cause: teammates wear identical uniforms, so appearance
      embeddings can't split exactly the crossings that cost clicks. Remaining
      levers re-ranked: footage zoom/4K, then span-prioritized queue.

## Review (housekeeping + re-ID probe, 2026-07-12)
Working tree fully committed in 7 commits (docs; pipeline Phase 0-lite ->
queue-resolution v2; DECISIONS/log; artifacts; probe; variant 2; verdict).
Suite 73 green before and after every change. The re-ID experiment cost two
read-only background runs and returned a decisive negative: the ~4x click
reduction does NOT come from a tracker config swap — recorded in §11 so no
future session re-buys it. Pipeline behavior unchanged (iter_tracks default
still bytetrack.yaml; caches/labels/resolutions untouched).
NEXT per handoff order: HARD queue-resolution session (user clicks, same flow
as TEST1), then the DAD DEMO (overdue; two named box scores + review
workflow), then color tiebreak per roadmap.

---

# QUEUE-RESOLUTION v2 (completed 2026-07-12)

Human resolves a queue item -> that identity's candidate/unknown span is
retro-credited via the SAME merge machinery as an OCR agree (same
contradiction check; merge records carry source: "ocr" | "human"). Rules:
- The review page's queue rows show crops across the identity's WHOLE span
  (the click vouches for everything credited).
- Resolutions live in {clip}_decisions.json ("queue_resolutions") keyed by
  (window, identity_id) + the WINDOW BOUNDARIES they were made against;
  stage7 REFUSES stale resolutions (boundaries mismatch = ids shifted).
- Human number must be on-roster (refused loud otherwise); "reject" = crowd/
  not-a-player: recorded, never credited, dropped from future queues.
- OCR-vs-human conflict on one identity = flagged, never silently resolved.
- Bundle PRE-POPULATES existing track labels so re-download keeps them.

- [x] R1. 5 queue-resolution tests written first — all passed on first
      implementation (suite 72)
- [x] R2. stage7_merge: human resolutions applied after OCR agrees via a
      SHARED _restamp path (same contradiction check, same credited-set
      bookkeeping, source "ocr"|"human"); stale-boundaries refusal;
      off-roster refusal; OCR-vs-human conflicts flagged; rejects recorded
- [x] R3. bundle Part 2: queue rows with WHOLE-SPAN crops + resolution
      buttons; existing labels PRE-POPULATED; window_boundaries embedded in
      the download. Regenerated: TEST1 = 20 tracks + 10 queue items;
      HARD = 27 + 22.
- [x] R4. TEST1 resolved by user: 10 resolutions -> 6 merged (30.8s
      recovered), 2 rejected (crowd), 2 REFUSED by the contradiction check
      (#13/#32 overlap -- ledger defended itself against human error, flagged
      for re-review). Final board: 10/10 named, 8 near-full coverage. BUG
      found by real use + fixed + tested (suite 73): stage8 now honors the
      event-level merge.number over the registry hypothesis (human-resolved
      identities without hypotheses were dropping into unnamed).
- [x] R5. DECISIONS §10 recorded incl. the scaling answer (clicks scale with
      tracker fragmentation -> re-ID tracker experiment = NEXT BUILD).
      HARD queue (22 items) awaits the user whenever.

---

# PHASE 3 — POSSESSION DETECTION v1 (completed 2026-07-12)

Signal: per-frame mean court-x of ON-COURT bodies, DENSE and FREE from the
oncourt cache (court_feet already stored per frame). Algorithm v1: half-court
side classification with a dead zone + hold-time hysteresis + minimum
possession length; degenerate detection falls back LOUDLY to the fixed
accumulation windows (abstention). No new cache: windows are derived
deterministically from the oncourt cache at load time; an inspection JSON
({clip}_possessions.json) is written every run for eyeball validation
(user/dad scrub video at the printed boundary timestamps).
Measurement discipline: MAX_ATTEMPTS stays 10 — one variable (window shape)
changes at a time. Acceptance metric: queue items per minute of film,
before/after. stage3_windows stays on fixed windows (containment diagnostic).

- [x] Q1-Q4 built (suite 64: 6 detector tests + 3 late-seed safety tests).
      PLUS a mid-phase design addition forced by real data: LATE SEEDING —
      with 1 long window, window-start-only seeding lost coverage of labeled
      tracks appearing mid-span (#44 vanished). Fix: a human label vouches
      for the TRACK, so a labeled ON-COURT track seeds at FIRST APPEARANCE —
      but ONLY if its identity is fresh UNKNOWN; a relinked CANDIDATE carries
      unvouched continuity history and is never late-seeded (tested).
- [x] Q5a. TEST1 (possessions + late seeding): detector says 1 L-possession
      (80% side agreement; matches left-heavy zone data — user to eyeball).
      Queue 24 -> 9 items (93 -> 35 per min of film, 2.7x). Confirms 3, all
      window-0. RETRO recovery 104 -> 669 frames (3.5s -> 22.3s, 6.4x): #5 =
      1.6 live + 13.5 RETRO; #13 = 6.3 + 8.8. 9/10 roster named; #44 in
      queue (his tracks relink as candidates; OCR never reads 44 -> honest
      queue item; queue-resolution v2 is the designed answer). NEW
      contradiction caught+refused: identity 37 #32 overlapping 48 frames
      with existing #32 (the multi-#32-label mess, contained).
- [x] Q5b. HARD: queue 55 -> 15 (165 -> 45/min, 3.7x); 1 confirm (#24 @1.00,
      merge correctly still refused on the 3-frame splice overlap); 7/9
      numbers named, totals honestly LOWER (the 2s regime was re-vouching
      everyone every 2s = weak provenance; possessions demand earned credit).
- [x] Q6. DECISIONS §9 recorded with measured before/after.

## Review (Phase 3 v1 — possessions + late seeding, 2026-07-12)
Windows now follow the game. Queue density fell 2.7-3.7x and every queue item
resolves a SPAN, not a 2s sliver; retro recovery per read grew 6.4x on TEST1.
HARD exposed the honest tradeoff: stricter provenance lowers free credit on
low-read-rate footage — the queue carries it. NEXT UNIT (now clearly the
highest-value piece): queue-resolution v2 — a human resolving a candidate
retro-credits its span through the SAME gate mechanics as an OCR agree. Then
dad demo. Boundary eyeball request OPEN: user to confirm both clips are
genuinely one left-side possession (timestamps in {clip}_possessions.json).

---

# TRACK-PURITY CHECK (completed 2026-07-12)

Two detectors for the two observed splice diseases:
A) INTRA-track (t49 case): OCR sweep across each labelable track's lifespan
   (once per clip, cached like oncourt). >=2 DIFFERENT confident on-roster
   numbers on one track => SPLICED: quarantined from labeling (bundle shows a
   warning instead of buttons) and its labels REFUSED at seed time (loud).
B) INTER-track (t6/t1496 case): stage8 counts each number's seconds as the
   UNION of solely-claimed frames; any frame where 2+ identities claim the
   same number simultaneously is DISPUTED — excluded from the line, surfaced
   as disputed_seconds (abstention: one of those bodies is wrong).
Plus: bundle crops now spread early/mid/late per track (a mid-track jersey
change is always visible to the human), via a shared seed-tracks helper.
Limitation noted: number-only detection can't see a splice between two
same-number players (dual-team #3/#23) — color tiebreak territory.

- [x] P1-P5 built (suite 55 green; 5 purity tests + 2 label-refusal tests +
      1 disputed test, all written first)
- [x] P6 verified on both clips. Honest results:
      * Detector A recall-limited: 0 convictions incl. known-spliced t49
        (conviction needs confident reads of BOTH numbers; rare at this read
        rate). Recorded in DECISIONS §8 — spread crops + human remain the
        intra-track defense.
      * Detector B (disputed frames) caught THREE real cases first run:
        HARD #24 0.7s (known splice tail, line now 18.2s honest);
        TEST1 #32 0.9s (PREVIOUSLY UNKNOWN label collision);
        HARD #23 2.2s disputed vs 2.7 counted (two REAL #23s, one per team,
        simultaneously on court — the dual-team ledger limitation live;
        strongest case yet for the color tiebreak).
      * Bundles regenerated with early/mid/late crops (79/160 embedded).
- [x] P7. DECISIONS §8 recorded.

## Review (track-purity unit, 2026-07-12)
Ledger integrity now enforced at three layers: gate-authorized merges,
contradiction refusal, and per-frame disputed exclusion. No disputed second
is ever counted; every one is printed and persisted. Known limits recorded:
detector A low recall (footage read rate), number-keyed ledger can't split
dual-team numbers (color tiebreak = next natural identity improvement, now
with a measured 2.2s real-world case). NEXT per agreed order: Phase 3
possession windows, then dad demo with both box scores.

---

# PHASE 2 — CLICK-SEEDING + REVIEW BUNDLE v1 (completed 2026-07-10/11)

Design: a generated SELF-CONTAINED HTML page (crops embedded as data URIs, no
server) listing every distinct ON-COURT seed-frame track, sorted by floor
time, with its 3 largest jersey crops + one roster-number button row. The
human labels what they can and downloads {clip}_decisions.json into
phase2/out/. roster.seed_number_for() then merges those labels (decisions
override legacy config seed_labels; off-roster labels REFUSED loud) — so
human clicks flow through the SAME seed gate as everything else. v1 scope =
TRACK labeling only (mid-window candidate resolution = v2, needs a post-hoc
human-confirm mechanism).

- [x] C1. roster.load_decisions + merge DONE (human labels override legacy
      config labels; off-roster REFUSED loud); 3 tests (suite 47)
- [x] C2. make_review_bundle.py DONE: TEST1 = 28 labelable tracks / 83 crops
      in one 335KB self-contained HTML; HARD = 56 tracks / 167 crops. The
      101/141 unnamed IDENTITIES collapse to 28/56 TRACKS — one label names
      all its window-instances.
- [x] C3. USER labeled TEST1 (19 track labels, incl. correctly marking a
      spliced track t49 as unsure). decisions.json placed at
      phase2/out/TEST1_decisions.json -> rerun stage6-8:
      confirms 0->4 (all EYEBALLED correct: #5x2, #32), 0 disagreements;
      merge restamped 104 candidate frames as confirmed_retroactive;
      BOX SCORE: 10/10 roster players NAMED (was 0). Unnamed 101->36
      identities (167.4s -> 62.7s). First "coach clicks, gets stat lines"
      artifact of the project.
- [x] C4. Records below. NEW FINDING: track-splice defect (t49) — human
      caught the tracker jumping #44->#13 mid-track via 2 different jersey
      numbers in one track's crops. Logged as a build item (purity check +
      3-crop spread in the bundle), not fixed yet.
- [x] C5. HARD labeled by user (18 labels + 12 refs + unsures). First run:
      1 confirm (#24 @1.00, EYEBALLED correct), 0 disagreements — and the
      SAFETY MACHINERY CAUGHT A LABEL ERROR two independent ways:
      (a) #24's box line = 21.9s in a 20s clip (impossible => double credit);
      (b) the merge CONTRADICTION check refused to re-credit #24 (3
      overlapping frames). Diagnosis via early/mid/late crops
      (HARD_check24.png): t6=#24 ✓, t1496=#24 ✓, but t7 = #23 (mislabel) AND
      white/red #23 was MISSING from HARD's Milford roster. Also found: t6 &
      t1496 overlap 37 frames though both genuinely #24 => a small splice
      tail (purity-check case #2, ~1.2s inflation, deferred to the check).
      User confirmed: t7 -> 23; Milford roster += 23 (now #3 AND #23 are
      dual-team -> color-tiebreak backlog grows). Corrected rerun in flight.

---

# PHASE 2 — JERSEY-KEYED BOX SCORE + PLAYER POSITIONS (completed 2026-07-07)

Design: phase2/stage8_box_score.py aggregates the MERGED events by JERSEY
NUMBER across windows (numbers are the stable key; identity_ids are per-
window). Counts confirmed (live) + confirmed_retroactive separately + jointly,
converts frames->seconds via cache fps. Court positions come FREE from the
oncourt cache (court_feet already stored per frame/track) -> per-player zone
time via phase1 zones. Honesty rules: identities WITHOUT a number surface as
an "unnamed confirmed" bucket (the click-seeding gap made visible, never
dropped); dual-team numbers (#3 on HARD) flagged team-ambiguous; presence
seconds on a short clip, not game stats. Outputs: JSON + CSV + console table.

- [x] B1. 7 aggregation tests written first — all passed on first stage run
      (suite 44 green)
- [x] B2. phase2/stage8_box_score.py DONE (pure build_box_score + JSON/CSV)
- [x] B3. run_clip wiring DONE (stage8 section)
- [x] B4. TEST1 box score: #5 (LM) 1.9s = 1.6 live + 0.3 RETRO, TOP_OF_KEY;
      #13 (Milford) 0.9s PAINT; unnamed 101 ids/167.4s surfaced; review
      counts surfaced. First coach-readable output of the project.
- [x] B5. HARD chain (stage6 registry rerun -> 7 -> 8): 0 merges by
      construction; empty player table; unnamed 141/182.5s = the honest
      click-seeding before-picture.
- [x] B6. DECISIONS.md §6 recorded.

## Review (Phase 2 unit 2 — box score, 2026-07-07)
The pipeline now ends in a coach-readable artifact: per-number lines with
live/retro seconds + zone time, unnamed bucket, review counts, honesty note.
Positions came free from the oncourt cache (no new geometry). NEXT unit:
click-seeding + review bundle v1 (the decisions JSON feeds seeds through the
SAME gate) — converts both clips' unnamed buckets into named lines. Then CSV
polish/exports are already half-done (CSV ships with stage8).

---

# PHASE 2 — RETROACTIVE STAT MERGE (completed 2026-07-06)

User resolved #30: hand-label was a mistake, no #30 in HARD — rosters stand.
User accepted sideline pickups (margin class, contained by abstention).

Design (from ROADMAP Phase 2 / REVIEW 7.5): when OCR AGREES (the ONLY trigger
— consumes gate-emitted confirmation records, never position scans), the
identity's candidate-stamped events re-stamp as `confirmed_retroactive` (NEW
event-level state; live vs retro distinguishable forever). LOST gaps stay
unattributed (never invent presence). CONTRADICTION check: if the same number
already has confirmed/retro frames overlapping the span in that window ->
NO merge + loud flag. Merge writes a NEW artifact ({clip}_player_events_
merged.json); stage5's raw artifact is never mutated.

- [x] M1. identity.py confirmation records DONE (gate-emitted; seed carries
      roster_number)
- [x] M2. stage6 identities registry DONE
- [x] M3. 11 merge tests written FIRST — all passed on first implementation
      run (suite 37 green, ~0.25s)
- [x] M4. phase2/stage7_merge.py DONE (pure merge_events + artifact main;
      loud abort on stage5/stage6 replay divergence)
- [x] M5. run_clip stage7 section + final state counts in INTEGRITY DONE
- [x] M6. TEST1 real-data merge DONE: w0 id5 #5 @1.00 -> 10 candidate frames
      re-credited as confirmed_retroactive; LOST f168-169 NOT invented;
      ledger exact (candidate 1587->1577, retro +10); 0 contradictions.
      Recorded as DECISIONS.md §5.

## Review (Phase 2 unit 1 — retroactive merge, 2026-07-06)
Merge is live and structurally safe: triggers are gate-emitted agree records
only (no position input exists in the code path); candidate-only restamps;
live vs retro distinguishable forever; contradiction = refuse + flag; raw
stage5 artifact preserved; canonical output is the merged JSON. Next Phase-2
units: jersey-keyed box score (aggregate confirmed+retro by number), per-
player court positions (tracks x homography join), CSV export, review bundle
v1. HARD needs a stage6 rerun (registry) before its merge runs — no-op today
(0 agrees).
- Deferred within Phase 2 (next units): jersey-keyed box score, per-player
  court positions, CSV export, review bundle v1. HARD stage6 rerun for
  registry when needed.

---

# HARD WIDE + ATTEMPT POLICY v2 (completed 2026-07-06)

- [x] HARD real roster entered (Milford 1,3,13,24,44 / Winton Woods 10,3,23,0,20;
      #3 on BOTH teams -> validate() warns; #30 contradiction flagged to user;
      old seed labels retired; rig flag removed). Span widened to 600..1200.
- [x] HARD wide chain: tracks 601f/260 ids; oncourt 601/601 mean 12.7; full
      run_clip green (containment 45->0, 0 continuity confirms, queue 56
      on-court / 174 crowd excluded).
- [x] HARD v1 baseline: 1%/frame confident, 2%/window (1/41) — cross-gym gap.
- [x] MONTAGE DIAGNOSIS (scratch crop tiling, both clips, eyeballed): HARD's
      close-camera zone = refs/coaches; players small across the pan; a close
      player's #24 was human-legible -> gap = DISTANCE + attempt selection,
      NOT jersey contrast. Montages copied to phase2/out/*_crops.png.
- [x] stage6 attempt policy v2 (best-crops-first, budget/threshold unchanged;
      v1 JSONs preserved): TEST1 12.5% -> 25% windows-with-reads; HARD 2% ->
      5%. 29 confident reads across clips, 0 disagreements. DECISIONS 4b/4c.

## Review (HARD wide + v2, 2026-07-06)
Fair protocol now proven on two gyms with a same-day protocol improvement
measured side-by-side. Read-rate levers ranked by evidence: crop selection
(done, ~2x), window length (Phase 3), hypothesis coverage (Phase 2), footage
zoom (product guidance). Safety unbroken across every run: 0 wrong confirms,
0 disagreements, 0 continuity confirms. NEXT: Phase 2 (retroactive merge +
click-seeding/review flow) as its own clean unit.

---

# FAIR REMEASURE — TEST1 WIDE SAMPLE (completed 2026-07-06)

Span widened 300..+120 -> 120..+461 (full validated pan, ~8 x 2.0s windows;
window size unchanged — same protocol, bigger sample). Both caches stale by
design; guards will refuse until rebuilt.

- [x] W1. cache_tracks TEST1 DONE: 461 frames, 122 distinct track_ids
- [x] W2. cache_oncourt TEST1 DONE: 461/461 anchored (0 unclassified, reproj
      <=0.69px), on-court mean 12.7 (min 10 max 15) stable across the whole
      pan. (First launch orphaned by a shell mistake; killed + relaunched.)
- [x] W3. full run_clip TEST1 wide DONE (exit 0, end-to-end): 24 on-court
      candidates / 8 windows; per-frame 8% any / 6% confident; per-2s-window
      12.5% (3/24); agree 1 (EYEBALLED correct #5@1.00 f=128), disagree 0,
      no_position_hypothesis 2 (perfect reads #5@1.00 + #24@0.993 correctly
      abstained — two-signal rule); containment 12->0; 0 continuity confirms;
      queue 28 on-court (124 crowd excluded). Far-window seed still (w7 f540)
      eyeballed: mask holds at pan end.
- [x] W4. DECISIONS.md 4a-WIDE recorded: 3 separate bottlenecks (read rate /
      window length / hypothesis coverage) + the seed-labels-don't-survive-
      re-tracking finding. G1 per-POSSESSION number still pending possession
      detection (Phase 3); 12.5% is per-2s-window, NOT the G1 rate.

## Review (fair remeasure TEST1, completed 2026-07-06)
Protocol is now fair on TEST1 (ROI pool + real roster) and recorded. Wide
sample: read rate 6%/frame confident (reads land at 0.99-1.00; dial not the
bottleneck); 12.5% per 2s window; confirms additionally gated by hypothesis
coverage (2 hand labels, stale after re-track). Safety perfect throughout:
0 wrong confirms (all eyeballed), 0 disagreements, 0 continuity confirms,
containment 12->0. Next per roadmap: retroactive merge + review/seeding
coverage (Phase 2), possession windows (Phase 3) -> that's where the true
G1 per-possession number comes from. HARD real roster still pending (user).

---

# FAIR REMEASURE — TEST1 (completed 2026-07-06) — roster half

User entered the REAL TEST1 rosters from film (2026-07-06):
  Milford (white/red): 3, 13, 23, 44, 10
  Little Miami (green/yellow): 24, 5, 32, 14, 30
(Open: HARD/Winton Woods roster still pending; also confirm whether these are
on-floor fives or full team rosters — matters for full games, not this span.)

- [x] 1. clip_config: TEST1 real roster entered; seed_labels {17:13, 6:5} still
      on-roster; HARD marked allow_off_roster_seeds=True (documented rig)
- [x] 2. ClipConfig.validate() added (video exists, span sane, roster ints
      0-99, seed⊆roster w/ rig escape, window>0, cross-team duplicate WARNING);
      wired into run_clip + cache_tracks + cache_oncourt. Verified: TEST1 ok,
      HARD ok via flag, off-roster seed refused loud. Suite 26 green.
- [x] 3. stage6 TEST1 fair rerun DONE: per-frame 13% any / 10% confident;
      per-POSSESSION 33% (2 of 6). Both confirms EYEBALLED CORRECT (#5 green
      LM jersey @1.00; #13 white/red Milford @0.99). 0 disagreements, 0
      continuity confirms. Roster 3->10 changed nothing (same 8 reads): the
      pool was the rig; remaining no-reads are unreadable crops, not filtering.
- [x] 4. Recorded as phase2/DECISIONS.md §4a with the DO-NOT-GATE-ON-n=6
      warning. Next sample-wideners: longer TEST1 span (pure compute) and/or
      HARD real roster (user input).

---

# ROI-MASK SEEDING (completed 2026-07-05) — first half of the fair remeasure

Design (approved defaults): per-frame on/off-court classification of the CACHED
tracks, computed ONCE per clip (root cache_oncourt.py -> phase2/out/
{clip}_oncourt.json), reusing the validated Phase-1 rules verbatim (anchor +
pixel_to_feet + on_court margin/horizon + frame-edge feet drop). Stages consume
a per-(window, track) MAJORITY vote (tie = off, conservative). Identity machine
mechanics UNTOUCHED — only who gets seeded / OCR'd / queued changes. Off-court
bodies are excluded from seeds+queue+OCR but COUNTED in prints and recorded in
the JSONs (abstention: nothing silently vanishes). Refs remain in the pool.

- [x] 1. phase2/oncourt.py DONE — build() + load_checked() + on_court_by_window()
- [x] 2. cache_oncourt.py root wrapper DONE (syncs both configs)
- [x] 3. Policy unit tests DONE (6 tests; suite 26 green)
- [x] 4. TEST1 on-court cache DONE: 120/120 frames anchored (0 unclassified,
      reproj 0.00-0.69px); per-frame on-court min=11 max=14 MEAN=13.3 — matches
      stage1's validated 13.1. Seed frame f=300: 13/32 on-court (19 crowd/bench
      previously being seeded).
- [x] 5. Stages wired DONE (stage4 queue+stills, stage5 seeds, stage6
      seeds+OCR pool+queue, run_clip precheck; stage4 docstring updated)
- [x] 6a. stage4 rerun: COACH QUEUE 25 -> 8 (26 off-court excluded+counted);
      seeds w0=14/w1=13 (players+refs band). Seed still eyeballed: players
      green, bleachers/bench red, refs green (by design), 1 borderline
      sideline coach inside MARGIN_FT (accepted, same class as refs).
      stage5 rerun: box score = on-court confirmed only.
- [x] 6b. stage6 TEST1 rerun DONE: OCR pool 19 -> 6 on-court (14 crowd excluded);
      crops 190 -> 60; per-frame confident 3% -> 10%; per-POSSESSION 11% -> 33%.
      PROOF the old number was pool-rigged: numerators IDENTICAL (same 8 reads,
      6 confident, same 2 confirms #5@1.00 f=310 / #13@0.99 f=322) — only the
      denominator changed. 0 disagreements, 0 continuity confirms. Queue after
      OCR: 6. STILL NOT the fair number (roster = 3-number stand-in).
- [x] 7. HARD DONE: on-court cache 120/120 anchored, mean 13.0 on-court (2nd gym,
      classifier generalizes). Full run_clip HARD end-to-end (exit 0): prechecks
      passed; calibration 20.4->0.7px / 0.75/1.75ft (matches baseline);
      containment 4->0 cross-window relinks (matches prior safety run); seeds
      15/13 on-court (19/14 skipped); coach queue 8 (21 off-court excluded);
      OCR pool 5 on-court (6 excluded), 0 reads/0 confirms — STRUCTURAL, the
      stand-in roster {5,13,24} can't emit HARD's real numbers (23/30); 0
      continuity confirms everywhere; HARD_ocr_confirms.json written; INTEGRITY
      prints the OCR line with pre-OCR caveat.

## Review (ROI-mask seeding, completed 2026-07-05)
Built: phase2/oncourt.py (per-frame on/off-court classification of cached
tracks, reusing Phase-1 court rules verbatim; strict-majority per (window,
track), tie=off) + cache_oncourt.py wrapper + 6 policy unit tests (suite: 26).
Wired: stage4 (seeds+queue+red/green stills), stage5 (seeds), stage6
(seeds+OCR pool+queue counts), run_clip precheck. Identity machine UNTOUCHED;
exclusions always counted+persisted, never silent.
Validated: TEST1 classifier mean 13.3 vs stage1's independent 13.1; seed still
eyeballed (players green, crowd/bench red, refs green by design, 1 borderline
sideline coach within MARGIN_FT — accepted); TEST1 queue 25->8; OCR pool
19->6; per-possession confident read 11%->33% with IDENTICAL numerators (same
2 confirms, same frames/confidences) — proof the old number was pool-rigged.
HARD generalized + full run_clip green. REMAINING RIG: the 3-number stand-in
roster. NEXT: real rosters for both clips (film work with dad) -> the fair
remeasure -> Gate G1.

---

# PHASE 0-LITE (completed 2026-07-05) — the 4-item slice before the fair remeasure

Agreed in session: cut Phase 0 to the items that protect the Phase-1 measurement,
then go to ROI-mask seeding + real rosters. Item 1 approved to build now.

- [x] 1. Identity safety tests DONE (tests/test_identity_safety.py, 19 tests,
      ~0.2s, no video). Covers: continuity can't confirm (6 provenances);
      occlusion->LOST attributes nothing; relink->CANDIDATE ceiling; ambiguous->
      UNKNOWN; gap-expiry; roster_number survives breaks; OCR agree/disagree/
      no-read/no-hypothesis; window boundary blocks cross-window relinks (with
      single-machine control). Net PROVEN: planted CONFIRMED-on-relink ->
      5 tests failed loudly -> reverted (git-verified identical) -> 19 green.
      pytest installed in .venv; .pytest_cache gitignored.
      Run:  .venv/Scripts/python -m pytest tests/ -v
- [x] 2. Cache guard DONE: run_clip._load_and_check_cache() refuses a missing,
      stale (clip/span mismatch), or empty cache BEFORE the slow calibration
      solve. Verified: real TEST1 cache accepted; span-999 config refused in ~1s
      with a plain-English message + the re-cache command.
- [x] 3. seed() DONE: unknown track_id now prints a WARNING and returns False
      (was a silent no-op); returns True on success. +2 tests (20 total green).
- [x] 4. stage6 persists ocr_confirms.json DONE (+ makedirs; run_clip INTEGRITY
      reports agree/disagree counts and labels the box score pre-OCR). Verified
      by standalone TEST1 stage6 run: reproduced the known rigged-pool numbers
      exactly (2 agrees #5@1.00 / #13@0.99, 0 disagreements, 3%/11%, 0
      continuity confirms) and wrote phase2/out/TEST1_ocr_confirms.json with
      all 4 outcome buckets + evidence + read bboxes.
- Then: ROI-mask seeding -> real rosters -> fair remeasure (ROADMAP Phase 1)

## Review (Phase 0-lite, completed 2026-07-05)
All 4 items done and verified. Changes: NEW tests/test_identity_safety.py (20
tests, ~0.2s, proven to catch a planted violation); run_clip.py gained
_load_and_check_cache() (refuses missing/stale/empty cache before calibration)
+ OCR line in INTEGRITY; identity.seed() warns + returns bool on unknown
track_id; stage6 writes {clip}_ocr_confirms.json (+ makedirs). Also: pytest in
.venv, .pytest_cache gitignored. Unchanged: set_confirmed lock, thresholds,
schemas, calibration, stage5 outputs. TEST1 stage6 rerun reproduced prior
numbers exactly (no behavior drift). Ready for ROI-mask seeding next.

---

# READ-ONLY CODEBASE REVIEW + ROADMAP (task of 2026-07-02)

READ-ONLY. No code changes, no commits. Deliverables: REVIEW.md + ROADMAP.md at repo
root (uncommitted; user reviews and commits). Note: prompt says "check in before
working" but the task is an explicit fully-specified review request run autonomously,
so the plan is recorded here and executed in one pass.

- [x] Read phase1/DECISIONS.md, phase2/DECISIONS.md (phase2/HANDOFF.md does NOT
      exist — doc-drift finding #1, also noted by the earlier RUN_REPORT.md)
- [x] Read pipeline path: run_clip.py, clip_config.py, cache_tracks.py
- [x] Read phase1 stages: stage1_court_roi, stage2_generate_events, refit_keyframes,
      team_event_schema, zones, stage3_*, stage4_overlay, stage2c_validate
- [x] Read phase2: identity, ocr_reader, roster, run_tracking, tracking, windows,
      stage1..stage6
- [x] Read supporting: spikes/clips_config.py, spikes engine files, src/camera_tracking.py
- [x] Read World-B leftovers: process_game.py, src/*, render_heatmaps.py
- [x] Audit: every entrance to CONFIRMED; dual-config guardrail; TEST1/frame
      hardcodes; tracks-cache staleness/missing behavior; read_jersey pluggability;
      ClipConfig-as-API readiness
- [x] Write REVIEW.md (file-by-file GOOD/BAD/NEEDS-IMPROVEMENT + feature gap analysis)
- [x] Write ROADMAP.md (standalone, phased, decision gates, principles card)
- [x] Review section added below when done

## Review (read-only review task, 2026-07-02)

Deliverables written, nothing committed (user reviews + commits):
- **REVIEW.md** — file-by-file review of the whole repo + the 6 requested audits
  + the 26-feature gap analysis (plus 10 extra gaps found, §7.26a-j).
- **ROADMAP.md** — standalone 9-phase roadmap (0–8) with decision gates G1–G5
  and the principles card.

Headline findings (full detail in REVIEW.md §0):
1. The dual-config "loud assertion" in run_clip is a TAUTOLOGY (checks the value
   it just wrote); real protection is unenforced import ordering. Fix: assert
   the imported stage BINDINGS against the config (~20 min).
2. Tracks cache (and refit npz) have NO staleness validation — stale cache =
   silently wrong crops/identity evidence. Highest-value fix in the repo.
3. Stage 6 OCR outcomes are never persisted; the canonical player_events.json
   (and run_clip's INTEGRITY report) is pre-OCR.
4. phase2/stage1_states.py IRON-RULE proof harness crashes (TypeError — stale
   stub-era call signature) and its printed claims are stale.
5. README.md still documents the rejected World-B draft as the product;
   phase2/HANDOFF.md never existed.
6. The safety property itself HOLDS by construction: `state = CONFIRMED` occurs
   exactly once in the repo, inside set_confirmed; all callers route through
   seed/second_signal (proof in REVIEW.md §6.1).
7. One-clip-per-Python-process is a hard unstated invariant (module-level
   config binding across ~10 files) — fine for CLI use, the #1 hazard for the
   future web worker.
8. The jersey-crop montage generator script is lost (only its PNGs remain);
   golden artifacts (TEST1 caches/events) are gitignored, so the byte-identical
   baseline lives only on this machine.

No code was changed. Recommended first step: ROADMAP Phase 0 (regression suite
+ fingerprints + guard fixes, ~2-4 evenings) BEFORE the fair OCR remeasure.

---

# Phase 1 — Team Stats (the reliable spine, no player identity)

Goal: cheapest useful layer. CV emits structured **team_events** (per-frame on-court
positions in court-feet). Deterministic code computes zone occupancy/heatmaps, team
possession counts, pace. NO LLM, NO player identity, NO track IDs, NO jersey OCR, NO
shot/made-missed, NO database, NO web/API. Single offline Python tool, CONFIG-driven,
runs on ONE validated test clip. Validate each stage by eye before the next.

(Prior task's court-calibration log archived in `tasks/calibration_generalization_log.md`.)

## What already exists (and how it's used)
- **Validated homography engine** (World A): `spikes/stage2..stage6` + `clips_config.py`
  + `src/camera_tracking.py`. Produces `H_court` = ref-keyframe px -> court **feet**
  (HS 84x50), mean ~0.25 ft (TEST1) / ~0.96 ft (HARD), arcs glued. **REUSE THIS.**
- **Validated detector**: YOLOv8m @ imgsz=1280, person class. (Note: env torch is CPU
  -> sample frames for the eyeball loop.)
- **World B = earlier rough draft we are NOT using**: `process_game.py`,
  `src/court_mapping.py` (weaker single-frame click, wrong 94x50 NBA court),
  `src/detection.py` (track IDs), `src/team_assignment.py` (jersey k-means),
  `src/team_stats.py`, `src/schema.py`, `render_heatmaps.py`, `out.json`, `heatmaps/`.
  These conflict with the Phase-1 architecture rule. Leave untouched; do not import the
  naive court_mapping. (team_stats/render logic may be cribbed later if clean.)

## Key reuse: getting per-frame pixel->court-feet on the test clip
Mirror `spikes/stage4_courtmap.py`:
1. `run_optimization()` -> `H_court` (ref-keyframe px -> court feet) from the clip's
   clicked landmarks in `clips_config.py`.
2. Camera-track the whole clip; per frame f, `T = frame_to_ref(f)` (nearest-keyframe
   anchored). Then **pix->feet for frame f = H_court @ T**.
   (stage4 uses the inverse, feet->px, to DRAW the court. We want px->feet to LOCATE
   players, so we use `H_court @ T` directly.)

---

## STAGE 1 — COURT ROI FILTER (riskiest; build + validate FIRST)
The "13-person rule" fix: drop refs/bench/coaches/crowd, keep the ~10 on-court.
- [x] New script `phase1/stage1_court_roi.py` (CONFIG-driven, deterministic).
- [x] Build per-frame `pix->feet` from the validated engine (H_court @ frame_to_ref).
      Fit recovered at mean 0.25 ft / max 0.54 ft (matches validated TEST1).
- [x] YOLOv8m@1280 person detection per sampled frame (16 frames, 120..570, the
      validated pan). Feet pixel = bbox bottom-center. Map -> court feet.
- [x] Court polygon = 84x50 + MARGIN_FT. Classify on/off court.
- [x] Two leak fixes found by eye/coords: (a) horizon guard (reject feet mapping
      behind the vanishing line, w-sign); (b) drop boxes clamped to the bottom
      frame edge (feet off-screen = fake ground point). Margin 3 -> 1.5 ft.
- [x] Render: on-court GREEN, discarded RED, court+margin overlay, per-frame count.
      Result: on-court mean 18.2 -> 13.1 (most frames 10-15 = ~10 players + 2-3
      on-court refs, which can't be removed without identity). Crowd/bench excluded.
- [ ] **STOP for user eyeball confirm.** Open question: refs inflate count to ~13;
      f=450/480 wide views run 14-19 — user to confirm those are real bodies, not
      leaks, and whether MARGIN_FT=1.5 is right.

## STAGE 2 — TEAM_EVENT SCHEMA  (Stage 1 signed off)
Architecture decisions captured in `phase1/DECISIONS.md` (read first).
- [x] Part A: `phase1/DECISIONS.md` — settled decisions for future sessions.
- [x] 2a SCHEMA: `phase1/team_event_schema.py` — record + JSON (de)serialization,
      prints schema + round-tripped example. No track_id, order-independent,
      team/identity default "unknown". homography_confidence = the 450 guardrail.
- [x] 2b GENERATE: `phase1/stage2_generate_events.py` — reuses Stage 1 unchanged;
      16 team_events -> `phase1/out/TEST1_team_events.json`. Confidence from direct
      nearest-keyframe SIFT match (all frames "ok"; 450 stays "ok" by direct match
      — the documented guardrail limitation vs the ORB-chain positions).
- [x] 2c ROUND-TRIP + identity-free proof: `phase1/stage2c_validate.py` — round-trip
      PASS; counts bodies+zones from a positions-only view (no identity field at
      all). "team stats computable without identity: YES".
- Note: detection deterministic within a run; cross-run borderline flicker (CPU)
  worst at the bad-homography frame 450 — flagged for the dense pipeline.
- Commit before + after each sub-stage. Do NOT compute real stats/heatmaps/
  possessions/pace, assign identity/team, build a DB, or touch mask/margins/homography.

## STAGE 3 — STATS + ZONE HEATMAPS (deterministic reads over Stage-2 events)
Reads team_events JSON only. NO new perception, NO identity, NO LLM. ~16 sampled
frames => proves RENDER + zone-mapping correctness, not real basketball numbers.
FRAME TRUST: skip homography_confidence.state=="low_confidence" AND explicitly skip
frame 450 (hardcoded; known bad ORB-chain homography scored "ok" — see DECISIONS.md).
- [x] 3a ZONES: `phase1/zones.py` — half-court zones folded to nearest basket;
      boundaries tied to FT line + 3pt arc-top; spot checks pass.
- [x] 3b STATS: `phase1/stage3_team_stats.py` — trust filter (low_confidence +
      explicit 450) -> 15 trusted frames; per-zone occupancy + mean on-court 12.8.
- [x] 3c HEATMAP: `phase1/stage3_heatmap.py` — geometrically-correct court + 2D
      occupancy heatmap + zone guides -> phase1/out/TEST1_stage3_heatmap.png.
- [x] 3d DEMO: `phase1/stage3_demo.py` — coach-legible artifact
      `phase1/out/TEST1_phase1_demo.png` (heatmap + summary). STOP for user confirm.
- Commit before + after each sub-stage. Do NOT compute possessions/pace/shooting,
  assign identity/team, build a DB, touch web/API/mask/margins/homography, or fix 450.

## STAGE 4 — DEMO ARTIFACT (phase validation gate)
- [ ] Coach-legible output: court heatmap image + team stat summary. STOP for user.

## Open decisions to confirm before Stage 1 (see check-in)
1. **Which clip** is the Phase-1 test clip? (needs a validated homography) — recommend
   HARD.mp4 (canonical baseline). Alt: Test1.mp4 (tighter 0.25 ft, different gym).
2. **World B**: confirm we build Stage 1–4 fresh on the validated engine and leave the
   existing `src/`+`process_game.py` draft untouched (not deleted, not imported).

## FIX — direct-anchor homography (component swap, pulled forward)
Diagnostic proved the ORB chain is broadly unreliable (74% of frames >40px off,
catastrophic mid-interval). Swap the per-frame court homography source from the
accumulated ORB chain to DIRECT nearest-keyframe SIFT match. DO NOT touch ROI mask
logic, schema, stats, or zone mapping — only WHERE the homography comes from.
- [x] A: `build_court_anchor()` — direct nearest-kf SIFT, no chain. Previously
      catastrophic frames (330=1191px, 350/550) now glued; on-court mean 12.9.
- [x] B: confidence from the same match. Inlier floor 1103 (median 2557);
      threshold 150 -> 0 flagged. Frame 450 now CORRECT (inl=2568, kf=420); stage3
      450-skip confirmed redundant, LEFT in place. 47-frame events regenerated.
- [x] C: handoff pops MEASURED (same frame, two anchors): max 515px @171 (120->220),
      else 37-137px. NOT auto-fixed — flagged for user decision (smoothing/keyframe
      consistency is separate). Means the keyframes aren't perfectly mutually consistent.
- [x] D: old live(chain) was 74% of frames >40px off vs direct; new live = direct,
      fits every frame at reproj 0.00-0.72px (47/47 ok). Broad unreliability gone.
- [x] E: heatmap + demo re-rendered over 46 trusted frames (was 15); whole-pan
      coverage. phase1/out/TEST1_phase1_demo.png. STOP for user eyeball.
- Commit before + after each stage. No temporal smoothing, no schema/stats/zone/mask
  changes, no Phase 2, do not remove the 450 skip yet.

## KEYFRAME-CONSISTENCY RE-FIT (fix the handoff pop at the cause)
Cause: run_optimization ties keyframes only at sparse clicked landmarks (clustered
on one basket), so keyframes diverge when extrapolated -> handoff pop. Fix: add
DENSE adjacent-keyframe SIFT correspondences to the SAME global least_squares so
keyframes agree across their overlap (one shared court coord system).
- [ ] 1: `phase1/refit_keyframes.py` — global least_squares = landmark residuals
      (as now) + dense adjacent-keyframe correspondence residuals. Print keyframe
      mutual-consistency error before/after. Wire into build_court_anchor.
- [x] 2: handoff pops re-measured. 120<->220: 515->2.6px; all 5 now <=8.7px max
      (<=3.1px mean). Target met.
- [x] 3: per-frame accuracy held — anchor reproj 0.00-0.72px, 0 frames >5/15/40/100px;
      landmark court-fit improved 0.25->0.14 ft. No regression.
- [x] 4: `phase1/stage4_overlay.py` — full-pan court overlay on refit homography
      -> phase1/out/TEST1_stage4_overlay.mp4 + handoff stills. Seams glued (171/541
      verified). USER EYEBALL GATE.
- [x] 5: frame-450 skip REMOVED from stage3 (redundant). Effect: +1 frame (450),
      +14 body-positions, zone shares shift <1%, mean on-court unchanged (13.0).
      Nothing else surfaced. Demo/heatmap re-rendered over 47 frames.
- Commit before + after each stage. No blending/smoothing, no mask/schema/stat/zone
  edits, no Phase 2.

## PHASE 2 OCR — jersey second signal (promote_via_second_signal seam)
STRICT auto-confirm. EasyOCR installed + confirmed (reads #13@0.95, #5@0.98).
Crop finding: legible when close+facing, NO number >half the time (turned/angled)
-> read OPPORTUNISTICALLY ACROSS the window, accumulate best on-roster read.
Partial CALIBRATION roster (not ground truth): {5,13,24} discovered by OCR.
- [x] roster.py (numbers + loose team color; hand-verified seed labels t17=13,t6=5)
- [x] ocr_reader.py: pluggable read_jersey(crop, roster) closed-set; ONE constant
      OCR_CONFIRM_THRESHOLD (autonomy dial), easyocr lazy-imported.
- [x] identity.py: implement promote_via_second_signal -> 3 outcomes (AGREE->confirm
      provenance=second_signal; DISAGREE->flag swap; NO-READ->stay candidate).
- [x] stage6 driver: temporal accumulation per candidate across window; measure
      per-frame vs per-possession readability; auto-confirm list; queue before/after;
      stills of OCR-confirmed players + disagreements + stayed-yellow.
- set_confirmed lock intact (seed|second_signal only). Commit before+after.

## Next sequence (decided)
1. Demo Phase 1 to dad (artifact: phase1/out/TEST1_phase1_demo.png). His reaction
   may reorder the roadmap.
2. Keyframe-consistency RE-FIT (fixes the handoff pop at the CAUSE — global
   optimization over keyframes, like calibration Stage 3). REQUIRED before Phase 2.
   Rejected: blending (confident-wrong), adding keyframes (band-aid). See DECISIONS.md §6.
   Validation gate: 5 handoff jumps drop to a few px + full-pan overlay glued & smooth.
3. Phase 2.

## PHASE 2 — individual player identity (abstention-first)
Sits on NEW tracks + player_events layers ON TOP OF the identity-free team_event
spine (do NOT modify team_events). No LLM. SAFETY: never silently attribute a stat
to the wrong player. Silent promotion candidate->confirmed is STRUCTURALLY
IMPOSSIBLE until a second signal (OCR, later step) exists.
- [ ] 1: port YOLOv8m@1280+ByteTrack into phase2/; define IdentityState
      (confirmed/lost/candidate/unknown) carried by every track every frame. IRON
      RULE in code: no candidate->confirmed auto path; promote_via_second_signal()
      stub unimplemented/unreachable. Print per-frame track states for a span.
- [x] 2: honest loss (occlusion->lost, attribute nothing) + candidate recovery
      (reappear->candidate w/ evidence: gap, distance, motion). No auto-confirm.
- [x] 3: per-possession (or fixed ~15s stand-in) re-seed boundary; contain errors.
- [x] 4: seeding (post-process, re-seed-on-loss) + coach review queue (all
      candidates/unknowns need one click). Queue will be LONG without OCR — correct.
- [x] 5: identity-stamped player_events; uncertainty propagates; box score trusts
      'confirmed' only. Print states propagating onto events.
- Commit before + after each stage. NO OCR/second signal, NO auto-promote, NO
  team_event/stat/calibration edits, NO LLM, do NOT shrink the queue by loosening.

## Review
- Phase 1 spine built staged: Stage 1 (court ROI mask), Stage 2 (identity-free
  team_event schema + JSON), Stage 3 (zones + occupancy + heatmap + coach demo).
- All reads deterministic over `phase1/out/TEST1_team_events.json`. No identity,
  no LLM, no DB, no web/API. Validated clip = TEST1.
- Frame 450 carried as a known-bad explicit skip (see DECISIONS.md); confidence
  guardrail in schema for the dense pipeline.
- ~16 sampled frames => the heatmap/zone mapping is PROVEN to render correctly;
  the numbers are not real basketball stats (dense pipeline produces those later).
- Coach demo gate: `phase1/out/TEST1_phase1_demo.png`.

# LOCAL BALL MODEL — GPU-trained weights adoption diligence (2026-07-17)

Continuation of the open "ADOPTION diligence" item above: TEST_LOG.md's
TEST 2 already proved the HOSTED fine-tuned model (basketball-players-fy4c2
v25) reproduces both HARD ground-truth shots via the Roboflow API. Cost
problem: hosted = ~72k API calls/game, free tier won't scale. This task
trains OWN weights on a rented GPU (RunPod) so the same detector can run
locally at game scale with no API dependency.

## Plan
- [x] RunPod pod + persistent Network Volume set up, dataset (basketball-
      players-fy4c2 v25, same as hosted) downloaded to /workspace, YOLOv8m
      trained @ imgsz=1280, 100 epochs. Result: best.pt, overall mAP50
      0.877, but Ball class specifically weaker (recall 0.567, mAP50 0.642)
      than other classes -- flagged to user, not glossed over.
- [ ] Download best.pt from the pod to the local repo (models/).
- [ ] Small fix to spikes/ball_spike.py: it hardcodes COCO class 32
      ("sports ball") for the ball filter, which only matches STOCK
      models. Our fine-tuned model's own class list has "Ball" at a
      different index (0), so class filtering needs to key off the loaded
      model's own name->id mapping instead of the hardcoded COCO id, with
      the COCO id kept as fallback for stock model runs (no behavior
      change for existing stock usage).
- [ ] Run ball_spike.py on HARD with the local weights (same clip/spans
      TEST_LOG.md's TEST 2 already used), then the SAME existing chain
      already in the repo (ball_trajectory.py -> shot_attempts.py) to
      check: does it reproduce the 2 verified shots (356-381 near,
      1188-1211 far) and correctly reject the known deflections
      (418-438, 1217-1250)? Mirrors TEST 2's protocol exactly, just with
      local weights instead of the hosted API.
- [ ] Report comparison honestly (local vs hosted vs stock) — nothing
      adopted as the pipeline default until this is measured and the user
      reviews it, same discipline as every other TEST_LOG entry.

Scope guard: this only touches spikes/ (probe territory) + a local models/
folder; no edits to committed pipeline defaults (run_clip.py, tracking.py,
clip_config.py) regardless of outcome.

# SHIP HANDOFF — state + critical path to a working demo (written 2026-07-19)

DJ directive: SHIP / working demo is the priority. Footage labeling
continues in parallel as a background improvement loop (labels are
cumulative; model files never expire) — it must NOT block demo work.

## Exact current state (ball/shot layer)
- Weights OWNED, zero API dependency: models/ball_finetuned_v1.pt (yolov8m
  recipe) and v2.pt (yolov8l recipe). TEST 8/9/10 in TEST_LOG.md are the
  record. v2 = best single model: HARD 2/2 shots + 2/2 deflections BUT one
  DJ-refuted FALSE POSITIVE (401-415 "layup" = actually rebound->dish);
  TEST1 4/5 (missing shot B 315-327). v1 and v2 have COMPLEMENTARY blind
  spots (union = 5/5 TEST1). Hosted API no longer needed for anything.
- Milestone 2 labeling IN PROGRESS: Roboflow project "my-footage-ball"
  (DJ's workspace, class "ball"), 231 targeted frames uploaded, ~3/4
  labeled as of 2026-07-19. When DJ finishes: Generate dataset version in
  Roboflow -> download via RF_KEY -> merge with public dataset v25 ->
  train v3 FROM v2 weights on pod -> same TEST 8/9-style gates (HARD must
  stay 2/2+0FP, TEST1 target 5/5, rebound/dish must NOT re-claim).
- RunPod workflow (see memory infra_runpod_gpu.md): volume persists,
  pods are disposable; direct-TCP SSH does exec+scp, proxy SSH does not;
  pick LOWER-cuda pytorch template (host-driver trap, hit once).

## Critical path to demo (maps to ROADMAP Phases 5->6->7)
1. INTEGRATE the spike chain into run_clip as real stages (ball detect
   with own weights -> ball_trajectory -> shot_attempts -> shot_location
   -> shot_outcome). Everything exists in spikes/, proven by gates; the
   work is porting to per-clip stages + ClipConfig fields (weights path,
   hoop anchors) + tests. Ball layer sits BESIDE team_events (Principle
   4). Needs DJ adoption call: v2-alone (simplest, one FP class to
   watch), or v1+v2 union (5/5 but 2x compute) — recommend deciding
   AFTER v3's numbers land.
2. Phase 6 full-game scale: G5 is now ANSWERED by practice (rented GPU
   per game, RunPod); needs streaming frame access + batch runner +
   resumable stages. Demo can be a single clip, so 6 can start minimal.
3. Phase 7 worker loop: jobs table + subprocess run_clip + artifacts to
   storage -> the coach-facing demo (dad uploads, box score + shot chart
   come back). Phase 8 narrative comes only after real outputs.

## Open DJ items (non-blocking, keep visible)
- Eyeball: TEST1 84-98 candidate (from TEST 1); HARD 1352-13xx candidate
  (~45.1-45.9s, claimed by hosted AND v2); TEST1 shotA marginal pass.
- Player-tracker plan (below) — item 2 (ID-switch ground truth session)
  is the highest-value unblock; can ride along with any labeling session.
- HARD 7 queue items unsure; TEST1 2 refused resolutions; t49 splice.

# CV UPGRADES CHAT — HANDOFF (2026-07-24)

**SCOPE OF THIS CHAT: CV quality/accuracy work ONLY** (ball model,
tracker safety, new signals like scoreboard OCR). Shipping/integration
work (run_clip wiring, Phase 6 scaling, Phase 7 web worker) happens in
the OTHER chat — see "SHIP HANDOFF" section above for that thread's
state. Don't cross the streams; this chat measures and improves CV
components, the other chat ships them.

Full raw detail for everything below lives in TEST_LOG.md (TESTs 8-14).
This is the plain-English index into that.

## Where things stand, thread by thread

**1. Ball model (own-trained, zero API dependency):**
models/ball_finetuned_v3.pt is the strongest so far — TEST1 5/5 verified
shots (full recovery), HARD 2/2 shots + both deflections correctly
rejected. NOT adopted yet: it also confidently misreads a real
rebound-then-outlet-pass as a made layup, and this is PROVEN to not be
fixable by more ball labels (TEST 11/13 follow-ups) — a caught rebound
and a thrown pass both look physically identical to a ball-position-only
classifier. Two more real non-shots got DJ-confirmed and permanently
added to ground truth (an inbounds hold, a cross-court pass) —
spikes/local_weights_check.py's GROUND_TRUTH dict is the living record.
NEXT: this needs the player-signal cross-check (item below), not more
ball training.

**2. Player tracker safety (match_thresh 0.8->0.9, the big fragmentation
win from earlier testing):** TEST 12 DJ-confirmed a REAL identity switch
this setting causes (cross-team, id=17, with picture proof) — NOT
adopted. TEST 13 tried a jersey-color check at the moment a track
reattaches after going missing: correctly catches the known switch, but
of 8 flagged moments only 2 were real switches — the other 6 were mostly
a DIFFERENT bug this same probe accidentally found (the player detector
sometimes tracks the scoreboard graphic and referees as if they were
players). NEXT: filter those two junk-detection sources out BEFORE
re-scoring the color check — should raise precision a lot, cheap to try.
Tools built + reusable: phase2/bytetrack_mt09.yaml,
spikes/render_tracker_overlay.py (watchable video from any tracks.json),
spikes/tracker_color_reattach_check.py.

**3. Scoreboard OCR as a make/miss signal (DJ's idea, works):**
spikes/scoreboard_ocr_probe.py reads the scoreboard graphic independently
of player/ball detection (it sits at a fixed screen position all game).
VALIDATED on HARD (correctly reads "never changes" — matches known
ground truth) and TEST1 (reads a sane, monotonic score progression).
Two real bugs found + fixed this session: (a) a naive per-frame
confidence gate rejected genuinely-correct-but-blurry digit reads,
delaying TEST1's first reliable lock by 26+ seconds; (b) re-seeking the
video on every single frame instead of reading sequentially made a dense
pass absurdly slow (25+ min and still not done) — fixed. OPEN, UNSOLVED:
attributing a specific make to a specific one of TEST1's clustered early
shots needs sampling every few frames right after each shot, but real
player bodies passing through that corner of the screen for several
frames in a row can fool even a spread-out vote (not just single-frame
noise — proven with a still image, a player is standing right in/near
the graphic at the moment of the bad "5-0" read). Paused here for a
decision, not resolved. NEXT ACTION IDEAS (neither built): detect when a
player bbox overlaps the scoreboard region and skip those frames
entirely rather than voting through them; or accept coarser attribution
(which shot-CLUSTER scored, not which exact shot) as good enough.
Also found: TEST1's one clean coarse-pass event (27s) falls AFTER all 5
verified shots' windows -- it belongs to some OTHER, untracked shot in
the clip, not one of the five. Bonus, not yet acted on: the scoreboard's
fixed region is a known false-detection source for the player detector
(ties to item 2's bonus finding) -- once this is trustworthy it's a
free exclusion zone.

## Open items ranked (my recommendation, not decided)
1. Player-signal cross-check (shooter posture/hands-already-there vs a
   real release) -- the only thing that fixes the ball model's remaining
   false-positive class. Needs player labels/tracking data.
2. Filter scoreboard+ref detections before re-scoring the TEST 13 color
   check -- cheap, likely raises precision a lot, no new data needed.
3. Scoreboard-to-shot attribution refinement (the occlusion-skip idea) --
   real but lower priority than #1; make/miss is a nice-to-have, shot
   detection itself already works.
4. Player labeling ("my-footage-players", 280 frames) continues in the
   background at DJ's pace, no deadline -- feeds #1 directly.

# PLAYER-TRACKER PLAN OF RECORD (DJ-approved 2026-07-17)

Synthesis of TEST_LOG Tests 3-7 + DECISIONS §11/§22/§23. The tracker's
enemy is FRAGMENTATION (122 fragments/15s; clicks scale with fragments;
paint fragmentation is the layup-attribution blocker). Measured lever
scoreboard lives in TEST_LOG's cross-test summary. Standing constraint:
fragmentation metrics CANNOT see wrong-merges (ID switches) — a
confidently-wrong track is worse than a fragmented one; nothing adopts
on proxy metrics alone.

- [ ] 1. COMBO PROBE (cheap, read-only, next session): BoT-SORT + GMC
      sparseOptFlow + match_thresh 0.9 (and 0.85 as cautious middle) —
      Tests 5/6 measured these levers separately, never together. Same
      probe harness (spikes/reid_fragment_probe.py pattern), same TEST1
      span, vs 122/105.8 baseline. Output: ceiling of the current-
      detector stack.
- [ ] 2. ID-SWITCH GROUND TRUTH (the missing measurement, unblocks the
      biggest lever): DJ labels true identities over a short span
      (15-30s, "this track is #23...") ONCE -> permanent measuring stick
      (the tracker-side analog of the verified-shots list). Then any
      variant gets a REAL ID-switch count, resolving Test 6's mt=0.9
      safety caveat with data instead of fear. HIGHEST VALUE ITEM.
- [ ] 3. FINE-TUNED PLAYER DETECTOR (rides along with ball Milestone 2):
      in the same labeling session as ball-miss frames, label PLAYERS in
      crowded-paint frames (where ByteTrack shatters, §22: 10 fragments
      in one layup). Retrain -> re-run combo probe with fine-tuned
      Player class as tracking input + Test 4's ref exclusion (refs out
      of review queue, per-clip ROI mask retired).
- [ ] 4. COLOR-CONSTRAINED FRAGMENT STITCHING (queued idea, never run):
      appearance re-ID failed because TEAMMATES look identical, but the
      shipped color-tiebreak centroids reliably separate the two TEAMS
      -> team color as a stitch-candidate constraint (green never
      stitches to white) halves the search space, provably-safe
      direction. Measured probe, same protocol.
- [ ] 5. SAFETY DEBTS -> PREREQUISITES before adopting longer tracks
      (more damage per splice when tracks lengthen):
      (a) spread review-page crops across early/mid/late track life
          (t49-class splices hide in one-stretch crops);
      (b) Part-1/Part-2 cross-check gap (two labeling channels never
          validate each other — DECISIONS §4 known debt);
      (c) disputed-frame color extension (HARD id7's real 2.7s currently
          zeroed — DECISIONS §12 finding 1).
- [ ] 6. END-TO-END ADOPTION GATE (product numbers, not tracker
      numbers): clicks-per-game, % roster named, retro-recovery seconds,
      wrong-confirms = EXACTLY ZERO, on BOTH clips, winning stack vs
      committed baseline. Only then does bytetrack.yaml change.

Ordering rationale: 1 is nearly free and bounds the ceiling; 2 converts
"looks better" into "is safe" and unblocks mt=0.9; 3-4 stack on the new
GPU + labeling assets; 5 must land before any adoption that lengthens
tracks; 6 is the only gate that counts.

