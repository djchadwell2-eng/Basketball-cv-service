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
- [ ] Compare detection rate on frames 0-450: existing 1280 log vs new
      1920 log. Did coverage rise from 32%? Did the 2 known shots gain
      more of their flight (closing the truncation gap that forced the
      arrival-extrapolation in DECISIONS 19)?
- [ ] Decide with user: adopt 1920 (full-clip rerun + re-run pipeline),
      climb to option 2 (tiling), or accept current detection as the
      footage-quality ceiling. DECISIONS 20 with the measurement.

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
