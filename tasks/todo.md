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
- [ ] roster.py (numbers + loose team color; hand-verified seed labels t17=13,t6=5)
- [ ] ocr_reader.py: pluggable read_jersey(crop, roster) closed-set; ONE constant
      OCR_CONFIRM_THRESHOLD (autonomy dial), easyocr lazy-imported.
- [ ] identity.py: implement promote_via_second_signal -> 3 outcomes (AGREE->confirm
      provenance=second_signal; DISAGREE->flag swap; NO-READ->stay candidate).
- [ ] stage6 driver: temporal accumulation per candidate across window; measure
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
