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

## STAGE 2 — TEAM_EVENT SCHEMA  (only after Stage 1 confirmed)
- [ ] Per-frame `team_event`: frame_index, timestamp, detections[] each with
      pixel_xy, court_xy (feet), state (confirmed/candidate/unknown — abstention is
      first-class). No identity. Team A/B left **unknown** unless trivially reliable.
- [ ] Write deterministically to JSON (no DB). Print schema + one example frame.
- [ ] STOP for user confirm.

## STAGE 3 — STATS + ZONE HEATMAPS (deterministic reads over Stage-2 events)
- [ ] Map court-feet -> standard zones for HS 84x50. Compute zone occupancy heatmap,
      (team) possession counts, pace — all deterministic.
- [ ] VALIDATION: render heatmap over a court diagram; print possession/pace. User
      eyeballs hot zones vs where play happened. STOP for user confirm.

## STAGE 4 — DEMO ARTIFACT (phase validation gate)
- [ ] Coach-legible output: court heatmap image + team stat summary. STOP for user.

## Open decisions to confirm before Stage 1 (see check-in)
1. **Which clip** is the Phase-1 test clip? (needs a validated homography) — recommend
   HARD.mp4 (canonical baseline). Alt: Test1.mp4 (tighter 0.25 ft, different gym).
2. **World B**: confirm we build Stage 1–4 fresh on the validated engine and leave the
   existing `src/`+`process_game.py` draft untouched (not deleted, not imported).

## Review
(to be filled in after work)
