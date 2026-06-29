# Generalization test — court-calibration engine across gyms

**Goal:** Map the *operating envelope* of the EXISTING engine (direct keyframe→keyframe
SIFT+RANSAC homographies → landmarks in one shared coord system → global
`least_squares` fit). DIAGNOSTIC ONLY. Do NOT change the engine/math/matcher/optimizer.
HARD.mp4 is the validated baseline (~0.95 ft, arcs glued). The per-clip LOG is the deliverable.

---

## Available clips (source: `C:\Users\djcha\Downloads\`, NOT `./clips` — no such dir)
- `EASY.mp4`, `MEDIUM.mp4`, `HARD.mp4` — same source family (difficulty = pan/zoom severity).
- `Milford vs Princeton - Tactical.mp4` (3.8 GB) — different gym. Old overlay exists in diagnostics/.
- `Milford ... Loveland ... Tactical.mp4` (3.1 GB, x2) — different gym. Old overlay exists.
- `Test1/2/3.mp4`, `1-Clip_at_20_20.mp4`, `1-Clip_at_32_32.mp4` — unidentified.
- **No "Mason" clip exists.** Need user to map real clips → the 3 difficulty categories.

## STEP 0 — De-hardcode HARD-specific values into a per-clip CONFIG  ✔ DONE
Moved into new `spikes/clips_config.py` (per-clip dict, `ACTIVE` selector, `active()`):
- [x] `VIDEO_PATH` — stage1 + stage2 now read `cfg.active()["video_path"]`
- [x] `KEYFRAMES`, `REFERENCE_POS` — stage2 reads from cfg
- [x] `EXCLUDE_REGIONS` (scorebug px) — stage1 + stage2 read from cfg (per-gym)
- [x] court dims — stage4 reads `cfg.active()["court"]` (length/width/lane/ft/circle)
- [x] `LANDMARKS` clicked-pixel dict — moved to cfg, stage2 reads it
- [x] output filenames — stage6 now prefixes with `cfg.ACTIVE` (HARD→same names; new clip→own files)
- [x] **Regression gate PASSED:** refactored HARD fit = mean 0.96 / max 1.75 ft, per-landmark
      residuals identical to committed baseline. Overlay logic untouched (only filename prefix).
- Note: stage2's display-only LANDMARK_TAGS palette + 94-ft ideal-draw left as-is (not used by
  the fit; stage4 COURT_MODEL is the real model). stage1 FRAME_PAIRS/VALIDATION_POINTS are
  standalone-diagnostic test data, not clip config — left in stage1.

## Per-clip workflow (mirror HARD exactly; nothing new)
For each selected clip, in increasing-difficulty order, ONE at a time:
- [ ] a. Set per-clip CONFIG (scorebug region, court dims, keyframes ~100–150 apart).
- [ ] b. Surface each keyframe so user confirms scorebug box + usability BEFORE clicking;
        user clicks landmarks via existing interactive tool.
- [ ] c. Run existing global `least_squares`; print per-landmark residuals worst→best + mean ft.
- [ ] d. Render court-map overlay (FT lines + arcs both ends); save for user to watch (the verdict).
- [ ] e. LOG: keyframe matching health, landmark assembly sanity, mean vs 0.95, overlay
        glue/slide/pop/shake + WHERE in pan it breaks + SPECIFIC cause. Honest uncertainty.
- [ ] f. Commit after the clip is done and user-confirmed.

## Clip order (run fully, one at a time — do NOT batch)
1. [ ] "Normal but different gym" (not low-texture, not odd scorebug) — does it generalize at all?
2. [ ] Mason / odd-scorebug clip — is scorebug config the ONLY HARD assumption? (no Mason clip → TBD)
3. [ ] Low-texture-floor clip — does SIFT survive? (riskiest; predicts auto-calib feasibility)

## Hard constraints (do NOT, mid-test)
- No LoFTR/SuperGlue/new matcher. No per-clip RANSAC/optimizer tuning. No auto court-line
  detection. Do NOT "fix" a broken clip — LOG the break, move on. Fixes decided by user after.

## Final deliverable
- [ ] Per-clip envelope table: holds / partial / breaks · cause · mean error.

## Decisions (confirmed by user)
1. Clip #1 = `Test1.mp4`. Run ONLY clip #1 fully for now; decide #2/#3 after.
2. Commit the confirmed HARD re-click cleanup as the validated baseline BEFORE refactoring. ✔
3. STEP 0 config shape = new `spikes/clips_config.py` (one per-clip dict; stages import active clip).

## Review
(to be filled in after work)
