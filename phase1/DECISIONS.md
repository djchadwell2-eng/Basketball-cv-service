# Phase 1 — settled architecture decisions

Plain-English record so a future session inherits these without re-deriving them.
These are DECIDED. Change them only with a new explicit decision.

## 1. Court ROI / the "13-person rule" — DONE (Stage 1, signed off)
- On-court vs off-court is decided by the **validated homography + the court
  polygon** (84×50 HS court in feet), not by classifying people.
- `MARGIN_FT = 1.5` ft of slack outside the painted lines counts as on-court.
- **Off-screen / frame-clipped feet are dropped**: if a detection's box bottom
  touches the frame edge, its feet are off-screen so the ground-contact point is
  fake — discard it (this removed the foreground spectators / scorer's table).
- **Horizon guard**: a detection whose feet map *behind the vanishing line*
  (homogeneous depth `w` flips sign) is rejected — kills the "past the horizon"
  wrap where bleacher bodies get fake in-bounds court coords.
- **Refs are counted as on-court — ACCEPTED Phase 1 behavior.** Removing refs
  needs player identity, which Phase 1 forbids. So the realistic *clean* on-court
  count is **~10 players + 2–3 refs ≈ 12–13**, NOT 10. A count of ~13 is healthy;
  the goal was excluding crowd/bench, which is done.

## 2. Frame 450 root cause — NOT a mask defect, NOT a wide-view weakness
- Frame 450's **direct** SIFT match to the reference keyframe is the *strongest
  in the clip* (2568 inliers, ratio 0.904, 0.61 px). The imagery is trivially
  matchable. The genuinely wider frames (480–570) also match healthily, and 480
  (wider than 450) was visually correct — so wide-ness does not predict failure.
- The live mask gets each frame's homography from an **accumulated ORB
  consecutive-frame chain** (frame 0 → … → frame, then nearest-keyframe anchored).
  That chain drifted/glitched between keyframe 420 and frame 450.
- **Cause = the chained-transform mechanism**, the same failure class the static
  multi-keyframe calibration rebuild already fixed. It is transient/localized, not
  structural to wide views.

## 3. Frame 450 fix — DONE (direct nearest-keyframe anchoring)
- **Anchor each frame by a DIRECT match to its nearest keyframe**, instead of the
  accumulated ORB chain. Implemented in `stage1_court_roi.build_court_anchor()`.
  Verified: old chain was off >40px on 74% of frames; new live fits every swept
  frame at 0.00-0.72px reproj. Frame 450 now correct (anchors to kf420).
- Temporal smoothing across neighbors was the weaker symptom-level fallback;
  direct anchoring was the real fix. The stage3 frame-450 skip is now redundant
  but intentionally LEFT in place (remove only when convenient).

## 4. Camera-track confidence guardrail — abstention applied to calibration
- The dense pipeline must carry a **per-frame homography quality signal** (inliers
  / reproj of the **direct anchor match**) so low-confidence frames are **FLAGGED,
  not silently fed into stats**.
- Made structural now in the Stage 2 schema as `homography_confidence`
  (state `ok` / `low_confidence`). NOTE: measured from the direct nearest-keyframe
  match — exactly the signal the chosen fix (decision 3) anchors on. On today's
  ORB-chain Stage 1 it reports frame *matchability*; once direct anchoring lands,
  the same number gates the actual positioning homography.

## 5. Schema decoupling principle — the spine is identity-free
- `team_events` (the per-frame on-court positions) **never reference identity**:
  no track_id, no player number, no cross-frame link, no team guess.
- Identity layers (`tracks`, `player_events`) **bolt on LATER as new top-level
  keys with zero rework** to team_events.
- **Deterministic stats never pass through an LLM.** No LLM anywhere in Phase 1.

## 6. Keyframe-handoff pop — FIXED (keyframe-consistency re-fit)
**DONE.** Implemented in `phase1/refit_keyframes.py`: added dense adjacent-keyframe
SIFT correspondences to the same global least_squares, so keyframes agree across
their overlap (one shared court system), warm-started from the existing fit and
wired into `build_court_anchor` via `refit()` (cached .npz). Results:
- keyframe mutual-consistency: mean 8.1px -> 0.6px, max 41.3px -> 2.7px
- handoff pops: 120<->220 515px -> 2.6px; all 5 handoffs now <=8.7px max
- no accuracy tradeoff: landmark court-fit improved 0.25/0.54ft -> 0.14/0.37ft;
  per-frame anchor reproj still 0.00-0.72px (0 frames >5px)
- the redundant frame-450 skip in stage3 was REMOVED (450 now a normal trusted
  frame; removing it shifts zone shares <1%, mean on-court unchanged at 13.0)
The original problem statement + rejected options are kept below for the record.

### (original entry) Keyframe-handoff pop — fix scheduled BEFORE Phase 2
- **Issue (continuity, not accuracy):** direct anchoring (decision 3) made each
  frame accurate *on its own*, but the keyframes are not mutually consistent. When
  a mid-interval frame switches which keyframe it anchors to, the court polygon
  jumps by however much those two keyframes disagree. MEASURED: 5 anchor-handoff
  frames pop 37-515px (worst: 120<->220 at frame 171). Per-frame matching is fine;
  the keyframes disagree with *each other*.
- **Same class as an already-solved problem:** the keyframe-level version of the
  landmark-consistency issue fixed in calibration Stage 3, where a global
  least_squares tightened shared-landmark consistency (~7px -> ~3.4px). Locally
  accurate, globally inconsistent anchors — same shape, one level up.
- **CHOSEN FIX:** re-fit the keyframes for mutual consistency (the same global-
  optimization move that worked at the landmark layer). Fixing the *cause*
  (inconsistent keyframes) improves all 5 handoffs at once.
- **REJECTED — blending across handoffs:** hides the pop by smearing a 515px jump
  into a gradual 515px slide; the court is still wrong mid-transition. Confident-
  wrong failure mode. **REJECTED — adding keyframes near the worst handoff:** a
  band-aid that shortens the interval but leaves the inconsistency intact (OK as a
  targeted patch, wrong as the general fix).
- **Why it doesn't block the demo:** affects 5 of 46 frames; matters for video
  overlay (jarring) and per-frame Phase 2 work (tracking hates discontinuities),
  but does NOT meaningfully distort the static heatmap (an aggregate over frames).
- **Validation gate for the fix:** after re-fitting, re-measure the 5 handoff jumps
  (should drop to a few px), then watch the full-pan overlay — court stays glued
  AND moves smoothly across the seams, no visible pops.
- **SEQUENCE:** (1) demo Phase 1 to dad -> (2) keyframe-consistency re-fit ->
  (3) Phase 2. Demo first (his reaction may reorder the roadmap); the re-fit is
  required before Phase 2 regardless.
