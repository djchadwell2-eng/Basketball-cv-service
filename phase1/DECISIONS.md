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

## 3. Frame 450 fix — scheduled for the dense/continuous pipeline (NOT now)
- **Anchor each frame by a DIRECT match to its nearest keyframe**, instead of the
  accumulated ORB chain. 450's direct match is the best in the clip, so this makes
  it correct outright. This is the **chosen structural fix**.
- Temporal smoothing across neighbors is only the weaker symptom-level fallback;
  direct anchoring is the real fix.

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
