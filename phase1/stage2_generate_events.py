"""STAGE 2b -- generate team_events from Stage 1, persist as JSON.

Reuses Stage 1's on-court classification UNCHANGED (same homography, same
MARGIN_FT, same off-screen/horizon guards) to get the on-court bodies per frame,
then wraps each frame in the Stage-2a schema and writes one JSON document.

It ALSO attaches the per-frame homography_confidence (the frame-450 guardrail):
the inliers + mean reproj of a DIRECT SIFT match of the frame to its NEAREST
keyframe -- exactly the direct-anchor signal the chosen 450 fix relies on
(see phase1/DECISIONS.md). state='low_confidence' flags a frame whose calibration
shouldn't be trusted; 'ok' otherwise.

Computes NO stats (that's Stage 3). Touches no mask/margin/homography.
"""

import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import stage1_court_roi as st          # reuse Stage 1 classification (unchanged)
import team_event_schema as schema     # Stage 2a records + JSON
import stage1_keyframe_match as s1     # validated SIFT matcher (spikes/)

# Confidence thresholds for the guardrail (kept simple, in one place).
INLIER_MIN = 150         # below this -> low_confidence
REPROJ_MAX_PX = 3.0      # above this -> low_confidence

OUT_JSON = os.path.join(_HERE, "out", f"{st.s2.cfg.ACTIVE}_team_events.json")
KEYFRAMES = st.s2.KEYFRAMES
REGIONS = st.s2.EXCLUDE_REGIONS


def nearest_keyframe(f):
    kf = np.array(KEYFRAMES)
    return int(kf[np.argmin(np.abs(kf - f))])


def direct_match_confidence(f_img, k_img):
    """inliers + mean reproj (px) of a DIRECT SIFT match frame -> nearest keyframe."""
    cv2.setRNGSeed(0)                                   # reproducible
    kp_k, kp_f, good = s1.detect_and_match(k_img, f_img, 0.75, REGIONS)
    H, mask, inliers = s1.estimate_homography(kp_k, kp_f, good, 3.0)
    if H is None or inliers == 0:
        return 0, float("inf")
    im = [m for m, keep in zip(good, mask.ravel()) if keep]
    pf = np.float32([kp_f[m.trainIdx].pt for m in im]).reshape(-1, 1, 2)
    pk = np.float32([kp_k[m.queryIdx].pt for m in im])
    proj = cv2.perspectiveTransform(pf, H).reshape(-1, 2)
    reproj = float(np.linalg.norm(proj - pk, axis=1).mean())
    return inliers, reproj


def main():
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    cv2.setRNGSeed(0)

    # Stage 1's per-frame pixel->court-feet (ORB-chain homography) + fps.
    H_court, frame_to_ref, fps, total = st.build_pixel_to_feet()

    # Frame images for sampled frames AND keyframes (single pass) -- used for both
    # YOLO detection and the direct-match confidence.
    need = sorted(set(st.SAMPLE_FRAMES) | set(KEYFRAMES))
    frames = st.s2.extract_frames(st.s2.VIDEO_PATH, need)

    model = st.YOLO(st.MODEL_NAME)
    events = []
    print(f"\nGenerating team_events for {len(st.SAMPLE_FRAMES)} sampled frames...")
    for f in st.SAMPLE_FRAMES:
        frame = frames[f]
        frame_h = frame.shape[0]

        # --- Stage 1 classification (reused unchanged) ---
        T = frame_to_ref(f)
        p2f = H_court @ T
        feet2px = np.linalg.inv(p2f)
        ctr_px = feet2px @ np.array([st.COURT_LEN / 2.0, st.COURT_WID / 2.0, 1.0])
        ctr_px = ctr_px / ctr_px[2]
        ref_w = float((p2f @ ctr_px)[2])

        res = model.predict(frame, imgsz=st.IMG_SIZE, classes=[st.PERSON_CLASS],
                            verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []

        dets = []
        for box in boxes:
            fx, fy = st.feet_pixel(box)
            if fy >= frame_h - st.FRAME_EDGE_PX:          # off-screen feet -> drop
                continue
            cx, cy, w = st.pixel_to_feet(p2f, fx, fy)
            if not st.on_court(cx, cy, w, ref_w):
                continue
            dets.append(schema.Detection(
                pixel=(round(float(fx), 1), round(float(fy), 1)),
                court_feet=(round(cx, 1), round(cy, 1))))   # identity/team default unknown

        # Frame-local CANONICAL SORT by position so the JSON is deterministic.
        # This is NOT identity: it's recomputed per frame and never linked across
        # frames (detection N in frame A != detection N in frame B).
        dets.sort(key=lambda d: (d.court_feet[0], d.court_feet[1]))

        # --- homography_confidence guardrail (direct nearest-keyframe match) ---
        k = nearest_keyframe(f)
        inliers, reproj = direct_match_confidence(frame, frames[k])
        state = "ok" if (inliers >= INLIER_MIN and reproj <= REPROJ_MAX_PX) \
            else "low_confidence"
        conf = schema.HomographyConfidence(
            inliers=int(inliers), reproj_px=round(reproj, 2),
            state=state, nearest_keyframe=k)

        events.append(schema.TeamEvent(
            frame_index=f, timestamp_s=round(f / fps, 3),
            homography_confidence=conf, detections=dets))
        print(f"  f={f:>4}  on_court={len(dets):>2}  "
              f"conf(inl={inliers:>4}, reproj={reproj:.2f}px, {state}, kf={k})")

    schema.write_events(events, OUT_JSON, clip=st.s2.cfg.ACTIVE, fps=fps)
    print(f"\nwrote {len(events)} team_events -> {OUT_JSON}")


if __name__ == "__main__":
    main()
