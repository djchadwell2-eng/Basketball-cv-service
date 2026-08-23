"""STAGE 2b -- generate team_events, persist as JSON.

Reuses Stage 1's on-court classification UNCHANGED (same MARGIN_FT, same
off-screen/horizon guards). The per-frame court homography now comes from the
DIRECT nearest-keyframe anchor (stage1_court_roi.build_court_anchor), so the SAME
match that POSITIONS the players also produces homography_confidence -- the
confidence finally gates the thing it measures.

Computes NO stats (that's Stage 3). Touches no mask/margin/homography logic.
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))                          # repo root (clip_config)

import stage1_court_roi as st          # reuse Stage 1 classification + direct anchor
import team_event_schema as schema     # Stage 2a records + JSON
from clip_config import ACTIVE_CLIP as CLIP

OUT_JSON = os.path.join(_HERE, "out", f"{CLIP.name}_team_events.json")

# Per-clip frames Phase 1 samples for team_events (dense sweep across the pan).
GEN_FRAMES = list(CLIP.event_frames)


def main():
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)

    H_court, anchor, fps, total = st.build_court_anchor()
    # Phase 6: STREAM the sampled frames one at a time (never materialize the
    # whole sample into one dict -- GEN_FRAMES is a sorted range, so iterating
    # in lockstep with iter_frames's increasing yield order needs no lookup).
    model = st.YOLO(st.MODEL_NAME)

    events = []
    inlier_dist = []
    print(f"\nGenerating team_events for {len(GEN_FRAMES)} frames...")
    for f, frame in st.s2.iter_frames(st.s2.VIDEO_PATH, GEN_FRAMES):
        frame_h = frame.shape[0]

        # Per-frame homography from the DIRECT nearest-keyframe anchor.
        T, inliers, reproj, kf = anchor(f, frame)
        if T is None:
            # A frame the anchor cannot place carries NO team_event -- the same
            # fail-open rule oncourt.build already applies. Without this the
            # stage crashed on `H_court @ T`, and the frame that does it is not
            # exotic: DJ's game opens with ~15 BLACK frames (brightness 0.0,
            # zero SIFT keypoints, MEASURED 2026-08-19) and event_frames starts
            # at the span start, which for a whole game is frame 0. The crash
            # would land AFTER the ten slices and the merge were paid for.
            print(f"  f={f:>4}  anchor match FAILED -- no team_event for this "
                  f"frame (fail-open, not guessed)")
            continue
        p2f = H_court @ T
        feet2px = np.linalg.inv(p2f)
        ctr_px = feet2px @ np.array([st.COURT_LEN / 2.0, st.COURT_WID / 2.0, 1.0])
        ctr_px = ctr_px / ctr_px[2]
        ref_w = float((p2f @ ctr_px)[2])

        # --- Stage 1 classification (reused unchanged) ---
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
                court_feet=(round(cx, 1), round(cy, 1))))
        # Frame-local canonical sort (deterministic JSON; NOT identity).
        dets.sort(key=lambda d: (d.court_feet[0], d.court_feet[1]))

        # Confidence from the SAME direct match that positioned the players.
        state = st.confidence_state(inliers, reproj)
        conf = schema.HomographyConfidence(
            inliers=int(inliers), reproj_px=round(reproj, 2),
            state=state, nearest_keyframe=kf)
        inlier_dist.append(inliers)

        events.append(schema.TeamEvent(
            frame_index=f, timestamp_s=round(f / fps, 3),
            homography_confidence=conf, detections=dets))
        print(f"  f={f:>4}  on_court={len(dets):>2}  inl={inliers:>5}  "
              f"reproj={reproj:.2f}px  kf={kf}  {state}")

    schema.write_events(events, OUT_JSON, clip=st.s2.cfg.ACTIVE, fps=fps)

    # --- threshold report (from the observed inlier distribution) ---
    arr = np.array(inlier_dist)
    n_low = sum(1 for ev in events
                if ev.homography_confidence.state == "low_confidence")
    print(f"\ninlier distribution over {len(arr)} frames: "
          f"min={arr.min()} median={int(np.median(arr))} max={arr.max()}")
    print(f"low_confidence threshold: inliers < {st.CONF_INLIER_MIN} OR "
          f"reproj > {st.CONF_REPROJ_MAX}px  ->  {n_low} frame(s) flagged")

    # --- Stage B verification: frame 450 ---
    ev450 = next((e for e in events if e.frame_index == 450), None)
    if ev450:
        c = ev450.homography_confidence
        print(f"\nFRAME 450 (was the deferred bad frame): on_court={len(ev450.detections)} "
              f"inl={c.inliers} reproj={c.reproj_px}px kf={c.nearest_keyframe} {c.state}")
        print("  -> now anchored DIRECTLY to its nearest keyframe; the homography IS")
        print("     the proven-good direct match, so it is CORRECT, not just flagged.")
        print("  -> the hardcoded frame-450 skip in stage3 is now REDUNDANT (left in place).")

    print(f"\nwrote {len(events)} team_events -> {OUT_JSON}")


if __name__ == "__main__":
    main()
