"""STAGE 4 -- full-pan court overlay on the REFIT homography (eyeball gate).

Draws the court model on frames across the whole pan using the direct-anchor +
consistency-refit homography, so the user can confirm by eye: court stays GLUED
and moves SMOOTHLY across the keyframe seams (no pops). Court lines only (no YOLO)
so it is fast. Dense every STEP frames, PLUS every frame in a window around each
of the 5 anchor-handoff seams (where pops used to happen).
"""

import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import stage1_court_roi as st          # build_court_anchor (refit) + court draw
import stage4_courtmap as s4

OUT_DIR = os.path.join(_HERE, "out")
STEP = 12                               # sparse sampling across the pan
HANDOFFS = [171, 271, 371, 461, 541]    # anchor-switch frames; render neighbours
WIN = 3                                 # +/- frames rendered around each handoff (seam)
DISP_W, DISP_H = 1280, 720


def frames_to_render():
    fs = set(range(120, 581, STEP))
    for h in HANDOFFS:
        fs |= set(range(h - WIN, h + WIN + 1))
    return sorted(f for f in fs if 120 <= f <= 580)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    H_court, anchor, fps, total = st.build_court_anchor()

    want = frames_to_render()
    frames = st.s2.extract_frames(st.s2.VIDEO_PATH, want)

    writer = cv2.VideoWriter(
        os.path.join(OUT_DIR, f"{st.s2.cfg.ACTIVE}_stage4_overlay.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (DISP_W, DISP_H))

    prev_kf = None
    print(f"Rendering {len(want)} frames (court overlay on refit homography)...")
    for f in want:
        frame = frames[f]
        T, inliers, reproj, kf = anchor(f, frame)
        feet2px = np.linalg.inv(H_court @ T)
        s4.draw_court(frame, feet2px)                      # court model -> this frame
        seam = "  <-- SEAM" if kf != prev_kf and prev_kf is not None else ""
        prev_kf = kf
        cv2.putText(frame, f"f={f}  anchor_kf={kf}  inl={inliers}  reproj={reproj:.2f}px",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        out = cv2.resize(frame, (DISP_W, DISP_H))
        writer.write(out)
        if f in HANDOFFS:
            cv2.imwrite(os.path.join(OUT_DIR, f"{st.s2.cfg.ACTIVE}_stage4_handoff_{f}.jpg"), out)
        if seam:
            print(f"  f={f:>4} anchor_kf={kf}{seam}")
    writer.release()
    print(f"\nsaved overlay video + handoff stills in {OUT_DIR}")


if __name__ == "__main__":
    main()
