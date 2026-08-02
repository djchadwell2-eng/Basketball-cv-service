"""RENDER AN OVERLAY DJ CAN ACTUALLY WATCH -- Part 4 step 6: "No number
substitutes." A good ft-error number already passed (TEST 36 also had numbers;
they just happened to be honest about failing). This is the real check.

SCOPED ON PURPOSE: rendering the whole 171,120-frame game frame-by-frame (the
way spikes/stage6_arc_overlay.py does for the much-shorter HARD.mp4) would take
hours. Instead this grabs ~10 REAL seconds of actual footage immediately after
each of the 4 clicked spots -- camera motion, players, the works -- and draws
the court + HS 3-point line on top using the calibration DJ's clicks just
produced. Uses cv2 seeking (not the frame-exact sequential read calibration
needs) because a preview being off by a couple of frames doesn't matter for a
human eyeball check.

HONEST SCOPE: this shows the calibration holds up NEAR each of the 4 clicked
spots. It does NOT show the 70-90 minute gaps BETWEEN them, because that much
footage cannot be reviewed in one sitting. If DJ wants a stretch of real game
action checked, tell me which minute and this can be pointed there directly.

Usage:  .venv/Scripts/python.exe spikes/render_chain_overlay_sample.py [CLIP_NAME]
        (default FULL_GAME_CHAIN; the second gym is FULL_GAME2)
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_ROOT, "phase1"))

import clips_config                                               # noqa: E402
clips_config.ACTIVE = sys.argv[1] if len(sys.argv) > 1 else "FULL_GAME_CHAIN"

import stage2_multikeyframe as s2                                 # noqa: E402
import stage4_courtmap as s4                                      # noqa: E402
import stage5_courtmap as s5                                      # noqa: E402
import stage6_arc_overlay as s6                                   # noqa: E402
import refit_keyframes as rk                                      # noqa: E402

NFEAT, RATIO, RANSAC_PX, MIN_INLIERS = 1500, 0.75, 3.0, 30
SECONDS_PER_SPOT = 10
OUT = os.path.join(_HERE, "out", f"{clips_config.ACTIVE}_sample_overlay.mp4")
W, H = 960, 540


def main():
    print("Loading DJ's refit calibration (from cache -- already computed)...")
    KF, ref_pos, Hs_opt, L_opt, tags = rk.refit(use_cache=True)
    H_court, per_err, mean_err, max_err = s4.compute_H_court(L_opt, tags)
    Hcourt_inv = np.linalg.inv(H_court)
    print(f"  landmark court-fit: mean={mean_err:.2f} ft max={max_err:.2f} ft")

    sift = cv2.SIFT_create(nfeatures=NFEAT)
    print(f"Extracting the {len(KF)} keyframes for matching (single accurate pass)...")
    kf_frames = s2.extract_frames(s2.VIDEO_PATH, KF)
    kf_db = []
    for pos, k in enumerate(KF):
        kp, des = s5.sift_of(kf_frames[k], sift)
        m = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        m.add([des]); m.train()
        kf_db.append((pos, kp, m))

    def court_to_frame(Hfk, pos):
        """court feet -> this frame's px, or None if the transform is degenerate.

        RANSAC can return a technically-non-None homography that is singular or
        near-singular (all its correspondences nearly collinear, or a wild
        extrapolation). Inverting that raises LinAlgError and, before this
        guard, killed the ENTIRE render on one bad frame -- gym #2 died 300
        frames into spot 2 of 7 after spot 1 rendered perfectly. A single
        unusable frame must degrade to "NO MATCH" for that frame only, exactly
        like a failed match already does.
        """
        T = Hs_opt[pos] @ Hfk
        if not np.all(np.isfinite(T)) or abs(np.linalg.det(T)) < 1e-12:
            return None
        try:
            return s5.signfix(np.linalg.inv(T) @ Hcourt_inv)
        except np.linalg.LinAlgError:
            return None

    cap = cv2.VideoCapture(s2.VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(SECONDS_PER_SPOT * fps)
    writer = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    for pos, k in enumerate(KF):
        print(f"\n-- spot {pos+1}/{len(KF)}: near frame {k} "
              f"({k/fps/60:.1f} min) --")
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        rendered = no_match = 0
        for i in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                break
            kp_f, des_f = s5.sift_of(frame, sift)
            best = None
            if des_f is not None and len(kp_f) >= 8:
                for (kpos, kp_k, matcher) in kf_db:
                    knn = matcher.knnMatch(des_f, k=2)
                    good = [a for a, b in (p for p in knn if len(p) == 2)
                            if a.distance < RATIO * b.distance]
                    if len(good) < 8:
                        continue
                    src = np.float32([kp_f[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
                    dst = np.float32([kp_k[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
                    Hfk, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
                    if Hfk is not None:
                        inl = int(mask.sum())
                        if best is None or inl > best[0]:
                            best = (inl, kpos, Hfk)
            M = None
            if best is not None and best[0] >= MIN_INLIERS:
                _, kpos, Hfk = best
                M = court_to_frame(Hfk, kpos)
            if M is not None:
                s4.draw_court(frame, M)
                s6.draw_arc(frame, M)
                rendered += 1
                txt = f"SPOT {pos+1} (kf {k})  t={k/fps+i/fps:6.1f}s  matched"
            else:
                no_match += 1
                txt = f"SPOT {pos+1} (kf {k})  t={k/fps+i/fps:6.1f}s  NO MATCH"
            cv2.putText(frame, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 2)
            writer.write(cv2.resize(frame, (W, H)))
        print(f"  rendered={rendered} no_match={no_match} / {n_frames}")
    writer.release()
    cap.release()
    print(f"\nwrote {OUT}")
    print(f"HONEST SCOPE: {SECONDS_PER_SPOT}s samples at each of the {len(KF)} "
          f"clicked spots only -- not the whole game, not the gaps between spots.")


if __name__ == "__main__":
    main()
