"""Stage 4 coverage refinement: anchor each frame to its BEST-MATCHING keyframe
(by SIFT), not the nearest by frame index.

Finding behind this: HARD uses ONE zoom level and the existing 7 keyframes
(600-1200) already sweep every view the camera uses (left, center, right). A
far-right frame at 2600 SIFT-matches keyframe 1200 at ~0.8 inlier ratio even
though it's 1400 frames away. So the Stage-4 "disappear / drift in far regions"
was purely the anchoring rule (nearest-by-frame -> long tracking hop). Here each
frame is matched DIRECTLY to its best-overlapping keyframe (Stage-1 accuracy),
then composed:  court feet -> frame px = (900->f) @ inv(H_court), where
    frame_f -> 900 = Hs_opt[best_kf] @ (frame_f -> best_kf)

No new keyframes, no new clicks, H_court and the optimization unchanged. Reuses
Stage 4. Deterministic.
"""

import os
import sys
import numpy as np
import cv2

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))
import stage1_keyframe_match as s1
import stage2_multikeyframe as s2
import stage4_courtmap as s4               # reuse optimization, H_court, drawing

NFEAT = 1500
RATIO = 0.75
RANSAC_PX = 3.0
MIN_INLIERS = 30                            # below this -> no confident match
OUT_DIR = s2.OUT_DIR
VIDEO = s2.VIDEO_PATH


def sift_of(img, sift):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = s1.build_sift_mask(img.shape, s2.EXCLUDE_REGIONS)   # drop scorebug
    return sift.detectAndCompute(gray, mask)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cv2.setRNGSeed(0)

    print("Recovering optimized transforms + H_court (Stage 3/4)...")
    KF, ref_pos, Hs_opt, L_opt, tags = s4.run_optimization()
    H_court, per_err, mean_err, max_err = s4.compute_H_court(L_opt, tags)
    Hcourt_inv = np.linalg.inv(H_court)
    print(f"H_court residual mean={mean_err:.2f} ft max={max_err:.2f} ft")

    # Precompute SIFT for each keyframe + a FLANN matcher trained on it.
    sift = cv2.SIFT_create(nfeatures=NFEAT)
    kf_frames = s2.extract_frames(VIDEO, KF)
    flann_params = dict(algorithm=1, trees=5)     # KDTREE for float SIFT
    kf_db = []                                    # (pos, kp, matcher)
    for pos, k in enumerate(KF):
        kp, des = sift_of(kf_frames[k], sift)
        m = cv2.FlannBasedMatcher(flann_params, dict(checks=50))
        m.add([des])
        m.train()
        kf_db.append((pos, kp, m))

    cap = cv2.VideoCapture(VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_path = os.path.join(OUT_DIR, "HARD_stage4_courtmap_bestmatch.mp4")
    W, H = 960, 540
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    saved_stills = []
    best_ratios = []
    no_match = 0
    for f in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        kp_f, des_f = sift_of(frame, sift)
        best = None                               # (inliers, pos, H_f_to_kf, ratio)
        if des_f is not None and len(kp_f) >= 8:
            for (pos, kp_k, matcher) in kf_db:
                knn = matcher.knnMatch(des_f, k=2)
                good = [a for a, b in (p for p in knn if len(p) == 2)
                        if a.distance < RATIO * b.distance]
                if len(good) < 8:
                    continue
                # src = frame pts, dst = keyframe pts -> H maps frame_f -> keyframe
                src = np.float32([kp_f[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
                dst = np.float32([kp_k[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
                Hfk, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
                if Hfk is None:
                    continue
                inl = int(mask.sum())
                if best is None or inl > best[0]:
                    best = (inl, pos, Hfk, inl / len(good))

        if best is not None and best[0] >= MIN_INLIERS:
            _, pos, Hfk, ratio = best
            best_ratios.append(ratio)
            T = Hs_opt[pos] @ Hfk                  # frame_f -> 900
            M = np.linalg.inv(T) @ Hcourt_inv      # court feet -> frame_f px
            M = M / M[2, 2]
            s4.draw_court(frame, M)
            tag = f"f={f} t={f/fps:04.1f}s  best_kf={KF[pos]} ratio={ratio:.2f}"
        else:
            no_match += 1
            tag = f"f={f}  NO MATCH"
        cv2.putText(frame, tag, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2)

        if f in s4.STILL_FRAMES:
            sp = os.path.join(OUT_DIR, f"HARD_stage4bm_still_{f:05d}.jpg")
            cv2.imwrite(sp, cv2.resize(frame, (W, H)))
            saved_stills.append(sp)
        writer.write(cv2.resize(frame, (W, H)))
        if f % 200 == 0:
            print(f"  ...{f}/{total}  (no_match so far: {no_match})", flush=True)
    writer.release()
    cap.release()

    br = np.array(best_ratios) if best_ratios else np.array([0.0])
    lines = []
    lines.append("Stage 4 best-match overlay -- per-frame anchoring to best keyframe")
    lines.append(f"keyframes (unchanged): {KF}")
    lines.append(f"H_court residual: mean={mean_err:.2f} ft, max={max_err:.2f} ft "
                 "(unchanged -- same landmarks/optimization)")
    lines.append(f"frames: {total}   frames with NO confident match: {no_match} "
                 f"({100*no_match/max(total,1):.1f}%)")
    lines.append(f"per-frame best-keyframe inlier ratio: min={br.min():.2f}  "
                 f"mean={br.mean():.2f}  (high+everywhere => full coverage)")
    summary = "\n".join(lines)
    print("\n================ SUMMARY ================")
    print(summary)
    print("=========================================")
    sp = os.path.join(OUT_DIR, "stage4_bestmatch_summary.txt")
    with open(sp, "w", encoding="utf-8") as fh:
        fh.write(summary + "\n")
    print(f"saved summary: {sp}")
    print(f"saved overlay: {out_path}")
    for s in saved_stills:
        print(f"saved still: {s}")


if __name__ == "__main__":
    main()
