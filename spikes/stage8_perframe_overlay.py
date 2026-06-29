"""Stage 8: tight AND smooth court overlay via a PER-FRAME pooled homography.

Stage 7 (per-view) was tight (~0.28 ft) but POPPED at keyframe switches because
each keyframe's homography was fit independently. Here we never switch keyframes:
for EACH frame we pool the landmarks from ALL keyframes that match it (each
transformed into this frame's pixels via its SIFT match), and fit ONE court->frame
homography (RANSAC, sign-fixed) from that pool. As the frame moves, keyframes
fade in/out of the pool gradually, so the fit changes smoothly -> no pops -- while
staying tight because it uses the local landmarks directly.

Reuses Stages 1-3 (optimized landmark set + keyframe SIFT) and the sign-fix and
the HS court model + 3-point arc. Stages 1-7 left intact. Deterministic.
"""

import os
import sys
import numpy as np
import cv2

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))
import stage2_multikeyframe as s2
import stage4_courtmap as s4
import stage5_courtmap as s5          # sift_of, signfix
import stage6_arc_overlay as s6       # draw_arc

NFEAT, RATIO, RANSAC_PX = 1500, 0.75, 3.0
MIN_POOL_INLIERS = 20                 # a keyframe joins the pool above this
MIN_CORR = 6                          # need this many pooled correspondences to fit
OUT_DIR, VIDEO = s2.OUT_DIR, s2.VIDEO_PATH
STILLS = [650, 800, 820, 900, 1100, 1500, 2000, 2700]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cv2.setRNGSeed(0)
    print("Recovering keyframes + landmarks (Stages 1-3)...")
    KF, ref_pos, Hs_opt, L_opt, tags = s4.run_optimization()
    # landmark pixel lists per keyframe (clicked), with court-feet targets
    kf_lms = {pos: [(s4.COURT_MODEL[t], (x, y)) for (t, x, y) in s2.LANDMARKS[KF[pos]]]
              for pos in range(len(KF))}

    sift = cv2.SIFT_create(nfeatures=NFEAT)
    kf_frames = s2.extract_frames(VIDEO, KF)
    kf_db = []
    for pos, k in enumerate(KF):
        kp, des = s5.sift_of(kf_frames[k], sift)
        m = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        m.add([des]); m.train()
        kf_db.append((pos, kp, m))

    cap = cv2.VideoCapture(VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_path = os.path.join(OUT_DIR, "HARD_perframe_overlay.mp4")
    W, H = 960, 540
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    rendered = no_render = 0
    pool_sizes, inlier_counts = [], []
    prev_center = None
    center_jumps = []        # frame-to-frame move of court point (42,25); pops = spikes
    stills = []
    for f in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        kp_f, des_f = s5.sift_of(frame, sift)
        court_pts, frame_pts, n_kf = [], [], 0
        if des_f is not None and len(kp_f) >= 8:
            for (pos, kp_k, matcher) in kf_db:
                knn = matcher.knnMatch(des_f, k=2)
                good = [a for a, b in (p for p in knn if len(p) == 2)
                        if a.distance < RATIO * b.distance]
                if len(good) < 8:
                    continue
                src = np.float32([kp_f[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
                dst = np.float32([kp_k[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
                Hfk, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
                if Hfk is None or int(mask.sum()) < MIN_POOL_INLIERS:
                    continue
                n_kf += 1
                Hinv = np.linalg.inv(Hfk)            # keyframe px -> frame px
                for (ft, (xk, yk)) in kf_lms[pos]:
                    p = Hinv @ np.array([xk, yk, 1.0])
                    if p[2] <= 1e-9:
                        continue
                    fx, fy = p[0] / p[2], p[1] / p[2]
                    if -1000 <= fx <= 2920 and -1000 <= fy <= 1980:
                        court_pts.append(ft); frame_pts.append((fx, fy))

        M = None
        if len(court_pts) >= MIN_CORR:
            cv2.setRNGSeed(0)
            src = np.array(court_pts, np.float64)      # court feet
            dst = np.array(frame_pts, np.float64)      # frame px
            Hf, mask = cv2.findHomography(src, dst, cv2.RANSAC, 15.0)  # court -> frame
            if Hf is not None:
                M = s5.signfix(Hf)
                pool_sizes.append(len(court_pts)); inlier_counts.append(int(mask.sum()))

        if M is not None:
            s4.draw_court(frame, M); s6.draw_arc(frame, M)
            rendered += 1
            c = M @ np.array([42.0, 25.0, 1.0])        # smoothness probe
            center = (c[0]/c[2], c[1]/c[2]) if c[2] > 1e-9 else None
            if center and prev_center:
                center_jumps.append(np.hypot(center[0]-prev_center[0],
                                             center[1]-prev_center[1]))
            prev_center = center
            txt = f"f={f} t={f/fps:04.1f}s  pooled {n_kf} kf  [per-frame +3pt]"
        else:
            no_render += 1
            prev_center = None
            txt = f"f={f}  NO FIT"
        cv2.putText(frame, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2)
        if f in STILLS:
            sp = os.path.join(OUT_DIR, f"HARD_perframe_still_{f:05d}.jpg")
            cv2.imwrite(sp, cv2.resize(frame, (W, H))); stills.append(sp)
        writer.write(cv2.resize(frame, (W, H)))
        if f % 200 == 0:
            print(f"  ...{f}/{total} rendered={rendered}", flush=True)
    writer.release(); cap.release()

    cj = np.array(center_jumps) if center_jumps else np.array([0.0])
    lines = [
        "Stage 8 -- per-frame POOLED court homography (tight + smooth)",
        "each frame fit from ALL matching keyframes' landmarks (pooled), RANSAC, "
        "sign-fixed; HS court model + 3pt arc; no hard keyframe switch",
        f"frames: {total}  rendered: {rendered} ({100*rendered/max(total,1):.1f}%)  "
        f"no-fit: {no_render}",
        f"pool per frame: correspondences median={int(np.median(pool_sizes))} "
        f"(range {min(pool_sizes)}-{max(pool_sizes)}), RANSAC inliers "
        f"median={int(np.median(inlier_counts))}",
        f"SMOOTHNESS (frame-to-frame move of court point (42,25); pops would be "
        f"isolated spikes): median={np.median(cj):.1f}px 95th={np.percentile(cj,95):.1f}px "
        f"max={cj.max():.1f}px",
        "  (vs Stage 7 per-view which popped 30-190px at keyframe switches)",
    ]
    summary = "\n".join(lines)
    print("\n================ SUMMARY ================\n" + summary +
          "\n=========================================")
    sp = os.path.join(OUT_DIR, "stage8_perframe_summary.txt")
    with open(sp, "w", encoding="utf-8") as fh:
        fh.write(summary + "\n")
    print(f"saved summary: {sp}")
    print(f"saved overlay: {out_path}")
    for s in stills:
        print(f"saved still: {s}")


if __name__ == "__main__":
    main()
