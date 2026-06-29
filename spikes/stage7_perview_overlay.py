"""Stage 7: tighten the overlay with PER-VIEW court homographies.

Stages 5-6 used ONE global H_court (mean 0.94 ft) -> the arc and FT line sit ~1 ft
off in places. Here each keyframe gets its OWN court homography (court feet ->
that keyframe's pixels), fit by RANSAC from neighbor-POOLED in-view landmarks,
then sign-normalized. Measured residual drops to ~0.2-0.3 ft per keyframe. The
court model (incl. the HS 3-point line) and best-match anchoring and the sign-fix
are all reused unchanged; only the final court homography goes global -> per-view.

Pooling: a keyframe alone can have a narrow landmark spread (left keyframes see
only x=0..19), which fits a poor homography. So each keyframe pools landmarks
from its +-2 neighbors, transformed into its space via the optimized inter-
keyframe transforms, keeping only those that land in/near its frame. RANSAC then
drops the loose clicks. Stages 1-6 intact. Deterministic.
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
import stage5_courtmap as s5          # signfix, sift_of
import stage6_arc_overlay as s6       # draw_arc, ARC geometry

NFEAT, RATIO, RANSAC_PX, MIN_INLIERS = 1500, 0.75, 3.0, 30
OUT_DIR, VIDEO = s2.OUT_DIR, s2.VIDEO_PATH
STILLS = [650, 900, 1100, 1500, 2000, 2700]
POOL_WINDOW = 2


def build_perview(KF, Hs_opt):
    """Per keyframe: court feet -> keyframe px (RANSAC, sign-fixed). Returns
    {pos: (H, residual_ft, n_inliers, n_total)}."""
    out = {}
    for pos in range(len(KF)):
        cpx, cft = [], []
        for npos in range(max(0, pos - POOL_WINDOW), min(len(KF), pos + POOL_WINDOW + 1)):
            R = np.linalg.inv(Hs_opt[pos]) @ Hs_opt[npos]   # kf_npos px -> kf_pos px
            for (tag, x, y) in s2.LANDMARKS[KF[npos]]:
                p = R @ np.array([x, y, 1.0])
                if p[2] <= 1e-9:
                    continue
                px = p[:2] / p[2]
                if -600 <= px[0] <= 2520 and -600 <= px[1] <= 1680:
                    cpx.append(px); cft.append(s4.COURT_MODEL[tag])
        cpx, cft = np.array(cpx), np.array(cft)
        cv2.setRNGSeed(0)
        H, mask = cv2.findHomography(cft, cpx, cv2.RANSAC, 12.0)
        if H[2, 0] * 42 + H[2, 1] * 25 + H[2, 2] < 0:        # sign-fix
            H = -H
        proj = cv2.perspectiveTransform(cpx.reshape(-1, 1, 2).astype(np.float32),
                                        np.linalg.inv(H)).reshape(-1, 2)
        err = np.linalg.norm(proj - cft, axis=1)[mask.ravel() == 1]
        out[pos] = (H, float(err.mean()), int(mask.sum()), len(cpx))
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cv2.setRNGSeed(0)
    print("Recovering transforms; building per-view court homographies...")
    KF, ref_pos, Hs_opt, L_opt, tags = s4.run_optimization()
    _, _, gmean, gmax = s4.compute_H_court(L_opt, tags)
    Hpv = build_perview(KF, Hs_opt)
    print(f"global H_court residual: {gmean:.2f} ft  (per-view below)")
    for pos, k in enumerate(KF):
        print(f"  kf {k}: per-view residual {Hpv[pos][1]:.2f} ft "
              f"({Hpv[pos][2]}/{Hpv[pos][3]} inliers)")

    sift = cv2.SIFT_create(nfeatures=NFEAT)
    kf_frames = s2.extract_frames(VIDEO, KF)
    kf_db = []
    for pos, k in enumerate(KF):
        kp, des = s5.sift_of(kf_frames[k], sift)
        m = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        m.add([des]); m.train()
        kf_db.append((pos, kp, m))

    def court_to_frame(Hfk, pos):
        return s5.signfix(np.linalg.inv(Hfk) @ Hpv[pos][0])   # feet -> frame px

    cap = cv2.VideoCapture(VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_path = os.path.join(OUT_DIR, "HARD_perview_overlay.mp4")
    W, H = 960, 540
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    seam_pts = np.array([(fx, fy) for fx in (10, 42, 74) for fy in (10, 25, 40)], float)
    rendered = no_match = 0
    seam = []
    stills = []
    for f in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        kp_f, des_f = s5.sift_of(frame, sift)
        cands = []
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
                if Hfk is not None:
                    cands.append((int(mask.sum()), pos, Hfk))
            cands.sort(reverse=True, key=lambda c: c[0])
        if cands and cands[0][0] >= MIN_INLIERS:
            _, pos, Hfk = cands[0]
            M = court_to_frame(Hfk, pos)
            s4.draw_court(frame, M); s6.draw_arc(frame, M)
            rendered += 1
            txt = f"f={f} t={f/fps:04.1f}s kf={KF[pos]}  [per-view +3pt]"
            if len(cands) > 1 and cands[1][0] >= MIN_INLIERS:
                M2 = court_to_frame(cands[1][2], cands[1][1])
                ds = []
                for (fx, fy) in seam_pts:
                    a = M @ np.array([fx, fy, 1.0]); b = M2 @ np.array([fx, fy, 1.0])
                    if a[2] > 1e-9 and b[2] > 1e-9:
                        ax, ay = a[0]/a[2], a[1]/a[2]
                        if 0 <= ax <= 1920 and 0 <= ay <= 1080:
                            ds.append(np.hypot(ax - b[0]/b[2], ay - b[1]/b[2]))
                if ds:
                    seam.append(max(ds))
        else:
            no_match += 1
            txt = f"f={f}  NO MATCH"
        cv2.putText(frame, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2)
        if f in STILLS:
            sp = os.path.join(OUT_DIR, f"HARD_perview_still_{f:05d}.jpg")
            cv2.imwrite(sp, cv2.resize(frame, (W, H))); stills.append(sp)
        writer.write(cv2.resize(frame, (W, H)))
        if f % 200 == 0:
            print(f"  ...{f}/{total} rendered={rendered}", flush=True)
    writer.release(); cap.release()

    pv_res = [Hpv[p][1] for p in range(len(KF))]
    lines = [
        "Stage 7 -- per-view court homographies (+ HS 3-point line)",
        f"court homography: PER-KEYFRAME (was ONE global). residual per keyframe "
        f"{min(pv_res):.2f}-{max(pv_res):.2f} ft (mean {np.mean(pv_res):.2f}) "
        f"vs OLD global {gmean:.2f} ft",
        "reused unchanged: best-match anchoring, sign-fix, HS court model + 3pt arc",
        f"frames: {total}  rendered: {rendered} ({100*rendered/max(total,1):.1f}%)  "
        f"no-match: {no_match}",
    ]
    if seam:
        lines.append(f"seam (visible court shift at keyframe switches): "
                     f"mean={np.mean(seam):.1f}px max={np.max(seam):.1f}px")
    summary = "\n".join(lines)
    print("\n================ SUMMARY ================\n" + summary +
          "\n=========================================")
    sp = os.path.join(OUT_DIR, "stage7_perview_summary.txt")
    with open(sp, "w", encoding="utf-8") as fh:
        fh.write(summary + "\n")
    print(f"saved summary: {sp}")
    print(f"saved overlay: {out_path}")
    for s in stills:
        print(f"saved still: {s}")


if __name__ == "__main__":
    main()
