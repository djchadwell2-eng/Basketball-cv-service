"""Stage 5: render the court overlay on BOTH baskets across the whole clip.

FINDING that reshaped this stage: the Stage-4 "blank right half" was NOT a
single-reference representation limit. It was a homography SIGN-convention bug --
a homography is defined only up to sign, and for right-basket views the visible
court points came out with w<0, so the draw guard (w<=0 -> skip) dropped them
even though they divide to the correct pixels. The global frame-900 H_court is
fine for both baskets once the transform is SIGN-NORMALIZED (flip so a central
court point has w>0). No per-view homographies, and therefore no seams to
reconcile -- it's one globally-consistent court model everywhere.

So this builds on the working pieces: the optimized landmarks + global H_court
(Stages 2-4) and best-match per-frame anchoring (the coverage fix). The only new
thing is the one-line sign normalization. Stages 1-4 logic unchanged; reused.
Deterministic.
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
import stage4_courtmap as s4

NFEAT, RATIO, RANSAC_PX, MIN_INLIERS = 1500, 0.75, 3.0, 30
OUT_DIR, VIDEO = s2.OUT_DIR, s2.VIDEO_PATH
STILLS = [650, 900, 1050, 1100, 1200, 1500, 2000, 2700]   # left, center, SEAM, right


def sift_of(img, sift):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = s1.build_sift_mask(img.shape, s2.EXCLUDE_REGIONS)
    return sift.detectAndCompute(gray, mask)


def signfix(M):
    """A homography is defined up to sign; flip so a central court point (42,25)
    is IN FRONT (w>0), so the visible court isn't wrongly dropped as w<=0."""
    w = M[2, 0] * 42.0 + M[2, 1] * 25.0 + M[2, 2]
    return -M if w < 0 else M


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cv2.setRNGSeed(0)
    print("Recovering optimized transforms + global H_court...")
    KF, ref_pos, Hs_opt, L_opt, tags = s4.run_optimization()
    H_court, per_err, mean_err, max_err = s4.compute_H_court(L_opt, tags)
    Hcourt_inv = np.linalg.inv(H_court)
    print(f"global H_court residual: mean={mean_err:.2f} ft max={max_err:.2f} ft")

    sift = cv2.SIFT_create(nfeatures=NFEAT)
    kf_frames = s2.extract_frames(VIDEO, KF)
    kf_db = []
    for pos, k in enumerate(KF):
        kp, des = sift_of(kf_frames[k], sift)
        m = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        m.add([des]); m.train()
        kf_db.append((pos, kp, m))

    def court_to_frame(Hfk, pos):
        """Sign-normalized court feet -> frame px, via best-keyframe pos."""
        T = Hs_opt[pos] @ Hfk                       # frame -> 900
        return signfix(np.linalg.inv(T) @ Hcourt_inv)

    def match_frame(des_f, kp_f):
        """Return list of (inliers, pos, Hfk, ratio) sorted best-first."""
        cands = []
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
                inl = int(mask.sum())
                cands.append((inl, pos, Hfk, inl / len(good)))
        cands.sort(reverse=True, key=lambda c: c[0])
        return cands

    cap = cv2.VideoCapture(VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_path = os.path.join(OUT_DIR, "HARD_stage5_courtmap_overlay.mp4")
    W, H = 960, 540
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    # court points used to measure how far the drawing moves if we switch anchor
    seam_pts = np.array([(0, 25), (42, 25), (84, 25), (42, 0), (42, 50)], np.float64)
    rendered = no_match = 0
    seam_shifts = []        # (frame, px_shift, ft_shift) when 2 keyframes both strong
    stills = []
    for f in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        kp_f, des_f = sift_of(frame, sift)
        cands = match_frame(des_f, kp_f) if des_f is not None else []
        if cands and cands[0][0] >= MIN_INLIERS:
            inl, pos, Hfk, ratio = cands[0]
            M = court_to_frame(Hfk, pos)
            s4.draw_court(frame, M)
            rendered += 1
            # SEAM check: if a 2nd keyframe is also strong, how far does the
            # court drawing move when anchored to it instead?
            if len(cands) > 1 and cands[1][0] >= MIN_INLIERS:
                M2 = court_to_frame(cands[1][2], cands[1][1])
                d_px = []
                for (fx, fy) in seam_pts:
                    a = M @ np.array([fx, fy, 1.0]); b = M2 @ np.array([fx, fy, 1.0])
                    if a[2] != 0 and b[2] != 0:
                        d_px.append(np.hypot(a[0]/a[2]-b[0]/b[2], a[1]/a[2]-b[1]/b[2]))
                if d_px:
                    seam_shifts.append((f, max(d_px), KF[pos], KF[cands[1][1]]))
            tagtxt = f"f={f} t={f/fps:04.1f}s  kf={KF[pos]} r={ratio:.2f}"
        else:
            no_match += 1
            tagtxt = f"f={f}  NO MATCH"
        cv2.putText(frame, tagtxt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2)
        if f in STILLS:
            sp = os.path.join(OUT_DIR, f"HARD_stage5_still_{f:05d}.jpg")
            cv2.imwrite(sp, cv2.resize(frame, (W, H))); stills.append(sp)
        writer.write(cv2.resize(frame, (W, H)))
        if f % 200 == 0:
            print(f"  ...{f}/{total} rendered={rendered} no_match={no_match}", flush=True)
    writer.release(); cap.release()

    # seam stats: convert worst px shift to feet using H_court scale (~px/ft local)
    shifts_px = [s[1] for s in seam_shifts]
    lines = []
    lines.append("Stage 5 -- both-basket court overlay (global H_court + sign fix)")
    lines.append(f"per-frame anchoring: best-match keyframe; court model: ONE global "
                 f"H_court (NOT per-view -- the right-half failure was a sign bug)")
    lines.append(f"global H_court residual: mean={mean_err:.2f} ft, max={max_err:.2f} ft "
                 f"(unchanged vs Stage 4)")
    lines.append(f"frames: {total}   RENDERED court: {rendered} "
                 f"({100*rendered/max(total,1):.1f}%)   no-match: {no_match}")
    lines.append("  (Stage 4 rendered the right half = blank; this renders both baskets)")
    if shifts_px:
        lines.append(f"SEAM disagreement (court draw shift when switching to the 2nd-"
                     f"best keyframe, frames where both are strong): "
                     f"n={len(shifts_px)} mean={np.mean(shifts_px):.1f}px "
                     f"max={np.max(shifts_px):.1f}px")
        worst = max(seam_shifts, key=lambda s: s[1])
        lines.append(f"  worst at f={worst[0]} ({worst[2]}<->{worst[3]}): {worst[1]:.1f}px")
    summary = "\n".join(lines)
    print("\n================ SUMMARY ================\n" + summary +
          "\n=========================================")
    sp = os.path.join(OUT_DIR, "stage5_summary.txt")
    with open(sp, "w", encoding="utf-8") as fh:
        fh.write(summary + "\n")
    print(f"saved summary: {sp}")
    print(f"saved overlay: {out_path}")
    for s in stills:
        print(f"saved still: {s}")


if __name__ == "__main__":
    main()
