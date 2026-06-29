"""Add the HIGH-SCHOOL 3-point line to the working Stage-5 court overlay.

Purely additive: the homography (global H_court), best-match anchoring, and the
sign-fix are reused UNCHANGED from Stage 5. The only new thing is extra court
geometry (the 3-point arcs + corner straights) drawn through the SAME
court->frame transform as the rest of the court.

HS (NFHS) 3-point dimensions used:
  arc radius        = 19.75 ft (19'9")  from the basket center
  basket center     = 5.25 ft from baseline (4 ft backboard + 1.25 ft ring),
                      centered on width -> left (5.25,25), right (78.75,25)
  corner straights  = 5.25 ft (5'3") from each sideline (y=5.25, y=44.75),
                      baseline -> arc tangent (which is the arc's widest point)
Court is HS 84 x 50 ft, consistent with the existing COURT_MODEL.
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
import stage5_courtmap as s5          # reuse signfix + sift_of (sign-fix unchanged)

NFEAT, RATIO, RANSAC_PX, MIN_INLIERS = 1500, 0.75, 3.0, 30
OUT_DIR, VIDEO = s2.OUT_DIR, s2.VIDEO_PATH
STILLS = [650, 900, 1100, 1500, 2000, 2700]   # left, center, mid, right x3
ARC_COLOR = (255, 0, 255)                      # magenta, to distinguish the 3pt line

# --- HS 3-point geometry, in court feet --------------------------------------
R3 = 19.75
HOOP_DX = 5.25                       # basket center distance from baseline
SIDE = 5.25                          # corner straight distance from sideline
CY = s4.COURT_WID / 2.0              # 25
L = s4.COURT_LEN                     # 84


def three_point_polys():
    polys = []
    # LEFT basket: center (HOOP_DX, 25), arc opens toward +x (theta -90..+90)
    arc_l = [(HOOP_DX + R3*np.cos(np.radians(d)), CY + R3*np.sin(np.radians(d)))
             for d in range(-90, 91, 4)]
    polys.append([(0.0, CY - R3), arc_l[0]])     # corner straight (near y=5.25)
    polys.append(arc_l)                          # the arc
    polys.append([arc_l[-1], (0.0, CY + R3)])    # corner straight (near y=44.75)
    # RIGHT basket: center (L-HOOP_DX, 25), arc opens toward -x (theta 90..270)
    hx = L - HOOP_DX
    arc_r = [(hx + R3*np.cos(np.radians(d)), CY + R3*np.sin(np.radians(d)))
             for d in range(90, 271, 4)]
    polys.append([arc_r[0], (L, CY + R3)])
    polys.append(arc_r)
    polys.append([(L, CY - R3), arc_r[-1]])
    return polys


ARC_POLYS = three_point_polys()


def draw_arc(frame, M):
    for poly in ARC_POLYS:
        pts = [s4.to_px(M, fx, fy) for (fx, fy) in poly]
        for j in range(len(pts) - 1):
            if pts[j] and pts[j + 1]:
                cv2.line(frame, pts[j], pts[j + 1], ARC_COLOR, 2)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cv2.setRNGSeed(0)
    print("Recovering transforms + global H_court (unchanged from Stage 5)...")
    KF, ref_pos, Hs_opt, L_opt, tags = s4.run_optimization()
    H_court, per_err, mean_err, max_err = s4.compute_H_court(L_opt, tags)
    Hcourt_inv = np.linalg.inv(H_court)
    print(f"global H_court residual: mean={mean_err:.2f} ft max={max_err:.2f} ft "
          "(unchanged)")

    sift = cv2.SIFT_create(nfeatures=NFEAT)
    kf_frames = s2.extract_frames(VIDEO, KF)
    kf_db = []
    for pos, k in enumerate(KF):
        kp, des = s5.sift_of(kf_frames[k], sift)
        m = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        m.add([des]); m.train()
        kf_db.append((pos, kp, m))

    def court_to_frame(Hfk, pos):
        T = Hs_opt[pos] @ Hfk
        return s5.signfix(np.linalg.inv(T) @ Hcourt_inv)   # SAME sign-fix as Stage 5

    cap = cv2.VideoCapture(VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_path = os.path.join(OUT_DIR, "HARD_arc_overlay.mp4")
    W, H = 960, 540
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    rendered = no_match = 0
    stills = []
    for f in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        kp_f, des_f = s5.sift_of(frame, sift)
        best = None
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
                    inl = int(mask.sum())
                    if best is None or inl > best[0]:
                        best = (inl, pos, Hfk, inl / len(good))
        if best is not None and best[0] >= MIN_INLIERS:
            _, pos, Hfk, ratio = best
            M = court_to_frame(Hfk, pos)
            s4.draw_court(frame, M)        # base court (unchanged)
            draw_arc(frame, M)             # + 3-point line
            rendered += 1
            txt = f"f={f} t={f/fps:04.1f}s kf={KF[pos]} r={ratio:.2f}  [+3pt]"
        else:
            no_match += 1
            txt = f"f={f}  NO MATCH"
        cv2.putText(frame, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2)
        if f in STILLS:
            sp = os.path.join(OUT_DIR, f"HARD_arc_still_{f:05d}.jpg")
            cv2.imwrite(sp, cv2.resize(frame, (W, H))); stills.append(sp)
        writer.write(cv2.resize(frame, (W, H)))
        if f % 200 == 0:
            print(f"  ...{f}/{total} rendered={rendered}", flush=True)
    writer.release(); cap.release()

    lines = [
        "Stage 6 -- Stage-5 court overlay + HS 3-point line (additive only)",
        f"HS 3pt: radius={R3} ft, basket {HOOP_DX} ft from baseline, "
        f"corner straights {SIDE} ft from sideline; court {L:.0f}x{s4.COURT_WID:.0f} ft",
        f"UNCHANGED: global H_court residual mean={mean_err:.2f} ft max={max_err:.2f} ft; "
        "best-match anchoring; sign-fix",
        f"frames: {total}   rendered: {rendered} ({100*rendered/max(total,1):.1f}%)   "
        f"no-match: {no_match}",
    ]
    summary = "\n".join(lines)
    print("\n================ SUMMARY ================\n" + summary +
          "\n=========================================")
    sp = os.path.join(OUT_DIR, "stage6_arc_summary.txt")
    with open(sp, "w", encoding="utf-8") as fh:
        fh.write(summary + "\n")
    print(f"saved summary: {sp}")
    print(f"saved overlay: {out_path}")
    for s in stills:
        print(f"saved still: {s}")


if __name__ == "__main__":
    main()
