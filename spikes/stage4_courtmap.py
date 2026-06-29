"""Stage 4 (payoff): compute the pixel->court homography from the Stage-3
optimized landmarks and overlay the court map on the WHOLE HARD.mp4, so we can
watch whether it sits on the real painted court through the full pan.

Pipeline reused:
  - Stage 2/3: the optimized landmark positions L (in reference frame-900 pixel
    space) and the keyframe->900 transforms Hs_opt.
  - The existing camera tracking (src/camera_tracking.CameraTracker) to relate
    every frame to frame 900.

Two homographies, kept explicit:
  H_court : frame-900 pixels -> court feet      (computed here from L + model)
  per-frame: court feet -> frame_f pixels  =  (900->f) @ inv(H_court)

Per-frame transform is KEYFRAME-ANCHORED: each frame is tied to its NEAREST
keyframe (whose ->900 transform is the accurate Stage-3 one), with only a short
camera-tracking hop in between -- so accumulated drift stays in the Stage-1
sweet spot instead of growing across the whole clip.

Deterministic. Prior stages untouched (their helpers are imported, not modified).
This stage does NOT change the optimization or the landmark set.
"""

import os
import sys
import numpy as np
import cv2
from scipy.optimize import least_squares

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))            # project root for src/
import stage2_multikeyframe as s2
import stage3_optimize as s3                           # reuse helpers only
from src.camera_tracking import CameraTracker

OUT_DIR = s2.OUT_DIR
VIDEO = s2.VIDEO_PATH

# ---------------------------------------------------------------------------
# 1. KNOWN COURT MODEL -- HIGH-SCHOOL (NFHS) dimensions, in FEET.
#    Origin at the left-baseline / near-sideline corner. x along length (0..84),
#    y across width (0..50).
#      court: 84 x 50 ft        (NBA is 94 -- using NBA would skew every tag)
#      lane (paint) width: 12 ft -> y = 19..31   (NBA lane is 16 -> y 17..33)
#      free-throw line: 19 ft from baseline       (same as NBA, by coincidence)
#      center circle radius: 6 ft
#    HS-vs-NBA MATTERS for: court length (center x, right-side x) and lane width
#    (lane/FT y). It does NOT change the left-baseline tags.
# ---------------------------------------------------------------------------
_COURT = s2.cfg.active()["court"]      # per-clip court dims (STEP 0)
COURT_LEN, COURT_WID = _COURT["length"], _COURT["width"]
LANE_Y0, LANE_Y1 = _COURT["lane_y0"], _COURT["lane_y1"]   # lane centered on 25
FT_X = _COURT["ft_x"]                   # left FT line; right FT at LEN-FT_X
CIRCLE_R = _COURT["circle_r"]
CX, CY = COURT_LEN / 2.0, COURT_WID / 2.0   # center (42, 25) for HS

COURT_MODEL = {
    "LB_side_near": (0.0, 0.0),          "LB_side_far": (0.0, 50.0),
    "L_lane_base_near": (0.0, LANE_Y0),  "L_lane_base_far": (0.0, LANE_Y1),
    "L_FT_near": (FT_X, LANE_Y0),        "L_FT_far": (FT_X, LANE_Y1),
    "center_near": (CX, 0.0),            "center_logo": (CX, CY),
    "center_far": (CX, 50.0),
    "R_FT_near": (COURT_LEN - FT_X, LANE_Y0),  "R_FT_far": (COURT_LEN - FT_X, LANE_Y1),
    "R_lane_base_near": (COURT_LEN, LANE_Y0),  "R_lane_base_far": (COURT_LEN, LANE_Y1),
    "RB_side_near": (COURT_LEN, 0.0),    "RB_side_far": (COURT_LEN, 50.0),
    "circle_top": (CX, CY + CIRCLE_R),   "circle_bottom": (CX, CY - CIRCLE_R),
    "circle_left": (CX - CIRCLE_R, CY),  "circle_right": (CX + CIRCLE_R, CY),
}

STILL_FRAMES = [300, 600, 750, 900, 1050, 1200, 1500, 2000, 2700]


# ---------------------------------------------------------------------------
# Stage-3 optimization (replicated using stage3's helpers so stage3 stays
# untouched) -> returns optimized keyframe transforms + landmark positions.
# ---------------------------------------------------------------------------
def run_optimization():
    KF = s2.KEYFRAMES
    n = len(KF)
    ref_pos = s2.REFERENCE_POS if s2.REFERENCE_POS is not None else n // 2
    frames = s2.extract_frames(VIDEO, KF)
    fwd, _, _ = s2.adjacent_homographies(frames, KF, s2.EXCLUDE_REGIONS)
    H_to_ref = s2.chain_to_reference(fwd, ref_pos, n)
    Hs_init = {p: (np.eye(3) if p == ref_pos else s2._norm(H_to_ref[p]))
               for p in range(n)}

    observations = []
    for pos, idx in enumerate(KF):
        for (tag, x, y) in s2.LANDMARKS.get(idx, []):
            observations.append((pos, tag, float(x), float(y)))
    tags = sorted({t for (_, t, _, _) in observations})

    tr0 = s3.transform_obs(Hs_init, observations, KF)
    by = {}
    for t in tr0:
        by.setdefault(t["tag"], []).append(t["ref_xy"])
    L_init = {tag: np.mean(np.array(by[tag]), axis=0) for tag in tags}

    nonref = [p for p in range(n) if p != ref_pos]

    def pack(Hs, L):
        x = []
        for p in nonref:
            x += s3.H_to8(Hs[p])
        for tag in tags:
            x += [float(L[tag][0]), float(L[tag][1])]
        return np.array(x)

    def unpack(x):
        Hs = {ref_pos: np.eye(3)}
        for k, p in enumerate(nonref):
            Hs[p] = s3.H_from8(x[8 * k:8 * k + 8])
        base = 8 * len(nonref)
        L = {tag: x[base + 2 * j: base + 2 * j + 2] for j, tag in enumerate(tags)}
        return Hs, L

    def residuals(x):
        Hs, L = unpack(x)
        r = np.empty(2 * len(observations))
        for i, (pos, tag, px, py) in enumerate(observations):
            qx, qy = s3.project(Hs[pos], px, py)
            r[2 * i] = qx - L[tag][0]
            r[2 * i + 1] = qy - L[tag][1]
        return r

    res = least_squares(residuals, pack(Hs_init, L_init), method="trf",
                        x_scale="jac", max_nfev=10000)
    Hs_opt, L_opt = unpack(res.x)
    return KF, ref_pos, Hs_opt, L_opt, tags


# ---------------------------------------------------------------------------
def compute_H_court(L_opt, tags):
    """H_court: frame-900 pixels -> court FEET (least-squares over ALL points)."""
    src = np.array([L_opt[t] for t in tags], dtype=np.float64)        # ref-900 px
    dst = np.array([COURT_MODEL[t] for t in tags], dtype=np.float64)  # court feet
    H_court, _ = cv2.findHomography(src, dst, method=0)               # all points
    proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H_court).reshape(-1, 2)
    err = np.linalg.norm(proj - dst, axis=1)                          # FEET
    return H_court, dict(zip(tags, err)), float(err.mean()), float(err.max())


def to_px(M, fx, fy):
    """Map a court point (feet) to image px via M; None if behind the camera."""
    d = M[2, 0] * fx + M[2, 1] * fy + M[2, 2]
    if d <= 1e-9:
        return None
    x = (M[0, 0] * fx + M[0, 1] * fy + M[0, 2]) / d
    y = (M[1, 0] * fx + M[1, 1] * fy + M[1, 2]) / d
    if abs(x) > 1e5 or abs(y) > 1e5:
        return None
    return (int(round(x)), int(round(y)))


def court_polylines():
    L, W = COURT_LEN, COURT_WID
    polys = [
        [(0, 0), (L, 0), (L, W), (0, W), (0, 0)],                 # boundary
        [(CX, 0), (CX, W)],                                       # center line
        [(0, LANE_Y0), (FT_X, LANE_Y0), (FT_X, LANE_Y1), (0, LANE_Y1), (0, LANE_Y0)],
        [(L, LANE_Y0), (L - FT_X, LANE_Y0), (L - FT_X, LANE_Y1), (L, LANE_Y1), (L, LANE_Y0)],
    ]
    circ = [(CX + CIRCLE_R * np.cos(t), CY + CIRCLE_R * np.sin(t))
            for t in np.linspace(0, 2 * np.pi, 48)]
    polys.append(circ)
    return polys


POLYS = court_polylines()


def draw_court(frame, M):
    """Draw the court model (feet) onto the frame using M: feet -> frame px."""
    for poly in POLYS:
        pts = [to_px(M, fx, fy) for (fx, fy) in poly]
        for j in range(len(pts) - 1):
            if pts[j] and pts[j + 1]:
                cv2.line(frame, pts[j], pts[j + 1], (0, 0, 255), 2)
    for tag, (fx, fy) in COURT_MODEL.items():
        p = to_px(M, fx, fy)
        if p:
            cv2.circle(frame, p, 5, (0, 255, 255), -1)


# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cv2.setRNGSeed(0)

    print("Running Stage-3 optimization to recover landmarks + transforms...")
    KF, ref_pos, Hs_opt, L_opt, tags = run_optimization()
    ref_idx = KF[ref_pos]

    H_court, per_err, mean_err, max_err = compute_H_court(L_opt, tags)
    Hcourt_inv = np.linalg.inv(H_court)   # court feet -> frame-900 px
    print(f"\nH_court (frame-{ref_idx} px -> court feet) reprojection residual: "
          f"mean={mean_err:.2f} ft  max={max_err:.2f} ft")
    for t in sorted(per_err):
        print(f"    {t:<18}{per_err[t]:>6.2f} ft")

    # --- PASS 1: camera tracking over the whole clip (frame_f -> frame_0) -----
    print("\nPASS 1: camera tracking over the whole clip...")
    cap = cv2.VideoCapture(VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = CameraTracker(np.eye(3))
    A = [None] * total
    A_kf = {}
    kf_set = set(KF)
    f = 0
    while f < total:
        ok, frame = cap.read()
        if not ok:
            break
        A[f] = tracker.update(frame, []).copy()   # full-res; no player masking
        if f in kf_set:
            A_kf[f] = A[f].copy()
        if f % 300 == 0:
            print(f"  ...{f}/{total}", flush=True)
        f += 1
    cap.release()
    total = f

    # Per-frame: tie each frame to its NEAREST keyframe. The relative transform
    # inv(A_kf[k]) @ A[f] only involves the steps between f and k, so the global
    # tracking drift (from frame 0) cancels -> drift is bounded by |f - k|.
    KF_arr = np.array(KF)

    def frame_to_900(f):
        k = int(KF_arr[np.argmin(np.abs(KF_arr - f))])    # nearest keyframe
        pos = KF.index(k)
        rel = np.linalg.inv(A_kf[k]) @ A[f]               # frame_f -> keyframe_k
        return Hs_opt[pos] @ rel                           # keyframe_k -> 900

    # --- PASS 2: draw court onto every frame, write video + stills -----------
    print("\nPASS 2: drawing court overlay...")
    out_path = os.path.join(OUT_DIR, "HARD_stage4_courtmap_overlay.mp4")
    W_out, H_out = 960, 540
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                             (W_out, H_out))
    cap = cv2.VideoCapture(VIDEO)
    saved_stills = []
    for f in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        T = frame_to_900(f)                               # frame_f -> 900
        # court feet -> frame_f px = (900 -> f) @ (feet -> 900 px)
        M = np.linalg.inv(T) @ Hcourt_inv
        M = M / M[2, 2]
        draw_court(frame, M)
        nearest = int(KF_arr[np.argmin(np.abs(KF_arr - f))])
        cv2.putText(frame, f"f={f}  t={f/fps:05.1f}s  nearest_kf={nearest}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        if f in STILL_FRAMES:
            sp = os.path.join(OUT_DIR, f"HARD_stage4_still_{f:05d}.jpg")
            cv2.imwrite(sp, cv2.resize(frame, (W_out, H_out)))
            saved_stills.append(sp)
        writer.write(cv2.resize(frame, (W_out, H_out)))
        if f % 300 == 0:
            print(f"  ...{f}/{total}", flush=True)
    writer.release()
    cap.release()

    # --- report --------------------------------------------------------------
    lines = []
    lines.append("Stage 4 -- pixel->court homography + full-clip court overlay")
    lines.append(f"court model: HIGH SCHOOL {COURT_LEN:.0f}x{COURT_WID:.0f} ft, "
                 f"12-ft lane (y {LANE_Y0:.0f}-{LANE_Y1:.0f}), FT {FT_X:.0f} ft, "
                 f"circle r{CIRCLE_R:.0f}")
    lines.append(f"reference frame: {ref_idx}")
    lines.append("")
    lines.append("H_court (frame-900 px -> court feet):")
    lines.append(np.array2string(H_court, precision=6, suppress_small=True))
    lines.append(f"reprojection residual on landmarks: mean={mean_err:.2f} ft, "
                 f"max={max_err:.2f} ft")
    for t in sorted(per_err):
        lines.append(f"    {t:<18}{per_err[t]:>6.2f} ft")
    lines.append("")
    lines.append(f"overlay video: {out_path}")
    lines.append(f"stills: {len(saved_stills)} at frames {STILL_FRAMES}")
    summary = "\n".join(lines)
    print("\n================ SUMMARY ================")
    print(summary)
    print("=========================================")
    spath = os.path.join(OUT_DIR, "stage4_summary.txt")
    with open(spath, "w", encoding="utf-8") as fh:
        fh.write(summary + "\n")
    print(f"saved summary: {spath}")
    print(f"saved overlay: {out_path}")
    for sp in saved_stills:
        print(f"saved still: {sp}")


if __name__ == "__main__":
    main()
