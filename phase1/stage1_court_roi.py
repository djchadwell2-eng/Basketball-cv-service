"""STAGE 1 -- Court ROI filter (the "13-person rule" fix).

Raw person detection sees refs, bench, coaches, and crowd -- not just the 10
on-court players. This stage uses the EXISTING, validated homography to decide,
per detection, whether a body is standing ON the painted court (keep) or OFF it
(discard the bench/crowd/refs).

HOW IT REUSES THE VALIDATED ENGINE (World A, spikes/):
  1. run_optimization()  -> H_court : reference-keyframe px -> court FEET
     (from the clip's clicked landmarks in clips_config.py).
  2. DIRECT-ANCHOR each frame: SIFT-match the frame to its NEAREST keyframe and
     compose with that keyframe's known court homography (Hs_opt). This REPLACED
     the old accumulated ORB consecutive-frame chain, which a diagnostic proved
     broadly unreliable (74% of frames off >40px). The direct match was proven
     glued on every swept frame (see phase1/DECISIONS.md). No chain, no drift.
  3. PER FRAME:  pixel -> court feet  =  H_court @ T   (T = frame px -> reference px)
     (stage4 uses the inverse, feet->px, to DRAW the court; we use px->feet to
      LOCATE players. Same two matrices, opposite direction.)

A detection's ground-contact point is the bottom-center of its box (the feet are
on the floor; the head is well above the ground plane and would map wrong).

THE FILTER: map feet->court-feet, keep it if it lands inside the 84x50 court
rectangle plus a small MARGIN_FT of slack (feet positions wobble near the lines).

VALIDATION IS BY EYE: we render sampled frames with on-court boxes GREEN and
discarded boxes RED, overlay the court + margin envelope, and print the on-court
count per frame (should hover near 10, not 13+). Tune MARGIN_FT only after seeing
where it leaks.

Detection config = the validated one: YOLOv8m @ imgsz=1280, person class only.
NO tracking IDs (not needed to decide on/off court), NO team assignment here.
Deterministic, CONFIG-driven, single offline tool.
"""

import os
import sys

import cv2
import numpy as np

# --- make the validated engine importable (spikes/ modules + src/) ------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "spikes"))   # stage2/stage4 + clips_config
sys.path.insert(0, _ROOT)                            # src/ (camera_tracking)

import stage1_keyframe_match as s1         # noqa: E402  validated SIFT matcher
import stage2_multikeyframe as s2          # noqa: E402  per-clip CONFIG + helpers
import stage4_courtmap as s4               # noqa: E402  H_court + court drawing

import refit_keyframes                     # noqa: E402  consistency-refit keyframes
from ultralytics import YOLO               # noqa: E402

# ==============================================================================
# CONFIG (the only knobs)
# ==============================================================================
MARGIN_FT = 1.5          # slack outside the painted lines still counted on-court
FRAME_EDGE_PX = 4        # box bottoms within this many px of the frame edge have
                         # their FEET off-screen -> ground point unreliable, drop
IMG_SIZE = 1280          # validated detector resolution
MODEL_NAME = os.path.join(_ROOT, "yolov8m.pt")
PERSON_CLASS = 0

# Direct-anchor matcher settings (the same the validated diagnostic used).
ANCHOR_RATIO = 0.75
ANCHOR_RANSAC_PX = 3.0
# Homography-confidence thresholds. The observed direct-match inlier floor on this
# clip is ~1000; 150 flags only near-failed matches (an order of magnitude below).
CONF_INLIER_MIN = 150
CONF_REPROJ_MAX = 2.0    # mean reproj px on inliers (RANSAC caps inliers at 3px)

# Sample frames for the eyeball loop (CPU-only torch -> don't run YOLO on every
# frame). Stay INSIDE the clip's validated pan range; clips_config notes TEST1's
# clean L->C->R pan is ~100..585.
SAMPLE_FRAMES = list(range(120, 581, 30))

OUT_DIR = os.path.join(_HERE, "out")
DISP_W, DISP_H = 1280, 720    # still/video output size

COURT_LEN = s4.COURT_LEN      # 84 (HS), from the active clip's court dims
COURT_WID = s4.COURT_WID      # 50


# ==============================================================================
def build_court_anchor():
    """Return (H_court, anchor, fps, total) for the ACTIVE clip.

    H_court : reference-keyframe px -> court feet (validated engine).
    anchor(f, frame_bgr) -> (T, inliers, reproj_px, kf):
        T          = frame px -> reference px, via a DIRECT SIFT match of the frame
                     to its NEAREST keyframe composed with that keyframe's known
                     transform (NO accumulated ORB chain).  pixel->feet = H_court @ T
        inliers    = RANSAC inliers of that direct match (homography confidence)
        reproj_px  = mean reprojection error of the inliers
        kf         = which keyframe the frame anchored to
    """
    print("Recovering consistency-refit keyframes + H_court...")
    KF, ref_pos, Hs_opt, L_opt, tags = refit_keyframes.refit()   # mutually-consistent
    H_court, per_err, mean_err, max_err = s4.compute_H_court(L_opt, tags)
    print(f"  H_court reprojection: mean={mean_err:.2f} ft  max={max_err:.2f} ft "
          f"(ref keyframe {KF[ref_pos]})")

    cap = cv2.VideoCapture(s2.VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    kf_imgs = s2.extract_frames(s2.VIDEO_PATH, KF)   # keyframe images for matching
    KF_arr = np.array(KF)

    def anchor(f, frame_bgr):
        k = int(KF_arr[np.argmin(np.abs(KF_arr - f))])      # nearest keyframe
        pos = KF.index(k)
        cv2.setRNGSeed(0)                                   # reproducible RANSAC
        # detect_and_match(A=keyframe, B=frame) -> estimate gives H: frame -> kf.
        kp_k, kp_f, good = s1.detect_and_match(kf_imgs[k], frame_bgr,
                                               ANCHOR_RATIO, s2.EXCLUDE_REGIONS)
        H, mask, inliers = s1.estimate_homography(kp_k, kp_f, good, ANCHOR_RANSAC_PX)
        if H is None or inliers == 0:
            return None, 0, float("inf"), k
        im = [m for m, keep in zip(good, mask.ravel()) if keep]
        pf = np.float32([kp_f[m.trainIdx].pt for m in im]).reshape(-1, 1, 2)
        pk = np.float32([kp_k[m.queryIdx].pt for m in im])
        proj = cv2.perspectiveTransform(pf, H).reshape(-1, 2)
        reproj = float(np.linalg.norm(proj - pk, axis=1).mean())
        T = _norm(Hs_opt[pos] @ H)                          # frame px -> reference px
        return T, int(inliers), reproj, k

    return H_court, anchor, fps, total


def confidence_state(inliers, reproj):
    """Map a direct-match quality to the schema's ok/low_confidence state."""
    return "ok" if (inliers >= CONF_INLIER_MIN and reproj <= CONF_REPROJ_MAX) \
        else "low_confidence"


def _norm(H):
    return H / H[2, 2] if H[2, 2] != 0 else H


def feet_pixel(box):
    """Ground-contact point of a person box = bottom-center (feet on the floor)."""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


def pixel_to_feet(pix_to_feet, px, py):
    """Map one pixel to court feet. Returns (cx, cy, w) where w is the homogeneous
    depth -- its SIGN tells which side of the horizon the point is on."""
    v = pix_to_feet @ np.array([px, py, 1.0])
    w = v[2]
    ws = w if w != 0 else 1e-9
    return float(v[0] / ws), float(v[1] / ws), float(w)


def on_court(cx, cy, w, ref_w):
    """True if a court-feet point is inside the 84x50 rectangle + MARGIN_FT AND
    in FRONT of the horizon (same depth-sign as a known on-court point).

    The horizon guard kills the "past the vanishing line" trap: a body high in
    the bleachers projects ABOVE the court's horizon, where the homogeneous depth
    w flips sign and the point wraps to fake in-bounds court coords.
    """
    if w * ref_w <= 0:
        return False
    m = MARGIN_FT
    return (-m <= cx <= COURT_LEN + m) and (-m <= cy <= COURT_WID + m)


def draw_margin_envelope(frame, feet_to_px):
    """Draw the on-court acceptance rectangle (court + MARGIN_FT) in yellow."""
    m = MARGIN_FT
    corners = [(-m, -m), (COURT_LEN + m, -m),
               (COURT_LEN + m, COURT_WID + m), (-m, COURT_WID + m), (-m, -m)]
    pts = [s4.to_px(feet_to_px, fx, fy) for (fx, fy) in corners]
    for j in range(len(pts) - 1):
        if pts[j] and pts[j + 1]:
            cv2.line(frame, pts[j], pts[j + 1], (0, 255, 255), 2)


# ==============================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cv2.setRNGSeed(0)

    H_court, anchor, fps, total = build_court_anchor()
    model = YOLO(MODEL_NAME)

    cap = cv2.VideoCapture(s2.VIDEO_PATH)
    writer = cv2.VideoWriter(
        os.path.join(OUT_DIR, f"{s2.cfg.ACTIVE}_stage1_roi.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), 6.0, (DISP_W, DISP_H))

    sample_set = set(SAMPLE_FRAMES)
    counts = []
    print(f"\nDetecting + filtering on {len(SAMPLE_FRAMES)} sampled frames...")
    for f in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        if f not in sample_set:
            continue

        T, inliers, reproj, kf = anchor(f, frame)   # DIRECT nearest-keyframe match
        p2f = H_court @ T                       # pixel -> court feet
        feet2px = np.linalg.inv(p2f)            # court feet -> pixel (for drawing)

        # Depth-sign of a definitely-on-court point (court center), to orient the
        # horizon guard for THIS frame's homography.
        ctr_px = feet2px @ np.array([COURT_LEN / 2.0, COURT_WID / 2.0, 1.0])
        ctr_px = ctr_px / ctr_px[2]
        ref_w = float((p2f @ ctr_px)[2])

        # Court lines (boundary/lane/center/circle) + the margin envelope.
        s4.draw_court(frame, feet2px)
        draw_margin_envelope(frame, feet2px)

        res = model.predict(frame, imgsz=IMG_SIZE, classes=[PERSON_CLASS],
                            verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else []

        frame_h = frame.shape[0]
        on_ct = 0
        for box in boxes:
            fx, fy = feet_pixel(box)
            # Feet clamped to the bottom frame edge -> off-screen ground point,
            # don't trust it (these are cut-off foreground spectators/table).
            feet_off_screen = fy >= frame_h - FRAME_EDGE_PX
            cx, cy, w = pixel_to_feet(p2f, fx, fy)
            keep = on_court(cx, cy, w, ref_w) and not feet_off_screen
            color = (0, 255, 0) if keep else (0, 0, 255)   # green keep / red drop
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (int(fx), int(fy)), 4, color, -1)
            cv2.putText(frame, f"({cx:.0f},{cy:.0f})", (x1, y2 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            if keep:
                on_ct += 1
        counts.append((f, on_ct, len(boxes)))

        cv2.putText(frame, f"f={f}  on-court={on_ct}  total={len(boxes)}  "
                           f"anchor_kf={kf} inl={inliers}",
                    (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        out = cv2.resize(frame, (DISP_W, DISP_H))
        cv2.imwrite(os.path.join(OUT_DIR, f"{s2.cfg.ACTIVE}_stage1_{f:05d}.jpg"), out)
        writer.write(out)
        print(f"  f={f:>4}  on-court={on_ct:>2}  total={len(boxes):>2}  "
              f"anchor_kf={kf}  inl={inliers}  reproj={reproj:.2f}px")
    writer.release()
    cap.release()

    # ---- summary -------------------------------------------------------------
    lines = ["Stage 1 -- court ROI filter", f"clip: {s2.cfg.ACTIVE}  ({s2.VIDEO_PATH})",
             f"margin: {MARGIN_FT} ft   court: {COURT_LEN:.0f}x{COURT_WID:.0f}",
             f"sampled frames: {SAMPLE_FRAMES}", "",
             "frame   on_court   total_detected"]
    for (f, on_ct, tot) in counts:
        lines.append(f"{f:>5}   {on_ct:>8}   {tot:>14}")
    if counts:
        on_vals = [c[1] for c in counts]
        lines.append("")
        lines.append(f"on-court count: min={min(on_vals)} max={max(on_vals)} "
                     f"mean={sum(on_vals)/len(on_vals):.1f}  (target ~10)")
    summary = "\n".join(lines)
    spath = os.path.join(OUT_DIR, f"{s2.cfg.ACTIVE}_stage1_summary.txt")
    with open(spath, "w", encoding="utf-8") as fh:
        fh.write(summary + "\n")
    print("\n================ SUMMARY ================")
    print(summary)
    print("=========================================")
    print(f"saved stills + {s2.cfg.ACTIVE}_stage1_roi.mp4 + summary in {OUT_DIR}")


if __name__ == "__main__":
    main()
