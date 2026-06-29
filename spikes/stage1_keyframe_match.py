"""Stage 1 spike: relate two WELL-SEPARATED keyframes DIRECTLY with a single
homography, exclude burned-in broadcast graphics, and measure how keyframe
SEPARATION affects accuracy.

Background: the court is planar, so two views of it relate by ONE homography.
The previous approach chained consecutive-frame homographies and accumulated
drift; this tests estimating the homography DIRECTLY between distant keyframes.
A verified problem: burned-in graphics (scorebug, timestamp) are STATIC on
screen, so SIFT+RANSAC latches onto them (zero motion) instead of the moving
court. So we now MASK those fixed-screen regions out before SIFT, then measure
inlier quality and reprojection error across increasing keyframe separation.

Scope is deliberately tiny and unchanged in spirit: classic SIFT only, ONE direct
homography per pair, hand-clicked INDEPENDENT validation landmarks. No chaining,
no YOLO/player masking, no graphics auto-detection (regions are clicked by hand),
no multi-keyframe stitching / keyframe graph / bundle adjustment, no learned
matchers, no web/API. Run from CONFIG; reproducible (RANSAC RNG seed is fixed).
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clips_config as cfg            # per-clip CONFIG (STEP 0 de-hardcode)

# ==============================================================================
# CONFIG  -- edit, then re-run. Runs are reproducible (fixed RANSAC seed).
# Per-clip values (video, scorebug) come from clips_config.py (cfg.active()).
# ==============================================================================
VIDEO_PATH = cfg.active()["video_path"]

# Keyframe pairs at INCREASING separation (close -> very far). Separation is just
# |b - a| frames. Larger separation = less view overlap = the thing we're testing.
FRAME_PAIRS = [
    (760, 1120),   # ~360 (medium) -- BOTH frames clearly show the center "M"
                   # so the overlap contains a clickable landmark to validate.
    # Earlier separations (restore if you want to re-measure them):
    # (900, 1000),   # ~100  (close)
    # (900, 1300),   # ~400  (medium, but overlap had no clickable landmark)
    # (900, 1800),   # ~900  (far)
    # (600, 1900),   # ~1300 (very far -- no overlap)
]

RANSAC_THRESH_PX = 3.0     # max reprojection error (px) for a RANSAC inlier
RATIO_TEST = 0.75          # Lowe's ratio: keep match if best < 0.75 * second-best

# Burned-in graphics to ignore (FIXED screen overlays: scorebug, timestamp, etc.)
# Pixel rectangles (x1, y1, x2, y2) in NATIVE frame coords. These are masked out
# of BOTH frames before SIFT so no keypoints are detected inside them.
# Leave EMPTY to click them once (prints a block to paste here). HARD-specific.
EXCLUDE_REGIONS = cfg.active()["exclude_regions"]   # per-clip scorebug box(es)
EXCLUDE_REF_FRAME = 900     # frame shown for the exclude-region click session

# Independent validation landmarks, keyed PER PAIR: same physical COURT point's
# pixel location in A and B -> (ax, ay, bx, by). Court structure (line/hash/logo),
# NOT SIFT features, so it tests H independently.
#   - pair ABSENT from this dict  -> interactive click mode for that pair
#   - pair maps to []             -> skip validation for that pair (no clicks)
#   - pair maps to [points]       -> deterministic validation
VALIDATION_POINTS = {
    # (900, 1300): [ (ax, ay, bx, by), ... ],
}

DISPLAY_MAX_W = 1280        # display scale for click windows only (clicks stored native)
SEED = 0                    # RANSAC RNG seed (per pair) for reproducibility
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


# ==============================================================================
def extract_frame(video_path, idx):
    """Read frame `idx` (native resolution), FRAME-ACCURATELY.

    Decodes sequentially (grab past 0..idx-1, then read idx) instead of
    cap.set(CAP_PROP_POS_FRAMES, idx), which is NOT reliably frame-accurate (it
    can snap to a keyframe) and would silently return the wrong frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    for _ in range(idx):
        if not cap.grab():
            cap.release()
            raise RuntimeError(f"Video ended before frame {idx}: {video_path}")
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {idx} from {video_path}")
    return frame


def build_sift_mask(shape, regions):
    """255 where SIFT may find features, 0 inside excluded graphics rectangles."""
    h, w = shape[:2]
    mask = np.full((h, w), 255, np.uint8)
    for (x1, y1, x2, y2) in regions:
        xa, xb = sorted((int(x1), int(x2)))
        ya, yb = sorted((int(y1), int(y2)))
        xa, ya = max(0, xa), max(0, ya)
        xb, yb = min(w, xb), min(h, yb)
        mask[ya:yb, xa:xb] = 0
    return mask


def point_in_regions(x, y, regions):
    for (x1, y1, x2, y2) in regions:
        if min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
            return True
    return False


def detect_and_match(img_a, img_b, ratio, regions):
    """SIFT keypoints (excluded regions masked) + Lowe's-ratio-filtered matches.

    Matches are oriented query=A, train=B: each DMatch links kp_a[queryIdx] to
    kp_b[trainIdx]. Returns (kp_a, kp_b, good_matches).
    """
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    mask_a = build_sift_mask(img_a.shape, regions)
    mask_b = build_sift_mask(img_b.shape, regions)

    sift = cv2.SIFT_create()
    kp_a, des_a = sift.detectAndCompute(gray_a, mask_a)
    kp_b, des_b = sift.detectAndCompute(gray_b, mask_b)
    print(f"  keypoints (excluded regions masked): A={len(kp_a)}, B={len(kp_b)}")
    if des_a is None or des_b is None:
        return kp_a, kp_b, []

    # k=2 NN per descriptor so we can apply the ratio test.
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(des_a, des_b, k=2)

    # Lowe's ratio test: trust a match only if the best neighbour is clearly
    # closer than the second-best (ambiguous matches discarded).
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    print(f"  matches passing ratio test ({ratio}): {len(good)}")
    return kp_a, kp_b, good


def estimate_homography(kp_a, kp_b, good, ransac_thresh):
    """H_BA: maps a point in frame B -> the corresponding point in A.

    We want to take points FROM B and land them IN A, so
    cv2.findHomography(src=pts_B, dst=pts_A); the result satisfies pt_A ~ H * pt_B.
    """
    if len(good) < 4:
        print("  not enough matches to estimate a homography (need >= 4).")
        return None, None, 0
    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H_ba, mask = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, ransac_thresh)
    if H_ba is None:
        print("  findHomography failed (degenerate matches).")
        return None, None, 0
    inliers = int(mask.sum())
    ratio = inliers / len(good) if good else 0.0
    print(f"  RANSAC inliers: {inliers} / {len(good)}  (inlier ratio = {ratio:.3f})")
    return H_ba, mask, inliers


def report_inlier_locations(kp_a, good, mask, regions):
    """Confirm inliers are NOT inside the excluded graphics (should be ~0 now)."""
    inlier_matches = [m for m, keep in zip(good, mask.ravel()) if keep]
    inside = sum(1 for m in inlier_matches
                 if point_in_regions(*kp_a[m.queryIdx].pt, regions))
    total = len(inlier_matches)
    frac = (inside / total) if total else 0.0
    print(f"  inliers inside excluded regions: {inside}/{total} ({frac:.1%}) "
          f"-- outside: {total - inside}")
    return inside, total


def save_inlier_match_viz(img_a, kp_a, img_b, kp_b, good, mask, path):
    """drawMatches of ONLY the RANSAC inlier matches."""
    inlier_matches = [m for m, keep in zip(good, mask.ravel()) if keep]
    viz = cv2.drawMatches(
        img_a, kp_a, img_b, kp_b, inlier_matches, None,
        matchColor=(0, 255, 0), singlePointColor=(0, 0, 255),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    cv2.imwrite(path, viz)
    print(f"  saved inlier-match viz: {path}")


# ------------------------------------------------------------------------------
# Interactive click helpers (only run when CONFIG lists are missing entries).
# ------------------------------------------------------------------------------
def _click_points(img, title, banner):
    """Show `img` (scaled to fit), collect left-clicks in NATIVE px coords."""
    h, w = img.shape[:2]
    scale = min(1.0, DISPLAY_MAX_W / float(w))
    disp0 = cv2.resize(img, (int(w * scale), int(h * scale)))
    pts = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x / scale, y / scale))   # back to native coords

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, on_mouse)
    print(f"\n  [{title}] {banner}")
    while True:
        disp = disp0.copy()
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(disp, banner, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        for i, (px, py) in enumerate(pts):
            d = (int(px * scale), int(py * scale))
            cv2.circle(disp, d, 5, (0, 255, 0), -1)
            cv2.putText(disp, str(i), (d[0] + 6, d[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow(title, disp)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("u") and pts:
            pts.pop()
        elif k in (13, ord("q")):
            break
    cv2.destroyWindow(title)
    return pts


def collect_exclude_regions(img):
    """Click TWO opposite corners per graphic box; many boxes; returns rects."""
    pts = _click_points(
        img, "EXCLUDE REGIONS",
        "click 2 opposite corners per graphic (scorebug/timestamp); u=undo ENTER=done")
    if len(pts) % 2 != 0:
        print(f"  WARNING: {len(pts)} corner clicks (odd) -- dropping the last one.")
        pts = pts[:-1]
    regions = []
    for i in range(0, len(pts), 2):
        (x1, y1), (x2, y2) = pts[i], pts[i + 1]
        regions.append((round(min(x1, x2), 1), round(min(y1, y2), 1),
                        round(max(x1, x2), 1), round(max(y1, y2), 1)))
    print("\n  ---- paste into CONFIG.EXCLUDE_REGIONS ----")
    print("EXCLUDE_REGIONS = [")
    for r in regions:
        print(f"    {r},")
    print("]")
    print("  -------------------------------------------")
    return regions


def collect_validation_points(img_a, img_b, pair):
    """Click the SAME landmarks in A then B (same order); return (ax,ay,bx,by) list."""
    print("\n  >> Pick COURT landmarks (line/hash intersections, logo corners).")
    print("  >> SPREAD them across the frame -- include points NEAR THE EDGES,")
    print("  >> do NOT bunch them in the center (edges expose homography error).")
    a_pts = _click_points(img_a, f"FRAME A {pair[0]}",
                          "click landmarks, SPREAD to edges; u=undo ENTER=done")
    b_pts = _click_points(img_b, f"FRAME B {pair[1]}",
                          "click the SAME landmarks in the SAME order; u=undo ENTER=done")
    n = min(len(a_pts), len(b_pts))
    if len(a_pts) != len(b_pts):
        print(f"  WARNING: {len(a_pts)} in A vs {len(b_pts)} in B; using first {n}.")
    combined = [(a_pts[i][0], a_pts[i][1], b_pts[i][0], b_pts[i][1]) for i in range(n)]
    print(f"\n  ---- paste into CONFIG.VALIDATION_POINTS (keyed by pair {pair}) ----")
    print(f"    {pair}: [")
    for (ax, ay, bx, by) in combined:
        print(f"        ({ax:.1f}, {ay:.1f}, {bx:.1f}, {by:.1f}),")
    print("    ],")
    print("  -------------------------------------------------------------------")
    return combined


def validate(H_ba, img_a, points, path):
    """Project B-landmarks through H_BA and measure pixel error vs A-landmarks."""
    a_pts = np.float32([[ax, ay] for (ax, ay, bx, by) in points])
    # perspectiveTransform requires shape (N, 1, 2) float32.
    b_pts = np.float32([[bx, by] for (ax, ay, bx, by) in points]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(b_pts, H_ba).reshape(-1, 2)
    errors = np.linalg.norm(proj - a_pts, axis=1)
    print("  independent validation (B-landmark projected into A vs clicked A):")
    for i, (e, a, p) in enumerate(zip(errors, a_pts, proj)):
        print(f"    pt {i}: A=({a[0]:.1f},{a[1]:.1f}) proj=({p[0]:.1f},{p[1]:.1f}) "
              f"err={e:.1f}px")
    mean_err, max_err = float(errors.mean()), float(errors.max())
    print(f"    MEAN error = {mean_err:.2f}px   MAX error = {max_err:.2f}px")

    viz = img_a.copy()
    for (a, p) in zip(a_pts, proj):
        ai, pi = (int(a[0]), int(a[1])), (int(p[0]), int(p[1]))
        cv2.line(viz, ai, pi, (255, 255, 0), 1)
        cv2.circle(viz, ai, 6, (0, 255, 0), -1)   # GREEN = clicked A (truth)
        cv2.circle(viz, pi, 6, (0, 0, 255), -1)   # RED   = projected from B
    cv2.imwrite(path, viz)
    print(f"  saved validation viz: {path}")
    return mean_err, max_err


# ==============================================================================
def process_pair(a_idx, b_idx, regions, saved_frames):
    """Run the full pipeline for one keyframe pair; return a summary row dict."""
    sep = abs(b_idx - a_idx)
    print(f"\n================ PAIR A={a_idx} B={b_idx}  (separation {sep}) ================")
    cv2.setRNGSeed(SEED)   # per-pair determinism regardless of order

    img_a = extract_frame(VIDEO_PATH, a_idx)
    img_b = extract_frame(VIDEO_PATH, b_idx)
    for idx, img in ((a_idx, img_a), (b_idx, img_b)):
        if idx not in saved_frames:
            p = os.path.join(OUT_DIR, f"frame_{idx}.png")
            cv2.imwrite(p, img)
            saved_frames.add(idx)
            print(f"  saved frame {idx}: {p}")

    kp_a, kp_b, good = detect_and_match(img_a, img_b, RATIO_TEST, regions)
    H_ba, mask, inliers = estimate_homography(kp_a, kp_b, good, RANSAC_THRESH_PX)
    row = {"sep": sep, "a": a_idx, "b": b_idx, "matches": len(good),
           "inliers": inliers, "inlier_ratio": (inliers / len(good)) if good else 0.0,
           "mean_err": None, "max_err": None}
    if H_ba is None:
        print("  -> no homography for this pair.")
        return row

    report_inlier_locations(kp_a, good, mask, regions)
    save_inlier_match_viz(img_a, kp_a, img_b, kp_b, good, mask,
                          os.path.join(OUT_DIR, f"matches_sep{sep}_{a_idx}_{b_idx}.png"))

    # Validation: provided -> deterministic; [] -> skip; absent -> click.
    pair = (a_idx, b_idx)
    if pair in VALIDATION_POINTS:
        points = VALIDATION_POINTS[pair]
        if not points:
            print("  validation skipped for this pair (empty list in CONFIG).")
    else:
        print("  no validation points in CONFIG for this pair -> click mode.")
        points = collect_validation_points(img_a, img_b, pair)

    if points:
        vpath = os.path.join(OUT_DIR, f"validation_sep{sep}_{a_idx}_{b_idx}.png")
        row["mean_err"], row["max_err"] = validate(H_ba, img_a, points, vpath)
    return row


def print_and_save_table(rows):
    lines = []
    lines.append("Stage 1 -- keyframe separation vs accuracy (exclusion applied)")
    lines.append(f"video: {VIDEO_PATH}")
    lines.append(f"exclude_regions: {EXCLUDE_REGIONS_RUNTIME}")
    lines.append("")
    header = (f"{'sep':>5} {'A':>6} {'B':>6} {'matches':>8} {'inliers':>8} "
              f"{'inlier_ratio':>13} {'mean_err':>9} {'max_err':>8}")
    lines.append(header)
    lines.append("-" * len(header))
    for r in sorted(rows, key=lambda x: x["sep"]):
        me = "-" if r["mean_err"] is None else f"{r['mean_err']:.2f}"
        mx = "-" if r["max_err"] is None else f"{r['max_err']:.2f}"
        lines.append(f"{r['sep']:>5} {r['a']:>6} {r['b']:>6} {r['matches']:>8} "
                     f"{r['inliers']:>8} {r['inlier_ratio']:>13.3f} {me:>9} {mx:>8}")
    table = "\n".join(lines)
    print("\n" + table)
    path = os.path.join(OUT_DIR, "separation_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(f"\nsaved summary table: {path}")


# Filled in at runtime so the table can record which regions were used.
EXCLUDE_REGIONS_RUNTIME = []


def main():
    global EXCLUDE_REGIONS_RUNTIME
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Video: {VIDEO_PATH}")
    print(f"Frame pairs: {FRAME_PAIRS}")

    # Exclude regions: provided -> use; empty -> click once on a reference frame.
    regions = EXCLUDE_REGIONS
    if not regions:
        print(f"\nEXCLUDE_REGIONS empty -> click mode on frame {EXCLUDE_REF_FRAME}.")
        ref = extract_frame(VIDEO_PATH, EXCLUDE_REF_FRAME)
        regions = collect_exclude_regions(ref)
    EXCLUDE_REGIONS_RUNTIME = regions
    print(f"\nusing exclude regions: {regions}")

    saved_frames = set()
    rows = [process_pair(a, b, regions, saved_frames) for (a, b) in FRAME_PAIRS]
    print_and_save_table(rows)


if __name__ == "__main__":
    main()
