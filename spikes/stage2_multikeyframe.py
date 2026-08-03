"""Stage 2: assemble landmarks from SEVERAL closely-spaced keyframes across the
HARD.mp4 pan into ONE shared coordinate system -- building the well-separated
landmark set that no single frame contains.

Stage 1 proved a direct two-keyframe SIFT+RANSAC homography (scorebug excluded)
is accurate for NEARBY keyframes (~3px @ 100 apart, ~8px @ 360, overlap gone
past ~400). So here we chain a FEW STRONG hops between closely-spaced keyframes
(~100-150 apart) -- not the hundreds of weak consecutive-frame hops that drifted
before -- to express every keyframe's landmarks in one reference frame's space.

Stage 2 ENDS at "landmarks combined + verified geometrically sane". It does NOT
compute the final pixel->court homography and does NOT draw a court-map overlay
(later stage). Small per-hop errors still accumulate across the chain -- that is
EXPECTED and SURFACED here (the consistency distances); Stage 3 will correct it.

Reuses Stage 1's SIFT/RANSAC/exclusion (imported; stage1 left intact). No bundle
adjustment, no YOLO/players, no deep matchers, no auto-detected landmarks, no
web/API. CONFIG-driven and deterministic.
"""

import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage1_keyframe_match as s1   # reuse detect_and_match / estimate_homography
import clips_config as cfg           # per-clip CONFIG (STEP 0 de-hardcode)

# ==============================================================================
# CONFIG  -- per-clip values come from clips_config.py (cfg.active()). Switch
# clips by setting clips_config.ACTIVE. The engine/math below is unchanged.
# ==============================================================================
_CLIP = cfg.active()
VIDEO_PATH = _CLIP["video_path"]

# Keyframe chain spanning the pan, ~100 apart so consecutive frames overlap
# strongly (e.g. HARD pans left-basket -> center -> right side).
KEYFRAMES = _CLIP["keyframes"]

# Reference keyframe POSITION in KEYFRAMES (None = middle -> fewest chain hops).
REFERENCE_POS = _CLIP["reference_pos"]

# Scorebug / burned-in graphics to mask out of SIFT -- DIFFERENT per gym.
EXCLUDE_REGIONS = _CLIP["exclude_regions"]
RATIO_TEST = 0.75
RANSAC_THRESH_PX = 3.0
WEAK_INLIER_RATIO = 0.6            # adjacent pairs below this are FLAGGED

# Canonical court landmark palette: (tag, court_x_ft, court_y_ft). You pick from
# this list by keypress while clicking, so the SAME physical point gets the SAME
# tag in every keyframe. Also used as the ideal court model for the comparison.
LANDMARK_TAGS = [
    ("LB_side_near",      0.0,  0.0),
    ("LB_side_far",       0.0, 50.0),
    ("L_lane_base_near",  0.0, 17.0),
    ("L_lane_base_far",   0.0, 33.0),
    ("L_FT_near",        19.0, 17.0),
    ("L_FT_far",         19.0, 33.0),
    ("center_near",      47.0,  0.0),
    ("center_logo",      47.0, 25.0),
    ("center_far",       47.0, 50.0),
    ("R_FT_near",        75.0, 17.0),
    ("R_FT_far",         75.0, 33.0),
    ("R_lane_base_near", 94.0, 17.0),
    ("R_lane_base_far",  94.0, 33.0),
    ("RB_side_near",     94.0,  0.0),
    ("RB_side_far",      94.0, 50.0),
    # --- Stage 3 bridge points: center circle (radius 6 ft @ (47,25)). left/
    # right are OFF the center line, so they actually constrain the wing-to-core
    # transform (collinear center-line points cannot). top/bottom are the clean
    # center-line x circle intersections. ---
    ("circle_top",       47.0, 31.0),
    ("circle_bottom",    47.0, 19.0),
    ("circle_left",      41.0, 25.0),
    ("circle_right",     53.0, 25.0),
]

# Clicked landmarks per keyframe: { frame_idx: [(tag, x, y), ...] }. Native px.
#   - keyframe ABSENT -> interactive click mode for it
#   - keyframe maps to [] -> no landmarks in that frame (skip)
# Per-clip; lives in clips_config.py.
LANDMARKS = _CLIP["landmarks"]

DISPLAY_MAX_W = 1280
SEED = 0
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")

COURT_LEN, COURT_WID = 94.0, 50.0   # for the ideal-court reference drawing


# ==============================================================================
def extract_frames(video_path, indices):
    """Frame-accurate read of the requested indices.

    SEEKS, then VERIFIES it landed on the exact frame, falling back to a
    sequential scan for any frame it missed (../fast_frames.py). It used to
    ALWAYS scan sequentially from frame 0 -- correct, but on a 171k-frame game
    that is minutes per call, and the calibration calls it twice.

    Measured 2026-07-31 on both full games: seeking is 5.4x faster, hits the
    exact frame every time, and returns PIXEL-IDENTICAL frames. The old comment
    here warned that H.264 seeks "can be frame-inaccurate"; that was never
    measured and is false for this footage. The verify-and-fall-back keeps the
    guarantee for any file where it is not.
    """
    need = sorted(set(indices))
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    import fast_frames
    frames = fast_frames.read_frames(video_path, need, verify=True)
    missing = [i for i in need if i not in frames]
    if missing:
        raise RuntimeError(f"Could not read frames {missing} from {video_path}")
    return frames


def iter_frames(video_path, indices):
    """Same single-pass frame-accurate read as extract_frames, but YIELDS
    (idx, frame) in increasing order instead of materializing every
    requested frame into one dict first -- O(1) frames held at a time
    instead of O(len(indices)). Phase 6: the full-game memory fix for
    every caller that only needs frames in order (never needs to hold
    more than the current one)."""
    need = sorted(set(indices))
    if not need:
        return
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # SEEK to the first wanted frame instead of grabbing every frame from zero.
    # extract_frames already does this via fast_frames (5.4x faster, landed
    # exactly 60/60 times, pixel-identical -- measured 2026-07-31); this
    # iterator never got the same treatment, and it is the one the per-frame
    # stages actually use. On a 95-minute game analysed in ten parallel slices
    # that mattered enormously: the worker handling the last tenth decoded
    # ~154,000 frames it then threw away just to reach its own.
    #
    # Same safety rule as fast_frames: seek, then CHECK where we landed. If the
    # file does not seek accurately we fall back to the scan from zero, so the
    # output is identical either way -- fast when that is correct, slow only
    # when it has to be.
    start = need[0]
    idx = 0
    if start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        landed = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if landed == start:
            idx = start
        else:
            cap.release()
            cap = cv2.VideoCapture(video_path)      # seek missed: scan instead

    it = iter(need)
    nxt = next(it, None)
    yielded = set()
    while nxt is not None:
        if not cap.grab():
            break
        if idx == nxt:
            ok, fr = cap.retrieve()
            if ok:
                yield idx, fr
                yielded.add(idx)
            nxt = next(it, None)
        idx += 1
    cap.release()
    missing = [i for i in need if i not in yielded]
    if missing:
        raise RuntimeError(f"Could not read frames {missing} from {video_path}")


def adjacent_homographies(frames, keyframes, regions):
    """For each consecutive pair, the FORWARD homography fwd[i]: K_i px -> K_{i+1} px.

    Stage1.detect_and_match(a, b) is query=a/train=b and estimate returns H mapping
    b -> a. So calling it with a=K_{i+1}, b=K_i gives H: K_i -> K_{i+1} = fwd[i].
    """
    fwd, stats, weak = [], [], []
    for i in range(len(keyframes) - 1):
        a_idx, b_idx = keyframes[i], keyframes[i + 1]
        print(f"\n-- adjacent pair {a_idx} -> {b_idx} (sep {b_idx - a_idx}) --")
        cv2.setRNGSeed(SEED)
        kp_b1, kp_a1, good = s1.detect_and_match(frames[b_idx], frames[a_idx],
                                                 RATIO_TEST, regions)
        H, mask, inliers = s1.estimate_homography(kp_b1, kp_a1, good, RANSAC_THRESH_PX)
        ratio = (inliers / len(good)) if good else 0.0
        fwd.append(H)
        stats.append({"a": a_idx, "b": b_idx, "matches": len(good),
                      "inliers": inliers, "ratio": ratio})
        if H is None or ratio < WEAK_INLIER_RATIO:
            weak.append((a_idx, b_idx, ratio))
            print(f"  *** WEAK PAIR FLAG: {a_idx}->{b_idx} inlier ratio {ratio:.3f} "
                  f"< {WEAK_INLIER_RATIO} -- these keyframes are too far apart; "
                  f"consider adding an intermediate keyframe between them. ***")

    # A None homography means the pair could not be related AT ALL -- not merely
    # weakly. It used to be passed on regardless, and the chain then carried the
    # None into the optimiser, which died deep inside with
    # "'NoneType' object is not subscriptable" -- a stack trace that says
    # nothing about the actual problem. Fail HERE, naming the two frames, so the
    # caller can tell someone which part of the game does not connect.
    dead = [(s["a"], s["b"], s["matches"]) for s, H in zip(stats, fwd) if H is None]
    if dead:
        pairs = "; ".join(f"{a} and {b} (only {m} matching points)" for a, b, m in dead)
        raise ValueError(
            f"NO_OVERLAP: these frames show too little of the same court to be "
            f"connected: {pairs}. The camera was pointing somewhere different, or "
            f"the view is obscured. A frame BETWEEN them is needed to bridge the gap.")
    return fwd, stats, weak


def _norm(H):
    return H / H[2, 2] if (H is not None and H[2, 2] != 0) else H


def chain_to_reference(fwd, ref_pos, n):
    """H_to_ref[i]: maps K_i pixels -> reference keyframe pixel space.

    Recurrence (planar court, single homography per hop):
      ref side, i < ref:  K_i --fwd[i]--> K_{i+1} --H_to_ref[i+1]--> ref
                          => H_to_ref[i] = H_to_ref[i+1] @ fwd[i]
      ref side, i > ref:  K_i --inv(fwd[i-1])--> K_{i-1} --H_to_ref[i-1]--> ref
                          => H_to_ref[i] = H_to_ref[i-1] @ inv(fwd[i-1])
    A few strong hops only -- but small per-hop error still compounds with chain
    length (Stage 2 surfaces this; Stage 3 fixes it). None propagates if a hop
    failed, so we don't fabricate a chain through a broken pair.
    """
    H_to_ref = [None] * n
    H_to_ref[ref_pos] = np.eye(3)
    for i in range(ref_pos - 1, -1, -1):          # leftwards from ref
        if fwd[i] is not None and H_to_ref[i + 1] is not None:
            H_to_ref[i] = _norm(H_to_ref[i + 1] @ fwd[i])
    for i in range(ref_pos + 1, n):               # rightwards from ref
        if fwd[i - 1] is not None and H_to_ref[i - 1] is not None:
            inv = np.linalg.inv(fwd[i - 1])
            H_to_ref[i] = _norm(H_to_ref[i - 1] @ inv)
    return H_to_ref


# ------------------------------------------------------------------------------
# Interactive landmark collection (palette-tagged), per keyframe.
# ------------------------------------------------------------------------------
def _palette_keymap(tags):
    """Map a keyboard char to a tag index: 1..9 then a,b,c,..."""
    keymap = {}
    for k in range(len(tags)):
        ch = str(k + 1) if k < 9 else chr(ord("a") + (k - 9))
        keymap[ord(ch)] = k
    return keymap


def collect_landmarks_for_frame(img, frame_idx, tags):
    """Click landmarks in this keyframe; pick each one's tag from the palette."""
    keymap = _palette_keymap(tags)
    print(f"\n  Keyframe {frame_idx}: pick a tag key, then click that landmark.")
    print("  Palette (press the key to select, then click). Only click what's visible.")
    for code, k in keymap.items():
        print(f"    [{chr(code)}] {tags[k][0]}")
    print("  Keys: pick tag -> click; u=undo last; ENTER=done with this frame.")

    h, w = img.shape[:2]
    scale = min(1.0, DISPLAY_MAX_W / float(w))
    disp0 = cv2.resize(img, (int(w * scale), int(h * scale)))
    active = {"i": 0}
    pts = []   # (tag_index, x_native, y_native)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((active["i"], x / scale, y / scale))

    win = f"keyframe {frame_idx} -- tag(key)+click"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        disp = disp0.copy()
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(disp, f"kf {frame_idx}  active tag = {tags[active['i']][0]}"
                          f"   (u=undo ENTER=done)", (6, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        for (ti, px, py) in pts:
            d = (int(px * scale), int(py * scale))
            cv2.circle(disp, d, 5, (0, 255, 0), -1)
            cv2.putText(disp, tags[ti][0], (d[0] + 6, d[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.imshow(win, disp)
        key = cv2.waitKey(20) & 0xFF
        if key in keymap:
            active["i"] = keymap[key]
        elif key == ord("u") and pts:
            pts.pop()
        elif key in (13, ord("q")):
            break
    cv2.destroyWindow(win)
    return [(tags[ti][0], round(px, 1), round(py, 1)) for (ti, px, py) in pts]


def collect_landmarks(frames, keyframes, tags, existing):
    """Return { frame_idx: [(tag,x,y),...] }, clicking where CONFIG lacks entries."""
    out = {}
    for idx in keyframes:
        if idx in existing:
            out[idx] = existing[idx]
        else:
            out[idx] = collect_landmarks_for_frame(frames[idx], idx, tags)
    # Printable block for deterministic re-runs.
    print("\n  ---- paste into CONFIG.LANDMARKS ----")
    print("LANDMARKS = {")
    for idx in keyframes:
        print(f"    {idx}: [")
        for (tag, x, y) in out[idx]:
            print(f"        ({tag!r}, {x}, {y}),")
        print("    ],")
    print("}")
    print("  -------------------------------------")
    return out


# ------------------------------------------------------------------------------
def transform_landmarks(landmarks, H_to_ref, keyframes):
    """Map each clicked landmark into reference pixel space. Returns list of
    {tag, frame, ref_xy}. Skips keyframes whose chain hop failed (H is None)."""
    out = []
    for pos, idx in enumerate(keyframes):
        H = H_to_ref[pos]
        if H is None:
            if landmarks.get(idx):
                print(f"  (skipping {len(landmarks[idx])} landmarks in kf {idx}: "
                      f"chain broken, no transform)")
            continue
        for (tag, x, y) in landmarks.get(idx, []):
            p = H @ np.array([x, y, 1.0])
            out.append({"tag": tag, "frame": idx, "ref_xy": (p[0] / p[2], p[1] / p[2])})
    return out


def consistency_report(transformed):
    """For tags clicked in >=2 keyframes, distance between transformed positions."""
    by_tag = {}
    for t in transformed:
        by_tag.setdefault(t["tag"], []).append(t)
    print("\n  shared-landmark consistency (same tag from different keyframes):")
    all_d = []
    for tag, items in sorted(by_tag.items()):
        if len(items) < 2:
            continue
        pts = np.array([it["ref_xy"] for it in items])
        cen = pts.mean(axis=0)
        dists = [float(np.hypot(*(p - cen))) for p in pts]
        spread = max(np.hypot(*(pts[i] - pts[j]))
                     for i in range(len(pts)) for j in range(i + 1, len(pts)))
        frs = [it["frame"] for it in items]
        all_d.append(spread)
        print(f"    {tag:18s} seen in {frs} -> max pairwise dist = {spread:6.1f}px")
    if all_d:
        print(f"    -> shared landmarks: {len(all_d)}, "
              f"mean spread = {np.mean(all_d):.1f}px, max spread = {np.max(all_d):.1f}px")
    else:
        print("    (no landmark was clicked in more than one keyframe)")
    return all_d, by_tag


def plot_assembly(transformed, by_tag, keyframes, ref_idx, tags, path):
    """Left: assembled landmarks in reference pixel space (perspective).
    Right: ideal court (top-down) for topology comparison."""
    tag_court = {name: (cx, cy) for (name, cx, cy) in tags}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- left: assembled (reference camera perspective; spans beyond image) ---
    cmap = plt.cm.viridis
    for pos, idx in enumerate(keyframes):
        pts = [t["ref_xy"] for t in transformed if t["frame"] == idx]
        if pts:
            arr = np.array(pts)
            ax1.scatter(arr[:, 0], arr[:, 1], s=25,
                        color=cmap(pos / max(1, len(keyframes) - 1)),
                        label=f"kf {idx}" + (" (ref)" if idx == ref_idx else ""))
    for tag, items in by_tag.items():
        cen = np.array([it["ref_xy"] for it in items]).mean(axis=0)
        ax1.annotate(tag, (cen[0], cen[1]), fontsize=7, color="black")
    ax1.add_patch(plt.Rectangle((0, 0), 1920, 1080, fill=False, ec="red",
                                lw=1, ls="--"))
    ax1.set_title(f"Assembled landmarks in reference kf {ref_idx} pixel space\n"
                  "(perspective; red = ref image bounds; extends beyond it)")
    ax1.set_xlabel("ref px x"); ax1.set_ylabel("ref px y")
    ax1.invert_yaxis(); ax1.legend(fontsize=7, loc="best"); ax1.set_aspect("equal")

    # --- right: ideal court top-down for shape/topology comparison ------------
    L, W = COURT_LEN, COURT_WID
    ax2.plot([0, L, L, 0, 0], [0, 0, W, W, 0], "k-")
    ax2.plot([L / 2, L / 2], [0, W], "k-")
    ax2.add_patch(plt.Circle((L / 2, W / 2), 6, fill=False, ec="k"))
    for (x1, y1, x2, y2) in [(0, 17, 19, 33), (L - 19, 17, L, 33)]:
        ax2.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], "k-")
    seen_tags = set(by_tag.keys())
    for tag in seen_tags:
        if tag in tag_court:
            cx, cy = tag_court[tag]
            ax2.scatter([cx], [cy], s=30, color="tab:red")
            ax2.annotate(tag, (cx, cy), fontsize=7)
    ax2.set_title("Ideal court (top-down) -- compare ARRANGEMENT/topology\n"
                  "(not exact shape: left is perspective, this is bird's-eye)")
    ax2.set_xlabel("court ft x"); ax2.set_ylabel("court ft y")
    ax2.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"\nsaved assembly scatter: {path}")


# ==============================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    n = len(KEYFRAMES)
    ref_pos = REFERENCE_POS if REFERENCE_POS is not None else n // 2
    ref_idx = KEYFRAMES[ref_pos]
    print(f"Video: {VIDEO_PATH}")
    print(f"Keyframe chain: {KEYFRAMES}")
    print(f"Reference keyframe: {ref_idx} (position {ref_pos})")
    print(f"Exclude regions: {EXCLUDE_REGIONS}")

    frames = extract_frames(VIDEO_PATH, KEYFRAMES)

    fwd, stats, weak = adjacent_homographies(frames, KEYFRAMES, EXCLUDE_REGIONS)
    H_to_ref = chain_to_reference(fwd, ref_pos, n)
    broken = [KEYFRAMES[i] for i in range(n) if H_to_ref[i] is None]
    if broken:
        print(f"\n  WARNING: no chain to reference for keyframes {broken} "
              f"(a hop failed/was degenerate). Their landmarks are skipped.")

    landmarks = collect_landmarks(frames, KEYFRAMES, LANDMARK_TAGS, LANDMARKS)
    transformed = transform_landmarks(landmarks, H_to_ref, KEYFRAMES)
    spreads, by_tag = consistency_report(transformed)
    plot_assembly(transformed, by_tag, KEYFRAMES, ref_idx, LANDMARK_TAGS,
                  os.path.join(OUT_DIR, "stage2_assembled_landmarks.png"))

    # ----- summary -----
    lines = []
    lines.append("Stage 2 -- multi-keyframe landmark assembly")
    lines.append(f"video: {VIDEO_PATH}")
    lines.append(f"keyframe chain: {KEYFRAMES}")
    lines.append(f"reference keyframe: {ref_idx} (pos {ref_pos})")
    lines.append("")
    lines.append("adjacent-pair inlier ratios:")
    for s in stats:
        flag = "  <-- WEAK" if s["ratio"] < WEAK_INLIER_RATIO else ""
        lines.append(f"  {s['a']:>5} -> {s['b']:>5}  matches={s['matches']:>5}  "
                     f"inliers={s['inliers']:>5}  ratio={s['ratio']:.3f}{flag}")
    lines.append("")
    distinct = sorted({t["tag"] for t in transformed})
    lines.append(f"distinct landmarks collected (transformed): {len(distinct)}")
    lines.append(f"  {distinct}")
    if spreads:
        lines.append(f"shared-landmark consistency: count={len(spreads)}  "
                     f"mean_spread={np.mean(spreads):.1f}px  max_spread={np.max(spreads):.1f}px")
    else:
        lines.append("shared-landmark consistency: (none clicked in 2+ keyframes)")
    if weak:
        lines.append(f"WEAK PAIRS FLAGGED: {[(a, b, round(r, 3)) for (a, b, r) in weak]}")
    summary = "\n".join(lines)
    print("\n================ SUMMARY ================")
    print(summary)
    print("=========================================")
    spath = os.path.join(OUT_DIR, "stage2_summary.txt")
    with open(spath, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"saved summary: {spath}")


if __name__ == "__main__":
    main()
