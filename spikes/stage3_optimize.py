"""Stage 3: global (bundle-adjustment-style) refinement of the Stage 2 assembly.

Stage 2 chained per-keyframe homographies to one reference frame; the same
physical landmark seen in several keyframes lands ~8px apart because the chain
stacks small errors. Stage 3 adjusts the keyframe transforms AND the consolidated
landmark positions TOGETHER (least-squares) to minimize that disagreement -- the
most globally-consistent solution -- instead of trusting the one-directional
chain.

This is a focused least-squares refinement (scipy), NOT a SLAM framework. It does
NOT compute the final pixel->court homography and does NOT draw a video overlay
(that's Stage 4). It ends at "shared-landmark consistency reduced (or shown
near-optimal) and the court still geometrically sane."

Reuses Stage 2 (imported; stage2 left intact). Deterministic: it starts from
Stage 2's deterministic chained solution and least_squares has no randomness.
"""

import os
import sys
import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage2_multikeyframe as s2   # reuse frames, homographies, LANDMARKS, plot

OUT_DIR = s2.OUT_DIR


# ---- small homography helpers -----------------------------------------------
def H_from8(p):
    """3x3 homography from 8 free params (bottom-right fixed at 1)."""
    return np.array([[p[0], p[1], p[2]],
                     [p[3], p[4], p[5]],
                     [p[6], p[7], 1.0]])


def H_to8(H):
    H = H / H[2, 2]
    return [H[0, 0], H[0, 1], H[0, 2], H[1, 0], H[1, 1], H[1, 2], H[2, 0], H[2, 1]]


def project(H, x, y):
    d = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    return ((H[0, 0] * x + H[0, 1] * y + H[0, 2]) / d,
            (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / d)


def transform_obs(Hs, observations, keyframes):
    """Project every observation into reference space with the given transforms."""
    out = []
    for (pos, tag, x, y) in observations:
        qx, qy = project(Hs[pos], x, y)
        out.append({"tag": tag, "frame": keyframes[pos], "ref_xy": (qx, qy)})
    return out


def shared_consistency(transformed):
    """Same metric Stage 2 reports: per tag seen in >=2 keyframes, the max
    pairwise distance between its projected positions; then mean/max over tags."""
    by = {}
    for t in transformed:
        by.setdefault(t["tag"], []).append(t["ref_xy"])
    detail, spreads = {}, []
    for tag, pts in by.items():
        if len(pts) < 2:
            continue
        a = np.array(pts)
        s = max(np.hypot(*(a[i] - a[j]))
                for i in range(len(a)) for j in range(i + 1, len(a)))
        detail[tag] = s
        spreads.append(s)
    mean = float(np.mean(spreads)) if spreads else 0.0
    mx = float(np.max(spreads)) if spreads else 0.0
    return detail, mean, mx


def extent(transformed):
    """Bounding-box size of all projected points -- a collapse/explode guard."""
    a = np.array([t["ref_xy"] for t in transformed])
    return (a[:, 0].max() - a[:, 0].min(), a[:, 1].max() - a[:, 1].min())


# ==============================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    KF = s2.KEYFRAMES
    n = len(KF)
    ref_pos = s2.REFERENCE_POS if s2.REFERENCE_POS is not None else n // 2
    ref_idx = KF[ref_pos]
    print(f"Keyframes: {KF}   reference: {ref_idx} (pos {ref_pos})")

    # ---- INITIALIZE from Stage 2's chained solution (CRITICAL: start from the
    # ~8px assembled state, not from scratch; bad init is how BA fails) --------
    frames = s2.extract_frames(s2.VIDEO_PATH, KF)
    fwd, _, _ = s2.adjacent_homographies(frames, KF, s2.EXCLUDE_REGIONS)
    H_to_ref = s2.chain_to_reference(fwd, ref_pos, n)
    assert all(H is not None for H in H_to_ref), "a chain hop failed -- cannot init"
    Hs_init = {p: (np.eye(3) if p == ref_pos else s2._norm(H_to_ref[p]))
               for p in range(n)}

    observations = []
    for pos, idx in enumerate(KF):
        for (tag, x, y) in s2.LANDMARKS.get(idx, []):
            observations.append((pos, tag, float(x), float(y)))
    tags = sorted({t for (_, t, _, _) in observations})

    # BEFORE consistency + initial landmark positions (centroid of projections)
    tr_before = transform_obs(Hs_init, observations, KF)
    det_b, mean_b, max_b = shared_consistency(tr_before)
    by = {}
    for t in tr_before:
        by.setdefault(t["tag"], []).append(t["ref_xy"])
    L_init = {tag: np.mean(np.array(by[tag]), axis=0) for tag in tags}

    # ---- variables: 8 params per NON-reference keyframe + 2 per landmark ------
    # Reference keyframe stays identity (fixed anchor) -> defines the coordinate
    # frame and prevents the degenerate "collapse everything to a point" solution.
    nonref = [p for p in range(n) if p != ref_pos]

    def pack(Hs, L):
        x = []
        for p in nonref:
            x += H_to8(Hs[p])
        for tag in tags:
            x += [float(L[tag][0]), float(L[tag][1])]
        return np.array(x, dtype=np.float64)

    def unpack(x):
        Hs = {ref_pos: np.eye(3)}
        for k, p in enumerate(nonref):
            Hs[p] = H_from8(x[8 * k:8 * k + 8])
        base = 8 * len(nonref)
        L = {tag: x[base + 2 * j: base + 2 * j + 2] for j, tag in enumerate(tags)}
        return Hs, L

    def residuals(x):
        """For every observation: (projected_position - consolidated_landmark).
        Minimizing these reconciles repeated observations of the same point."""
        Hs, L = unpack(x)
        r = np.empty(2 * len(observations))
        for i, (pos, tag, px, py) in enumerate(observations):
            qx, qy = project(Hs[pos], px, py)
            Lx, Ly = L[tag]
            r[2 * i] = qx - Lx
            r[2 * i + 1] = qy - Ly
        return r

    x0 = pack(Hs_init, L_init)
    cost0 = 0.5 * float(np.sum(residuals(x0) ** 2))
    print(f"\n{len(observations)} observations, {len(tags)} landmarks, "
          f"{len(x0)} variables. Initial cost = {cost0:.1f}")

    # x_scale='jac' handles the very different magnitudes of homography entries
    # (translation ~1e2, perspective terms ~1e-5) and point coords (~1e3).
    result = least_squares(residuals, x0, method="trf", x_scale="jac",
                           max_nfev=10000)

    Hs_opt, _ = unpack(result.x)
    tr_after = transform_obs(Hs_opt, observations, KF)
    det_a, mean_a, max_a = shared_consistency(tr_after)

    # ---- regenerate the assembled scatter on OPTIMIZED transforms ------------
    by_after = {}
    for t in tr_after:
        by_after.setdefault(t["tag"], []).append(t)
    scatter_path = os.path.join(OUT_DIR, "stage3_optimized_landmarks.png")
    s2.plot_assembly(tr_after, by_after, KF, ref_idx, s2.LANDMARK_TAGS, scatter_path)

    # ---- report --------------------------------------------------------------
    ext_b, ext_a = extent(tr_before), extent(tr_after)
    lines = []
    lines.append("Stage 3 -- global bundle-adjustment refinement")
    lines.append(f"keyframes: {KF}   reference (fixed): {ref_idx}")
    lines.append(f"observations: {len(observations)}  landmarks: {len(tags)}  "
                 f"variables: {len(x0)}")
    lines.append("")
    lines.append(f"optimizer: scipy least_squares (trf)  converged={result.success}  "
                 f"status={result.status}  nfev={result.nfev}")
    lines.append(f"  message: {result.message}")
    lines.append(f"  cost: {cost0:.1f} -> {result.cost:.1f}")
    lines.append("")
    lines.append("shared-landmark consistency (px):")
    lines.append(f"  {'metric':<12}{'BEFORE':>10}{'AFTER':>10}")
    lines.append(f"  {'mean spread':<12}{mean_b:>10.2f}{mean_a:>10.2f}")
    lines.append(f"  {'max spread':<12}{max_b:>10.2f}{max_a:>10.2f}")
    lines.append("")
    lines.append("per-landmark spread (px)  BEFORE -> AFTER:")
    for tag in sorted(det_b):
        lines.append(f"  {tag:<18}{det_b[tag]:>7.1f} -> {det_a.get(tag, 0):>6.1f}")
    lines.append("")
    lines.append(f"assembly extent (px)  BEFORE: {ext_b[0]:.0f}x{ext_b[1]:.0f}  "
                 f"AFTER: {ext_a[0]:.0f}x{ext_a[1]:.0f}  "
                 f"(should stay similar -- guard against collapse/explode)")
    if mean_a > mean_b + 1e-6:
        lines.append("\n  *** NOTE: AFTER mean is WORSE than BEFORE -- this signals a "
                     "setup problem, not a result to accept. ***")
    summary = "\n".join(lines)
    print("\n================ SUMMARY ================")
    print(summary)
    print("=========================================")

    spath = os.path.join(OUT_DIR, "stage3_summary.txt")
    with open(spath, "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print(f"saved summary: {spath}")
    print(f"saved optimized scatter: {scatter_path}")


if __name__ == "__main__":
    main()
