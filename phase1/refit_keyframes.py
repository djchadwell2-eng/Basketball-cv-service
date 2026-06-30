"""KEYFRAME-CONSISTENCY RE-FIT -- fix the handoff pop at its cause.

The live per-frame court homography routes each frame through its NEAREST keyframe:
    pixel -> (SIFT) -> keyframe k px -> (Hs[k]) -> reference px -> (H_court) -> feet
Per-frame ACCURACY is excellent. The remaining problem is CONTINUITY: the keyframe
transforms Hs[k] are not mutually consistent, so when a frame switches which
keyframe it anchors to, the court polygon jumps (37-515px at the 5 handoffs).

WHY the keyframes disagree: the existing global fit (s4.run_optimization) ties the
keyframes together ONLY at the sparse CLICKED landmarks, which for some keyframes
are clustered on one basket. Away from those points each keyframe's homography is
under-constrained and extrapolates differently -> the keyframes diverge.

THE FIX (same global least_squares machinery, one level up -- like calibration
Stage 3 did for landmarks): add DENSE adjacent-keyframe SIFT correspondences as
extra residuals. Each matched scene point seen in keyframes i and i+1 must project
to the SAME reference location:  project(Hs[i], p) == project(Hs[i+1], q). These
spread-out constraints pin the keyframe homographies down across their whole
overlap, so they agree -> the handoffs stop popping. Warm-started from the existing
solution so it only tightens consistency.

This re-fits the CAUSE (inconsistent keyframes). It does NOT blend/smooth handoffs
(rejected) and does NOT add keyframes (band-aid). See phase1/DECISIONS.md.
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "spikes"))

import stage1_keyframe_match as s1     # SIFT matcher (correspondences)
import stage2_multikeyframe as s2      # frames, LANDMARKS, chain init
import stage3_optimize as s3           # H_to8/H_from8/project (proven machinery)
import stage4_courtmap as s4           # run_optimization (warm-start) + compute_H_court

# Tunables (one place).
N_CORR_PER_PAIR = 80     # dense adjacent-keyframe correspondences per pair
CORR_WEIGHT = 1.0        # residual weight on consistency vs landmark anchoring
RANSAC_PX = 3.0
RATIO = 0.75

CACHE = os.path.join(_HERE, "out", f"{s2.cfg.ACTIVE}_keyframe_refit.npz")


def _project_many(H, pts):
    """Project Nx2 pixel points through H. Returns Nx2."""
    x, y = pts[:, 0], pts[:, 1]
    d = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    return np.stack([(H[0, 0]*x + H[0, 1]*y + H[0, 2]) / d,
                     (H[1, 0]*x + H[1, 1]*y + H[1, 2]) / d], axis=1)


def adjacent_correspondences(frames, KF, regions, n_per_pair):
    """RANSAC-inlier point pairs between each adjacent keyframe pair.
    Returns list of (i, j, pa[Nx2] in kf_i px, pb[Nx2] in kf_j px)."""
    rng = np.random.default_rng(0)
    corr = []
    for i in range(len(KF) - 1):
        a, b = KF[i], KF[i + 1]
        cv2.setRNGSeed(0)
        kp_a, kp_b, good = s1.detect_and_match(frames[a], frames[b], RATIO, regions)
        H, mask, inl = s1.estimate_homography(kp_a, kp_b, good, RANSAC_PX)
        inliers = [m for m, keep in zip(good, mask.ravel()) if keep]
        pa = np.array([kp_a[m.queryIdx].pt for m in inliers], dtype=np.float64)
        pb = np.array([kp_b[m.trainIdx].pt for m in inliers], dtype=np.float64)
        if len(pa) > n_per_pair:                       # spread sample (fixed seed)
            sel = rng.choice(len(pa), n_per_pair, replace=False)
            pa, pb = pa[sel], pb[sel]
        corr.append((i, i + 1, pa, pb))
        print(f"  kf {a}<->{b}: {len(inliers)} inliers, using {len(pa)} correspondences")
    return corr


def keyframe_consistency(Hs, corr):
    """Per adjacent pair, disagreement (ref px) when its correspondences are
    projected via Hs[i] vs Hs[j]. This is the keyframe MUTUAL-consistency error."""
    out = []
    for (i, j, pa, pb) in corr:
        qa = _project_many(Hs[i], pa)
        qb = _project_many(Hs[j], pb)
        d = np.linalg.norm(qa - qb, axis=1)
        out.append((i, j, float(d.mean()), float(d.max())))
    return out


def _setup():
    """Shared setup: KF, ref, frames, observations, tags, correspondences, and the
    warm-start solution (Hs0, L0) from the existing global fit."""
    KF, ref_pos, Hs0, L0, tags = s4.run_optimization()      # warm start
    frames = s2.extract_frames(s2.VIDEO_PATH, KF)
    observations = []
    for pos, idx in enumerate(KF):
        for (tag, x, y) in s2.LANDMARKS.get(idx, []):
            observations.append((pos, tag, float(x), float(y)))
    corr = adjacent_correspondences(frames, KF, s2.EXCLUDE_REGIONS, N_CORR_PER_PAIR)
    return KF, ref_pos, Hs0, L0, tags, observations, corr


def _solve(KF, ref_pos, Hs0, L0, tags, observations, corr, w):
    """Global least_squares: landmark residuals + dense correspondence residuals."""
    n = len(KF)
    nonref = [p for p in range(n) if p != ref_pos]

    def pack(Hs, L):
        x = []
        for p in nonref:
            x += s3.H_to8(Hs[p])
        for tag in tags:
            x += [float(L[tag][0]), float(L[tag][1])]
        return np.array(x, dtype=np.float64)

    def unpack(x):
        Hs = {ref_pos: np.eye(3)}
        for k, p in enumerate(nonref):
            Hs[p] = s3.H_from8(x[8 * k:8 * k + 8])
        base = 8 * len(nonref)
        L = {tag: x[base + 2 * j: base + 2 * j + 2] for j, tag in enumerate(tags)}
        return Hs, L

    def residuals(x):
        Hs, L = unpack(x)
        r = []
        for (pos, tag, px, py) in observations:           # landmark anchoring
            qx, qy = s3.project(Hs[pos], px, py)
            r += [qx - L[tag][0], qy - L[tag][1]]
        for (i, j, pa, pb) in corr:                       # keyframe consistency
            diff = (_project_many(Hs[i], pa) - _project_many(Hs[j], pb)) * w
            r += diff.ravel().tolist()
        return np.array(r)

    res = least_squares(residuals, pack(Hs0, L0), method="trf",
                        x_scale="jac", max_nfev=20000)
    Hs_opt, L_opt = unpack(res.x)
    return Hs_opt, L_opt, res


def refit(use_cache=True):
    """Return (KF, ref_pos, Hs_opt, L_opt, tags) with mutually-consistent keyframes.
    Drop-in for s4.run_optimization. Cached to a small .npz (delete to recompute)."""
    if use_cache and os.path.exists(CACHE):
        d = np.load(CACHE, allow_pickle=True)
        tags = list(d["tags"])
        Hs = {int(k): d["Hs"][int(k)] for k in range(len(d["Hs"]))}
        L = {t: d["L"][()][t] for t in tags}
        return list(d["KF"]), int(d["ref_pos"]), Hs, L, tags

    KF, ref_pos, Hs0, L0, tags, observations, corr = _setup()
    Hs_opt, L_opt, _res = _solve(KF, ref_pos, Hs0, L0, tags, observations, corr,
                                 CORR_WEIGHT)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, KF=np.array(KF), ref_pos=ref_pos, tags=np.array(tags, dtype=object),
             Hs=np.array([Hs_opt[p] for p in range(len(KF))]),
             L=np.array({t: L_opt[t] for t in tags}, dtype=object))
    return KF, ref_pos, Hs_opt, L_opt, tags


def _landmark_fit_ft(L, tags):
    """H_court reprojection residual (ft) for a given landmark solution."""
    H_court, per, mean, mx = s4.compute_H_court(L, tags)
    return mean, mx


def main():
    KF, ref_pos, Hs0, L0, tags, observations, corr = _setup()
    before = keyframe_consistency(Hs0, corr)
    Hs_opt, L_opt, res = _solve(KF, ref_pos, Hs0, L0, tags, observations, corr,
                                CORR_WEIGHT)
    after = keyframe_consistency(Hs_opt, corr)

    print("\n=========== KEYFRAME MUTUAL-CONSISTENCY (ref px) ===========")
    print(f"{'pair':>14} {'before_mean':>11} {'after_mean':>10} "
          f"{'before_max':>11} {'after_max':>10}")
    for (b, a) in zip(before, after):
        i, j = b[0], b[1]
        print(f"  kf{KF[i]:>4}<->{KF[j]:<4} {b[2]:>11.1f} {a[2]:>10.1f} "
              f"{b[3]:>11.1f} {a[3]:>10.1f}")
    bm = np.mean([b[2] for b in before]); am = np.mean([a[2] for a in after])
    bx = np.max([b[3] for b in before]); ax = np.max([a[3] for a in after])
    print(f"  {'OVERALL':>12} {bm:>11.1f} {am:>10.1f} {bx:>11.1f} {ax:>10.1f}")

    mb, xb = _landmark_fit_ft(L0, tags)
    ma, xa = _landmark_fit_ft(L_opt, tags)
    print(f"\nlandmark court-fit (ft): BEFORE mean={mb:.2f}/max={xb:.2f}  "
          f"AFTER mean={ma:.2f}/max={xa:.2f}  (must stay tight -- court accuracy)")
    print(f"optimizer: converged={res.success} nfev={res.nfev} "
          f"cost {res.cost:.1f}")

    # write the cache so downstream stages reuse this solution
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, KF=np.array(KF), ref_pos=ref_pos, tags=np.array(tags, dtype=object),
             Hs=np.array([Hs_opt[p] for p in range(len(KF))]),
             L=np.array({t: L_opt[t] for t in tags}, dtype=object))
    print(f"\nsaved refit -> {CACHE}")


if __name__ == "__main__":
    main()
