"""IS THE CAMERA REALLY A TRIPOD? -- the premise every click-saving idea rests on.

THE CLAIM (tasks/calibration_scale_options.md, M4): a camera rotating about a
fixed point with fixed intrinsics relates its views by a PURE-ROTATION
homography -- 3 degrees of freedom, 4 with zoom -- not the general 8. If true,
the per-keyframe unknowns drop from 8 to 4, those 4 come free from SIFT, and
clicks stop scaling with the keyframe count. That is where a claimed 12-18x
reduction comes from.

IF THE PREMISE IS FALSE the whole family of ideas weakens at once, so it is
worth an hour before a day of rewriting.

THE TEST COSTS ALMOST NOTHING because the answer is already on disk. The working
8-DOF solution (phase1/refit_keyframes) has fitted one homography per keyframe,
mapping that keyframe's pixels into the reference frame. So: try to explain
those SAME homographies with the restricted model

    H_i  ~=  K_ref . R_i . K_i^-1

where K is a pinhole with a shared principal point, R_i is a rotation (3
params), and K_i carries a per-keyframe focal length so zoom is allowed. Then
report, IN PIXELS, how far the restricted model's prediction sits from the
homography that actually fits DJ's clicks.

READING THE RESULT (bar stated before running):
    under ~2 px   the tripod model explains the camera -> M4/M9/M3 are sound
    2-10 px       partially; distortion or a small translation is present
    over ~10 px   the camera is NOT a pure rotation; those methods lose their
                  foundation and the honest move is to say so

Usage:  .venv/Scripts/python.exe spikes/tripod_premise_test.py TEST1
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.optimize import least_squares

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT, os.path.join(_ROOT, "phase1")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

W, H = 1920.0, 1080.0


def rot(v):
    """Rodrigues rotation from a 3-vector."""
    t = np.linalg.norm(v)
    if t < 1e-12:
        return np.eye(3)
    k = v / t
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(t) * Kx + (1 - np.cos(t)) * (Kx @ Kx)


def K_of(f, cx, cy):
    return np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])


def corner_residual(Ha, Hb, pts):
    """Mean distance (px) between two homographies' effect on the same points.
    Both are normalised first -- a homography is only defined up to scale."""
    out = []
    for Hm in (Ha, Hb):
        q = (Hm @ pts.T)
        out.append((q[:2] / q[2]).T)
    return np.linalg.norm(out[0] - out[1], axis=1)


def main():
    clip = sys.argv[1] if len(sys.argv) > 1 else "TEST1"
    import clips_config
    clips_config.ACTIVE = clip
    import clip_config
    clip_config.ACTIVE_CLIP = getattr(clip_config, f"{clip}_CLIP")

    import refit_keyframes as rk
    KF, ref_pos, Hs, _L, _tags = rk.refit()      # the WORKING 8-DOF solution
    n = len(KF)
    print(f"\n=== TRIPOD PREMISE TEST -- {clip} ===")
    print(f"  {n} keyframes {KF}, reference = kf {KF[ref_pos]}")

    # a grid of image points to measure agreement on (corners + centre + edges)
    gx, gy = np.meshgrid(np.linspace(0, W, 5), np.linspace(0, H, 5))
    pts = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)], axis=1)

    # NO NONLINEAR OPTIMISER. Two attempts failed -- the first diverged to a
    # 7-million-pixel focal, the second hit its iteration cap on all three clips
    # (converged=False). BOTH would have reported a false "premise fails". A
    # blind start is hopeless: keyframes 460 frames apart sit nowhere near the
    # identity rotation the optimiser begins from.
    #
    # Use the closed form panorama stitchers use instead. For a pure rotation
    # H = K_ref . R . K_i^-1, so K_ref^-1 . H . K_i MUST BE A ROTATION MATRIX.
    # Form that product and project it to the nearest rotation by SVD (R = U V^T).
    # HOW FAR IT HAD TO MOVE IS THE ANSWER: if the camera really is a tripod, it
    # barely moves. Focals come from a deterministic scan, so nothing has to
    # converge and the result is reproducible.
    cx, cy = W / 2.0, H / 2.0
    FOCALS = np.exp(np.linspace(np.log(500.0), np.log(12000.0), 40))

    def nearest_rotation(M):
        U, _s, Vt = np.linalg.svd(M)
        R = U @ Vt
        if np.linalg.det(R) < 0:                    # keep it a proper rotation
            U = U.copy(); U[:, -1] *= -1
            R = U @ Vt
        return R

    def fit_one(Hd, f_ref, f_i):
        Kr, Ki = K_of(f_ref, cx, cy), K_of(f_i, cx, cy)
        M = np.linalg.inv(Kr) @ Hd @ Ki
        s = abs(np.linalg.det(M)) ** (1.0 / 3.0)
        if s < 1e-12:
            return None, None
        R = nearest_rotation(M / s)
        Hm = Kr @ R @ np.linalg.inv(Ki)
        if abs(Hm[2, 2]) < 1e-12:
            return None, None
        return Hm / Hm[2, 2], R

    best = None
    for f_ref in FOCALS:
        per, tot = {}, 0.0
        for i in range(n):
            Hd = Hs[i] / Hs[i][2, 2]
            bi = None
            for f_i in FOCALS:                      # per-keyframe zoom allowed
                Hm, R = fit_one(Hd, f_ref, f_i)
                if Hm is None:
                    continue
                e = float(corner_residual(Hd, Hm, pts).mean())
                if bi is None or e < bi[0]:
                    bi = (e, f_i, Hm, R)
            if bi is None:
                tot = float("inf"); break
            per[i] = bi
            tot = max(tot, bi[0])                   # judged on the WORST keyframe
        if best is None or tot < best[0]:
            best = (tot, f_ref, per)

    worst, f_ref, per = best
    print(f"  closed-form solve: reference focal {f_ref:.0f} px "
          f"(principal point at image centre)")
    # CENTRE vs FULL FRAME -- the diagnostic that separates two very different
    # explanations for a large residual:
    #   small in the centre, large at the edges  -> LENS DISTORTION (this model
    #       is a pinhole and a wide-angle gym camera is not one). The tripod
    #       premise would still be sound; it just needs a radial term.
    #   large everywhere                          -> the camera genuinely is not
    #       a pure rotation (it translates), and M4/M9/M3 lose their basis.
    cgx, cgy = np.meshgrid(np.linspace(W * 0.3, W * 0.7, 4),
                           np.linspace(H * 0.3, H * 0.7, 4))
    cpts = np.stack([cgx.ravel(), cgy.ravel(), np.ones(cgx.size)], axis=1)

    print("")
    print(f"  {'kf':>6} {'zoom x':>8} {'rot deg':>8} {'centre px':>10} "
          f"{'full px':>9} {'max px':>8}")
    centre_worst = 0.0
    for i in range(n):
        e, f_i, Hm, R = per[i]
        Hd = Hs[i] / Hs[i][2, 2]
        d = corner_residual(Hd, Hm, pts)
        dc = corner_residual(Hd, Hm, cpts)
        ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2.0, -1, 1)))
        centre_worst = max(centre_worst, float(dc.mean()))
        print(f"  {int(KF[i]):>6} {f_i / f_ref:>8.3f} {ang:>8.2f} "
              f"{dc.mean():>10.2f} {d.mean():>9.2f} {d.max():>8.2f}")
    print(f"\n  CENTRE-ONLY worst: {centre_worst:.2f} px   FULL-FRAME worst: "
          f"{worst:.2f} px   ratio {worst / max(centre_worst, 1e-9):.1f}x")
    print(f"  -> a big ratio points at LENS DISTORTION (fixable with a radial "
          f"term); a ratio near 1 means the rotation model is simply wrong.")

    verdict = ("PREMISE HOLDS -- a tripod rotation explains this camera"
               if worst <= 2.0 else
               "PARTIAL -- distortion or a small translation is present"
               if worst <= 10.0 else
               "PREMISE FAILS -- not a pure rotation; M4/M9/M3 lose their basis")
    print(f"\n  WORST keyframe mean: {worst:.2f} px  =>  {verdict}")
    print(f"  (for scale: the 8-DOF solution's own keyframe mutual consistency "
          f"is ~0.6 px, and a court landmark sits ~0.15 ft = a few px)")


if __name__ == "__main__":
    main()
