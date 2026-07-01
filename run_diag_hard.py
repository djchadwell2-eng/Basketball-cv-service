"""THIN READ-ONLY diagnostic harness: run the current system's config-driven parts
on the FULL HARD.mp4 and surface integration/safety seams. Creates NO new logic and
modifies NO Phase 1 / Phase 2 internals -- it only (a) selects the HARD clip via the
existing ACTIVE knob at runtime, (b) invokes existing functions to profile the
calibration across the whole pan, and (c) reports the seams that block a full
end-to-end P1+P2 run. It patches nothing; where a seam breaks the run, it reports.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "spikes"))
sys.path.insert(0, os.path.join(_ROOT, "phase1"))
sys.path.insert(0, os.path.join(_ROOT, "phase2"))

import clips_config as cfg
cfg.ACTIVE = "HARD"                      # the ONE exposed knob: select the clip at runtime

import numpy as np                       # noqa: E402
import cv2                               # noqa: E402
import stage1_keyframe_match as s1       # noqa: E402
import stage2_multikeyframe as s2        # noqa: E402
import stage4_courtmap as s4             # noqa: E402
import refit_keyframes as rk             # noqa: E402

VIDEO = cfg.active()["video_path"]
KF = cfg.active()["keyframes"]
SAMPLES = list(range(300, 2746, 100))    # profile the WHOLE clip (~91s @30fps)


def norm(H):
    return H / H[2, 2] if (H is not None and H[2, 2] != 0) else H


def anchor_profile(Hs, H_court, kf_imgs):
    """Per sampled frame: direct-anchor match quality to the nearest HARD keyframe."""
    KF_arr = np.array(KF)
    frames = s2.extract_frames(VIDEO, [f for f in SAMPLES])
    rows = []
    for f in SAMPLES:
        k = int(KF_arr[np.argmin(np.abs(KF_arr - f))])
        pos = KF.index(k)
        cv2.setRNGSeed(0)
        kp_k, kp_f, good = s1.detect_and_match(kf_imgs[k], frames[f], 0.75,
                                               s2.EXCLUDE_REGIONS)
        H, mask, inl = s1.estimate_homography(kp_k, kp_f, good, 3.0)
        if H is None or inl == 0:
            rows.append((f, k, abs(f - k), 0, None))
            continue
        im = [m for m, keep in zip(good, mask.ravel()) if keep]
        pf = np.float32([kp_f[m.trainIdx].pt for m in im]).reshape(-1, 1, 2)
        pk = np.float32([kp_k[m.queryIdx].pt for m in im])
        proj = cv2.perspectiveTransform(pf, H).reshape(-1, 2)
        reproj = float(np.linalg.norm(proj - pk, axis=1).mean())
        rows.append((f, k, abs(f - k), int(inl), reproj))
    return rows


def main():
    print(f"=== FULL-CLIP DIAGNOSTIC on {cfg.ACTIVE} ({VIDEO}) ===")
    cap = cv2.VideoCapture(VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"clip: {total} frames @ {fps:.0f}fps = {total/fps:.1f}s   keyframes={KF}")

    # --- Phase 1 CALIBRATION (config-driven -> runs on HARD) ---
    print("\n[1] Keyframe-consistency re-fit on HARD ...")
    KFr, ref_pos, Hs0, L0, tags, obs, corr = rk._setup()
    before = rk.keyframe_consistency(Hs0, corr)
    Hs1, L1, res = rk._solve(KFr, ref_pos, Hs0, L0, tags, obs, corr, rk.CORR_WEIGHT)
    after = rk.keyframe_consistency(Hs1, corr)
    H_court, per, mean_ft, max_ft = s4.compute_H_court(L1, tags)
    bm = np.mean([b[2] for b in before]); am = np.mean([a[2] for a in after])
    print(f"  keyframe mutual-consistency (ref px): mean {bm:.1f} -> {am:.1f}")
    print(f"  landmark court-fit: mean {mean_ft:.2f} ft / max {max_ft:.2f} ft")

    # --- calibration hold ACROSS THE WHOLE PAN ---
    print("\n[2] Direct-anchor calibration profile across the clip "
          "(inliers/reproj to nearest keyframe):")
    kf_imgs = s2.extract_frames(VIDEO, KF)
    rows = anchor_profile(Hs1, H_court, kf_imgs)
    print(f"  {'frame':>6} {'t(s)':>6} {'kf':>5} {'dist':>5} {'inliers':>8} {'reproj_px':>10}")
    for (f, k, d, inl, reproj) in rows:
        rp = "FAIL" if reproj is None else f"{reproj:.2f}"
        flag = "  <-- weak/failed" if (inl < 200) else ""
        print(f"  {f:>6} {f/fps:>6.1f} {k:>5} {d:>5} {inl:>8} {rp:>10}{flag}")

    ok = [r for r in rows if r[3] >= 200]
    print(f"\n  frames with a strong anchor (>=200 inliers): {len(ok)}/{len(rows)}")
    covered = [r[0] for r in ok]
    if covered:
        print(f"  strong-anchor frame range: {min(covered)}..{max(covered)} "
              f"(keyframes span {min(KF)}..{max(KF)})")

    print("\n[3] Downstream P1-team-events + P2-identity: NOT RUN -- see RUN_REPORT.md")
    print("    (frame ranges, roster, seed labels, and tracks cache are TEST1-hardcoded;")
    print("     no HARD equivalents exist. Reported as seams, not patched.)")


if __name__ == "__main__":
    main()
