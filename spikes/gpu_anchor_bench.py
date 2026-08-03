"""gpu_anchor_bench.py -- put the CAMERA ANCHOR on the GPU, and prove it.

WHY. Working out where the camera points is the pipeline's real scaling wall:
3.35 s/frame on DJ's laptop for a 15-second clip, ~179 s/frame on a full game
under load. YOLO detection was 68 h -> 31 min on the endpoint's 4090; this
stage never moved, because it calls OpenCV's CPU SIFT.

WHAT THIS DOES. The same job -- match a frame to its nearest keyframe, estimate
the homography -- with feature detection and matching done in torch on the GPU
(kornia's SIFT), then RANSAC on the handful of matched points (cheap, stays on
the CPU where OpenCV is fine).

WHAT IT REPORTS. Speed is the easy half. The half that decides whether this is
usable is AGREEMENT: for each frame, where would a player standing at a known
spot be placed by the GPU anchor versus the CPU one? Measured in FEET, because
feet is what the pipeline decides in and what DJ judges a court by (0.21 ft
"utter perfection", 0.94 ft "broken"). A fast anchor that moves players a foot
is not a win.

Runs on the GPU worker (mode "anchorbench"), because nothing local has CUDA.
"""

from __future__ import annotations

import os
import sys
import time

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "spikes"),
           os.path.join(_ROOT, "phase1"), os.path.join(_ROOT, "phase2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Same probe spots the subsampling test uses: spread over the floor, away from
# the horizon where any homography's error explodes and basketball never happens.
PROBE_FEET = [(x, y) for x in (10, 25, 42, 59, 74) for y in (8, 25, 42)]

RANSAC_PX = 3.0


def _gpu_sift(device, n_features=4000):
    import kornia.feature as KF
    return KF.SIFTFeature(num_features=n_features, device=device).eval()


def _to_tensor(img_bgr, device):
    import torch
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return torch.from_numpy(gray)[None, None].to(device)


def _describe(sift, t):
    """Keypoint centres + descriptors for one image, on the GPU."""
    import torch
    with torch.inference_mode():
        lafs, _, desc = sift(t)
    import kornia.feature as KF
    return KF.get_laf_center(lafs)[0], desc[0]


def _match(desc_a, pts_a, desc_b, pts_b):
    import torch
    import kornia.feature as KF
    with torch.inference_mode():
        _, idxs = KF.match_smnn(desc_a, desc_b, 0.9)
    return (pts_a[idxs[:, 0]].cpu().numpy(), pts_b[idxs[:, 1]].cpu().numpy())


def _gpu_match(sift, ta, tb):
    """Both images described from scratch -- the naive version, kept so the
    keyframe-caching win can be measured against it rather than assumed."""
    pa, da = _describe(sift, ta)
    pb, db = _describe(sift, tb)
    return _match(da, pa, db, pb)


def _feet_error(H_court, T_ref, T_test):
    """How far apart do the two anchors place the same court spots? In feet."""
    try:
        f2p_ref = np.linalg.inv(H_court @ T_ref)
        f2p_test = np.linalg.inv(H_court @ T_test)
    except np.linalg.LinAlgError:
        return None
    p2f = H_court @ T_ref
    errs = []
    for (x, y) in PROBE_FEET:
        a = f2p_ref @ np.array([x, y, 1.0])
        b = f2p_test @ np.array([x, y, 1.0])
        if abs(a[2]) < 1e-9 or abs(b[2]) < 1e-9:
            continue
        fa = p2f @ np.array([a[0] / a[2], a[1] / a[2], 1.0])
        fb = p2f @ np.array([b[0] / b[2], b[1] / b[2], 1.0])
        if abs(fa[2]) < 1e-9 or abs(fb[2]) < 1e-9:
            continue
        errs.append(float(np.hypot(fa[0] / fa[2] - fb[0] / fb[2],
                                   fa[1] / fa[2] - fb[1] / fb[2])))
    return errs or None


def subsample(clip: str, starts: list, n_frames: int = 300, ns=(2, 5, 10, 15, 30, 60)) -> dict:
    """THE DECIDING TEST: can we anchor every Nth frame instead of every frame?

    Anchoring every frame is 27.9 h of GPU work for a 95-minute game; even
    across ten workers that is 2.8 h, and the owner's bar is 30 minutes. Every
    other saving on the table is already counted. This is the step that decides
    whether the target is reachable at all.

    The yardstick is the GPU anchor run on EVERY frame -- itself already shown
    to agree with the original CPU anchor to 0.008 ft mean. Error is reported
    in FEET at spots spread over the floor, for the frames that would be
    SKIPPED. Two ways of filling a gap are compared: reuse the last anchor
    ("hold"), or blend the two either side ("interp").

    Run at several places in the game on purpose: a still camera during a free
    throw proves nothing about a fast break, and the honest number is the worst
    one, not the average of the easy ones.
    """
    import torch
    import clip_config
    import clips_config
    cfg = clip_config.get_clip(clip)
    if cfg is None:
        return {"error": f"no clip {clip}"}
    clip_config.ACTIVE_CLIP = cfg
    clips_config.ACTIVE = clip

    import stage1_court_roi as st
    import stage2_multikeyframe as s2
    import refit_keyframes

    device = "cuda" if torch.cuda.is_available() else "cpu"
    H_court, _anchor, fps, total = st.build_court_anchor()
    KF_list, ref_pos, Hs_opt, L_opt, tags = refit_keyframes.refit()
    kf_imgs = s2.extract_frames(s2.VIDEO_PATH, KF_list)
    KF_arr = np.array(KF_list)

    sift = _gpu_sift(device)
    kf_desc = {k: _describe(sift, _to_tensor(img, device)) for k, img in kf_imgs.items()}

    spots = []
    for start in starts:
        frames = list(range(start, start + n_frames))
        truth, failed = {}, 0
        t0 = time.time()
        for f, im in s2.iter_frames(cfg.video_path, frames):
            k = int(KF_arr[np.argmin(np.abs(KF_arr - f))])
            pts_k, desc_k = kf_desc[k]
            pts_f, desc_f = _describe(sift, _to_tensor(im, device))
            pk, pf = _match(desc_k, pts_k, desc_f, pts_f)
            if len(pk) < 10:
                failed += 1
                continue
            H, _ = cv2.findHomography(pf.reshape(-1, 1, 2), pk.reshape(-1, 1, 2),
                                      cv2.RANSAC, RANSAC_PX)
            if H is None:
                failed += 1
                continue
            T = Hs_opt[KF_list.index(k)] @ H
            truth[f] = T / T[2, 2]
        per_frame = (time.time() - t0) / max(1, len(frames))

        ok = sorted(truth)
        rows = []
        for N in ns:
            anchored = [f for i, f in enumerate(ok) if i % N == 0]
            if len(anchored) < 2:
                continue
            for mode in ("hold", "interp"):
                errs = []
                for f in ok:
                    if f in anchored:
                        continue
                    prev = max([a for a in anchored if a <= f], default=None)
                    nxt = min([a for a in anchored if a >= f], default=None)
                    if prev is None and nxt is None:
                        continue
                    if mode == "hold" or nxt is None or prev is None or prev == nxt:
                        T_est = truth[prev if prev is not None else nxt]
                    else:
                        w = (f - prev) / (nxt - prev)
                        T_est = (1 - w) * truth[prev] + w * truth[nxt]
                        T_est = T_est / T_est[2, 2]
                    e = _feet_error(H_court, truth[f], T_est)
                    if e:
                        errs.extend(e)
                if errs:
                    rows.append({"N": N, "fill": mode,
                                 "mean_ft": round(float(np.mean(errs)), 3),
                                 "p95_ft": round(float(np.percentile(errs, 95)), 3),
                                 "max_ft": round(float(np.max(errs)), 3),
                                 "gpu_hours_full_game": round(171120 * per_frame / N / 3600, 2)})
        spots.append({"start": start, "minute": round(start / 30 / 60, 1),
                      "anchored_ok": len(ok), "failed": failed,
                      "gpu_s_per_frame": round(per_frame, 3), "rows": rows})

    return {"clip": clip, "device": device, "frames_per_spot": n_frames, "spots": spots}


def bench(clip: str, start: int, n_frames: int = 20, cpu_frames: int | None = None) -> dict:
    """cpu_frames: how many frames to ALSO do the slow way. The CPU path costs
    ~49 s/frame on a full game, so timing it on every frame turns a 2-minute
    measurement into a 2-hour one. A handful is enough to establish both its
    rate and whether the GPU agrees with it."""
    import torch
    import clip_config
    import clips_config
    cfg = clip_config.get_clip(clip)
    if cfg is None:
        return {"error": f"no clip {clip}"}
    clip_config.ACTIVE_CLIP = cfg
    clips_config.ACTIVE = clip

    import stage1_court_roi as st
    import stage2_multikeyframe as s2
    import refit_keyframes

    device = "cuda" if torch.cuda.is_available() else "cpu"
    H_court, anchor, fps, total = st.build_court_anchor()

    # The keyframe images the CPU anchor matches against -- reused here so both
    # paths are doing the SAME job, not two different ones.
    KF_list, ref_pos, Hs_opt, L_opt, tags = refit_keyframes.refit()
    kf_imgs = s2.extract_frames(s2.VIDEO_PATH, KF_list)
    KF_arr = np.array(KF_list)

    frames = list(range(start, start + n_frames))
    imgs = [(f, im) for f, im in s2.iter_frames(cfg.video_path, frames)]
    if not imgs:
        return {"error": "no frames read"}

    # --- CPU baseline -------------------------------------------------------
    cpu_n = min(cpu_frames or len(imgs), len(imgs))
    t0 = time.time()
    cpu_T = {}
    for f, im in imgs[:cpu_n]:
        T, inl, reproj, kf = anchor(f, im)
        cpu_T[f] = T
    cpu_per = (time.time() - t0) / max(1, cpu_n)

    # --- GPU ----------------------------------------------------------------
    sift = _gpu_sift(device)
    kf_tensors = {k: _to_tensor(img, device) for k, img in kf_imgs.items()}
    # one warm-up so CUDA init and kernel compilation are not counted as speed
    _gpu_match(sift, kf_tensors[KF_list[0]], _to_tensor(imgs[0][1], device))

    # THE KEYFRAMES ARE DESCRIBED ONCE. There are five of them and they never
    # change, yet the naive loop re-analysed the same photo for every frame of
    # the game -- re-reading the map at every step instead of once before
    # setting off. Doing it up front removes half the per-frame work at zero
    # cost to the result: the descriptors are identical either way.
    kf_desc = {k: _describe(sift, t) for k, t in kf_tensors.items()}

    t0 = time.time()
    gpu_T, gpu_fail = {}, 0
    for f, im in imgs:
        k = int(KF_arr[np.argmin(np.abs(KF_arr - f))])
        pos = KF_list.index(k)
        pts_k, desc_k = kf_desc[k]
        pts_f, desc_f = _describe(sift, _to_tensor(im, device))
        pk, pf = _match(desc_k, pts_k, desc_f, pts_f)
        if len(pk) < 10:
            gpu_fail += 1
            gpu_T[f] = None
            continue
        H, mask = cv2.findHomography(pf.reshape(-1, 1, 2), pk.reshape(-1, 1, 2),
                                     cv2.RANSAC, RANSAC_PX)
        if H is None:
            gpu_fail += 1
            gpu_T[f] = None
            continue
        T = Hs_opt[pos] @ H
        gpu_T[f] = T / T[2, 2]
    gpu_per = (time.time() - t0) / len(imgs)

    # --- do they agree? -----------------------------------------------------
    all_errs = []
    compared = 0
    for f, _ in imgs:
        if cpu_T.get(f) is None or gpu_T.get(f) is None:
            continue
        e = _feet_error(H_court, cpu_T[f], gpu_T[f])
        if e:
            all_errs.extend(e)
            compared += 1

    full_game = 171120
    return {
        "clip": clip, "start": start, "frames": len(imgs), "device": device,
        "cpu_frames_timed": cpu_n,
        "cpu_s_per_frame": round(cpu_per, 3),
        "gpu_s_per_frame": round(gpu_per, 3),
        "speedup": round(cpu_per / gpu_per, 1) if gpu_per else None,
        "cpu_full_game_hours": round(full_game * cpu_per / 3600, 1),
        "gpu_full_game_hours": round(full_game * gpu_per / 3600, 1),
        "gpu_failed_frames": gpu_fail,
        "frames_compared": compared,
        "agreement_mean_ft": round(float(np.mean(all_errs)), 3) if all_errs else None,
        "agreement_max_ft": round(float(np.max(all_errs)), 3) if all_errs else None,
    }
