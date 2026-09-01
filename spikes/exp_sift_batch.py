"""exp_sift_batch.py -- is the camera anchor's SIFT wasting the GPU?

THE CASE FOR ASKING. The anchor is the single biggest compute item in the
pipeline: MEASURED 0.086 s/frame, which is 4.09 GPU-hours for a 95-minute game
and ~24 minutes of every slice's ~35. Of that 86 ms, MEASURED breakdown says ~5
ms is decoding the frame and ~2.2 ms is RANSAC at 4,000 points -- so roughly 70
ms is GPU work being fed ONE 1920x1080 image at a time, on a card built to take
batches. If describing several frames in one call amortises that, it comes
straight off every slice.

WHAT IS AND IS NOT BATCHED. Only the DESCRIPTION -- kornia's SIFTFeature over a
stack of frames. The matching stays per-frame, because each frame matches its own
NEAREST keyframe and those differ; and RANSAC stays on the CPU where it is
already cheap. Description is the expensive part, so it is the part worth
changing.

THE BAR IS NOT SPEED. Same descriptors or not, what matters is where the anchor
puts the court. The GPU anchor earned its place at 0.008 ft mean / 0.11 ft max
against the CPU one, on a floor DJ calls perfect at 0.21 ft and broken at 0.94.
A batched anchor that disagrees by more than a hundredth of a foot is a different
anchor, and this reports the disagreement IN FEET before it reports a speed.

Batch sizes climb until the card refuses; an out-of-memory batch is a reported
result, not a crash.

    .venv/Scripts/python.exe lab.py spikes/exp_sift_batch.py --clip <CLIP> \
        --args '{"clip": "<CLIP>", "start": 60000, "frames": 48}'
"""

from __future__ import annotations

import sys
import time


def run(clip=None, start=60000, frames=48, batches=(1, 2, 4, 8, 16), **_kwargs):
    for p in ("/app", "/app/spikes", "/app/phase1", "/app/phase2"):
        if p not in sys.path:
            sys.path.insert(0, p)
    import numpy as np
    import torch
    import clip_config
    import clips_config
    import gpu_anchor

    clip = clip or clips_config.ACTIVE
    cfg = clip_config.get_clip(clip)
    if cfg is None:
        return {"error": f"clip {clip} not installed"}
    import run_clip
    run_clip._sync_and_guard(cfg)          # both selectors, before the stage imports

    if not gpu_anchor.available():
        return {"error": "no CUDA on this worker"}

    import stage1_court_roi as st
    import stage2_multikeyframe as s2
    import refit_keyframes
    import gpu_anchor_bench as bench

    H_court, _anchor, _fps, _total = st.build_court_anchor()
    KF, ref_pos, Hs_opt, L_opt, tags = refit_keyframes.refit()
    kf_imgs = s2.extract_frames(s2.VIDEO_PATH, KF)

    want = list(range(int(start), int(start) + int(frames)))
    imgs = [im for _f, im in s2.iter_frames(cfg.video_path, want)]
    if not imgs:
        return {"error": "no frames read"}
    fidx = want[:len(imgs)]

    ga = gpu_anchor.GpuAnchor(kf_imgs, KF, Hs_opt, s2.EXCLUDE_REGIONS)
    KF_arr = np.array(KF)

    def match_from(pts_f, desc_f, f):
        """The rest of the anchor, once a frame is described -- unchanged."""
        import cv2
        k = int(KF_arr[np.argmin(np.abs(KF_arr - f))])
        pos = KF.index(k)
        pts_k, desc_k = ga._kf[k]
        with torch.inference_mode():
            _, idxs = ga._KF.match_smnn(desc_k, desc_f, gpu_anchor.LOWE_RATIO)
        if idxs.shape[0] < gpu_anchor.MIN_MATCHES:
            return None
        pk = pts_k[idxs[:, 0]].cpu().numpy()
        pf = pts_f[idxs[:, 1]].cpu().numpy()
        H, mask = cv2.findHomography(pf.reshape(-1, 1, 2), pk.reshape(-1, 1, 2),
                                     cv2.RANSAC, gpu_anchor.RANSAC_PX)
        if H is None:
            return None
        T = Hs_opt[pos] @ H
        return T / T[2, 2]

    def describe_batch(chunk):
        """Describe several frames in ONE call. Returns [(pts, desc), ...]."""
        t = torch.cat([ga._tensor(im) for im in chunk], dim=0)
        with torch.inference_mode():
            lafs, _resp, desc = ga.sift(t)
        centers = ga._KF.get_laf_center(lafs)
        return [(centers[i], desc[i]) for i in range(len(chunk))]

    out = {"clip": clip, "frames": len(imgs), "start": int(start),
           "gpu": torch.cuda.get_device_name(0), "runs": {}}
    truth = None
    for B in batches:
        B = int(B)
        try:
            torch.cuda.synchronize(); torch.cuda.empty_cache()
            t0 = time.time()
            Ts = {}
            for i in range(0, len(imgs), B):
                chunk = imgs[i:i + B]
                for j, (pts, desc) in enumerate(describe_batch(chunk)):
                    T = match_from(pts, desc, fidx[i + j])
                    if T is not None:
                        Ts[fidx[i + j]] = T
            torch.cuda.synchronize()
            per = (time.time() - t0) / max(1, len(imgs))
        except RuntimeError as e:          # out of memory is an ANSWER
            out["runs"][str(B)] = {"error": str(e)[:140]}
            torch.cuda.empty_cache()
            continue

        row = {"seconds_per_frame": round(per, 4),
               "placed": len(Ts),
               "full_game_gpu_hours": round(171120 * per / 3600, 2),
               "minutes_per_slice_of_10": round(17112 * per / 60, 1)}
        if truth is None:
            truth = Ts                     # batch 1 IS the current pipeline
            row["note"] = "this is today's anchor -- the yardstick"
        else:
            errs = []
            for f, T in Ts.items():
                if f in truth:
                    e = bench._feet_error(H_court, truth[f], T)
                    if e:
                        errs.extend(e)
            row["feet_vs_batch1"] = (
                {"mean": round(float(np.mean(errs)), 4),
                 "p95": round(float(np.percentile(errs, 95)), 4),
                 "max": round(float(np.max(errs)), 4), "n": len(errs)}
                if errs else "no comparable frames")
            row["frames_only_batch1_placed"] = len(set(truth) - set(Ts))
            row["frames_only_this_placed"] = len(set(Ts) - set(truth))
        out["runs"][str(B)] = row

    base = out["runs"].get("1", {}).get("seconds_per_frame")
    if base:
        for k, v in out["runs"].items():
            if "seconds_per_frame" in v:
                v["speedup"] = round(base / v["seconds_per_frame"], 2)
    out["bar"] = ("the GPU anchor earned its place at 0.008 ft mean vs the CPU; "
                  "a batched anchor disagreeing by more than a hundredth of a "
                  "foot is a different anchor, whatever it costs")
    return out
