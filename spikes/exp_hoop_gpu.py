"""exp_hoop_gpu.py -- does the GPU rim tracker put the rim where the CPU does?

WHAT IS BEING RISKED. The rim is the origin of every shot: shot_attempts asks
whether an arc reached it, shot_location asks where the shooter stood relative
to it. Move the rim by a few pixels and every shot outcome moves with it,
silently. An earlier session proposed sharing the court anchor's cache with the
rim tracker precisely because both "do SIFT"; a review caught that they use
DIFFERENT rules (nearest keyframe vs best of all) and it would have moved the
rim without anything failing.

So speed is not the question here -- 47-49 s/frame on the CPU means shots simply
cannot run, and any working GPU path wins. The question is AGREEMENT, in pixels,
on the same frames.

WHAT IT REPORTS, per frame:
    which keyframe each path chose        (they may legitimately differ)
    the far rim's pixel, both ways        and the distance between them
    the near rim's pixel, both ways
    how many frames one path placed and the other did not

The bar to beat: the court anchor earned its GPU path at 0.008 ft mean / 0.11 ft
max against the CPU. A rim is a point, not a court, so it is judged in pixels --
the rim is ~20 px across on this footage, so single-digit pixels is the shape of
an acceptable answer and tens of pixels is not.

    .venv/Scripts/python.exe lab.py spikes/exp_hoop_gpu.py --clip <CLIP> \
        --args '{"start": 154008, "frames": 40}'
"""

from __future__ import annotations

import os
import sys
import time


def run(clip=None, start=154008, frames=40, **_kwargs):
    for p in ("/app", "/app/spikes", "/app/phase1", "/app/phase2"):
        if p not in sys.path:
            sys.path.insert(0, p)
    import numpy as np
    import cv2
    import clip_config
    import clips_config
    import gpu_anchor

    # BOTH SELECTORS, BEFORE THE STAGE IMPORTS. stage4_courtmap binds its VIDEO
    # from clips_config.ACTIVE at IMPORT time, and _install_uploaded_clip
    # reloads that module -- which resets ACTIVE to its default, TEST1. Reading
    # ACTIVE here therefore returned TEST1 and the job died on
    # "could not open video: C:\\Users\\djcha\\Downloads\\Test1.mp4".
    # run_clip._sync_and_guard sets both and fails loud if they disagree; it is
    # the sanctioned way and the fourth experiment to need it.
    clip = clip or clips_config.ACTIVE
    cfg = clip_config.get_clip(clip)
    if cfg is None:
        return {"error": f"clip {clip} not installed"}
    if not cfg.hoop_anchors:
        return {"error": "no rims marked on this clip"}
    import run_clip
    run_clip._sync_and_guard(cfg)

    import hoop_anchor as HA
    import stage2_multikeyframe as s2
    import stage4_courtmap as s4

    KF, ref_pos, Hs_opt, L_opt, tags = s4.run_optimization()
    kf_imgs = s2.extract_frames(cfg.video_path, KF)
    sift = cv2.SIFT_create(nfeatures=HA.NFEAT)
    kf_db = HA._keyframe_db(cfg.video_path, KF, sift)

    if not gpu_anchor.multi_available():
        return {"error": "no CUDA on this worker"}
    gm = gpu_anchor.GpuMultiAnchor(kf_imgs, KF, HA.NFEAT, HA.RATIO,
                                   HA.RANSAC_PX, HA.MIN_INLIERS)

    # the two rims in reference-900 pixels, exactly as build_hoop_track does
    def ref900(anchor):
        kf, px = anchor
        if kf in KF:
            return HA.project_point(Hs_opt[KF.index(kf)], *px)
        cap = cv2.VideoCapture(cfg.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(kf))
        ok, img = cap.read()
        cap.release()
        m = HA._match_frame(img, sift, kf_db)
        pos, Hfk, _i, _r = m
        return HA.project_point(Hs_opt[pos] @ Hfk, *px)

    rim_far = ref900(cfg.hoop_anchors["far"])
    rim_near = ref900(cfg.hoop_anchors["near"])

    want = list(range(int(start), int(start) + int(frames)))
    rows, far_d, near_d = [], [], []
    only_cpu = only_gpu = neither = 0
    t_cpu = t_gpu = 0.0
    for f, img in s2.iter_frames(cfg.video_path, want):
        t0 = time.time(); mc = HA._match_frame(img, sift, kf_db); t_cpu += time.time() - t0
        t0 = time.time(); mg = gm.match(img);                     t_gpu += time.time() - t0
        if mc is None and mg is None:
            neither += 1
            continue
        if mc is None:
            only_gpu += 1
            continue
        if mg is None:
            only_cpu += 1
            continue
        row = {"frame": f, "kf_cpu": KF[mc[0]], "kf_gpu": KF[mg[0]],
               "inl_cpu": mc[2], "inl_gpu": mg[2]}
        for label, rim, acc in (("far", rim_far, far_d), ("near", rim_near, near_d)):
            pc = HA.project_point(np.linalg.inv(Hs_opt[mc[0]] @ mc[1]), *rim)
            pg = HA.project_point(np.linalg.inv(Hs_opt[mg[0]] @ mg[1]), *rim)
            if pc is None or pg is None:
                row[f"{label}_px"] = None
                continue
            d = float(np.hypot(pc[0] - pg[0], pc[1] - pg[1]))
            acc.append(d)
            row[f"{label}_px"] = round(d, 2)
        rows.append(row)

    def stat(a):
        return None if not a else {"mean": round(float(np.mean(a)), 2),
                                   "p95": round(float(np.percentile(a, 95)), 2),
                                   "max": round(float(np.max(a)), 2), "n": len(a)}

    n = max(1, len(rows))
    return {
        "clip": clip, "frames_asked": len(want), "frames_compared": len(rows),
        "far_rim_pixel_difference": stat(far_d),
        "near_rim_pixel_difference": stat(near_d),
        "same_keyframe_chosen": sum(1 for r in rows if r["kf_cpu"] == r["kf_gpu"]),
        "cpu_only_matched": only_cpu, "gpu_only_matched": only_gpu,
        "neither_matched": neither,
        "cpu_s_per_frame": round(t_cpu / n, 3),
        "gpu_s_per_frame": round(t_gpu / n, 3),
        "speedup": round(t_cpu / max(1e-6, t_gpu), 1),
        "full_game_hours_cpu": round(171120 * (t_cpu / n) / 3600, 1),
        "full_game_hours_gpu": round(171120 * (t_gpu / n) / 3600, 2),
        "sample": rows[:10],
        "bar": ("the rim is ~20 px across on this footage; single-digit pixel "
                "agreement is acceptable, tens of pixels is not"),
    }
