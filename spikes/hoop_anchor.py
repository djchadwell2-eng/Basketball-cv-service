"""Phase 5 step 3, part A -- HOOP ANCHOR: carry a user-confirmed rim pixel
through the camera pan.

No calibration landmark can give an elevated rim pixel directly -- every
COURT_MODEL tag (stage4_courtmap.py) is a FLOOR point, and a floor
homography only holds on the floor plane. Instead this reuses the SAME
machinery that already draws the full-clip court overlay (stage4-6):
Hs_opt (keyframe -> reference-900 pixels) + per-frame best-match SIFT to a
keyframe (Hfk: frame -> keyframe pixels). Composing them carries ANY pixel,
elevated or not, because these are ROTATION-ONLY camera homographies -- for
a camera that only pans/tilts (no translation), one homography relates
every scene point regardless of depth. That's the same assumption stage4's
docstring already leans on for the floor; here it's exactly what makes an
elevated rim carryable at all.

ONE human-confirmed input: RIM_KEYFRAME / RIM_PIXEL, marked by eyeball and
confirmed by the user against spikes/out/HARD_hoop_anchor_f1100.jpg
(2026-07-13) -- same click-seeding philosophy as the rest of the project:
system proposes, human confirms, once per basket per clip.

Per-frame carrying can fail honestly (weak SIFT match -> no hoop pixel that
frame) -- that frame's hoop position is simply ABSENT from the output, not
guessed from a neighbor.
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "phase2"))

import clips_config as _cc                # noqa: E402
# KNOWN TRAP (DECISIONS.md KNOWN DEBT): stage1/2/4/5 read clip identity from
# clips_config.ACTIVE, bound AT IMPORT TIME -- must be set BEFORE importing
# them or they silently bind to whatever clip was last active (this module
# is HARD-only today: RIM_KEYFRAME/RIM_PIXEL above are HARD-specific).
_cc.ACTIVE = "HARD"

import stage1_keyframe_match as s1        # noqa: E402
import stage2_multikeyframe as s2         # noqa: E402
import stage4_courtmap as s4              # noqa: E402
import stage5_courtmap as s5              # noqa: E402  (sift_of, signfix)

RIM_KEYFRAME = 1100
RIM_PIXEL = (1855.0, 228.0)               # user-confirmed 2026-07-13 (DECISIONS 15)

NFEAT, RATIO, RANSAC_PX, MIN_INLIERS = 1500, 0.75, 3.0, 30


def project_point(M, x, y):
    """Apply a 3x3 homogeneous transform to one point. None on a degenerate
    (near-zero) denominator instead of dividing by it."""
    d = M[2, 0] * x + M[2, 1] * y + M[2, 2]
    if abs(d) < 1e-9:
        return None
    return ((M[0, 0] * x + M[0, 1] * y + M[0, 2]) / d,
            (M[1, 0] * x + M[1, 1] * y + M[1, 2]) / d)


def _keyframe_db(video_path, KF, sift):
    frames = s2.extract_frames(video_path, KF)
    db = []
    for pos, k in enumerate(KF):
        kp, des = s5.sift_of(frames[k], sift)
        matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        matcher.add([des])
        matcher.train()
        db.append((pos, kp, matcher))
    return db


def _match_frame(frame, sift, kf_db):
    """Best-match this frame against every keyframe (same policy as
    stage5/stage6); return (pos, Hfk: frame_px -> keyframe_pos_px, inliers,
    ratio) for the strongest match, or None if nothing clears MIN_INLIERS."""
    kp_f, des_f = s5.sift_of(frame, sift)
    if des_f is None or len(kp_f) < 8:
        return None
    best = None
    for (pos, kp_k, matcher) in kf_db:
        knn = matcher.knnMatch(des_f, k=2)
        good = [a for a, b in (p for p in knn if len(p) == 2)
                if a.distance < RATIO * b.distance]
        if len(good) < 8:
            continue
        src = np.float32([kp_f[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_k[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
        Hfk, mask = cv2.findHomography(src, dst, cv2.RANSAC, RANSAC_PX)
        if Hfk is not None:
            inl = int(mask.sum())
            if best is None or inl > best[2]:
                best = (pos, Hfk, inl, inl / len(good))
    if best is not None and best[2] >= MIN_INLIERS:
        return best
    return None


def build_hoop_track(video_path, span_start, span_len):
    """Return (rim_ref900, [{frame_index, hoop_px, matched_kf, inliers,
    ratio} or {frame_index, hoop_px: None} on no-match], KF, Hs_opt)."""
    print("[hoop_anchor] recovering optimized keyframe transforms (Hs_opt) ...")
    KF, ref_pos, Hs_opt, L_opt, tags = s4.run_optimization()
    rim_pos = KF.index(RIM_KEYFRAME)
    rim_ref900 = project_point(Hs_opt[rim_pos], *RIM_PIXEL)
    assert rim_ref900 is not None, "rim anchor projected to a degenerate point"
    print(f"[hoop_anchor] rim anchor: keyframe {RIM_KEYFRAME} px {RIM_PIXEL} "
          f"-> ref-900 px {tuple(round(v, 1) for v in rim_ref900)}")

    sift = cv2.SIFT_create(nfeatures=NFEAT)
    kf_db = _keyframe_db(video_path, KF, sift)

    cap = cv2.VideoCapture(video_path)
    for _ in range(span_start):
        cap.grab()
    out = []
    matched = 0
    for i in range(span_len):
        ok, frame = cap.read()
        if not ok:
            break
        f = span_start + i
        m = _match_frame(frame, sift, kf_db)
        if m is None:
            out.append({"frame_index": f, "hoop_px": None})
            continue
        pos, Hfk, inl, ratio = m
        T = Hs_opt[pos] @ Hfk                       # frame -> ref-900
        Tinv = np.linalg.inv(T)
        hoop_px = project_point(Tinv, *rim_ref900)   # ref-900 -> frame
        out.append({"frame_index": f, "hoop_px": hoop_px, "matched_kf": KF[pos],
                     "inliers": inl, "ratio": round(ratio, 2)})
        if hoop_px is not None:
            matched += 1
        if i % 60 == 0:
            print(f"  ...{i}/{span_len}", flush=True)
    cap.release()
    print(f"[hoop_anchor] hoop pixel found in {matched}/{len(out)} frames")
    return rim_ref900, out, KF, Hs_opt


def main():
    import clip_config
    CLIP = clip_config.HARD_CLIP
    clip_config.ACTIVE_CLIP = CLIP

    SPAN_START, SPAN_LEN = 1020, 360     # same span as the ball spike/trajectory
    rim_ref900, track, KF, Hs_opt = build_hoop_track(CLIP.video_path, SPAN_START, SPAN_LEN)

    out_json = os.path.join(_HERE, "out", f"{CLIP.name}_hoop_track.json")
    json.dump({"clip": CLIP.name, "span_start": SPAN_START, "span_len": SPAN_LEN,
               "rim_keyframe": RIM_KEYFRAME, "rim_pixel": list(RIM_PIXEL),
               "rim_ref900": list(rim_ref900), "frames": track},
              open(out_json, "w", encoding="utf-8"), indent=2)
    print(f"[hoop_anchor] wrote {out_json}")

    # --- eyeball overlay: hoop marker per frame + match confidence text ---
    out_video = os.path.join(_HERE, "out", f"{CLIP.name}_hoop_track_overlay.mp4")
    cap = cv2.VideoCapture(CLIP.video_path)
    for _ in range(SPAN_START):
        cap.grab()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    MAGENTA = (255, 0, 255)
    for i, rec in enumerate(track):
        ok, frame = cap.read()
        if not ok:
            break
        f = rec["frame_index"]
        if rec["hoop_px"] is not None:
            x, y = rec["hoop_px"]
            cv2.circle(frame, (int(x), int(y)), 30, MAGENTA, 3)
            cv2.drawMarker(frame, (int(x), int(y)), MAGENTA, cv2.MARKER_CROSS, 20, 2)
            txt = f"f={f} t={f/fps:04.1f}s kf={rec['matched_kf']} inl={rec['inliers']}"
        else:
            txt = f"f={f} t={f/fps:04.1f}s NO MATCH"
        cv2.putText(frame, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 2)
        writer.write(frame)
    writer.release()
    cap.release()
    print(f"[hoop_anchor] wrote {out_video}")


if __name__ == "__main__":
    main()
