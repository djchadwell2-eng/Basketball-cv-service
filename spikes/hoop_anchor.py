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
RIM_PIXEL = (1855.0, 228.0)               # FAR hoop, user-confirmed 2026-07-13 (DECISIONS 15)
RIM_KEYFRAME_NEAR = 600
RIM_PIXEL_NEAR = (633.0, 190.0)           # NEAR hoop, user-confirmed 2026-07-14 (DECISIONS 18)

NFEAT, RATIO, RANSAC_PX, MIN_INLIERS = 1500, 0.75, 3.0, 30


def project_point(M, x, y):
    """Apply a 3x3 homogeneous transform to one point. None on a degenerate
    (near-zero) denominator instead of dividing by it."""
    d = M[2, 0] * x + M[2, 1] * y + M[2, 2]
    if abs(d) < 1e-9:
        return None
    return ((M[0, 0] * x + M[0, 1] * y + M[0, 2]) / d,
            (M[1, 0] * x + M[1, 1] * y + M[1, 2]) / d)


# A SIFT match clearing MIN_INLIERS on OTHER visible features (floor lines,
# bleachers) does not guarantee the homography is well-conditioned for
# extrapolating to the RIM specifically, if the rim sits far outside the
# convex hull of matched keypoints for that view. Found on the full-clip
# harvest (DECISIONS 18): frames matched to keyframes 600-1000 (outside the
# span this was validated on) produced geometrically absurd hoop pixels
# (e.g. x=42625 in a 1920px-wide frame) -- a near-zero perspective-divide
# denominator, not a sign issue (negating a homogeneous matrix cannot change
# its projective output). PLAUSIBLE_BOUNDS_MARGIN catches this: a hoop
# position wildly outside the frame is not trustworthy even when the match
# that produced it "looked" confident -- abstain, same as a failed match.
PLAUSIBLE_BOUNDS_MARGIN = 0.5   # allow up to 50% of a frame dimension beyond its edge


def in_plausible_bounds(px, frame_w, frame_h, margin=PLAUSIBLE_BOUNDS_MARGIN):
    x, y = px
    mx, my = frame_w * margin, frame_h * margin
    return (-mx <= x <= frame_w + mx) and (-my <= y <= frame_h + my)


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
    """Carries BOTH hoop anchors (far + near) through every frame's SAME
    per-frame SIFT match -- one extra point projection per frame, no extra
    SIFT cost, since both anchors share the identical Hs_opt[pos] @ Hfk
    transform for a given frame. Return (rim_ref900_far, rim_ref900_near,
    [{frame_index, hoop_far_px, hoop_near_px, matched_kf, inliers, ratio}
    or {frame_index, hoop_far_px: None, hoop_near_px: None} on no-match],
    KF, Hs_opt)."""
    print("[hoop_anchor] recovering optimized keyframe transforms (Hs_opt) ...")
    KF, ref_pos, Hs_opt, L_opt, tags = s4.run_optimization()

    def _anchor_ref900(keyframe, pixel, label):
        pos = KF.index(keyframe)
        ref900 = project_point(Hs_opt[pos], *pixel)
        assert ref900 is not None, f"{label} rim anchor projected to a degenerate point"
        print(f"[hoop_anchor] {label} rim anchor: keyframe {keyframe} px {pixel} "
              f"-> ref-900 px {tuple(round(v, 1) for v in ref900)}")
        return ref900

    rim_ref900_far = _anchor_ref900(RIM_KEYFRAME, RIM_PIXEL, "far")
    rim_ref900_near = _anchor_ref900(RIM_KEYFRAME_NEAR, RIM_PIXEL_NEAR, "near")

    sift = cv2.SIFT_create(nfeatures=NFEAT)
    kf_db = _keyframe_db(video_path, KF, sift)

    cap = cv2.VideoCapture(video_path)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    for _ in range(span_start):
        cap.grab()
    out = []
    matched_far = matched_near = 0
    rejected_implausible = 0
    for i in range(span_len):
        ok, frame = cap.read()
        if not ok:
            break
        f = span_start + i
        m = _match_frame(frame, sift, kf_db)
        if m is None:
            out.append({"frame_index": f, "hoop_far_px": None, "hoop_near_px": None})
            continue
        pos, Hfk, inl, ratio = m
        T = Hs_opt[pos] @ Hfk                       # frame -> ref-900
        Tinv = np.linalg.inv(T)
        far_px = project_point(Tinv, *rim_ref900_far)     # ref-900 -> frame
        near_px = project_point(Tinv, *rim_ref900_near)
        if far_px is not None and not in_plausible_bounds(far_px, frame_w, frame_h):
            rejected_implausible += 1
            far_px = None
        if near_px is not None and not in_plausible_bounds(near_px, frame_w, frame_h):
            rejected_implausible += 1
            near_px = None
        out.append({"frame_index": f, "hoop_far_px": far_px, "hoop_near_px": near_px,
                     "matched_kf": KF[pos], "inliers": inl, "ratio": round(ratio, 2)})
        matched_far += far_px is not None
        matched_near += near_px is not None
        if i % 60 == 0:
            print(f"  ...{i}/{span_len}", flush=True)
    cap.release()
    print(f"[hoop_anchor] far hoop found in {matched_far}/{len(out)} frames, "
          f"near hoop found in {matched_near}/{len(out)} frames "
          f"({rejected_implausible} matched-but-implausible results rejected)")
    return rim_ref900_far, rim_ref900_near, out, KF, Hs_opt


def main():
    import clip_config
    CLIP = clip_config.HARD_CLIP
    clip_config.ACTIVE_CLIP = CLIP

    # Optional CLI override for a wider harvest (e.g. the full clip):
    # hoop_anchor.py <start> <len>. Default = same span as the ball spike.
    SPAN_START = int(sys.argv[1]) if len(sys.argv) > 1 else 1020
    SPAN_LEN = int(sys.argv[2]) if len(sys.argv) > 2 else 360
    rim_ref900_far, rim_ref900_near, track, KF, Hs_opt = build_hoop_track(
        CLIP.video_path, SPAN_START, SPAN_LEN)

    out_json = os.path.join(_HERE, "out", f"{CLIP.name}_hoop_track.json")
    json.dump({"clip": CLIP.name, "span_start": SPAN_START, "span_len": SPAN_LEN,
               "rim_keyframe_far": RIM_KEYFRAME, "rim_pixel_far": list(RIM_PIXEL),
               "rim_ref900_far": list(rim_ref900_far),
               "rim_keyframe_near": RIM_KEYFRAME_NEAR, "rim_pixel_near": list(RIM_PIXEL_NEAR),
               "rim_ref900_near": list(rim_ref900_near), "frames": track},
              open(out_json, "w", encoding="utf-8"), indent=2)
    print(f"[hoop_anchor] wrote {out_json}")

    # --- eyeball overlay: BOTH hoop markers per frame + match confidence text ---
    out_video = os.path.join(_HERE, "out", f"{CLIP.name}_hoop_track_overlay.mp4")
    cap = cv2.VideoCapture(CLIP.video_path)
    for _ in range(SPAN_START):
        cap.grab()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    MAGENTA, CYAN = (255, 0, 255), (255, 255, 0)
    for i, rec in enumerate(track):
        ok, frame = cap.read()
        if not ok:
            break
        f = rec["frame_index"]
        for px, color, label in ((rec["hoop_far_px"], MAGENTA, "far"),
                                 (rec["hoop_near_px"], CYAN, "near")):
            if px is not None:
                x, y = px
                cv2.circle(frame, (int(x), int(y)), 30, color, 3)
                cv2.drawMarker(frame, (int(x), int(y)), color, cv2.MARKER_CROSS, 20, 2)
        if "matched_kf" in rec:
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
