"""CALIBRATE a game set up through the web app, then RENDER PROOF a human can judge.

Runs after the coach has clicked their landmarks. Two outputs, and the second
one is not optional:

  1. THE NUMBERS -- court identified from the marks (never assumed), keyframe
     consistency, and the landmark court fit in feet.
  2. A SHORT VIDEO with the court drawn on real gameplay at each marked spot.

WHY (2) EXISTS. A good-looking number has passed while the calibration was
broken (TEST 36: the court identified cleanly at 0.23 ft while the full solve
was 15.45 ft out). The failure that mattered most on this project was caught by
DJ watching an overlay, not by a metric. So the pipeline does not treat a clip
as calibrated until a person has looked at it.

THE BARS, from real judgements on this footage:
    <= 0.30 ft   glued        (DJ on a 0.21 ft result: "utter perfection")
    ~  0.38 ft   usable       (DJ: "a little bit shaky... it wasn't glued")
    >= 0.94 ft   broken       (DJ judged this broken by eye)

Usage:  .venv/Scripts/python.exe calibrate_clip.py <CLIP_NAME>
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "spikes"))
sys.path.insert(0, os.path.join(_ROOT, "phase1"))

import clip_registry                                              # noqa: E402

GLUED_FT = 0.30
BROKEN_FT = 0.94
SECONDS_PER_SPOT = 8
W, H = 960, 540
MIN_INLIERS = 30
# A sanity floor, not the main guard. Nearest-in-time does the real work, so
# this only has to reject obvious rubbish -- set too high it starts discarding
# a frame's own legitimate keyframe and leaves holes in the overlay, which is a
# worse outcome than a slightly loose match to the correct view.
MIN_RATIO = 0.35
# Above this many inliers, the direct keyframe match is trusted outright and
# the temporal chain is re-anchored to it. Below it, the chain from the previous
# frame is more reliable -- consecutive frames barely differ, whereas a frame
# mid-pan can be far from the keyframe photo.
STRONG_INLIERS = 120


def log(m):
    print(m, flush=True)


def draw_court_clipped(frame, M, s4, s6):
    """Draw ONLY the court that is actually in view.

    THE BUG THIS FIXES. stage4.to_px accepts a projected point up to 100,000 px
    on a 1920-wide frame. A homography sends anything near the horizon to
    enormous coordinates, and those points were being joined up and drawn --
    producing long lines slashing across the picture at wild angles. That is
    what made the overlay look "atrocious" while every measured number was fine:
    the court was drawn CORRECTLY where the marks are, and garbage everywhere
    else, and the garbage is most of what you see.

    It is also dishonest to draw it. Those regions are outside the area any
    landmark constrains, so the line's position there is extrapolation, not
    measurement -- exactly the sort of confident-looking guess this project
    refuses everywhere else.

    So: a point counts only if it lands within one frame-width of the image.
    Segments with both ends outside are skipped entirely; a segment with one end
    inside is still drawn, and OpenCV clips it at the edge.
    """
    h, w = frame.shape[:2]
    mx, my = w, h                      # allow one frame-width of overhang

    def ok(p):
        return p is not None and -mx <= p[0] <= w + mx and -my <= p[1] <= h + my

    M = s4.signfix(M)
    for poly in s4.POLYS:
        pts = [s4.to_px(M, fx, fy) for (fx, fy) in poly]
        for a, b in zip(pts, pts[1:]):
            if ok(a) and ok(b):
                cv2.line(frame, a, b, (0, 0, 255), 2)
    for poly in s6.ARC_POLYS:
        pts = [s4.to_px(M, fx, fy) for (fx, fy) in poly]
        for a, b in zip(pts, pts[1:]):
            if ok(a) and ok(b):
                cv2.line(frame, a, b, s6.ARC_COLOR, 2)
    for tag, (fx, fy) in s4.COURT_MODEL.items():
        p = s4.to_px(M, fx, fy)
        if ok(p) and 0 <= p[0] < w and 0 <= p[1] < h:
            cv2.circle(frame, p, 5, (0, 255, 255), -1)


def _to_browser_playable(path):
    """Re-encode the proof clip to H.264 so it plays IN A BROWSER.

    OpenCV's VideoWriter writes MPEG-4 Part 2 ("mp4v"). Desktop players handle
    it, which is why every overlay reviewed by opening the file locally has been
    fine -- but Chrome and Edge do not support it, so an <video> tag renders a
    silent black box with no error and no duration. The proof video is useless
    if it cannot be watched where it is shown.

    ffmpeg is already a dependency of the web app (it compresses uploads).
    -movflags +faststart puts the index at the FRONT of the file, without which
    a browser must download the whole thing before it can start playing.
    Failure here is non-fatal: the original file is left in place, and a
    less-playable video beats a missing one.
    """
    import shutil
    import subprocess
    if not shutil.which("ffmpeg"):
        log("[proof] ffmpeg not found -- leaving the clip as-is; it may not play "
            "in a browser")
        return
    tmp = path + ".h264.mp4"
    r = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", path,
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-pix_fmt", "yuv420p",          # some players reject other layouts
         "-movflags", "+faststart", tmp],
        capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(tmp):
        log(f"[proof] could not convert for browser playback: {r.stderr.strip()[:200]}")
        return
    os.replace(tmp, path)
    log("[proof] converted to H.264 so it plays in the browser")


def main():
    name = sys.argv[1]
    doc = clip_registry.load(name)
    if not doc:
        raise SystemExit(f"no registry clip {name!r}")
    if not clip_registry.has_calibration(doc):
        raise SystemExit(f"{name} has no clicked landmarks yet")

    log(f"STAGE calibrating {name}")

    # --- WHICH FRAME DISAGREES WITH THE OTHERS? --------------------------------
    # Run BEFORE the solve, because when the court cannot be identified the
    # coach's only clue used to be "off by 2.88 ft", which says nothing about
    # what to do. Leave-one-frame-out pinpoints it in seconds and is how the
    # real cause was actually found: dropping one frame took the same marks
    # from "no court fits" to 0.22 ft. That frame had two points mirrored.
    import json as _json

    import court_detect
    import mark_diagnosis
    lm = {int(k): [(m[0], float(m[1]), float(m[2])) for m in v]
          for k, v in doc["landmarks"].items()}
    if not court_detect.identify(lm)["identified"]:
        log("[check] the marks don't agree -- finding which click is wrong")
        found = mark_diagnosis.diagnose(lm)
        if found:
            # One line the app can parse, so the coach is told the exact point
            # rather than a number they cannot act on.
            log("DIAGNOSIS " + _json.dumps(found))
            log(f"[check] frame {found['frame']}: {found['message']}")
        else:
            log("[check] no single click explains it -- several marks are off")
        # Persist it: going back to marking CLEARS the log, and the marking page
        # needs the diagnosis precisely then -- to open on the right frame with
        # the offending point already highlighted.
        clip_registry.update(name, mark_diagnosis=found, setup_stage="needs_more_marks")

    import clips_config
    clips_config._merge_registry_clips()      # pick up the freshly-saved marks
    clips_config.CLIPS[name] = clip_registry.to_calibration_entry(doc)
    clips_config.ACTIVE = name
    clips_config._RESOLVED.pop(name, None)    # re-solve the court from new marks

    import stage2_multikeyframe as s2
    import stage4_courtmap as s4
    import stage5_courtmap as s5
    import stage6_arc_overlay as s6
    import refit_keyframes as rk

    # A stale cache from an earlier attempt would silently calibrate the OLD
    # clicks and report them as the new result.
    if os.path.exists(rk.CACHE):
        os.remove(rk.CACHE)

    log("STAGE solving the court")
    KF, ref_pos, Hs0, L0, tags, obs, corr = rk._setup()
    before = rk.keyframe_consistency(Hs0, corr)
    Hs, L, res = rk._solve(KF, ref_pos, Hs0, L0, tags, obs, corr, rk.CORR_WEIGHT)
    after = rk.keyframe_consistency(Hs, corr)

    H_court, per, mean_ft, max_ft = s4.compute_H_court(L, tags)
    bm = float(np.mean([b[2] for b in before]))
    am = float(np.mean([a[2] for a in after]))
    log(f"[calib] keyframe agreement {bm:.1f}px -> {am:.1f}px")
    log(f"[calib] court fit  mean {mean_ft:.2f} ft   max {max_ft:.2f} ft")
    verdict = ("glued" if mean_ft <= GLUED_FT else
               "broken" if mean_ft >= BROKEN_FT else "usable")
    log(f"[calib] verdict: {verdict}")

    np.savez(rk.CACHE, KF=np.array(KF), ref_pos=ref_pos,
             tags=np.array(tags, dtype=object),
             Hs=np.array([Hs[p] for p in range(len(KF))]),
             L=np.array({t: L[t] for t in tags}, dtype=object))

    # ---- the overlay a human actually judges -------------------------------
    log("STAGE rendering proof video")
    Hcourt_inv = np.linalg.inv(H_court)
    sift = cv2.SIFT_create(nfeatures=1500)
    kf_frames = s2.extract_frames(s2.VIDEO_PATH, KF)
    kf_db = []
    for pos, k in enumerate(KF):
        kp, des = s5.sift_of(kf_frames[k], sift)
        m = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
        m.add([des]); m.train()
        kf_db.append((pos, kp, m))

    def court_to_frame(Hfk, pos):
        T = Hs[pos] @ Hfk
        # RANSAC can return a technically-valid but singular homography;
        # inverting it used to kill an entire render on one bad frame.
        if not np.all(np.isfinite(T)) or abs(np.linalg.det(T)) < 1e-12:
            return None
        try:
            return s5.signfix(np.linalg.inv(T) @ Hcourt_inv)
        except np.linalg.LinAlgError:
            return None

    cap = cv2.VideoCapture(s2.VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_path = os.path.join(_ROOT, "spikes", "out", f"{name}_calibration_proof.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    drawn = missed = 0
    for pos, k in enumerate(KF):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        # TEMPORAL CHAIN. Matching every frame straight back to its keyframe
        # works while the camera is near where that photo was taken and DEGRADES
        # as it pans away -- which is why the first and last spots were glued
        # (little movement) and the middle ones drifted by up to 430 px
        # (hard pans). That drift is the "camera moved but the court didn't".
        #
        # Consecutive frames are nearly identical, so a frame-to-PREVIOUS-frame
        # match is easy and accurate even mid-pan. Carry the previous frame's
        # solution forward and compose. The direct keyframe match is still
        # preferred whenever it is STRONG, which re-anchors and stops the chain
        # accumulating drift -- so this is continuity plus correction, not dead
        # reckoning.
        prev_gray = None      # previous frame's SIFT (kp, des)
        prev_H = None         # previous frame -> keyframe
        for i in range(int(SECONDS_PER_SPOT * fps)):
            ok, frame = cap.read()
            if not ok:
                break
            kp_f, des_f = s5.sift_of(frame, sift)
            # ANCHOR TO THE NEAREST GOOD KEYFRAME, NOT THE ONE WITH THE MOST
            # MATCHES. This was the bug that made the overlay look atrocious
            # while every number said the calibration was fine.
            #
            # Both ends of a basketball court are near-identical -- same key,
            # same arc, same circle. SIFT+RANSAC will happily find a large,
            # geometrically CONSISTENT set of matches that maps one end of the
            # floor onto the other end of a different keyframe. Measured on the
            # real failure: frame 151320 scored 216 inliers at ratio 0.742
            # against keyframe 158700 (7,500 frames away) and 162 against its
            # own keyframe 151200. Picking on count alone chose 158700 and drew
            # a court half a floor out of place; anchoring to 151200 drew it
            # correctly. More matches did NOT mean the right view.
            #
            # The camera moves continuously, so the keyframe nearest IN TIME is
            # overwhelmingly the most likely correct view. Quality still has to
            # clear a bar -- this prefers the nearest ADEQUATE match, it does
            # not accept a bad one just for being close.
            cands = []
            if des_f is not None and len(kp_f) >= 8:
                for (kpos, kp_k, matcher) in kf_db:
                    knn = matcher.knnMatch(des_f, k=2)
                    good = [a for a, b in (p for p in knn if len(p) == 2)
                            if a.distance < 0.75 * b.distance]
                    if len(good) < 8:
                        continue
                    src = np.float32([kp_f[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
                    dst = np.float32([kp_k[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
                    Hfk, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
                    if Hfk is None:
                        continue
                    inl = int(mask.sum())
                    ratio = inl / len(good)
                    if inl >= MIN_INLIERS and ratio >= MIN_RATIO:
                        cands.append((abs(KF[kpos] - (k + i)), -inl, kpos, Hfk))
            cands.sort()          # nearest in time first; more inliers breaks ties

            chosen_H = chosen_pos = None
            if cands:
                _, neg_inl, kpos0, H0 = cands[0]
                # STRONG direct match -> trust it and re-anchor the chain.
                if -neg_inl >= STRONG_INLIERS:
                    chosen_H, chosen_pos = H0, kpos0
                    prev_H = H0
                else:
                    chosen_H, chosen_pos = H0, kpos0

            # Weak or missing direct match: walk forward from the previous
            # frame instead, which is a far easier match mid-pan.
            if prev_gray is not None and prev_H is not None and (
                    not cands or -cands[0][1] < STRONG_INLIERS):
                kp_p, des_p = prev_gray
                if des_f is not None and des_p is not None and len(kp_f) >= 8:
                    mp = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
                    mp.add([des_p]); mp.train()
                    knn = mp.knnMatch(des_f, k=2)
                    g = [a for a, b in (p for p in knn if len(p) == 2)
                         if a.distance < 0.75 * b.distance]
                    if len(g) >= 12:
                        src = np.float32([kp_f[x.queryIdx].pt for x in g]).reshape(-1, 1, 2)
                        dst = np.float32([kp_p[x.trainIdx].pt for x in g]).reshape(-1, 1, 2)
                        Hstep, mk = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
                        if Hstep is not None and int(mk.sum()) >= 20:
                            chained = prev_H @ Hstep      # frame -> prev -> keyframe
                            chosen_H, chosen_pos = chained, pos
                            prev_H = chained

            M = court_to_frame(chosen_H, chosen_pos) if chosen_H is not None else None
            prev_gray = (kp_f, des_f)
            if M is not None:
                draw_court_clipped(frame, M, s4, s6)
                drawn += 1
                txt = f"spot {pos+1}/{len(KF)}  {k/fps/60:.1f} min"
            else:
                missed += 1
                txt = f"spot {pos+1}/{len(KF)}  NO MATCH"
            cv2.putText(frame, txt, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                        (255, 255, 255), 2)
            writer.write(cv2.resize(frame, (W, H)))
        log(f"[proof] spot {pos+1}/{len(KF)} done")
    writer.release(); cap.release()
    _to_browser_playable(out_path)

    clip_registry.update(
        name,
        court=s2.cfg.active()["court"],
        calibration={"mean_ft": round(float(mean_ft), 3),
                     "max_ft": round(float(max_ft), 3),
                     "keyframe_px": round(am, 2),
                     "verdict": verdict,
                     "frames_drawn": drawn, "frames_no_match": missed},
        setup_stage="awaiting_approval")
    log(f"[proof] wrote {out_path}  ({drawn} frames drawn, {missed} no-match)")
    log("STAGE awaiting your approval of the overlay")
    log("DONE")


if __name__ == "__main__":
    main()
