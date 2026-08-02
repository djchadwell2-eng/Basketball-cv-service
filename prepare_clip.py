"""AUTOMATIC SETUP for a game uploaded through the web app.

Runs with NO human input and produces the one thing a human must actually do:
the short list of frames to click. Steps, in order:

  1. FIND THE BURNED-IN GRAPHIC (scorebug / player overlay) by itself.
  2. PLAN a keyframe chain -- walk the game, jumping as far as each view still
     matches the last.
  3. VERIFY every adjacent pair at FULL RESOLUTION with that graphic masked.
  4. BRIDGE any weak link and re-verify.
  5. EXPORT those frames as JPEGs for the browser clicker.

WHY STEP 1 EXISTS. Every clip so far had its overlay rectangle typed in by
hand, and the two full games needed different ones. A burned-in graphic is the
worst possible input to this matching: it is pixel-identical in every frame, so
it produces a pile of perfect false matches and can make two totally different
camera views look related. Left in, it does not just add noise -- it can turn a
broken chain into one that scores well. It has to go, and asking a coach to
drag a box round a scorebug is not a setup step worth having.

WHY STEP 3 IS SEPARATE FROM STEP 2. Planning matches on downscaled frames
because it must test many pairs. That shortcut has produced confidently wrong
answers twice (TEST 36: a bridge scored 0.781 downscaled and 0.056 at full
resolution). So the plan is a PROPOSAL and the full-resolution pass is the
verdict. Nothing reaches a human to click until it has passed at full size.

Usage:  .venv/Scripts/python.exe prepare_clip.py <CLIP_NAME>
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import os
import sys
import time

import cv2
import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "spikes"))

import clip_registry                                              # noqa: E402
import fast_frames                                                # noqa: E402
import stage1_keyframe_match as s1                                # noqa: E402

MIN_RATIO = 0.6            # the project's own weak-pair bar (stage2_multikeyframe)
PLAN_SCALE = 0.35          # planning only -- never the verdict
SEED = 0                   # RANSAC is random; every pair must reseed or results drift
CACHE_STRIDE = 600         # ~20s at 30fps
MAX_FRAMES = 8             # DJ's ceiling: "as long as there [are] only 5-8 frames"


def log(msg):
    print(msg, flush=True)


def _ratio(earlier, later, regions):
    """Inlier ratio, measured the way the calibration pipeline measures it.

    stage2_multikeyframe.adjacent_homographies -- whose WEAK PAIR FLAG is the
    real gate -- calls detect_and_match(LATER, EARLIER). SIFT matching is not
    symmetric, so the argument order changes the answer (measured: 0.590 one
    way vs 0.659 the other on the same pair). Mirror it exactly.
    """
    cv2.setRNGSeed(SEED)
    with contextlib.redirect_stdout(io.StringIO()):
        kp_l, kp_e, good = s1.detect_and_match(later, earlier, 0.75, regions)
        if len(good) < 15:
            return 0.0
        _H, _m, inl = s1.estimate_homography(kp_l, kp_e, good, 3.0)
    return (inl / len(good)) if inl else 0.0


# ---------------------------------------------------------------------------
# STEP 1 -- the graphic mask. NOT auto-detected: that was tried and MEASURED
# to be impossible by the obvious method. Kept here so nobody rebuilds it.
#
# THE IDEA THAT FAILED: "a burned-in scorebug never changes, so find the frozen
# pixels." Measured across both full games, sampling 12 frames spread over the
# whole video, comparing the KNOWN overlay rectangle against open court:
#
#     Full_Game    overlay mean spread 191.0   court 135.2
#     Full_Game2   overlay mean spread 196.2   court 184.1
#
# The overlay is the MOST-CHANGING part of the frame, not the least -- these
# bugs carry a live clock, a score and a video thumbnail, and frames sampled
# minutes apart share nothing. Zero frozen pixels were found in either. The
# premise is simply false for real broadcast overlays.
#
# THE MASK STILL MATTERS, also measured -- Full_Game2, masked vs unmasked:
#     165000 -> 190500    0.745 masked    0.565 unmasked
# Unmasked, that healthy pair drops BELOW the 0.6 bar and would be "repaired"
# with a bridge frame it never needed. The overlay does not create false
# matches (it changes too much for that); it dilutes the inlier ratio.
#
# SO: run WITHOUT a mask here, which costs at most an extra frame to click, and
# let the coach drag a box over the graphic on the clicking page -- where they
# are already looking at these exact frames. That keeps setup automatic (DJ:
# "no extra steps") while the mask still reaches the calibration, which is
# where accuracy actually counts.


# ---------------------------------------------------------------------------
# STEP 2 -- plan a chain (downscaled; a PROPOSAL, never the verdict)
# ---------------------------------------------------------------------------
def plan_chain(video_path, regions, stride=CACHE_STRIDE):
    t0 = time.time()
    # SEEKED, not scanned. Measured 5.4x faster, and for planning the exact
    # frame index is irrelevant anyway -- frames 1/30s apart are the same view.
    cache, fps, total = fast_frames.sample_frames(video_path, stride, scale=PLAN_SCALE)
    order = sorted(cache)
    log(f"[plan] {total:,} frames / {total/fps/60:.1f} min -- sampled "
        f"{len(order)} frames in {time.time()-t0:.0f}s")

    # Scale the mask to match the downscaled frames.
    small_regions = [tuple(v * PLAN_SCALE for v in r) for r in regions]

    marks, pos = [order[0]], 0
    while pos < len(order) - 1:
        cur = order[pos]
        lo, hi, best = pos + 1, len(order) - 1, None
        while lo <= hi:
            mid = (lo + hi) // 2
            if _ratio(cache[cur], cache[order[mid]], small_regions) >= MIN_RATIO:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        pos = best if best is not None else pos + 1
        marks.append(order[pos])
        log(f"[plan] mark {len(marks)}: frame {order[pos]} ({order[pos]/fps/60:.1f} min)")
        if len(marks) >= MAX_FRAMES:
            log(f"[plan] hit the {MAX_FRAMES}-frame ceiling; stopping here")
            break
    return marks, fps, total


# ---------------------------------------------------------------------------
# STEP 3/4 -- verify at FULL resolution, bridge what fails
# ---------------------------------------------------------------------------
def read_frames(video_path, idxs):
    """Frame-accurate read of specific frames -- seeked, then VERIFIED, with a
    sequential fallback for any frame the seek missed. See fast_frames.py for
    the measurements that justify trusting the seek."""
    return fast_frames.read_frames(video_path, idxs, verify=True)


def verify(video_path, marks, regions):
    frames = read_frames(video_path, marks)
    marks = [m for m in marks if m in frames]
    pairs = []
    for a, b in zip(marks, marks[1:]):
        r = _ratio(frames[a], frames[b], regions)
        pairs.append({"a": a, "b": b, "ratio": round(r, 3), "weak": r < MIN_RATIO})
        log(f"[verify] {a} -> {b}  ratio {r:.3f}{'  <-- WEAK' if r < MIN_RATIO else '  OK'}")
    return marks, pairs, frames


def bridge(video_path, a, b, regions, stride=1500):
    """Find an intermediate frame that both sides actually match, at full res.

    NOT the arithmetic midpoint: frame-number proximity is not view similarity.
    Measured on a real gap -- the midpoint scored 0.266 and 0.354 against a 0.6
    bar, WORSE than the single broken link it was meant to repair.
    """
    cands = list(range(a + stride, b, stride))
    if not cands:
        return None
    frames = read_frames(video_path, [a, b] + cands)
    if a not in frames or b not in frames:
        return None
    best, best_score = None, 0.0
    for c in cands:
        if c not in frames:
            continue
        left = _ratio(frames[a], frames[c], regions)
        if left < MIN_RATIO:
            continue
        right = _ratio(frames[c], frames[b], regions)
        score = min(left, right)
        if score > best_score:
            best, best_score = c, score
    if best:
        log(f"[bridge] {a} -> {best} -> {b}  (weakest side {best_score:.3f})")
    else:
        log(f"[bridge] no single frame repairs {a} -> {b}")
    return best


# ---------------------------------------------------------------------------
def main():
    name = sys.argv[1]
    doc = clip_registry.load(name)
    if not doc:
        raise SystemExit(f"no registry clip {name!r}")
    video = doc["video_path"]
    if not os.path.exists(video):
        raise SystemExit(f"video not found: {video}")

    log(f"STAGE setup starting for {name}")

    regions = [tuple(r) for r in (doc.get("exclude_regions") or [])]
    if regions:
        log(f"[graphic] masking the graphic the coach marked: {regions}")
    else:
        log("[graphic] no graphic mask yet -- planning without one. A scoreboard "
            "overlay dilutes match scores, so a link may look weaker than it is "
            "and cost one extra frame to click. Drag a box over the graphic on "
            "the clicking page and the calibration will use it.")

    log("STAGE planning which frames to mark")
    marks, fps, total = plan_chain(video, regions)

    log("STAGE verifying at full resolution")
    marks, pairs, _ = verify(video, marks, regions)

    # One repair pass. If a link still fails after bridging, the frame set goes
    # to the human WITH the failure stated rather than quietly pretending.
    if any(p["weak"] for p in pairs):
        log("STAGE bridging weak links")
        added = []
        for p in [q for q in pairs if q["weak"]]:
            b = bridge(video, p["a"], p["b"], regions)
            if b:
                added.append(b)
        if added:
            marks = sorted(set(marks + added))
            log("STAGE re-verifying")
            marks, pairs, _ = verify(video, marks, regions)

    # DROP DEAD FRAMES. A frame that shares nothing with its neighbour (a black
    # intro, a cutaway, lights-down) cannot be tied into the court at all, so
    # asking someone to mark it wastes their time and then breaks the solve.
    # Real case: frame 0 of a game scored ratio 0.000 against frame 600, was
    # offered anyway, went unmarked, and crashed the optimiser.
    dead = {p["a"] for p in pairs if p["ratio"] <= 0.05} | \
           {p["b"] for p in pairs if p["ratio"] <= 0.05}
    if dead:
        # Only drop a frame if EVERY link it has is dead -- a frame between two
        # others may be weak on one side and fine on the other.
        for f in sorted(dead):
            links = [p for p in pairs if p["a"] == f or p["b"] == f]
            if links and all(p["ratio"] <= 0.05 for p in links) and len(marks) > 2:
                log(f"[plan] dropping frame {f} -- it shows nothing the other "
                    f"frames share (blank or a cutaway), so it cannot be used")
                marks = [m for m in marks if m != f]
        pairs = [p for p in pairs if p["a"] in marks and p["b"] in marks]
        marks, pairs, _ = verify(video, marks, regions)

    weak = [p for p in pairs if p["weak"]]
    log(f"[result] {len(marks)} frames to mark, {len(weak)} weak link(s)")

    log("STAGE exporting frames to click")
    frames = read_frames(video, marks)
    shots = []
    for m in marks:
        fr = frames.get(m)
        if fr is None:
            continue
        ok, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 88])
        if ok:
            shots.append({"frame": m, "t": m / fps, "w": fr.shape[1], "h": fr.shape[0],
                          "b64": base64.b64encode(buf).decode("ascii")})

    out_dir = os.path.join(_ROOT, "spikes", "out")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}_click_frames.json"), "w", encoding="utf-8") as fh:
        json.dump({"clip": name, "fps": fps, "total_frames": total,
                   "exclude_regions": [list(r) for r in regions],
                   "pairs": pairs, "weak": len(weak), "shots": shots}, fh)

    clip_registry.update(name, keyframes=marks, chain_pairs=pairs,
                         chain_weak=len(weak), setup_stage="awaiting_clicks")
    log(f"STAGE ready for clicks ({len(marks)} frames)")
    log("DONE")


if __name__ == "__main__":
    main()
