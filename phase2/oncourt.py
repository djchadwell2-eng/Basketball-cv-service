"""On-court ROI classification of cached tracks -- the seed-pool fix.

WHY: Phase-2 seeding was "seed everyone present at a window start", so the
candidate pool (and the rigged-low OCR numbers) was mostly crowd/refs/bench.
This module classifies every cached track on/off court so the identity layer
only auto-trusts bodies standing ON the court. It reuses the validated Phase-1
rules VERBATIM (direct nearest-keyframe anchor, pixel->feet, 84x50 + MARGIN_FT,
horizon guard, frame-edge feet drop) -- no new geometry, no new knobs.

Three parts:
  build()               HEAVY (SIFT per span frame, ~1-3s each; run ONCE per
                        clip via root cache_oncourt.py). Reads the tracks cache
                        + video, writes phase2/out/{clip}_oncourt.json with a
                        per-frame on/off verdict + court_feet per track.
  load_checked(config)  Light. Loads the JSON and REFUSES loud if missing or
                        stale (clip/span mismatch) -- same abstention rule as
                        the tracks-cache guard in run_clip.
  on_court_by_window()  Light, pure. Policy: a track counts as ON-COURT for a
                        window if a strict MAJORITY of its classified frames in
                        that window are on-court. Ties count OFF (conservative:
                        a body we can't place confidently is not auto-trusted;
                        it lands in the review queue, never in the box score).

The identity state machine is untouched: off-court tracks still flow through
it honestly (UNKNOWN/LOST/CANDIDATE); they are only excluded from SEEDING, OCR
attempts, and the coach review queue -- and every exclusion is counted and
persisted, never silent.
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))                          # repo root (clip_config)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "phase1"))  # stage1_court_roi
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "spikes"))  # calibration engine

from clip_config import ACTIVE_CLIP as CLIP

OUT_JSON = os.path.join(_HERE, "out", f"{CLIP.name}_oncourt.json")


# --------------------------------------------------------------------------
# readers (light imports -- safe for tests and for run_clip's precheck)
# --------------------------------------------------------------------------

def load_checked(config):
    """Load {clip}_oncourt.json; REFUSE loud if missing or stale."""
    path = os.path.join(_HERE, "out", f"{config.name}_oncourt.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"No on-court cache at {path}.\n"
            f"Run:  python -c \"import cache_oncourt, clip_config; "
            f"cache_oncourt.cache(clip_config.{config.name}_CLIP)\"  first.")
    doc = json.load(open(path, encoding="utf-8"))
    got = (doc.get("clip"), doc.get("span_start"), doc.get("span_len"))
    want = (config.name, config.tracking_span_start, config.tracking_span_len)
    if got != want:
        raise SystemExit(
            f"STALE/WRONG ON-COURT CACHE -- refusing to run.\n"
            f"  cache file:   {path}\n"
            f"  cache says:   clip={got[0]!r} span={got[1]}..+{got[2]}\n"
            f"  config wants: clip={want[0]!r} span={want[1]}..+{want[2]}\n"
            f"Re-run:  python -c \"import cache_oncourt, clip_config; "
            f"cache_oncourt.cache(clip_config.{config.name}_CLIP)\"")
    return doc


def _window_fn(span_start=None, win_frames=None, boundaries=None):
    """frame -> window index, for fixed windows or possession boundaries."""
    if boundaries is not None:
        from bisect import bisect_right
        bs = sorted(boundaries)
        return lambda f: max(0, bisect_right(bs, f) - 1)
    return lambda f: (f - span_start) // win_frames


def on_court_by_window(doc, span_start=None, win_frames=None, boundaries=None):
    """{window: set(track_id)} -- strict-majority on-court per (window, track).

    Windows: fixed (span_start + win_frames) or possession `boundaries`.
    Frames whose anchor failed carry no votes (fail-open at the vote level;
    the builder prints a warning per such frame). A (window, track) with a
    tie or an off-majority is NOT in the set."""
    win_of = _window_fn(span_start, win_frames, boundaries)
    votes = {}                                    # (win, tid) -> [on, off]
    for fr in doc["frames"]:
        win = win_of(fr["frame_index"])
        for tid_s, info in fr["tracks"].items():
            v = votes.setdefault((win, int(tid_s)), [0, 0])
            v[0 if info["on"] else 1] += 1
    out = {}
    for (win, tid), (n_on, n_off) in votes.items():
        if n_on > n_off:                          # strict majority; tie = OFF
            out.setdefault(win, set()).add(tid)
    return out


def seed_frame_tracks(tdoc, odoc, win_frames=None, boundaries=None):
    """{track_id: set(windows)} for tracks that are ON-COURT at any window-start
    seed frame -- the tracks that can carry a jersey label into the pipeline.
    Shared by the review bundle and the purity check so they agree."""
    span_start = tdoc["span_start"]
    onc = on_court_by_window(odoc, span_start=span_start,
                             win_frames=win_frames, boundaries=boundaries)
    win_of = _window_fn(span_start, win_frames, boundaries)
    out: dict = {}
    cur = None
    for fr in tdoc["frames"]:
        win = win_of(fr["frame_index"])
        if win != cur:
            cur = win
            on = onc.get(win, set())
            for t in fr["tracks"]:
                if t["track_id"] in on:
                    out.setdefault(t["track_id"], set()).add(win)
    return out


# --------------------------------------------------------------------------
# builder (heavy -- SIFT anchor per span frame; run once per clip)
# --------------------------------------------------------------------------

def build():
    import numpy as np
    import stage1_court_roi as st       # anchor + on_court rules (Phase 1, reused verbatim)

    with open(CLIP.tracks_cache_path, encoding="utf-8") as f:
        tdoc = json.load(f)
    got = (tdoc.get("clip"), tdoc.get("span_start"), tdoc.get("span_len"))
    want = (CLIP.name, CLIP.tracking_span_start, CLIP.tracking_span_len)
    if got != want:
        raise SystemExit(f"tracks cache is stale ({got} != {want}) -- "
                         f"re-run cache_tracks first.")

    print(f"[oncourt] {CLIP.name}: classifying {len(tdoc['frames'])} frames "
          f"(SIFT anchor per frame, CPU -- a few minutes, once per clip)")
    H_court, anchor, fps, total = st.build_court_anchor()
    fidxs = [fr["frame_index"] for fr in tdoc["frames"]]
    imgs = st.s2.extract_frames(CLIP.video_path, fidxs)

    out_frames, unclassified, on_counts = [], 0, []
    for i, fr in enumerate(tdoc["frames"]):
        f = fr["frame_index"]
        frame = imgs[f]
        T, inl, reproj, kf = anchor(f, frame)
        if T is None:
            unclassified += 1
            print(f"  WARNING f={f}: anchor match FAILED -- frame carries no "
                  f"on/off votes (fail-open)")
            out_frames.append({"frame_index": f, "anchor": None, "tracks": {}})
            continue
        p2f = H_court @ T
        feet2px = np.linalg.inv(p2f)
        ctr = feet2px @ np.array([st.COURT_LEN / 2.0, st.COURT_WID / 2.0, 1.0])
        ctr = ctr / ctr[2]
        ref_w = float((p2f @ ctr)[2])             # horizon-guard orientation
        frame_h = frame.shape[0]

        tracks_out, n_on = {}, 0
        for t in fr["tracks"]:
            x1, y1, x2, y2 = t["bbox"]
            fx, fy = (x1 + x2) / 2.0, y2          # feet = bbox bottom-center
            cx, cy, w = st.pixel_to_feet(p2f, fx, fy)
            feet_off_screen = fy >= frame_h - st.FRAME_EDGE_PX
            on = bool(st.on_court(cx, cy, w, ref_w) and not feet_off_screen)
            tracks_out[str(t["track_id"])] = {
                "on": on, "court_feet": [round(cx, 1), round(cy, 1)]}
            n_on += on
        on_counts.append(n_on)
        out_frames.append({"frame_index": f,
                           "anchor": {"inliers": int(inl),
                                      "reproj_px": round(reproj, 2), "kf": int(kf)},
                           "tracks": tracks_out})
        if i % 20 == 0:
            print(f"  ...{i}/{len(fidxs)}  f={f}  on-court {n_on}/{len(fr['tracks'])} "
                  f"(inl={inl} reproj={reproj:.2f}px)")

    doc = {"clip": CLIP.name, "span_start": tdoc["span_start"],
           "span_len": tdoc["span_len"], "fps": tdoc["fps"],
           "margin_ft": st.MARGIN_FT, "video_path": CLIP.video_path,
           "frames": out_frames}
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2)

    print(f"\n[oncourt] per-frame on-court count: min={min(on_counts)} "
          f"max={max(on_counts)} mean={sum(on_counts)/len(on_counts):.1f}  "
          f"(expect ~10 players + 2-3 refs; crowd/bench excluded)")
    if unclassified:
        print(f"[oncourt] WARNING: {unclassified} frame(s) unclassified (anchor failed)")
    print(f"[oncourt] wrote {len(out_frames)} frames -> {OUT_JSON}")


if __name__ == "__main__":
    build()
