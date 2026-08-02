"""anchor_subsample_test.py -- can we anchor the camera LESS often?

THE PROBLEM. phase2/oncourt.build calls stage1_court_roi.anchor once per frame
to work out where the camera points. Measured: 3.35 s/frame. A 95-minute game
is 171,120 frames = 159 HOURS, which is why the pipeline can only do ~15-second
clips today. It is the real scaling wall -- 300x more expensive than the YOLO
detection the GPU already sped up 131x.

THE IDEA. The camera pans smoothly at 30 fps, so consecutive frames barely
differ. Anchor every Nth frame and fill the gaps in, and the cost divides by N.

WHAT THIS MEASURES. Not "does it look fine" -- the ERROR, in feet on the court,
against anchoring every single frame. For each skipped frame we compare where a
player standing at a given spot WOULD be placed by the cheap estimate versus the
full one. Feet, because feet is what the pipeline's decisions are made in
(on/off court, which zone a shot came from) and what DJ judges a court by:
0.21 ft was "utter perfection", 0.38 ft "a little shaky", 0.94 ft "broken".

Two ways to fill a gap, both tested:
  hold    -- reuse the last anchored frame's transform
  interp  -- blend the transforms either side, weighted by distance

Usage:  .venv/Scripts/python.exe spikes/anchor_subsample_test.py [CLIP] [FRAMES]
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "spikes"),
           os.path.join(_ROOT, "phase1"), os.path.join(_ROOT, "phase2")):
    sys.path.insert(0, _p)

# Sample points spread over the floor, in court feet. Errors are measured at
# these spots rather than at image corners: a homography's worst error is
# usually near the horizon, which is not where basketball happens.
PROBE_FEET = [(x, y) for x in (10, 25, 42, 59, 74) for y in (8, 25, 42)]


def _to_px(M, pts_ft):
    """Court feet -> pixels through a feet->px matrix."""
    out = []
    for (x, y) in pts_ft:
        v = M @ np.array([x, y, 1.0])
        out.append((v[0] / v[2], v[1] / v[2]) if abs(v[2]) > 1e-9 else (np.nan, np.nan))
    return np.array(out)


def main():
    clip = sys.argv[1] if len(sys.argv) > 1 else "TEST1"
    n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 150
    # Optional: several START frames, comma separated. A 15-second clip where
    # the camera barely moves proves nothing about a 95-minute game -- the
    # gaps BETWEEN marked spots, and moments of hard panning, are where a
    # cheap estimate would fall apart. So the same test runs at several places.
    starts = ([int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else None)

    import clip_config
    import clips_config
    cfg = clip_config.get_clip(clip)
    clip_config.ACTIVE_CLIP = cfg
    clips_config.ACTIVE = clip
    import stage1_court_roi as st

    print(f"[anchor_test] {clip}: building court anchor ...", flush=True)
    H_court, anchor, fps, total = st.build_court_anchor()

    for start in (starts or [cfg.tracking_span_start]):
        run_one(cfg, st, H_court, anchor, start, n_frames)


def run_one(cfg, st, H_court, anchor, start, n_frames):
    frames = list(range(start, start + n_frames))
    print(f"\n{'=' * 60}\n[anchor_test] SPOT at frame {start} "
          f"({start / 30 / 60:.1f} min into the game)\n{'=' * 60}", flush=True)
    print(f"[anchor_test] anchoring {len(frames)} frames the SLOW way "
          f"(the baseline everything is compared against) ...", flush=True)

    t0 = time.time()
    truth = {}
    for f, img in st.s2.iter_frames(cfg.video_path, frames):
        T, inl, reproj, kf = anchor(f, img)
        truth[f] = T
    per_frame = (time.time() - t0) / max(1, len(frames))
    print(f"[anchor_test] {per_frame:.2f} s/frame  "
          f"-> full game (171,120 frames) = {171120 * per_frame / 3600:.0f} h", flush=True)

    ok = [f for f in frames if truth.get(f) is not None]
    failed = len(frames) - len(ok)
    if failed:
        # A frame the anchor cannot match carries no on/off-court votes at all.
        # Worth seeing per spot: dark or crowd-only frames are exactly where a
        # full game differs from a hand-picked 15-second clip.
        print(f"[anchor_test] {failed} of {len(frames)} frames FAILED to anchor", flush=True)
    if len(ok) < 10:
        print(f"[anchor_test] only {len(ok)} frames anchored -- skipping this spot")
        return

    # feet -> px for a given frame: invert (H_court @ T), which maps px -> feet
    def feet_to_px(T):
        return np.linalg.inv(H_court @ T)

    print(f"\n{'N':>4}  {'fill':<7} {'mean ft':>8} {'max ft':>8} {'hours':>7}  verdict")
    print("-" * 60)
    for N in (2, 5, 10, 15, 30, 60):
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

                # Where does each probe spot LAND, cheap estimate vs full?
                px_true = _to_px(feet_to_px(truth[f]), PROBE_FEET)
                px_est = _to_px(feet_to_px(T_est), PROBE_FEET)
                # Turn the pixel disagreement back into feet by pushing both
                # through the TRUE px->feet map: feet is the unit that matters.
                p2f = H_court @ truth[f]
                for a, b in zip(px_true, px_est):
                    if not (np.isfinite(a).all() and np.isfinite(b).all()):
                        continue
                    fa = p2f @ np.array([a[0], a[1], 1.0])
                    fb = p2f @ np.array([b[0], b[1], 1.0])
                    if abs(fa[2]) < 1e-9 or abs(fb[2]) < 1e-9:
                        continue
                    errs.append(np.hypot(fa[0] / fa[2] - fb[0] / fb[2],
                                         fa[1] / fa[2] - fb[1] / fb[2]))
            if not errs:
                continue
            mean_ft, max_ft = float(np.mean(errs)), float(np.max(errs))
            hours = 171120 * per_frame / N / 3600
            verdict = ("glued" if mean_ft <= 0.30 else
                       "usable" if mean_ft <= 0.60 else
                       "shaky" if mean_ft <= 0.94 else "BROKEN")
            print(f"{N:>4}  {mode:<7} {mean_ft:>8.2f} {max_ft:>8.2f} {hours:>7.1f}  {verdict}")


if __name__ == "__main__":
    main()
