"""How often is the scoreboard graphic ACTUALLY on screen?

DJ observed (2026-07-26) that the broadcast scorebug fades in and out on at
least two clips. That matters because a signal which is intermittently ABSENT
is dangerous in a specific way: absence must produce "unknown", never a value.
TEST 14 already hit this from a different cause -- it had no score data for 4
of TEST1's 5 shots and the matcher would have reported all four as MISSES,
which is the confident-wrong failure this project exists to avoid.

This measures the fade instead of arguing about it. v3 already carries the
scoreboard element classes from the public dataset (Team Points, Time
Remaining, Period, Shot Clock), so no new model and no region config is
needed -- if the graphic is on screen the classes fire, if it has faded out
they do not.

Sampled, not every frame: presence is a slow-moving property and TEST 14
already sampled at this stride.

Run:  .venv/Scripts/python spikes/scoreboard_presence.py TEST4
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

STRIDE = 15          # 0.5s, same sampling TEST 14 used
CONF = 0.40          # the usable-confidence bar used throughout this project
IMGSZ = 1280

VIDEOS = {"TEST4": r"C:\Users\djcha\New folder\Throw away repos"
                   r"\Basketball Analyer CV System Test\clips\Test4.mp4"}


def main():
    import cv2
    from ultralytics import YOLO

    clip = sys.argv[1] if len(sys.argv) > 1 else "TEST4"
    video = VIDEOS.get(clip)
    if video is None:
        import clip_config
        video = getattr(clip_config, f"{clip}_CLIP").video_path

    model = YOLO(os.path.join(_ROOT, "models", "ball_finetuned_v3.pt"))
    names = model.model.names
    want = {i: str(n) for i, n in names.items()
            if str(n) in ("Team Points", "Time Remaining", "Period", "Shot Clock")}
    print(f"[presence] {clip}: watching classes {sorted(want.values())} "
          f"at conf>={CONF}, every {STRIDE}th frame")

    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    seen, runs, cur = [], [], 0
    per_class = {v: 0 for v in want.values()}
    n = 0
    for f in range(0, total, STRIDE):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            break
        res = model.predict(img, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
        hits = {want[int(b.cls)] for b in res.boxes if int(b.cls) in want}
        for h in hits:
            per_class[h] += 1
        present = bool(hits)
        seen.append((f, present))
        n += 1
        if present:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
        if n % 100 == 0:
            print(f"  ...{n} samples", flush=True)
    if cur:
        runs.append(cur)
    cap.release()

    n_present = sum(1 for _, p in seen if p)
    print(f"\n[presence] {n_present}/{n} sampled frames have the scorebug "
          f"({100 * n_present / max(n, 1):.1f}%)")
    for k, v in sorted(per_class.items()):
        print(f"    {k:16s} {v:4d}/{n}  ({100 * v / max(n, 1):.1f}%)")

    # the product-relevant question is not the average, it is how long the
    # BLACKOUTS last -- a 20s hole loses a whole possession's confirmations
    gaps, cur = [], 0
    for _, p in seen:
        if not p:
            cur += 1
        elif cur:
            gaps.append(cur)
            cur = 0
    if cur:
        gaps.append(cur)
    if gaps:
        gaps.sort()
        secs = lambda k: k * STRIDE / fps
        print(f"\n[presence] {len(gaps)} blackout(s); longest {secs(gaps[-1]):.1f}s, "
              f"median {secs(gaps[len(gaps) // 2]):.1f}s")
        print(f"    all blackouts (s): {[round(secs(g), 1) for g in gaps]}")
    else:
        print("\n[presence] no blackouts -- the graphic never left the screen")


if __name__ == "__main__":
    main()
