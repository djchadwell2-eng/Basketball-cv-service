"""Pre-draw player/ref boxes on the harvested labelling frames so DJ CORRECTS
boxes instead of drawing them from scratch.

Why: the player-labelling queue (280 crowded-paint frames from
harvest_player_frames.py) was running at ~10 frames/day = ~25 days remaining,
and the #1 open CV problem (the ball model's false positives, TEST 16) is
gated behind it. v3 already carries the Player and Ref classes from the public
dataset (TEST 4 measured them as sane: ~10 players + ~3 refs per frame at
conf>=0.4), so the boxes can be proposed automatically. Correcting pre-drawn
boxes is typically 3-5x faster than drawing them.

This is a PROPOSAL, not a label. Every box still gets DJ's eye -- the point is
to remove the drawing, not the judgement.

Writes a Roboflow-ready YOLO folder (images + labels + data.yaml). Nothing in
the repo's caches is touched.

Run:  .venv/Scripts/python spikes/prefill_player_labels.py
"""
from __future__ import annotations

import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

SRC = os.path.join(_HERE, "out", "label_harvest_players")
DST = os.path.join(_HERE, "out", "label_prefill_players")
WEIGHTS = os.path.join(_ROOT, "models", "ball_finetuned_v3.pt")
IMGSZ = 1280
CONF = 0.25        # deliberately generous: deleting a wrong box is far faster
                   # than noticing a missing one and drawing it
OUT_CLASSES = ["player", "ref"]     # must match the Roboflow project's classes


def main():
    from ultralytics import YOLO

    if not os.path.isdir(SRC):
        print(f"[prefill] no harvest folder at {SRC}")
        return
    frames = sorted(f for f in os.listdir(SRC) if f.lower().endswith((".jpg", ".png")))
    if not frames:
        print("[prefill] no images found")
        return

    model = YOLO(WEIGHTS)
    names = {str(v).lower(): k for k, v in model.model.names.items()}
    src_player, src_ref = names.get("player"), names.get("ref")
    if src_player is None:
        print(f"[prefill] ABORT: no Player class in the weights ({list(names)})")
        return
    remap = {src_player: 0}
    if src_ref is not None:
        remap[src_ref] = 1

    img_dir = os.path.join(DST, "train", "images")
    lbl_dir = os.path.join(DST, "train", "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    n_box = n_ref = 0
    empty = []
    for i, fn in enumerate(frames):
        src_path = os.path.join(SRC, fn)
        res = model.predict(src_path, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
        h, w = res.orig_shape
        lines = []
        for b in res.boxes:
            cls = int(b.cls)
            if cls not in remap:
                continue
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            cx, cy = ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h)
            bw, bh = ((x2 - x1) / w, (y2 - y1) / h)
            lines.append(f"{remap[cls]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            n_box += 1
            n_ref += (remap[cls] == 1)
        if not lines:
            empty.append(fn)
        shutil.copy2(src_path, os.path.join(img_dir, fn))
        stem = os.path.splitext(fn)[0]
        with open(os.path.join(lbl_dir, stem + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + ("\n" if lines else ""))
        if (i + 1) % 40 == 0:
            print(f"[prefill] {i + 1}/{len(frames)} frames...", flush=True)

    with open(os.path.join(DST, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("train: train/images\nval: train/images\n"
                f"nc: {len(OUT_CLASSES)}\nnames: {OUT_CLASSES}\n")

    n_img = len(frames)
    print(f"\n[prefill] {n_box} proposed boxes over {n_img} frames "
          f"({n_box / n_img:.1f}/frame; {n_box - n_ref} player, {n_ref} ref)")
    if empty:
        print(f"[prefill] {len(empty)} frames got NO boxes (check these by hand): "
              f"{', '.join(empty[:6])}{' ...' if len(empty) > 6 else ''}")
    print(f"[prefill] Roboflow-ready folder -> {DST}")
    print("[prefill] REMINDER: these are proposals. Every box still needs DJ's eye;")
    print("          the point is to remove the drawing, not the judgement.")


if __name__ == "__main__":
    main()
