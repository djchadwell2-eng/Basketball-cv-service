"""Render actual crops from spikes/pose_facing_test.py's WIN/LOSS piles so a
human can eyeball whether "facing score" really tracks "squared up to the
camera" -- the project's own rule (HANDOFF_NAMING_2026_09_07 S10): rendering
crops has corrected a measurement-only conclusion five times before.

Deliberately includes the AWKWARD examples (low-facing WINs, high-facing
LOSSes), not just the confirming ones -- a montage of only confirming cases
proves nothing.

Usage:
    .venv/Scripts/python.exe spikes/render_pose_facing_examples.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "phase2")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stage2_multikeyframe as s2mk
import clip_config

CELL_W, CELL_H = 220, 300
ROW_TITLE_H = 30
OUT_PATH = os.path.join(_HERE, "out", "pose_facing_examples.jpg")

# (row key, title). Key = (is_win, take_from_low_end).
ROWS = [
    ((True, False), "WIN, high facing score (should look squared-on)"),
    ((True, True), "WIN, low facing score (the awkward cases -- read anyway)"),
    ((False, True), "LOSS, low facing score (should look side-on)"),
    ((False, False), "LOSS, high facing score (squared-on but STILL no read)"),
]


def pick_examples():
    rows = json.load(open(os.path.join(_HERE, "out", "pose_facing_test.json"),
                         encoding="utf-8"))
    have = [r for r in rows if r["facing_score"] is not None]
    win = sorted([r for r in have if r["is_positive"]], key=lambda r: -r["facing_score"])
    loss = sorted([r for r in have if not r["is_positive"]], key=lambda r: -r["facing_score"])
    return {
        (True, False): win[:4],
        (True, True): list(reversed(win[-4:])),
        (False, True): list(reversed(loss[-4:])),
        (False, False): loss[:4],
    }


def body_crop(frame, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    H, W = frame.shape[:2]
    px, py = int(0.15 * (x2 - x1)), int(0.08 * (y2 - y1))
    x1, y1 = max(0, x1 - px), max(0, y1 - py)
    x2, y2 = min(W, x2 + px), min(H, y2 + py)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def make_cell(img, label_lines):
    h, w = img.shape[:2]
    scale = min(CELL_W / w, (CELL_H - 60) / h)
    resized = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
    cell = np.full((CELL_H, CELL_W, 3), 255, dtype="uint8")
    rh, rw = resized.shape[:2]
    ox, oy = (CELL_W - rw) // 2, 5
    cell[oy:oy + rh, ox:ox + rw] = resized
    y = CELL_H - 50
    for line in label_lines:
        cv2.putText(cell, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 0, 0), 1, cv2.LINE_AA)
        y += 16
    return cell


def main():
    examples = pick_examples()

    need_by_clip = defaultdict(set)
    for exs in examples.values():
        for r in exs:
            need_by_clip[r["clip"]].add(r["frame"])

    imgs = {}   # (clip, frame) -> image
    for clip, frames in need_by_clip.items():
        video = clip_config.get_clip(clip).video_path
        for f, im in s2mk.iter_frames(video, sorted(frames)):
            imgs[(clip, f)] = im

    row_h = CELL_H + ROW_TITLE_H
    grid = np.full((row_h * len(ROWS), CELL_W * 4 + 15, 3), 255, dtype="uint8")
    for ri, (key, title) in enumerate(ROWS):
        cv2.putText(grid, title, (5, ri * row_h + 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1, cv2.LINE_AA)
        for ci, r in enumerate(examples[key]):
            im = imgs.get((r["clip"], r["frame"]))
            crop = body_crop(im, r["bbox"]) if im is not None else None
            if crop is None or crop.size == 0:
                continue
            label = [f"{r['clip']} t{r['track_id']} f{r['frame']}",
                    f"facing={r['facing_score']:.3f}"]
            cell = make_cell(crop, label)
            y0 = ri * row_h + ROW_TITLE_H
            x0 = ci * (CELL_W + 5)
            grid[y0:y0 + CELL_H, x0:x0 + CELL_W] = cell

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    cv2.imwrite(OUT_PATH, grid)
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
