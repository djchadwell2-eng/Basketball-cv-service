"""CLICK THE COURT LANDMARKS on Full_Game.mp4's 5 chosen frames.

Same two-stage flow as spikes/reclick_ft.py: click roughly on the whole frame,
then a ZOOMED crop opens so the point can be placed precisely. A little court
diagram in the corner shows which point is being asked for, because the tag names
mean nothing on their own.

WHY THESE 5 FRAMES: TEST 35 measured that 5 marks cover 99% of this 95-minute
game (the first one alone covers 50%). The other 10 marks the sweep produced were
the pre-game introductions with the house lights off -- no lit floor, no
basketball, nothing to calibrate.

CONTROLS
    left click   place / move the point
    ENTER        accept and go to the next landmark
    s            SKIP -- this landmark is not visible in this frame
    u            undo the point
    b            back to the previous landmark
    q            save what is done so far and quit

Aim for at least 5 landmarks per frame, SPREAD OUT. Five points bunched at one
basket cannot pin the court down; corners plus centre can.

Writes spikes/out/FULLGAME_landmarks.json in the same shape as
spikes/clips_config.py LANDMARKS, so it can be pasted straight in.

Usage:  .venv/Scripts/python.exe spikes/click_fullgame_landmarks.py
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

VIDEO = os.path.join(_ROOT, "Full_Game.mp4")
OUT = os.path.join(_HERE, "out", "FULLGAME_landmarks.json")

# The 5 frames TEST 35 selected, best coverage first.
FRAMES = [200, 16000, 65800, 79200, 169000]

# Landmarks to offer, spread along the whole floor. (tag, court_x, court_y,
# plain-English description). Court is x = 0..84 along the length (or 94 --
# the dimension is SOLVED from these clicks later, not assumed), y = 0..50 across.
LANDMARKS = [
    ("LB_side_far",   0.0, 50.0, "LEFT baseline meets the FAR sideline (far corner of the court, left end)"),
    ("L_FT_near",    19.0, 19.0, "LEFT free-throw line meets the NEAR edge of the lane"),
    ("L_FT_far",     19.0, 31.0, "LEFT free-throw line meets the FAR edge of the lane"),
    ("center_near",  42.0,  0.0, "HALF-COURT line meets the NEAR sideline"),
    ("center_logo",  42.0, 25.0, "EXACT CENTRE of the centre circle"),
    ("center_far",   42.0, 50.0, "HALF-COURT line meets the FAR sideline"),
    ("R_FT_near",    65.0, 19.0, "RIGHT free-throw line meets the NEAR edge of the lane"),
    ("R_FT_far",     65.0, 31.0, "RIGHT free-throw line meets the FAR edge of the lane"),
    ("RB_side_far",  84.0, 50.0, "RIGHT baseline meets the FAR sideline (far corner, right end)"),
]

FIT_W, FIT_H = 1500, 820          # window budget so it fits on screen
ZOOM_R, ZOOM_F = 130, 3.5         # zoom crop half-size (native px), upscale


def court_diagram(w, h, target):
    """Small schematic of the floor with the requested point ringed."""
    img = np.full((h, w, 3), 40, np.uint8)
    pad = 14
    sx, sy = (w - 2 * pad) / 84.0, (h - 2 * pad) / 50.0

    def P(fx, fy):
        return int(pad + fx * sx), int(pad + fy * sy)

    cv2.rectangle(img, P(0, 0), P(84, 50), (200, 200, 200), 1)
    cv2.line(img, P(42, 0), P(42, 50), (200, 200, 200), 1)
    cv2.circle(img, P(42, 25), int(6 * sx), (200, 200, 200), 1)
    for x0, x1 in ((0, 19), (65, 84)):                      # the two lanes
        cv2.rectangle(img, P(x0, 19), P(x1, 31), (200, 200, 200), 1)
    tx, ty = P(target[0], target[1])
    cv2.circle(img, (tx, ty), 9, (0, 255, 255), 2)
    cv2.drawMarker(img, (tx, ty), (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
    return img


def label(img, text, y, color=(0, 255, 255), scale=0.62):
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, 1, cv2.LINE_AA)


def pick_point(frame, tag, desc, target, fi, n_frames, li, n_marks, placed):
    """Two-stage pick on one frame. -> (x, y) native px, "skip", or "quit"."""
    fh, fw = frame.shape[:2]
    s = min(FIT_W / fw, FIT_H / fh)
    disp0 = cv2.resize(frame, (int(fw * s), int(fh * s)))
    diag = court_diagram(230, 150, target)
    dh, dw = diag.shape[:2]
    st = {"pt": None}

    def on_mouse(ev, x, y, _f, _p):
        if ev == cv2.EVENT_LBUTTONDOWN:
            st["pt"] = (x, y)

    win = "CLICK THE COURT LANDMARK"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, disp0.shape[1], disp0.shape[0])
    cv2.setMouseCallback(win, on_mouse)

    coarse = None
    while coarse is None:
        d = disp0.copy()
        cv2.rectangle(d, (0, 0), (d.shape[1], 96), (0, 0, 0), -1)
        label(d, f"FRAME {fi+1}/{n_frames}   LANDMARK {li+1}/{n_marks}   "
                 f"placed so far on this frame: {placed}", 26, (255, 255, 255), 0.6)
        label(d, f"{tag}", 54, (0, 255, 0), 0.8)
        label(d, desc, 80, (0, 255, 255), 0.6)
        label(d, "click roughly = zoom in | s=skip | b=back | q=save+quit",
              d.shape[0] - 14, (180, 180, 180), 0.55)
        d[d.shape[0] - dh - 8:d.shape[0] - 8, d.shape[1] - dw - 8:d.shape[1] - 8] = diag
        for (px, py) in st.get("done", []):
            cv2.circle(d, (int(px * s), int(py * s)), 5, (0, 140, 255), 2)
        cv2.imshow(win, d)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("s"):
            cv2.destroyWindow(win)
            return "skip"
        if k == ord("q"):
            cv2.destroyWindow(win)
            return "quit"
        if k == ord("b"):
            cv2.destroyWindow(win)
            return "back"
        if st["pt"]:
            coarse = (st["pt"][0] / s, st["pt"][1] / s)

    # ---- stage 2: zoomed refine ----
    cx, cy = coarse
    x0, y0 = int(max(0, cx - ZOOM_R)), int(max(0, cy - ZOOM_R))
    x1, y1 = int(min(fw, cx + ZOOM_R)), int(min(fh, cy + ZOOM_R))
    crop = cv2.resize(frame[y0:y1, x0:x1],
                      (int((x1 - x0) * ZOOM_F), int((y1 - y0) * ZOOM_F)),
                      interpolation=cv2.INTER_CUBIC)
    st["pt"] = (int((cx - x0) * ZOOM_F), int((cy - y0) * ZOOM_F))
    win2 = "ZOOM -- place it exactly"
    cv2.namedWindow(win2, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win2, on_mouse)
    while True:
        d = crop.copy()
        cv2.rectangle(d, (0, 0), (d.shape[1], 58), (0, 0, 0), -1)
        label(d, f"{tag} -- place it EXACTLY", 24, (0, 255, 0), 0.7)
        label(d, "ENTER=accept | u=undo | s=skip | q=save+quit", 48,
              (180, 180, 180), 0.55)
        if st["pt"]:
            p = st["pt"]
            cv2.drawMarker(d, p, (0, 255, 0), cv2.MARKER_CROSS, 26, 2)
            cv2.circle(d, p, 13, (0, 255, 0), 1)
        cv2.imshow(win2, d)
        k = cv2.waitKey(20) & 0xFF
        if k == ord("u"):
            st["pt"] = None
        elif k == ord("s"):
            cv2.destroyWindow(win2); cv2.destroyWindow(win)
            return "skip"
        elif k == ord("q"):
            cv2.destroyWindow(win2); cv2.destroyWindow(win)
            return "quit"
        elif k in (13, 10) and st["pt"] is not None:
            break
    px = x0 + st["pt"][0] / ZOOM_F
    py = y0 + st["pt"][1] / ZOOM_F
    cv2.destroyWindow(win2)
    cv2.destroyWindow(win)
    return (round(px, 1), round(py, 1))


def main():
    if not os.path.exists(VIDEO):
        raise SystemExit(f"missing {VIDEO}")
    print("Loading the 5 frames (seeking a 3.6 GB file, takes a moment)...",
          flush=True)
    cap = cv2.VideoCapture(VIDEO)
    frames = {}
    for f in FRAMES:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if ok:
            frames[f] = img
            print(f"  got frame {f}", flush=True)
    cap.release()

    result = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    print(f"\n{len(LANDMARKS)} landmarks offered per frame. Aim for 5+ SPREAD OUT.")
    print("Skip anything you cannot clearly see in that frame.\n")

    quit_all = False
    for fi, f in enumerate(FRAMES):
        if f not in frames or quit_all:
            continue
        got = {t: (x, y) for (t, x, y) in
               [tuple(v) for v in result.get(str(f), [])]}
        li = 0
        while li < len(LANDMARKS):
            tag, ux, uy, desc = LANDMARKS[li]
            if tag in got:
                li += 1
                continue
            r = pick_point(frames[f], tag, desc, (ux, uy), fi, len(FRAMES),
                           li, len(LANDMARKS), len(got))
            if r == "quit":
                quit_all = True
                break
            if r == "back":
                li = max(0, li - 1)
                prev = LANDMARKS[li][0]
                got.pop(prev, None)
                continue
            if r != "skip":
                got[tag] = r
                print(f"  frame {f}: {tag} -> {r}")
            li += 1
        result[str(f)] = [[t, v[0], v[1]] for t, v in got.items()]
        print(f"  == frame {f}: {len(got)} landmarks placed ==\n")
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(result, open(OUT, "w", encoding="utf-8"), indent=2)

    print(f"\nsaved -> {OUT}")
    for f in FRAMES:
        n = len(result.get(str(f), []))
        flag = "" if n >= 5 else "   <- fewer than 5, the fit may be weak"
        print(f"  frame {f}: {n} landmarks{flag}")
    print("\nRe-run this script any time; frames already done are kept and "
          "skipped.")


if __name__ == "__main__":
    main()
