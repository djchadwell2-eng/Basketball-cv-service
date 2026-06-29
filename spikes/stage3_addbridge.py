"""Stage 3 Fix #1: add BRIDGING landmarks so the right wing is anchored to the
core by 4+ well-spread (non-collinear) shared points instead of 2 collinear ones.

Opens click mode for the right wing + neighbors. Your already-saved landmarks are
drawn in RED for reference; you ADD new ones (green) using the same tag palette.
The point: click the SAME off-center features (center-circle edges g/h/i/j, and
the right free-throw corners a/b if visible) in BOTH a core frame (1000) AND a
wing frame (1100) so the wing-to-core transform is actually constrained.

Saves the MERGED landmark set to spikes/out/merged_landmarks.json (and prints it)
so it can be written straight back into stage2's CONFIG. Does NOT optimize or
change stage1/stage2 logic.
"""

import os
import sys
import json
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage2_multikeyframe as s2

# Right wing (1100, 1200) + their neighbors toward the anchored core (900, 1000).
BRIDGE_KEYFRAMES = [900, 1000, 1100, 1200]

HINTS = {
    900: "REFERENCE frame (center-LEFT). Add center-circle edges (g/h/i/j) -- they "
         "also appear in 1100, bridging the wing straight to the anchor.",
    1000: "CORE frame (center). KEY BRIDGE: click the center-circle edges "
          "g=top h=bottom i=left j=right, and R_FT corners a/b IF the right "
          "free-throw line is visible at the right edge. Click the SAME ones in 1100.",
    1100: "WING frame (center-RIGHT). Click the SAME center-circle edges (g/h/i/j) "
          "and R_FT (a/b) you clicked in 1000 -- this overlap ties the wing to the core.",
    1200: "WING frame (right). Add center-circle edges if still visible; helps extend "
          "the bridge.",
}


def collect_bridge(img, idx, tags, existing):
    """Draw EXISTING landmarks (red) for reference; collect NEW ones (green)."""
    keymap = s2._palette_keymap(tags)
    print(f"\n  === Keyframe {idx} ===")
    print(f"  {HINTS.get(idx, '')}")
    print("  Existing saved points are shown in RED. Add NEW bridge points (green).")
    print("  Pick a tag key, then click. Palette:")
    for code, k in keymap.items():
        mark = "  <- BRIDGE" if tags[k][0].startswith(("circle_", "R_FT")) else ""
        print(f"    [{chr(code)}] {tags[k][0]}{mark}")
    print("  u=undo last new point; ENTER=done with this frame.")

    h, w = img.shape[:2]
    scale = min(1.0, s2.DISPLAY_MAX_W / float(w))
    disp0 = cv2.resize(img, (int(w * scale), int(h * scale)))
    # bake the existing (red) reference points into the base image
    for (tag, x, y) in existing:
        d = (int(x * scale), int(y * scale))
        cv2.circle(disp0, d, 5, (0, 0, 255), 1)
        cv2.putText(disp0, tag, (d[0] + 6, d[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    active = {"i": 0}
    pts = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((active["i"], x / scale, y / scale))

    win = f"keyframe {idx} -- ADD bridge points (red=existing)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        disp = disp0.copy()
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(disp, f"kf {idx}  NEW tag = {tags[active['i']][0]}  "
                          f"(u=undo ENTER=done)", (6, 19),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        for (ti, px, py) in pts:
            d = (int(px * scale), int(py * scale))
            cv2.circle(disp, d, 5, (0, 255, 0), -1)
            cv2.putText(disp, tags[ti][0], (d[0] + 6, d[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
        cv2.imshow(win, disp)
        key = cv2.waitKey(20) & 0xFF
        if key in keymap:
            active["i"] = keymap[key]
        elif key == ord("u") and pts:
            pts.pop()
        elif key in (13, ord("q")):
            break
    cv2.destroyWindow(win)
    return [(tags[ti][0], round(px, 1), round(py, 1)) for (ti, px, py) in pts]


def main():
    os.makedirs(s2.OUT_DIR, exist_ok=True)
    frames = s2.extract_frames(s2.VIDEO_PATH, BRIDGE_KEYFRAMES)
    merged = {idx: list(s2.LANDMARKS.get(idx, [])) for idx in s2.KEYFRAMES}

    total_new = 0
    for idx in BRIDGE_KEYFRAMES:
        new = collect_bridge(frames[idx], idx, s2.LANDMARK_TAGS,
                             s2.LANDMARKS.get(idx, []))
        merged[idx] = list(s2.LANDMARKS.get(idx, [])) + new
        total_new += len(new)
        print(f"  added {len(new)} new point(s) to keyframe {idx}")

    # save merged set (JSON) for deterministic write-back into CONFIG
    jpath = os.path.join(s2.OUT_DIR, "merged_landmarks.json")
    with open(jpath, "w", encoding="utf-8") as f:
        json.dump({str(idx): merged[idx] for idx in s2.KEYFRAMES}, f, indent=2)

    print(f"\n  total new bridge points added: {total_new}")
    if total_new == 0:
        print("  *** You added NO bridge points. If that's because there are no "
              "clean OFF-center features in the 1000/1100 overlap, that is itself "
              "the finding: the right wing is inherently under-constrained from "
              "this pan. ***")

    # printable merged block
    print("\n  ---- merged LANDMARKS (also saved to merged_landmarks.json) ----")
    print("LANDMARKS = {")
    for idx in s2.KEYFRAMES:
        print(f"    {idx}: [")
        for (tag, x, y) in merged[idx]:
            print(f"        ({tag!r}, {x}, {y}),")
        print("    ],")
    print("}")
    print(f"\nsaved merged landmarks: {jpath}")


if __name__ == "__main__":
    main()
