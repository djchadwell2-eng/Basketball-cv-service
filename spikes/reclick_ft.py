"""Re-click the occluded L_FT corners as careful best-estimates (zoomed), to stop
them dragging the global fit -- WITHOUT removing them (keeps region spread).
The clean instances (L_FT_far@1000, L_FT_near@600/900) are NOT touched.
Saves new pixel positions to spikes/out/reclick_results.json. Read-only on CONFIG;
the assistant applies the results."""
import os, sys, json
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage2_multikeyframe as s2

# occluded instances to re-estimate (clean ones deliberately excluded)
RECLICK = [(600, "L_FT_far"), (700, "L_FT_far"), (800, "L_FT_far"), (900, "L_FT_far"),
           (700, "L_FT_near"), (800, "L_FT_near"), (1000, "L_FT_near")]
R, F = 210, 2.4                      # crop half-size (native px), display upscale
OUT = os.path.join(s2.OUT_DIR, "reclick_results.json")


def clicked_pos(kf, tag):
    for (t, x, y) in s2.LANDMARKS[kf]:
        if t == tag:
            return (x, y)
    return None


def main():
    os.makedirs(s2.OUT_DIR, exist_ok=True)
    frames = s2.extract_frames(s2.VIDEO_PATH, sorted({k for k, _ in RECLICK}))
    results = {}
    print("\nRe-click each FT corner = where the FREE-THROW LINE meets the LANE EDGE.")
    print("The old click is the small RED dot. Place the GREEN one precisely.")
    print("Keys: left-click=set, u=undo, s=skip(keep old), ENTER=confirm.\n")
    for kf, tag in RECLICK:
        old = clicked_pos(kf, tag)
        fr = frames[kf]
        x0, y0 = int(max(0, old[0]-R)), int(max(0, old[1]-R))
        x1, y1 = int(min(1920, old[0]+R)), int(min(1080, old[1]+R))
        crop = fr[y0:y1, x0:x1]
        disp0 = cv2.resize(crop, (int((x1-x0)*F), int((y1-y0)*F)))
        oldd = (int((old[0]-x0)*F), int((old[1]-y0)*F))
        state = {"pt": None}

        def on_mouse(ev, x, y, flags, param):
            if ev == cv2.EVENT_LBUTTONDOWN:
                state["pt"] = (x, y)

        win = f"{tag} @ kf{kf}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, on_mouse)
        while True:
            d = disp0.copy()
            cv2.circle(d, oldd, 6, (0, 0, 255), 1)           # old
            cv2.rectangle(d, (0, 0), (d.shape[1], 26), (0, 0, 0), -1)
            cv2.putText(d, f"{tag} @ kf{kf}: click FT-line x lane-edge corner "
                           "(u=undo s=skip ENTER=ok)", (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
            if state["pt"]:
                cv2.circle(d, state["pt"], 7, (0, 255, 0), 2)
            cv2.imshow(win, d)
            k = cv2.waitKey(20) & 0xFF
            if k == ord("u"):
                state["pt"] = None
            elif k == ord("s"):
                state["pt"] = "skip"; break
            elif k in (13, ord("q")) and state["pt"] is not None:
                break
        cv2.destroyWindow(win)
        if state["pt"] == "skip" or state["pt"] is None:
            print(f"  {tag}@{kf}: skipped (keeping old {old})")
            continue
        nx, ny = x0 + state["pt"][0]/F, y0 + state["pt"][1]/F
        results[f"{kf}|{tag}"] = [round(nx, 1), round(ny, 1)]
        print(f"  {tag}@{kf}: {tuple(round(v,1) for v in old)} -> ({nx:.1f},{ny:.1f})")

    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nsaved {OUT}  ({len(results)} re-clicks)")
    print("Tell your assistant it's saved; they'll apply it + remove R_FT_far@1200 + re-run.")


if __name__ == "__main__":
    main()
