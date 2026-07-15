"""READ-ONLY probe (TEST 3, gated protocol): does the fine-tuned model's
Hoop class agree with the hand-clicked, homography-carried rim anchors?

Samples every STRIDE-th frame of a clip, logs ALL Hoop-class detections
(any conf), for comparison against {clip}_hoop_track.json's per-frame
hoop_far_px / hoop_near_px. Key from env (RF_KEY), never committed.

Run:  RF_KEY=... .venv/Scripts/python spikes/roboflow_hoop_probe.py CLIP [STRIDE]
"""
from __future__ import annotations
import json, os, sys, time
import cv2

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
import clip_config

KEY = os.environ["RF_KEY"]
WORKSPACE, PROJECT, VERSION = "roboflow-universe-projects", "basketball-players-fy4c2", 25

CLIP_NAME = sys.argv[1] if len(sys.argv) > 1 else "TEST1"
STRIDE = int(sys.argv[2]) if len(sys.argv) > 2 else 10
CLIP = getattr(clip_config, f"{CLIP_NAME}_CLIP")
OUT_JSON = os.path.join(_HERE, "out", f"{CLIP_NAME}_hoop_probe_roboflow.json")
_TMP = os.path.join(_HERE, "out", "_rf_hoop_frame.jpg")


def main():
    from roboflow import Roboflow
    model = Roboflow(api_key=KEY).workspace(WORKSPACE).project(PROJECT).version(VERSION).model
    cap = cv2.VideoCapture(CLIP.video_path)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_out = []
    print(f"[hoop_probe] {CLIP_NAME}: {n_total} frames, stride {STRIDE} "
          f"-> {len(range(0, n_total, STRIDE))} samples")
    for f in range(0, n_total, STRIDE):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            break
        cv2.imwrite(_TMP, img)
        for attempt in range(3):
            try:
                r = model.predict(_TMP, confidence=1).json()
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
        hoops = [{"cx": p["x"], "cy": p["y"], "w": p["width"], "h": p["height"],
                  "conf": p["confidence"]}
                 for p in r["predictions"] if p["class"] == "Hoop"]
        frames_out.append({"frame_index": f, "hoops": hoops})
        if (f // STRIDE) % 20 == 0:
            print(f"  ...frame {f} ({len(hoops)} hoop dets)", flush=True)
    cap.release()
    json.dump({"clip": CLIP_NAME, "stride": STRIDE, "model": f"{PROJECT}_v{VERSION}",
               "frames": frames_out}, open(OUT_JSON, "w", encoding="utf-8"), indent=2)
    print(f"[hoop_probe] wrote {OUT_JSON} ({len(frames_out)} sampled frames)")


if __name__ == "__main__":
    main()
