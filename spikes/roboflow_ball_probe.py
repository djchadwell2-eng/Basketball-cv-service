"""READ-ONLY probe: does a FINE-TUNED basketball model (Roboflow) see the
ball better than stock YOLOv8m? (DECISIONS 24)

Stock COCO "sports ball" is the §20/§21 bottleneck. This tests the one
untried model lever the camera-fixed constraint leaves: a detector trained
on basketballs. Logs Ball detections in the SAME schema as ball_spike.py so
ball_trajectory.py can consume it and we compare ARCS (not raw coverage).

API key comes from the env (RF_KEY) -- NEVER hardcoded/committed. Hosted
inference = one call per frame.

Run:  RF_KEY=... .venv/Scripts/python spikes/roboflow_ball_probe.py CLIP START LEN
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
BALL_CLASS_NAME = "Ball"
# percent floor; 5 matches the stock spike's 0.05 (apples-to-apples) and cuts
# the 1%-conf junk flood -- the real ball sits ~0.85, junk <0.10 (DECISIONS 24).
CLIP_NAME = sys.argv[1] if len(sys.argv) > 1 else "TEST1"
START = int(sys.argv[2]) if len(sys.argv) > 2 else 0
LEN = int(sys.argv[3]) if len(sys.argv) > 3 else 450
CONF = int(sys.argv[4]) if len(sys.argv) > 4 else 5
CLIP = getattr(clip_config, f"{CLIP_NAME}_CLIP")
OUT_JSON = os.path.join(_HERE, "out", f"{CLIP_NAME}_ball_spike_log_roboflow.json")
_TMP = os.path.join(_HERE, "out", "_rf_frame.jpg")


def main():
    from roboflow import Roboflow
    model = Roboflow(api_key=KEY).workspace(WORKSPACE).project(PROJECT).version(VERSION).model
    print(f"[rf_probe] {CLIP_NAME} frames {START}..{START + LEN}, model {PROJECT} v{VERSION}")

    cap = cv2.VideoCapture(CLIP.video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, START)
    frames_out = []
    n_ball = 0
    for i in range(LEN):
        ok, img = cap.read()
        if not ok:
            break
        cv2.imwrite(_TMP, img)
        for attempt in range(3):
            try:
                r = model.predict(_TMP, confidence=CONF).json()
                break
            except Exception as e:
                if attempt == 2:
                    raise
                time.sleep(2)                 # transient/rate-limit backoff
        dets = []
        for p in r["predictions"]:
            if p["class"] != BALL_CLASS_NAME:
                continue
            w, h = p["width"], p["height"]
            cx, cy = p["x"], p["y"]
            dets.append({"bbox": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                         "conf": p["confidence"]})
        if dets:
            n_ball += 1
        frames_out.append({"frame_index": START + i, "detections": dets})
        if i % 25 == 0:
            print(f"  ...{i}/{LEN}  ({len(dets)} ball dets)", flush=True)
    cap.release()

    doc = {"clip": CLIP_NAME, "span_start": START, "span_len": len(frames_out),
           "fps": 30.0, "detector": f"{PROJECT}_v{VERSION}", "frames": frames_out}
    json.dump(doc, open(OUT_JSON, "w", encoding="utf-8"), indent=2)
    tot = len(frames_out)
    print(f"\n[rf_probe] {tot} frames, ball seen in {n_ball} ({100*n_ball/max(tot,1):.0f}%) "
          f"-> {OUT_JSON}")


if __name__ == "__main__":
    main()
