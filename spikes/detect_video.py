"""Standalone ball+hoop detection over ANY video file.

Exists so a brand-new clip can be measured without first inventing a
ClipConfig for it -- a holdout clip has no roster and no calibration, and
needs neither. Same detection settings as every previous ball run
(conf=0.05, imgsz=1280, v3 weights) so the logs are comparable, and the
same JSON schema local_weights_check.py already reads.

Run:  .venv/Scripts/python spikes/detect_video.py <video> <NAME>
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)

CONF = 0.05
IMGSZ = 1280


def main():
    from ultralytics import YOLO

    video = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else "CLIP"
    weights = os.path.join(_ROOT, "models", "ball_finetuned_v3.pt")

    model = YOLO(weights)
    names = model.model.names
    ball_id = next(i for i, n in names.items()
                   if str(n).lower() in ("ball", "sports ball"))
    hoop_id = next(i for i, n in names.items() if str(n).lower() == "hoop")

    frames, hoops = [], []
    n_ball = n_hoop = 0
    for i, res in enumerate(model.predict(source=video, classes=[ball_id, hoop_id],
                                          imgsz=IMGSZ, conf=CONF, stream=True,
                                          verbose=False)):
        bd, hd = [], []
        for b in res.boxes:
            box = [round(float(v), 2) for v in b.xyxy[0].tolist()]
            c = round(float(b.conf), 4)
            (bd if int(b.cls) == ball_id else hd).append({"bbox": box, "conf": c})
        frames.append({"frame_index": i, "t_sec": round(i / 30.0, 2),
                       "detections": bd})
        hoops.append({"frame_index": i, "detections": hd})
        n_ball += bool(bd)
        n_hoop += bool(hd)
        if (i + 1) % 500 == 0:
            print(f"  ...{i + 1}", flush=True)

    n = len(frames)
    out = os.path.join(_HERE, "out")
    json.dump({"clip": name, "span_start": 0, "span_len": n, "fps": 30.0,
               "conf_threshold": CONF, "imgsz": IMGSZ,
               "model": "ball_finetuned_v3.pt", "frames": frames},
              open(os.path.join(out, f"{name}_ball_spike_log_ball_finetuned_v3_gpu.json"), "w"))
    json.dump({"clip": name, "frames": hoops},
              open(os.path.join(out, f"{name}_hoop_dets_v3.json"), "w"))
    print(f"[detect] {n} frames | ball {n_ball} ({100*n_ball/max(n,1):.1f}%) "
          f"| hoop {n_hoop} ({100*n_hoop/max(n,1):.1f}%)")


if __name__ == "__main__":
    main()
