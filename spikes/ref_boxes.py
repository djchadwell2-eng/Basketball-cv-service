"""TEST 15 helper: dump REFEREE boxes for a clip's tracking span.

TEST 13 found the person detector sometimes tracks a referee as if they were a
player (1 of its 8 colour flags). TEST 4 already measured that the fine-tuned
model's Ref class separates them cleanly (~3/frame at conf>=0.4, = the real
number of officials on court), so this just writes those boxes out for
tracker_color_reattach_check.py to filter against.

Read-only over the video; writes one json. Nothing else consumes it.

Run:  .venv/Scripts/python spikes/ref_boxes.py [CLIP_NAME] [WEIGHTS]
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "phase2"))

CONF = 0.4          # TEST 4's strict-conf figure -- below this the Ref class is junk-flooded
IMGSZ = 1280        # the project's proven optimum (DECISIONS 20); do not change


def main():
    clip_name = sys.argv[1] if len(sys.argv) > 1 else "TEST1"
    weights = sys.argv[2] if len(sys.argv) > 2 else \
        os.path.join(_ROOT, "models", "ball_finetuned_v3.pt")

    import clip_config
    CLIP = getattr(clip_config, f"{clip_name}_CLIP")
    clip_config.ACTIVE_CLIP = CLIP
    import run_tracking
    from ultralytics import YOLO

    span_start, span_len = CLIP.tracking_span_start, CLIP.tracking_span_len
    subclip, fps, n = run_tracking.extract_subclip(CLIP.video_path, span_start, span_len)

    model = YOLO(weights)
    names = model.model.names
    ref_ids = [i for i, nm in names.items() if str(nm).lower() in ("ref", "referee")]
    if not ref_ids:
        print(f"[ref-boxes] ABORT: no Ref class in {os.path.basename(weights)} "
              f"-- classes are {list(names.values())}")
        return
    print(f"[ref-boxes] {os.path.basename(weights)} Ref class id={ref_ids} "
          f"conf>={CONF} imgsz={IMGSZ} over {clip_name} frames "
          f"{span_start}..{span_start + span_len - 1}")

    out: dict = {}
    total = 0
    for i, res in enumerate(model.predict(subclip, conf=CONF, imgsz=IMGSZ,
                                          stream=True, verbose=False)):
        frame_index = span_start + i
        boxes = []
        for b in res.boxes:
            if int(b.cls) in ref_ids:
                boxes.append([float(v) for v in b.xyxy[0].tolist()])
        if boxes:
            out[str(frame_index)] = boxes
            total += len(boxes)

    path = os.path.join(_HERE, "out", f"{clip_name}_ref_boxes.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f)
    frames_seen = i + 1
    print(f"[ref-boxes] {total} ref boxes across {len(out)}/{frames_seen} frames "
          f"({total / max(frames_seen, 1):.1f}/frame) -> {os.path.basename(path)}")


if __name__ == "__main__":
    main()
