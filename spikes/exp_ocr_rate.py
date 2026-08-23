"""exp_ocr_rate.py -- how fast does ONE jersey crop actually read on the worker?

WHY THIS EXISTS. The rehearsal measured the real number: ~150,000 jersey crops
on a full game. Everything else about the tail is now known and comfortable --
12 minutes, 13.5 GB of a 62 GB box. Whether the game is finishable at all comes
down to a rate nobody has measured on this machine, and guessing it twice
already produced two wrong answers 8x apart.

WHAT IT MEASURES, on REAL crops cut from DJ's own film:
  1. seconds per crop, EasyOCR on the CPU, at several pool sizes -- the worker
     has 128 cores and the pipeline uses a pool of 6, so the question is
     whether the other 122 are worth anything here.
  2. seconds per crop, EasyOCR on the GPU -- there is a 4090 sitting completely
     idle through the whole identity tail, and easyocr takes a gpu flag.
  3. WHETHER THE GPU AND CPU READERS AGREE, crop for crop. A faster reader that
     returns different numbers is not a speed-up, it is a different pipeline,
     and this project does not ship those without a diff.

Deliberately NOT the whole stage: no window logic, no attempt policy, no
early exit. Those decide HOW MANY crops get read; this decides how long one
takes, and the two multiply.

Run:  .venv/Scripts/python.exe lab.py spikes/exp_ocr_rate.py --args '{...}'
"""

import json
import os
import sys
import time

PER_SLICE = 17112
MIN_OCR_HEIGHT = 90        # same bar stage6 uses -- below it a number is unreadable


def _crops(clip, n_crops):
    """Real jersey crops, cut from real player boxes, spread across a slice."""
    import cv2  # noqa: F401
    import ocr_reader
    import stage2_multikeyframe as s2mk
    import clip_config

    cfg = clip_config.get_clip(clip)
    tp = os.path.join("/runpod-volume", "chunks", clip, "000_tracks.json")
    with open(tp, encoding="utf-8") as fh:
        doc = json.load(fh)

    # biggest boxes first is what stage6 does, but spread them over the slice so
    # the frame reads are representative rather than all from one moment
    picks = []
    for fr in doc["frames"]:
        for t in fr["tracks"]:
            x1, y1, x2, y2 = t["bbox"]
            if (y2 - y1) >= MIN_OCR_HEIGHT:
                picks.append((fr["frame_index"], t["bbox"], y2 - y1))
    if not picks:
        return [], 0
    picks.sort(key=lambda p: p[0])
    step = max(1, len(picks) // n_crops)
    chosen = picks[::step][:n_crops]
    del doc, picks

    by_frame = {}
    for (f, bb, _h) in chosen:
        by_frame.setdefault(f, []).append(bb)
    crops = []
    for f, img in s2mk.iter_frames(cfg.video_path, sorted(by_frame)):
        for bb in by_frame[f]:
            c = ocr_reader.jersey_crop(img, bb)
            if c is not None and c.size and c.shape[0] >= ocr_reader.MIN_CROP_HEIGHT_PX:
                crops.append(c.copy())
    return crops, len(by_frame)


def _time_reader(reader, crops, roster, pool):
    """Seconds per crop at a given pool size, plus what it read."""
    from concurrent.futures import ThreadPoolExecutor
    import cv2

    def one(c):
        up = cv2.resize(c, (c.shape[1] * 4, c.shape[0] * 4), interpolation=cv2.INTER_CUBIC)
        return [(int(t), float(cf))
                for (_b, t, cf) in reader.readtext(up, allowlist="0123456789")
                if t.isdigit() and int(t) in roster]

    reader_warm = one(crops[0])                       # model load is not the rate
    t0 = time.time()
    if pool <= 1:
        reads = [one(c) for c in crops]
    else:
        with ThreadPoolExecutor(max_workers=pool) as ex:
            reads = list(ex.map(one, crops))
    dt = time.time() - t0
    del reader_warm
    return dt / len(crops), reads


def run(config=None, n_crops=200, pools=(1, 6, 32, 96), full_game_crops=150000, **_kwargs):
    for p in ("/app", "/app/spikes", "/app/phase1", "/app/phase2"):
        if p not in sys.path:
            sys.path.insert(0, p)
    import serverless_handler as sh

    if not config:
        return {"error": "no clip config passed in args"}
    clip = config["name"]
    cfg_doc = dict(config)
    cfg_doc["ball_span_len"] = 0
    sh._install_uploaded_clip(clip, cfg_doc, (0, PER_SLICE))

    import roster as roster_mod
    import clips_config
    clips_config.ACTIVE = clip

    t0 = time.time()
    crops, n_frames = _crops(clip, int(n_crops))
    if not crops:
        return {"error": "no eligible crops in slice 0"}
    out = {"clip": clip, "image": sh.image_sha(), "crops": len(crops),
           "frames_read": n_frames, "cut_seconds": round(time.time() - t0, 1),
           "cpu": {}, "roster_size": len(roster_mod.ROSTER_NUMBERS)}
    roster = roster_mod.ROSTER_NUMBERS

    import easyocr
    import torch
    out["cuda"] = torch.cuda.is_available()
    out["cpu_count"] = os.cpu_count()

    cpu_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    cpu_reads = None
    for pool in pools:
        try:
            per, reads = _time_reader(cpu_reader, crops, roster, int(pool))
            out["cpu"][str(pool)] = {
                "seconds_per_crop": round(per, 4),
                "full_game_minutes": round(full_game_crops * per / 60, 1)}
            if cpu_reads is None:
                cpu_reads = reads
        except Exception as e:
            out["cpu"][str(pool)] = {"error": str(e)[:200]}

    # THE IDLE 4090. easyocr takes a gpu flag and the card does nothing at all
    # during the identity tail.
    if out["cuda"]:
        try:
            gpu_reader = easyocr.Reader(["en"], gpu=True, verbose=False)
            per, gpu_reads = _time_reader(gpu_reader, crops, roster, 1)
            out["gpu"] = {"seconds_per_crop": round(per, 4),
                          "full_game_minutes": round(full_game_crops * per / 60, 1)}
            # DOES IT READ THE SAME THING? A faster reader that disagrees is a
            # different pipeline, not a speed-up.
            same = sum(1 for a, b in zip(cpu_reads, gpu_reads)
                       if sorted(a) == sorted(b))
            out["gpu"]["agrees_with_cpu"] = f"{same}/{len(gpu_reads)}"
            out["gpu"]["disagreements"] = [
                {"cpu": a, "gpu": b} for a, b in zip(cpu_reads, gpu_reads)
                if sorted(a) != sorted(b)][:8]
        except Exception as e:
            out["gpu"] = {"error": str(e)[:300]}

    hits = sum(1 for r in (cpu_reads or []) if r)
    out["read_rate"] = f"{hits}/{len(cpu_reads or [])} crops produced an on-roster read"
    out["note"] = ("seconds_per_crop x the real crop count is the whole question; "
                   "the rehearsal measured ~150,000 crops for a full game")
    return out
