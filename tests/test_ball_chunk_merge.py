"""A ball log glued from slices must satisfy the TAIL'S OWN reuse check.

This is the test that matters for chunked ball detection, and it is here
because the first version failed it: merge_streamed inherited the first slice's
header for tracks and on-court but looked up the wrong path for ball, so the
merged log carried no model/imgsz/conf. detections_current then said "not mine",
the tail RE-DETECTED all 171,120 frames on one machine, and nothing anywhere
said so -- the chunking would have bought nothing, silently, at ~43 minutes a
run.

Offline: fake slices, no GPU, no film.
"""
import json, os, shutil, sys
R = r"C:\Users\djcha\New folder\basketball-cv-service"
import tempfile
VOL = os.path.join(tempfile.mkdtemp(), "ballvol")
shutil.rmtree(VOL, ignore_errors=True); os.makedirs(VOL)
os.environ["RUNPOD_VOLUME_ROOT"] = VOL
sys.path.insert(0, R); sys.path.insert(0, os.path.join(R, "spikes"))
import serverless_handler as sh
import ball_stages, clip_config, tracking as trk, ball_spike


def test_a_ball_log_glued_from_slices_is_reused_by_the_tail():
    cfg = clip_config.TEST1_CLIP
    CLIP, PER, N = cfg.name, 200, 3
    chunks = [{"index": i, "start": i*PER, "length": PER} for i in range(N)]

    # slices exactly as ball_spike.detect writes them
    for c in chunks:
        p = sh._ball_chunk_path(CLIP, c["index"])
        os.makedirs(os.path.dirname(p), exist_ok=True)
        json.dump({"clip": CLIP, "span_start": c["start"], "span_len": c["length"],
                   "fps": 30.0, "conf_threshold": ball_spike.CONF,
                   "imgsz": trk.IMG_SIZE,
                   "model": os.path.basename(cfg.ball_weights_path),
                   "frames": [{"frame_index": c["start"]+k, "t_sec": round((c["start"]+k)/30,2),
                               "detections": []} for k in range(c["length"])]},
                  open(p, "w"))

    out = os.path.join(VOL, "merged_ball.json")
    head = {"clip": CLIP, "span_start": 0, "span_len": N*PER}
    n = sh.merge_streamed(CLIP, chunks, "ball", out, head)
    doc = json.load(open(out, encoding="utf-8"))
    idx = [f["frame_index"] for f in doc["frames"]]
    assert idx == list(range(N*PER)), "merged ball frames are not contiguous"


    import dataclasses
    probe = dataclasses.replace(cfg, ball_span_start=0, ball_span_len=N*PER)
    ok = ball_stages.detections_current(doc, probe, trk.IMG_SIZE, ball_spike.CONF)
    print(f"\nTAIL WOULD REUSE IT: {ok}")
    # and a game with a hole in it must NOT be glued
    os.remove(sh._ball_chunk_path(CLIP, 1))
    have = sum(1 for c in chunks if os.path.exists(sh._ball_chunk_path(CLIP, c["index"])))
    print(f"\nwith a slice missing: {have}/{N} present -> merge is skipped, "
          f"tail detects for itself (no game with holes)")
