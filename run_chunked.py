"""run_chunked.py -- analyse a game on TEN GPUs at once instead of one.

WHY. Measured on DJ's own game: detection is 31 min on one 4090 and the camera
anchor is 27.9 h even after the GPU rewrite. One machine chewing a 95-minute
game was never going to hit his bar (under 30 min; under 10 is "beautiful").

The expensive stages look at one frame at a time and never look back, so they
split cleanly. Ten workers on a tenth of the game each finish in about a tenth
of the time -- and it costs the SAME, because RunPod bills by the second: ten
machines for three minutes is one machine for thirty.

WHAT SPLITS AND WHAT DOES NOT.
  splits    tracking bodies, and anchoring the camera        (per-frame work)
  does not  who each player IS                               (whole-game work)
So the workers build caches only, and identity runs once at the end over the
merged result. Track ids are offset per slice rather than guessed at across
seams -- see merge_chunks in serverless_handler.py.

Usage:
    .venv/Scripts/python.exe run_chunked.py <CLIP> [SLICES] [START] [LENGTH]
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

import clip_registry  # noqa: E402

APP_ENV = os.path.join(os.path.dirname(_ROOT), "Basketball Analysis App", ".env.local")


def _creds():
    """RunPod key + endpoint. Read from the app's env file, which is where they
    already live -- one copy, not two that can drift apart."""
    key = os.environ.get("RUNPOD_API_KEY")
    eid = os.environ.get("RUNPOD_ENDPOINT_ID")
    if not (key and eid) and os.path.exists(APP_ENV):
        env = dict(re.findall(r"^([A-Z_]+)=(.*)$", open(APP_ENV, encoding="utf-8").read(), re.M))
        key = key or env.get("RUNPOD_API_KEY", "").strip().strip('"')
        eid = eid or env.get("RUNPOD_ENDPOINT_ID", "").strip().strip('"')
    if not (key and eid):
        raise SystemExit("RUNPOD_API_KEY / RUNPOD_ENDPOINT_ID not found")
    return key, eid


def _post(url, key, body):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))


def _get(url, key):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    return json.load(urllib.request.urlopen(req))


def slices(start: int, length: int, n: int):
    """Even slices, with the remainder given to the last one so no frame is
    dropped -- a missing frame here is a missing second of a player's floor
    time later."""
    per = length // n
    out = []
    for i in range(n):
        s = start + i * per
        ln = per if i < n - 1 else length - per * (n - 1)
        out.append({"index": i, "start": s, "length": ln})
    return out


def run(clip: str, n: int = 10, start: int | None = None, length: int | None = None) -> dict:
    key, eid = _creds()
    doc = clip_registry.load(clip)
    if doc is None:
        raise SystemExit(f"no clips/{clip}.json")
    if not doc.get("volume_key"):
        raise SystemExit(f"{clip}: film is not on the GPU volume yet -- run upload_film.py")

    start = doc.get("tracking_span_start", 0) if start is None else start
    if length is None:
        length = doc.get("tracking_span_len")
        if not length:
            raise SystemExit(f"{clip}: no tracking span set, and none given")

    parts = slices(start, length, n)
    print(f"[chunked] {clip}: {length} frames in {n} slices of ~{parts[0]['length']}", flush=True)

    jobs = []
    for p in parts:
        r = _post(f"https://api.runpod.ai/v2/{eid}/run", key,
                  {"input": {"mode": "chunk", "clip": clip, "config": doc, **p}})
        jobs.append({"id": r["id"], **p})
        print(f"[chunked]   slice {p['index']} -> {r['id']}", flush=True)

    t0 = time.time()
    done, failed = {}, {}
    while len(done) + len(failed) < len(jobs):
        time.sleep(15)
        for j in jobs:
            if j["index"] in done or j["index"] in failed:
                continue
            d = _get(f"https://api.runpod.ai/v2/{eid}/status/{j['id']}", key)
            st = d.get("status")
            if st == "COMPLETED":
                o = d.get("output") or {}
                (done if o.get("ok") else failed)[j["index"]] = o
                print(f"[chunked]   slice {j['index']} {st} "
                      f"({o.get('seconds')}s)  {len(done)}/{len(jobs)} ok", flush=True)
            elif st in ("FAILED", "CANCELLED", "TIMED_OUT"):
                failed[j["index"]] = d.get("output") or {"error": st}
                print(f"[chunked]   slice {j['index']} {st}", flush=True)

    wall = time.time() - t0
    print(f"[chunked] slices finished in {wall / 60:.1f} min "
          f"({len(done)} ok, {len(failed)} failed)", flush=True)
    if failed:
        # A missing slice is a hole in the game, not a smaller game. Refuse.
        for i, f in sorted(failed.items()):
            print(f"[chunked]   slice {i} error: {str(f.get('error'))[:200]}")
        raise SystemExit("[chunked] refusing to merge with slices missing")

    print("[chunked] merging + identity over the whole game ...", flush=True)
    r = _post(f"https://api.runpod.ai/v2/{eid}/run", key,
              {"input": {"mode": "merge", "clip": clip, "config": doc, "chunks": parts}})
    mid = r["id"]
    while True:
        d = _get(f"https://api.runpod.ai/v2/{eid}/status/{mid}", key)
        if d.get("status") in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            break
        time.sleep(20)

    o = d.get("output") or {}
    total = time.time() - t0
    if not o.get("ok"):
        print(f"[chunked] MERGE FAILED: {o.get('error')}")
        print((o.get("traceback") or "")[-1500:])
        raise SystemExit(1)

    stats = o["measured_stats"]
    out_path = os.path.join(_ROOT, "spikes", "out", f"{clip}_measured_stats.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    print(f"\n[chunked] DONE in {total / 60:.1f} min total "
          f"({wall / 60:.1f} slices + {(total - wall) / 60:.1f} merge)")
    print(f"[chunked] {len(stats.get('box_score', []))} named players, "
          f"{len(stats.get('shots', []))} shots -> {out_path}")
    return stats


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python run_chunked.py <CLIP> [SLICES] [START] [LENGTH]")
    run(sys.argv[1],
        int(sys.argv[2]) if len(sys.argv) > 2 else 10,
        int(sys.argv[3]) if len(sys.argv) > 3 else None,
        int(sys.argv[4]) if len(sys.argv) > 4 else None)
