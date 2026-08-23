"""lab.py -- run an experiment on the GPU in about a minute instead of twenty.

THE PROBLEM THIS FIXES. Every measurement used to cost ~20 minutes of overhead
before it measured anything: edit a file, commit, wait for GitHub Actions to
build a multi-gigabyte image, repoint the RunPod template, pay a cold start.
That capped us at two or three experiments a day. DJ, 2026-08-03: "these tests
are slowing me down like crazy... it needs to be more."

HOW. Experiment code is uploaded to the shared network volume (a second) and
the worker just runs it (serverless_handler mode "exec"). No build, no deploy.
The cycle becomes the experiment itself.

The script must define run(**kwargs) and return something JSON-able.

    .venv/Scripts/python.exe lab.py spikes/my_experiment.py --clip Full_Game_9eb8bf2a \
        --args '{"frames": 50}'

Anything that earns its keep gets committed into the image properly afterwards;
this is for measuring, not for shipping.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

import clip_registry  # noqa: E402
import upload_film    # noqa: E402  (reuses its S3 client + volume id)

APP_ENV = os.path.join(os.path.dirname(_ROOT), "Basketball Analysis App", ".env.local")


def _creds():
    key = os.environ.get("RUNPOD_API_KEY")
    eid = os.environ.get("RUNPOD_ENDPOINT_ID")
    if not (key and eid) and os.path.exists(APP_ENV):
        # [A-Z_0-9], not [A-Z_]: the S3 keys are RUNPOD_S3_ACCESS_KEY and
        # RUNPOD_S3_SECRET_KEY, and a name class without digits cannot match the
        # "3". So the two variables this file needs to UPLOAD anything were the
        # exact two it could never read, and every lab run died at the upload
        # with "MISSING S3 CREDENTIALS" while the credentials sat in the file.
        env = dict(re.findall(r"^([A-Z_0-9]+)=(.*)$", open(APP_ENV, encoding="utf-8").read(), re.M))
        key = key or env.get("RUNPOD_API_KEY", "").strip().strip('"')
        eid = eid or env.get("RUNPOD_ENDPOINT_ID", "").strip().strip('"')
        for k in ("RUNPOD_S3_ACCESS_KEY", "RUNPOD_S3_SECRET_KEY"):
            os.environ.setdefault(k, env.get(k, "").strip().strip('"'))
    if not (key and eid):
        raise SystemExit("RUNPOD_API_KEY / RUNPOD_ENDPOINT_ID not found")
    return key, eid


def push(script_path: str) -> str:
    """Put the experiment on the volume. Returns its name."""
    name = os.path.splitext(os.path.basename(script_path))[0]
    if not re.match(r"^[A-Za-z0-9_-]+$", name):
        raise SystemExit(f"experiment name must be simple: {name!r}")
    upload_film._client().upload_file(script_path, upload_film.VOLUME_ID,
                                      f"experiments/{name}.py")
    return name


def run(script_path: str, clip: str | None = None, args: dict | None = None,
        timeout_min: int = 60) -> dict:
    key, eid = _creds()
    name = push(script_path)
    print(f"[lab] uploaded {name}.py", flush=True)

    payload = {"mode": "exec", "script": name, "args": args or {}}
    if clip:
        doc = clip_registry.load(clip)
        if doc is None:
            raise SystemExit(f"no clips/{clip}.json")
        payload["clip"] = clip
        payload["config"] = doc

    t0 = time.time()
    req = urllib.request.Request(
        f"https://api.runpod.ai/v2/{eid}/run", data=json.dumps({"input": payload}).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    jid = json.load(urllib.request.urlopen(req))["id"]
    print(f"[lab] job {jid}", flush=True)

    while time.time() - t0 < timeout_min * 60:
        d = json.load(urllib.request.urlopen(urllib.request.Request(
            f"https://api.runpod.ai/v2/{eid}/status/{jid}",
            headers={"Authorization": f"Bearer {key}"})))
        if d.get("status") in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            o = d.get("output") or {}
            print(f"[lab] {d['status']} in {(time.time() - t0) / 60:.1f} min", flush=True)
            if not o.get("ok"):
                print("[lab] ERROR:", o.get("error"))
                print((o.get("traceback") or "")[-1500:])
                return o
            print(json.dumps(o.get("result"), indent=1)[:4000])
            return o
        time.sleep(10)
    raise SystemExit(f"[lab] still running after {timeout_min} min (job {jid})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("--clip")
    ap.add_argument("--args", default="{}")
    ap.add_argument("--timeout", type=int, default=60)
    a = ap.parse_args()
    run(a.script, a.clip, json.loads(a.args), a.timeout)
