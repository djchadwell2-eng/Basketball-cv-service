"""upload_film.py -- put a game's film where the GPU can reach it.

WHY A NETWORK VOLUME AND NOT THE APP'S OWN STORAGE. The obvious route was
Supabase, which the app already uses -- but it refuses the file: a 87 MB test
upload came back 413 "EntityTooLarge", and a real game is 3.4 GB. RunPod's
network volume has no such cap, is already paid for (50 GB, "BasketBall-
Training"), and lives in the SAME datacenter as the workers, so the film is
mounted at /runpod-volume rather than downloaded again on every job.

Credentials are S3 API keys made in the RunPod console (Settings -> S3 API
Keys); the ordinary RunPod API key does NOT work for S3 (verified: it returns
SignatureDoesNotMatch). Set them in the environment:

    RUNPOD_S3_ACCESS_KEY, RUNPOD_S3_SECRET_KEY

Usage:  .venv/Scripts/python.exe upload_film.py <CLIP_NAME>

Uploading a 3.4 GB file over a home connection is slow, and that is the point
of doing it at SETUP time: it runs while the coach marks the court, so the
wait overlaps with work they were doing anyway.
"""

from __future__ import annotations

import os
import sys
import threading
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

import clip_registry  # noqa: E402

# The volume and its datacenter. Both are fixed by which volume the serverless
# endpoint mounts -- an object written to another region's endpoint would be
# invisible to the workers.
# US-NC-1, not the original US-IL-1. A network volume PINS every worker to its
# datacenter, and US-IL-1 ran out of GPUs: workers were allocated and then
# throttled indefinitely -- no error, no queue movement, for hours. Detaching
# the volume let workers start in minutes, which is what proved it. US-NC-1
# answered a probe with zero queue delay, and (unlike US-TX-3, which took the
# volume and then had nowhere to upload to) it has an S3 endpoint.
VOLUME_ID = os.environ.get("RUNPOD_VOLUME_ID", "r5hc0v2j7v")
DATACENTER = os.environ.get("RUNPOD_VOLUME_DC", "US-NC-1")
S3_ENDPOINT = f"https://s3api-{DATACENTER.lower()}.runpod.io/"

# Where the worker will find it: /runpod-volume/<key>
KEY_PREFIX = "films"


def _client():
    ak = os.environ.get("RUNPOD_S3_ACCESS_KEY")
    sk = os.environ.get("RUNPOD_S3_SECRET_KEY")
    if not ak or not sk:
        raise SystemExit(
            "MISSING S3 CREDENTIALS. Create them in the RunPod console under\n"
            "  Settings -> S3 API Keys -> Create key\n"
            "then set RUNPOD_S3_ACCESS_KEY and RUNPOD_S3_SECRET_KEY.\n"
            "(The normal RUNPOD_API_KEY does not work for S3 -- it fails with\n"
            "SignatureDoesNotMatch.)")
    import boto3
    from boto3.s3.transfer import TransferConfig  # noqa: F401  (import check)
    return boto3.client(
        "s3", endpoint_url=S3_ENDPOINT, region_name=DATACENTER,
        aws_access_key_id=ak, aws_secret_access_key=sk)


class _Progress:
    """Percent to stdout, so the web app's log tail can show a real bar."""

    def __init__(self, total: int):
        self.total = total
        self.seen = 0
        self.last = 0.0
        self.lock = threading.Lock()

    def __call__(self, chunk: int):
        with self.lock:
            self.seen += chunk
            pct = 100.0 * self.seen / max(1, self.total)
            now = time.time()
            if pct - self.last >= 1.0 or self.seen == self.total:
                self.last = pct
                print(f"[upload_film] PROGRESS {pct:.0f}% "
                      f"({self.seen / 1e9:.2f} / {self.total / 1e9:.2f} GB)", flush=True)


def upload(clip: str) -> str:
    doc = clip_registry.load(clip)
    if doc is None:
        raise SystemExit(f"[upload_film] FAILED: no clips/{clip}.json")
    src = doc.get("video_path")
    if not src or not os.path.exists(src):
        raise SystemExit(f"[upload_film] FAILED: film not found at {src!r}")

    key = f"{KEY_PREFIX}/{clip}{os.path.splitext(src)[1] or '.mp4'}"
    size = os.path.getsize(src)
    print(f"[upload_film] STAGE upload {clip} ({size / 1e9:.2f} GB) -> "
          f"{VOLUME_ID}:{key}", flush=True)

    # SEQUENTIAL, single-threaded on purpose. Uploading the parts in parallel
    # (max_concurrency=4) got 3.4 GB across and then failed at the last step
    # with "part 1 is missing; cannot complete multipart upload; 4 parts
    # missing" -- RunPod's S3 does not reliably register concurrently-uploaded
    # parts. One at a time costs little here (a home connection is the limit,
    # not request latency) and it actually completes.
    from boto3.s3.transfer import TransferConfig
    cfg = TransferConfig(multipart_threshold=64 * 1024 * 1024,
                         multipart_chunksize=64 * 1024 * 1024,
                         max_concurrency=1, use_threads=False)
    t0 = time.time()
    _client().upload_file(src, VOLUME_ID, key, Config=cfg, Callback=_Progress(size))
    dt = time.time() - t0

    # Recorded on the clip itself: the job input needs it, and a re-run must not
    # upload 3.4 GB a second time.
    clip_registry.update(clip, volume_key=key, volume_id=VOLUME_ID,
                         volume_bytes=size, uploaded_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"))
    print(f"[upload_film] STAGE done in {dt / 60:.1f} min "
          f"({size / 1e6 / max(1, dt):.1f} MB/s)", flush=True)
    return key


def already_uploaded(clip: str) -> bool:
    """True when this clip's film is already on the volume, same size."""
    doc = clip_registry.load(clip) or {}
    if not doc.get("volume_key"):
        return False
    src = doc.get("video_path")
    try:
        return bool(src and os.path.getsize(src) == doc.get("volume_bytes"))
    except OSError:
        return False


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if not clip_registry.valid_name(name):
        raise SystemExit("usage: python upload_film.py <CLIP_NAME>")
    if already_uploaded(name):
        print(f"[upload_film] {name} already on the volume -- nothing to do", flush=True)
    else:
        upload(name)
