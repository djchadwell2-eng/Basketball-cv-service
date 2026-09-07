"""Put a commit on the GPU workers -- and REFUSE to do it unsafely.

WHY THIS EXISTS. Every deploy so far was hand-typed, and two of them cost real
money for nothing:

  * a commit was pushed, called "shipped", and its build had FAILED at the
    push step -- so the change that removes the temp-mp4 round trip, the single
    biggest saving of the session, never existed in any image;
  * the template was pointed at a commit that had no image at all, so every
    job failed with nothing to show for it.

Both are the same mistake: trusting the git log about what the workers are
running. The git log does not know. Only the registry knows whether a tag can
be pulled, and only a worker knows what it actually loaded. So this asks both,
in order, and stops at the first honest "no":

  1. is this sha PULLABLE from GHCR?         (else: refuse -- nothing changes)
  2. point the template at it
  3. run a `version` job and make the worker say its own sha back

Nothing here spends more than the few seconds of GPU that step 3 costs, and
that cost is the point: it is the difference between "deployed" and "believed
to be deployed".

    .venv/Scripts/python.exe deploy_worker.py            # deploy HEAD
    .venv/Scripts/python.exe deploy_worker.py <sha>      # deploy a chosen commit
    .venv/Scripts/python.exe deploy_worker.py --check    # what is deployed NOW?
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

from run_chunked import _creds                 # one copy of the credentials, not two

REPO = "djchadwell2-eng/basketball-cv-service"
IMAGE = f"ghcr.io/{REPO}"
# rest.runpod.io sits behind Cloudflare, which answers urllib's default
# User-Agent with a 403 (error 1010). The job API (api.runpod.ai) does not care,
# which is why only the admin calls below need this.
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) basketball-cv/1.0"


def _api(path, key, body=None, method=None, base="https://rest.runpod.io/v1"):
    req = urllib.request.Request(
        base + path, method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json",
                 "User-Agent": UA})
    return json.load(urllib.request.urlopen(req, timeout=60))


def pullable(sha: str) -> bool:
    """Does an image with this tag actually EXIST in the registry?

    A green build is not proof: the build can go green on an earlier step and
    still fail to push. The registry is the only thing that knows.
    """
    tok = json.load(urllib.request.urlopen(
        f"https://ghcr.io/token?scope=repository:{REPO}:pull&service=ghcr.io",
        timeout=45))["token"]
    req = urllib.request.Request(
        f"https://ghcr.io/v2/{REPO}/manifests/{sha}",
        headers={"Authorization": f"Bearer {tok}",
                 "Accept": "application/vnd.oci.image.index.v1+json,"
                           "application/vnd.docker.distribution.manifest.v2+json,"
                           "application/vnd.oci.image.manifest.v1+json"})
    try:
        urllib.request.urlopen(req, timeout=45)
        return True
    except urllib.error.HTTPError:
        return False


def deployed(key, eid) -> tuple[str, str]:
    """(template id, image the template currently points at)."""
    tmpl = _api(f"/endpoints/{eid}", key)["templateId"]
    return tmpl, _api(f"/templates/{tmpl}", key).get("imageName", "")


def worker_says(key, eid, wait_s=600) -> str:
    """Ask a REAL worker which image it loaded. The last word on the subject."""
    def v2(path, body=None):
        return _api(f"/{eid}/{path}", key, body, base="https://api.runpod.ai/v2")
    job = v2("run", {"input": {"mode": "version"}})
    t0 = time.time()
    while time.time() - t0 < wait_s:
        d = v2(f"status/{job['id']}")
        if d.get("status") == "COMPLETED":
            return (d.get("output") or {}).get("image", "")
        if d.get("status") in ("FAILED", "CANCELLED", "TIMED_OUT"):
            raise SystemExit(f"version job {d.get('status')}: {d.get('output')}")
        time.sleep(10)
    raise SystemExit(f"version job never ran in {wait_s}s -- no worker capacity?")


def main(argv):
    key, eid = _creds()
    if "--check" in argv:
        tmpl, img = deployed(key, eid)
        print(f"template {tmpl} -> {img}")
        print(f"worker reports -> {worker_says(key, eid)}")
        return 0

    sha = argv[0] if argv else subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
        check=True).stdout.strip()
    sha = subprocess.run(["git", "rev-parse", sha], capture_output=True, text=True,
                         check=True).stdout.strip()

    # A file that only exists on the laptop does not exist in the cloud, and
    # neither does a commit. Say so BEFORE spending anything on a version job.
    unpushed = subprocess.run(["git", "branch", "-r", "--contains", sha],
                              capture_output=True, text=True).stdout.strip()
    if not unpushed:
        raise SystemExit(f"REFUSING: {sha[:12]} is not on any remote branch -- "
                         f"GitHub has never seen it, so no image was ever built. "
                         f"Push it first.")

    print(f"asking the registry about {sha[:12]} ...")
    if not pullable(sha):
        raise SystemExit(
            f"REFUSING: no image at {IMAGE}:{sha[:12]}.\n"
            f"The build for this commit either failed or has not finished. "
            f"Check the Actions tab; pointing the template here would make "
            f"every job fail.")
    print(f"  pullable")

    tmpl, before = deployed(key, eid)
    _api(f"/templates/{tmpl}", key, {"imageName": f"{IMAGE}:{sha}"}, method="PATCH")
    _tmpl, after = deployed(key, eid)
    if after != f"{IMAGE}:{sha}":
        raise SystemExit(f"template did not take the change (still {after})")
    print(f"  template {before.rsplit(':', 1)[-1][:12]} -> {sha[:12]}")

    print(f"asking a worker what it actually loaded ...")
    got = worker_says(key, eid)
    if not sha.startswith(got.rsplit(":", 1)[-1]):
        raise SystemExit(f"WORKER DISAGREES: it reports {got!r}, not {sha[:12]}. "
                         f"Do not run a paid job until this is explained.")
    print(f"  worker confirms {got}")
    print(f"\nDEPLOYED {sha[:12]} -- checked at the registry, the template, "
          f"and a running worker.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
