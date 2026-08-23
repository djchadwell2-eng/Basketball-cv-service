"""exp_machine.py -- what is a worker actually made of?

THE QUESTION THIS ANSWERS. Worker RAM is this project's largest [UNKNOWN]. The
only figure ever established is ">=3.85 GB", and that is not a limit -- it is
the high-water mark of one tail run that happened to survive. Every sizing
decision about the merge job (which holds a whole game's tracking in memory)
has been resting on it.

MEASURED on a full-game-sized cache, laptop, 2026-08-22: the identity tail peaks
at ~0.83 GB per slice, so a ten-slice game is ~8.5 GB. Whether that is fine or
fatal depends entirely on a number nobody has ever read.

Run through lab.py so it needs NO image rebuild -- the script goes on the volume
and a warm worker executes it:

    .venv/Scripts/python.exe lab.py spikes/exp_machine.py

Deliberately reads /proc and shutil only: a new dependency for one measurement
is a new way for a cold start to fail.
"""

import os


def run(**_kwargs):
    out = {}

    # RAM. MemTotal is the box; MemAvailable is what a job can actually get.
    try:
        mem = {}
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                k, _, v = line.partition(":")
                mem[k] = v.strip()
        for key, field in (("MemTotal", "ram_total_gb"),
                           ("MemAvailable", "ram_available_gb")):
            if key in mem:
                out[field] = round(int(mem[key].split()[0]) / 1e6, 2)
    except Exception as e:
        out["ram_error"] = str(e)

    # A CONTAINER MAY BE CAPPED BELOW THE BOX. /proc/meminfo reports the host, so
    # a worker can read "64 GB" and still be killed at 4 -- the cgroup limit is
    # the number that actually ends a job.
    for path, field in (
            ("/sys/fs/cgroup/memory.max", "cgroup_limit_gb"),                 # v2
            ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "cgroup_limit_gb")):  # v1
        try:
            with open(path, encoding="utf-8") as fh:
                raw = fh.read().strip()
            if raw and raw != "max":
                n = int(raw)
                if n < (1 << 62):                 # v1 writes a huge sentinel for "none"
                    out[field] = round(n / 1e9, 2)
            else:
                out[field] = "max (uncapped)"
            break
        except Exception:
            continue

    out["cpu_count"] = os.cpu_count()

    # Disk. The merged caches are written to the CONTAINER, not the volume --
    # ~2 GB pretty-printed for a ten-slice game -- and that limit is unknown too.
    import shutil
    for label, path in (("container", "/app"),
                        ("volume", os.environ.get("RUNPOD_VOLUME_ROOT", "/runpod-volume")),
                        ("tmp", "/tmp")):
        try:
            du = shutil.disk_usage(path)
            out[f"{label}_disk_free_gb"] = round(du.free / 1e9, 2)
            out[f"{label}_disk_total_gb"] = round(du.total / 1e9, 2)
        except Exception:
            out[f"{label}_disk_free_gb"] = None

    try:
        import torch
        out["cuda"] = torch.cuda.is_available()
        if out["cuda"]:
            out["gpu"] = torch.cuda.get_device_name(0)
            out["gpu_vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    except Exception as e:
        out["torch_error"] = str(e)

    # THE ACTUAL QUESTION, answered in one line rather than left as arithmetic
    # for whoever reads this next.
    limit = out.get("cgroup_limit_gb")
    usable = limit if isinstance(limit, (int, float)) else out.get("ram_total_gb")
    if usable:
        out["tail_needs_gb_measured"] = 8.5          # ~0.83 GB/slice x 10 slices
        out["verdict"] = ("the identity tail FITS" if usable >= 10
                          else "TOO SMALL -- the tail must stream its caches "
                               "before a full game can be merged")
        out["headroom_gb"] = round(usable - 8.5, 2)
    return out
