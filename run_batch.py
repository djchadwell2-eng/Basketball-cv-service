"""run_batch.py -- run one or more clips through run_clip.py, one clip per
SUBPROCESS (the one-clip-per-process invariant: module-level config binding
means a second run() call in the same process would silently use the first
clip's state -- see run_clip.py's docstring). Writes a manifest so a batch
run has a record: which clips ran, how long, pass/fail, where the artifacts
landed. This is the exact seam Phase 7's worker will call per job -- no
queue framework, just a loop.

Usage:
    .venv/Scripts/python run_batch.py TEST1 HARD
    .venv/Scripts/python run_batch.py            # defaults to every ClipConfig
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

_ROOT = os.path.dirname(os.path.abspath(__file__))
_VENV_PY = os.path.join(_ROOT, ".venv", "Scripts", "python.exe")
PYTHON = _VENV_PY if os.path.exists(_VENV_PY) else sys.executable

MANIFEST_PATH = os.path.join(_ROOT, "batch_manifest.json")
LOG_DIR = os.path.join(_ROOT, "spikes", "out")


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=_ROOT,
                              capture_output=True, text=True, check=True
                              ).stdout.strip()
    except Exception as e:                     # git absent/not a repo -- non-fatal
        return f"unknown ({e})"


def _config_fingerprint(config):
    """Enough to know what ran without re-deriving it from the log."""
    fp = {"video_path": config.video_path,
          "tracking_span": [config.tracking_span_start, config.tracking_span_len]}
    if config.ball_span_len:
        fp["ball_weights"] = os.path.basename(config.ball_weights_path)
        fp["ball_span"] = [config.ball_span_start, config.ball_span_len]
    return fp


def _artifact_paths(config):
    out2 = os.path.join(_ROOT, "phase2", "out")
    paths = {
        "box_score_json": os.path.join(out2, f"{config.name}_box_score.json"),
        "box_score_csv": os.path.join(out2, f"{config.name}_box_score.csv"),
    }
    if config.ball_span_len:
        outb = os.path.join(_ROOT, "spikes", "out")
        paths.update({
            "shot_attempts": os.path.join(outb, f"{config.name}_shot_attempts.json"),
            "shot_locations": os.path.join(outb, f"{config.name}_shot_locations.json"),
            "shot_outcomes": os.path.join(outb, f"{config.name}_shot_outcomes.json"),
            "shot_chart": os.path.join(outb, f"{config.name}_shot_chart.png"),
        })
    return {k: v for k, v in paths.items() if os.path.exists(v)}


def run_one(config):
    """Run one clip in its OWN subprocess; return its manifest entry."""
    log_path = os.path.join(LOG_DIR, f"{config.name}_batch_run.log")
    code = (f"import run_clip, clip_config; "
            f"run_clip.run(clip_config.{config.name}_CLIP)")
    print(f"[run_batch] {config.name}: starting (log -> {log_path})")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.run([PYTHON, "-c", code], cwd=_ROOT,
                              stdout=log_f, stderr=subprocess.STDOUT)
    duration_s = round(time.time() - t0, 1)

    # Exit code alone isn't proof of completion (abstention-first: verify the
    # pipeline's own completion marker, not just that the process didn't crash).
    with open(log_path, encoding="utf-8") as f:
        log_text = f.read()
    completed = "[run_clip] DONE" in log_text
    passed = proc.returncode == 0 and completed

    print(f"[run_batch] {config.name}: exit={proc.returncode} "
          f"completed_marker={completed} duration={duration_s}s "
          f"-> {'PASS' if passed else 'FAIL'}")

    return {
        "clip": config.name,
        "exit_code": proc.returncode,
        "completed_marker_found": completed,
        "passed": passed,
        "duration_s": duration_s,
        "log": log_path,
        "config_fingerprint": _config_fingerprint(config),
        "artifacts": _artifact_paths(config),
    }


def run_batch(configs):
    entries = [run_one(c) for c in configs]
    manifest = {
        "git_commit": _git_commit(),
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "clips": entries,
        "n_pass": sum(1 for e in entries if e["passed"]),
        "n_total": len(entries),
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[run_batch] {manifest['n_pass']}/{manifest['n_total']} passed "
          f"-> {MANIFEST_PATH}")
    return manifest


def main():
    import clip_config
    names = sys.argv[1:]
    if not names:
        names = [n[:-len("_CLIP")] for n in dir(clip_config) if n.endswith("_CLIP")]
    configs = [getattr(clip_config, f"{n}_CLIP") for n in names]
    run_batch(configs)


if __name__ == "__main__":
    main()
