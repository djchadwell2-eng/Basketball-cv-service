"""run_batch.py -- pure-logic tests (fingerprint/artifact helpers) + run_one's
pass/fail decision with subprocess.run stubbed out (no real clip processing,
no video needed). The real end-to-end batch run is exercised separately
against the actual clips, not in the unit suite."""

import json
import os
import sys
from dataclasses import dataclass

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import run_batch  # noqa: E402


@dataclass
class _FakeConfig:
    name: str
    video_path: str
    tracking_span_start: int
    tracking_span_len: int
    ball_weights_path: str = ""
    ball_span_start: int = 0
    ball_span_len: int = 0


def test_config_fingerprint_omits_ball_fields_when_not_configured():
    cfg = _FakeConfig("X", "video.mp4", 0, 10)
    fp = run_batch._config_fingerprint(cfg)
    assert fp == {"video_path": "video.mp4", "tracking_span": [0, 10]}


def test_config_fingerprint_includes_ball_fields_when_configured():
    cfg = _FakeConfig("X", "video.mp4", 0, 10,
                      ball_weights_path="models/ball_finetuned_v2.pt",
                      ball_span_start=0, ball_span_len=605)
    fp = run_batch._config_fingerprint(cfg)
    assert fp["ball_weights"] == "ball_finetuned_v2.pt"
    assert fp["ball_span"] == [0, 605]


def test_artifact_paths_only_lists_files_that_exist(tmp_path, monkeypatch):
    monkeypatch.setattr(run_batch, "_ROOT", str(tmp_path))
    out2 = tmp_path / "phase2" / "out"
    out2.mkdir(parents=True)
    (out2 / "X_box_score.json").write_text("{}")
    # box_score.csv deliberately NOT created -> must be absent from the result
    cfg = _FakeConfig("X", "video.mp4", 0, 10)
    paths = run_batch._artifact_paths(cfg)
    assert set(paths) == {"box_score_json"}


def test_artifact_paths_includes_shot_layer_when_ball_configured(tmp_path, monkeypatch):
    monkeypatch.setattr(run_batch, "_ROOT", str(tmp_path))
    out2 = tmp_path / "phase2" / "out"
    outb = tmp_path / "spikes" / "out"
    out2.mkdir(parents=True)
    outb.mkdir(parents=True)
    (out2 / "X_box_score.json").write_text("{}")
    (outb / "X_shot_attempts.json").write_text("{}")
    cfg = _FakeConfig("X", "video.mp4", 0, 10, ball_span_len=605)
    paths = run_batch._artifact_paths(cfg)
    assert "shot_attempts" in paths
    assert "shot_locations" not in paths     # not written -> not listed


def test_run_one_fails_on_nonzero_exit_even_with_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(run_batch, "_ROOT", str(tmp_path))
    monkeypatch.setattr(run_batch, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(run_batch, "PYTHON", sys.executable)

    def fake_run(cmd, cwd, stdout, stderr):
        stdout.write("[run_clip] DONE -- X ran end-to-end.\n")
        class R:
            returncode = 1
        return R()
    monkeypatch.setattr(run_batch.subprocess, "run", fake_run)

    cfg = _FakeConfig("X", "video.mp4", 0, 10)
    entry = run_batch.run_one(cfg)
    assert entry["exit_code"] == 1
    assert entry["completed_marker_found"] is True
    assert entry["passed"] is False    # exit code alone can't override a nonzero exit


def test_run_one_fails_on_exit_zero_without_completion_marker(tmp_path, monkeypatch):
    """Abstention-first: a clean exit code is NOT proof of completion -- a
    crash mid-pipeline that happens to exit 0 (e.g. a caught exception that
    swallows the error) must still fail the batch."""
    monkeypatch.setattr(run_batch, "_ROOT", str(tmp_path))
    monkeypatch.setattr(run_batch, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(run_batch, "PYTHON", sys.executable)

    def fake_run(cmd, cwd, stdout, stderr):
        stdout.write("some partial output, no completion line\n")
        class R:
            returncode = 0
        return R()
    monkeypatch.setattr(run_batch.subprocess, "run", fake_run)

    cfg = _FakeConfig("X", "video.mp4", 0, 10)
    entry = run_batch.run_one(cfg)
    assert entry["completed_marker_found"] is False
    assert entry["passed"] is False


def test_run_one_passes_on_exit_zero_with_completion_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(run_batch, "_ROOT", str(tmp_path))
    monkeypatch.setattr(run_batch, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(run_batch, "PYTHON", sys.executable)

    def fake_run(cmd, cwd, stdout, stderr):
        stdout.write("[run_clip] DONE -- X ran end-to-end.\n")
        class R:
            returncode = 0
        return R()
    monkeypatch.setattr(run_batch.subprocess, "run", fake_run)

    cfg = _FakeConfig("X", "video.mp4", 0, 10)
    entry = run_batch.run_one(cfg)
    assert entry["passed"] is True


def test_run_batch_writes_manifest_with_summary_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(run_batch, "_ROOT", str(tmp_path))
    monkeypatch.setattr(run_batch, "LOG_DIR", str(tmp_path))
    monkeypatch.setattr(run_batch, "MANIFEST_PATH", str(tmp_path / "batch_manifest.json"))
    monkeypatch.setattr(run_batch, "_git_commit", lambda: "deadbeef")

    calls = {"n": 0}
    def fake_run_one(config):
        calls["n"] += 1
        return {"clip": config.name, "passed": calls["n"] == 1}
    monkeypatch.setattr(run_batch, "run_one", fake_run_one)

    cfgs = [_FakeConfig("A", "a.mp4", 0, 1), _FakeConfig("B", "b.mp4", 0, 1)]
    manifest = run_batch.run_batch(cfgs)
    assert manifest["n_pass"] == 1
    assert manifest["n_total"] == 2
    assert manifest["git_commit"] == "deadbeef"
    on_disk = json.load(open(tmp_path / "batch_manifest.json"))
    assert on_disk == manifest
