"""env_local -- the loader that makes .env.local actually reach the code.

THE BUG BEING PINNED (2026-08-02): every Gemini/Gemma consumer reads
os.environ.get("GEMINI_API_KEY") and nothing ever set it, so .env.local was
read by nobody and the universal scoreboard reader silently never ran. It did
not error -- a missing key and a present one behaved identically, which is why
it hid for a whole session.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import env_local  # noqa: E402


def write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(body, encoding="utf-8")
    return str(p)


# ------------------------------------------------------------------ parsing --

def test_plain_assignment(tmp_path):
    p = write(tmp_path, "a.env", "GEMINI_API_KEY=abc123\n")
    assert env_local.parse(p)["GEMINI_API_KEY"] == "abc123"


def test_quotes_are_stripped(tmp_path):
    p = write(tmp_path, "a.env", 'A="dq"\nB=\'sq\'\n')
    got = env_local.parse(p)
    assert got["A"] == "dq" and got["B"] == "sq"


def test_export_prefix_and_spacing(tmp_path):
    p = write(tmp_path, "a.env", "export A = spaced\n")
    assert env_local.parse(p)["A"] == "spaced"


def test_comments_and_blank_lines_are_ignored(tmp_path):
    p = write(tmp_path, "a.env", "# a comment\n\nA=1\n# B=2\n")
    got = env_local.parse(p)
    assert got["A"] == "1" and "B" not in got


def test_a_value_containing_equals_survives(tmp_path):
    """Real keys and URLs carry '=' padding."""
    p = write(tmp_path, "a.env", "A=abc=def==\n")
    assert env_local.parse(p)["A"] == "abc=def=="


def test_a_missing_file_is_not_an_error(tmp_path):
    assert env_local.parse(str(tmp_path / "nope.env")) == {}


def test_an_unreadable_file_does_not_take_down_the_run(tmp_path):
    """A broken env file must not fail a whole pipeline run -- same discipline
    clip_registry uses when it skips a malformed clip."""
    assert env_local.parse(str(tmp_path)) == {}      # a directory, not a file


# ------------------------------------------------------------------ loading --

def test_load_sets_missing_vars_and_reports_the_source(tmp_path, monkeypatch):
    p = write(tmp_path, "a.env", "TEST_EL_KEY=fromfile\n")
    monkeypatch.setattr(env_local, "ENV_FILES", (p,))
    monkeypatch.setattr(env_local, "_loaded", False)
    monkeypatch.setattr(env_local, "_sources", {})
    monkeypatch.delenv("TEST_EL_KEY", raising=False)

    env_local.load(verbose=False)
    assert os.environ["TEST_EL_KEY"] == "fromfile"
    assert env_local.found("TEST_EL_KEY") == p


def test_a_real_environment_variable_always_wins(tmp_path, monkeypatch):
    """So a deployment can override without editing files on disk."""
    p = write(tmp_path, "a.env", "TEST_EL_KEY=fromfile\n")
    monkeypatch.setattr(env_local, "ENV_FILES", (p,))
    monkeypatch.setattr(env_local, "_loaded", False)
    monkeypatch.setattr(env_local, "_sources", {})
    monkeypatch.setenv("TEST_EL_KEY", "fromenv")

    env_local.load(verbose=False)
    assert os.environ["TEST_EL_KEY"] == "fromenv"
    assert env_local.found("TEST_EL_KEY") == "environment"


def test_the_first_file_wins_over_the_second(tmp_path, monkeypatch):
    """The two .env.local files on this machine held DIFFERENT keys. The
    service copy is listed first and must win."""
    a = write(tmp_path, "a.env", "TEST_EL_KEY=service\n")
    b = write(tmp_path, "b.env", "TEST_EL_KEY=app\n")
    monkeypatch.setattr(env_local, "ENV_FILES", (a, b))
    monkeypatch.setattr(env_local, "_loaded", False)
    monkeypatch.setattr(env_local, "_sources", {})
    monkeypatch.delenv("TEST_EL_KEY", raising=False)

    env_local.load(verbose=False)
    assert os.environ["TEST_EL_KEY"] == "service"


def test_an_empty_value_is_not_treated_as_a_key(tmp_path, monkeypatch):
    """KEY= with nothing after it must not read as 'configured'."""
    p = write(tmp_path, "a.env", "TEST_EL_KEY=\n")
    monkeypatch.setattr(env_local, "ENV_FILES", (p,))
    monkeypatch.setattr(env_local, "_loaded", False)
    monkeypatch.setattr(env_local, "_sources", {})
    monkeypatch.delenv("TEST_EL_KEY", raising=False)

    env_local.load(verbose=False)
    assert "TEST_EL_KEY" not in os.environ


def test_found_is_none_for_something_we_do_not_have(monkeypatch):
    monkeypatch.delenv("TEST_EL_ABSENT", raising=False)
    assert env_local.found("TEST_EL_ABSENT") is None


def test_load_is_idempotent(tmp_path, monkeypatch):
    p = write(tmp_path, "a.env", "TEST_EL_KEY=one\n")
    monkeypatch.setattr(env_local, "ENV_FILES", (p,))
    monkeypatch.setattr(env_local, "_loaded", False)
    monkeypatch.setattr(env_local, "_sources", {})
    monkeypatch.delenv("TEST_EL_KEY", raising=False)

    first = env_local.load(verbose=False)
    second = env_local.load(verbose=False)
    assert first == second


# --------------------------------------------------------- the real machine --

def test_the_real_env_files_are_readable():
    """Not asserting a key exists -- that depends on the machine. Only that
    every configured path parses without blowing up."""
    for path in env_local.ENV_FILES:
        assert isinstance(env_local.parse(path), dict)
