"""Load API keys from .env.local files into the environment, out loud.

THE BUG THIS FIXES (found 2026-08-02). Every consumer of the Gemini/Gemma key
reads it with os.environ.get("GEMINI_API_KEY") -- measured_stats.py and all six
spikes/gemma_* and gemini_* scripts. NOTHING in this repo ever put it there.
The key sat in .env.local being read by nobody, so a plain command-line run
always fell through to the slower OCR reader. It never errored: measured_stats
treats a missing key as "Gemma not configured" and quietly uses the fallback,
so the universal scoreboard reader looked installed and was simply never
running.

WHY IT PRINTS. That silence is the actual defect -- a key that is missing and a
key that is present behave identically from the outside. So loading is never
silent: it says which file a key came from, and any consumer that wants to know
can ask found(). A fallback nobody can see is indistinguishable from a bug.

TWO FILES, AND WHY THIS ORDER. Keys live in two places on this machine:
    ./.env.local                            (this service)
    ../Basketball Analysis App/.env.local   (the web app)
run_chunked.py already reads RunPod credentials from the app's copy, with the
comment "one copy, not two that can drift apart". They HAVE drifted: on
2026-08-02 the two files held DIFFERENT GEMINI_API_KEY values. The service copy
wins here because it holds the key added when the Gemma reader was built and
verified, while the app copy predates it. A real environment variable still
beats both, so a deployment can override without editing files.

DELIBERATELY NOT python-dotenv: this repo already parses env files with one
regex in run_chunked.py, and a dependency for one line is not worth it. Same
regex, one place.
"""

from __future__ import annotations

import os
import re

_ROOT = os.path.dirname(os.path.abspath(__file__))

# Highest priority last-resort first: an already-set environment variable always
# wins, then these files in order.
ENV_FILES = (
    os.path.join(_ROOT, ".env.local"),
    os.path.join(os.path.dirname(_ROOT), "Basketball Analysis App", ".env.local"),
)

_LINE = re.compile(r"^\s*(?:export\s+)?([A-Z][A-Z0-9_]*)\s*=\s*(.*)$", re.M)

_loaded = False
_sources: dict[str, str] = {}      # var name -> which file it came from


def parse(path):
    """One .env file -> {NAME: value}. Missing or unreadable file -> {}.

    A broken env file must not take down a pipeline run, for the same reason
    clip_registry skips a malformed clip rather than failing every other one.
    """
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return {}
    out = {}
    for name, raw in _LINE.findall(text):
        val = raw.strip()
        if val[:1] in ('"', "'") and val[-1:] == val[:1] and len(val) > 1:
            val = val[1:-1]                    # strip matching quotes
        out[name] = val
    return out


def load(verbose=False):
    """Populate os.environ from the env files. Never overwrites a variable that
    is already set. Returns {name: source_path} for what this call added.

    Safe to call repeatedly -- it only does the work once per process.

    QUIET BY DEFAULT, deliberately. The env files hold a dozen unrelated
    secrets (Supabase, RunPod, OpenAI) and announcing all of them on every stats
    run is noise that buries the one line anybody cares about. The REPORTING
    belongs to the consumer: whoever needs a key prints where it came from, via
    found(). measured_stats.generate() does exactly that for GEMINI_API_KEY.
    verbose=True is for debugging this loader itself.
    """
    global _loaded
    if _loaded:
        return dict(_sources)
    _loaded = True

    for path in ENV_FILES:
        for name, val in parse(path).items():
            if not val or name in os.environ:
                continue                        # a real env var always wins
            os.environ[name] = val
            _sources[name] = path

    if verbose and _sources:
        for name, path in sorted(_sources.items()):
            where = os.path.relpath(path, _ROOT)
            print(f"[env_local] {name} loaded from {where}")
    return dict(_sources)


def found(name):
    """Where a variable came from: a file path, 'environment' if it was already
    set, or None if we do not have it at all. For printing WHY a layer is or is
    not running -- the thing whose absence hid this bug for a whole session."""
    if name not in os.environ:
        return None
    return _sources.get(name, "environment")
