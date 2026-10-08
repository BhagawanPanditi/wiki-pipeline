"""Shared helpers for the end-to-end tests.

Most tests run a stage script as a subprocess, with the same environment and the
same failure message. That logic lives here once, so every test fails in the
same, fully explained way.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def run_script(script, args, *, expect_rc=0, timeout=300):
    """Run a repository script as a subprocess and assert its exit code.

    Args:
        script: File name of the script at the repository root, for example
            ``"verify_extraction.py"``.
        args: Command-line arguments that follow the script name.
        expect_rc: The exit code the run should produce.
        timeout: Seconds to wait before the run is abandoned.

    Returns:
        The completed ``subprocess.CompletedProcess``, with stdout and stderr
        captured as text.

    Raises:
        AssertionError: If the exit code differs from ``expect_rc``. The message
            includes the arguments and the tail of stdout and stderr.
    """
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    # A stub path from one test must never leak into another's subprocess.
    env.pop("PYTHONPATH", None)
    proc = subprocess.run([sys.executable, str(REPO_ROOT / script), *args],
                          cwd=REPO_ROOT, env=env, capture_output=True, text=True,
                          timeout=timeout)
    assert proc.returncode == expect_rc, (
        f"exit {proc.returncode} (expected {expect_rc})\nARGS: {args}\n"
        f"STDOUT:\n{proc.stdout[-3000:]}\nSTDERR:\n{proc.stderr[-3000:]}")
    return proc
