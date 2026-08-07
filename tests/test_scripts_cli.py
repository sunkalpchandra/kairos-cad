"""CLI contract tests.

The scripts are the project's public surface — a broken argument parser or an
unhelpful failure on a missing checkpoint is a real defect, and one that unit
tests of the libraries never catch. These run each CLI as a subprocess.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = [
    "audit_dataset.py",
    "dataset_stats.py",
    "train_bc.py",
    "evaluate_bc.py",
    "replay_policy.py",
    "train_ppo.py",
    "evaluate_ppo.py",
]


def _run(script, *args, timeout=120):
    return subprocess.run(
        [sys.executable, str(REPO / "scripts" / script), *args],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=timeout,
    )


@pytest.mark.parametrize("script", SCRIPTS)
def test_help_works(script):
    """--help must not import-error or crash; it is the first thing anyone runs."""
    result = _run(script, "--help")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


@pytest.mark.parametrize("script", SCRIPTS)
def test_scripts_are_executable_files(script):
    path = REPO / "scripts" / script
    assert path.exists()
    assert path.read_text().startswith("#!/usr/bin/env python")


def test_unknown_flags_are_rejected():
    result = _run("train_ppo.py", "--not-a-real-flag")
    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr


def test_missing_bc_checkpoint_explains_the_fix(tmp_path):
    """A missing checkpoint should say what to run, not raise a traceback."""
    pytest.importorskip("torch", reason="requires the 'learn' extra")
    result = _run(
        "train_ppo.py", "--bc", str(tmp_path / "absent.pt"), "--iterations", "1"
    )
    assert result.returncode == 1
    assert "make train-bc" in result.stderr
    assert "Traceback" not in result.stderr


def test_evaluate_bc_reports_a_missing_checkpoint(tmp_path):
    pytest.importorskip("torch", reason="requires the 'learn' extra")
    result = _run("evaluate_bc.py", "--checkpoint", str(tmp_path / "absent.pt"))
    assert result.returncode == 1
    assert "no checkpoint" in result.stderr
    assert "Traceback" not in result.stderr


def test_audit_handles_an_empty_dataset_root(tmp_path):
    result = _run("audit_dataset.py", "--root", str(tmp_path))
    assert result.returncode == 0
    assert "complete designs:     0" in result.stdout


def test_dataset_stats_on_an_empty_root_does_not_crash(tmp_path):
    result = _run("dataset_stats.py", "--root", str(tmp_path))
    assert result.returncode == 0
    assert "KAIROS dataset" in result.stdout
