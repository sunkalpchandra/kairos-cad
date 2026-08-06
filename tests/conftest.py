"""Shared test configuration.

Tests marked ``cad`` need a live FreeCAD; they are skipped automatically when
the backend is unavailable so `pytest -m "not cad"` and full runs both work
from any interpreter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _freecad_available() -> bool:
    try:
        from kairos.cad.backend import freecad_available

        return freecad_available()
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    if _freecad_available():
        return
    skip = pytest.mark.skip(reason="FreeCAD backend not available in this interpreter")
    for item in items:
        if "cad" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def engine():
    """A fresh CADEngine, closed after the test."""
    from kairos.cad.engine import CADEngine

    eng = CADEngine("pytest")
    yield eng
    eng.close()
