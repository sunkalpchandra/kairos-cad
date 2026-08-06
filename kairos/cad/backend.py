"""FreeCAD discovery and bootstrap.

FreeCAD is not pip-installable; its Python modules ship inside the
application bundle. This module locates an installation, extends
``sys.path``, and imports ``FreeCAD`` exactly once.

Search order:

1. ``FreeCAD`` already importable (running under FreeCAD's bundled python).
2. ``$KAIROS_FREECAD_LIB`` — explicit path to the directory holding
   ``FreeCAD.so`` / ``FreeCAD.pyd``.
3. Well-known install locations per platform.

CAD-dependent tests and scripts should run under FreeCAD's bundled
interpreter (see the Makefile); mixing an external interpreter with the
bundle's binary modules only works when the Python ABI versions match.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

from kairos.cad.errors import BackendUnavailableError

_CANDIDATE_LIB_DIRS = (
    "/Applications/FreeCAD.app/Contents/Resources/lib",
    "/usr/lib/freecad-python3/lib",
    "/usr/lib/freecad/lib",
    "/usr/local/lib/freecad/lib",
)

_freecad = None
_import_error: str | None = None


def _candidate_dirs() -> list[Path]:
    dirs = []
    env = os.environ.get("KAIROS_FREECAD_LIB")
    if env:
        dirs.append(Path(env))
    dirs.extend(Path(p) for p in _CANDIDATE_LIB_DIRS)
    return [d for d in dirs if d.is_dir()]


def load_freecad():
    """Import and return the ``FreeCAD`` module, extending sys.path if needed.

    Raises:
        BackendUnavailableError: if FreeCAD cannot be imported.
    """
    global _freecad, _import_error
    if _freecad is not None:
        return _freecad

    try:
        _freecad = importlib.import_module("FreeCAD")
        return _freecad
    except ImportError as first_error:
        _import_error = str(first_error)

    for lib_dir in _candidate_dirs():
        if str(lib_dir) not in sys.path:
            sys.path.append(str(lib_dir))
        try:
            _freecad = importlib.import_module("FreeCAD")
            return _freecad
        except ImportError as err:
            _import_error = str(err)

    raise BackendUnavailableError(
        "FreeCAD could not be imported. Install FreeCAD (e.g. "
        "`brew install --cask freecad`) and run under its bundled python, "
        "or set KAIROS_FREECAD_LIB to the directory containing FreeCAD.so. "
        f"Last import error: {_import_error}"
    )


def freecad_available() -> bool:
    """Report whether FreeCAD can be imported in this interpreter."""
    try:
        load_freecad()
        return True
    except BackendUnavailableError:
        return False


def load_module(name: str):
    """Import a FreeCAD companion module (Part, Sketcher, Mesh, ...)."""
    load_freecad()
    try:
        return importlib.import_module(name)
    except ImportError as err:  # pragma: no cover - depends on install
        raise BackendUnavailableError(
            f"FreeCAD module {name!r} could not be imported: {err}"
        ) from err


def freecad_version() -> str:
    """Return the FreeCAD version string, e.g. '1.0.1'."""
    fc = load_freecad()
    parts = fc.Version()
    return ".".join(str(p) for p in parts[:3])
