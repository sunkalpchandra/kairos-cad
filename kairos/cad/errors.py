"""Exception hierarchy for the KAIROS CAD backend.

Every failure mode the RL environment must distinguish (invalid action,
broken geometry, backend missing) maps to a typed exception so the action
executor can convert failures into structured penalties instead of crashes.
"""

from __future__ import annotations


class CADError(Exception):
    """Base class for all CAD backend errors."""


class BackendUnavailableError(CADError):
    """FreeCAD could not be located or imported."""


class DocumentError(CADError):
    """Document lifecycle failure (creation, recompute, save)."""


class SketchError(CADError):
    """Sketch geometry or constraint operation failed."""


class FeatureError(CADError):
    """A parametric feature (pad, pocket, fillet, ...) failed to build."""


class GeometryInvalidError(CADError):
    """Resulting geometry failed validation (open shell, null shape, ...)."""


class ExportError(CADError):
    """STEP/STL/FCStd export failed."""


class MeasurementError(CADError):
    """A measurement was requested on missing or unsuitable geometry."""
