"""Controlled FreeCAD backend for KAIROS.

All CAD manipulation flows through this package. The RL agent never executes
arbitrary Python. It emits structured actions (see ``kairos.actions``) that
are validated and dispatched onto the typed API exposed here.
"""

from kairos.cad.errors import (
    BackendUnavailableError,
    CADError,
    DocumentError,
    ExportError,
    FeatureError,
    GeometryInvalidError,
    SketchError,
)

__all__ = [
    "CADError",
    "BackendUnavailableError",
    "DocumentError",
    "SketchError",
    "FeatureError",
    "GeometryInvalidError",
    "ExportError",
]
