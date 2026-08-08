"""Typed sketch constraint API.

Maps KAIROS constraint operations onto ``Sketcher.Constraint``. Constraint
targets are sketch geometry indices as returned by ``kairos.cad.sketches``;
point positions use FreeCAD conventions (1 = start, 2 = end, 3 = center).
"""

from __future__ import annotations

from kairos.cad.backend import load_module
from kairos.cad.errors import SketchError

#: Constraint kinds supported in Phase 1, with their argument signatures.
#: geo = geometry index, pos = point position on that geometry, value = mm/deg.
CONSTRAINT_SIGNATURES: dict[str, str] = {
    "Horizontal": "(geo)",
    "Vertical": "(geo)",
    "Parallel": "(geo1, geo2)",
    "Perpendicular": "(geo1, geo2)",
    "Tangent": "(geo1, geo2)",
    "Equal": "(geo1, geo2)",
    "Coincident": "(geo1, pos1, geo2, pos2)",
    "Distance": "(geo1, pos1, geo2, pos2, value) | (geo, value)",
    "DistanceX": "(geo, pos, value)",
    "DistanceY": "(geo, pos, value)",
    "Radius": "(geo, value)",
    "Diameter": "(geo, value)",
    "Symmetric": "(geo1, pos1, geo2, pos2, geo_axis)",
}


def _constraint(kind: str, *args):
    sketcher = load_module("Sketcher")
    try:
        return sketcher.Constraint(kind, *args)
    except Exception as err:
        signature = CONSTRAINT_SIGNATURES.get(kind, "?")
        raise SketchError(
            f"could not build {kind} constraint with args {args} "
            f"(expected {signature}): {err}"
        ) from err


def add_constraint(sketch, kind: str, *args) -> int:
    """Add a constraint by kind name; returns the constraint index."""
    if kind not in CONSTRAINT_SIGNATURES:
        raise SketchError(
            f"unsupported constraint kind {kind!r}; "
            f"supported: {sorted(CONSTRAINT_SIGNATURES)}"
        )
    constraint = _constraint(kind, *args)
    try:
        return sketch.addConstraint(constraint)
    except Exception as err:
        raise SketchError(f"sketch rejected {kind} constraint {args}: {err}") from err
