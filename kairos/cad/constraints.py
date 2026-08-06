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


# ------------------------------------------------------------ typed wrappers


def add_horizontal(sketch, geo: int) -> int:
    return add_constraint(sketch, "Horizontal", int(geo))


def add_vertical(sketch, geo: int) -> int:
    return add_constraint(sketch, "Vertical", int(geo))


def add_parallel(sketch, geo1: int, geo2: int) -> int:
    return add_constraint(sketch, "Parallel", int(geo1), int(geo2))


def add_perpendicular(sketch, geo1: int, geo2: int) -> int:
    return add_constraint(sketch, "Perpendicular", int(geo1), int(geo2))


def add_tangent(sketch, geo1: int, geo2: int) -> int:
    return add_constraint(sketch, "Tangent", int(geo1), int(geo2))


def add_equal(sketch, geo1: int, geo2: int) -> int:
    return add_constraint(sketch, "Equal", int(geo1), int(geo2))


def add_coincident(sketch, geo1: int, pos1: int, geo2: int, pos2: int) -> int:
    return add_constraint(sketch, "Coincident", int(geo1), int(pos1), int(geo2), int(pos2))


def add_distance(sketch, geo1: int, pos1: int, geo2: int, pos2: int, value: float) -> int:
    if value <= 0:
        raise SketchError(f"distance must be positive, got {value}")
    return add_constraint(
        sketch, "Distance", int(geo1), int(pos1), int(geo2), int(pos2), float(value)
    )


def add_length(sketch, geo: int, value: float) -> int:
    """Constrain the length of a line segment."""
    if value <= 0:
        raise SketchError(f"length must be positive, got {value}")
    return add_constraint(sketch, "Distance", int(geo), float(value))


def add_radius(sketch, geo: int, value: float) -> int:
    if value <= 0:
        raise SketchError(f"radius must be positive, got {value}")
    return add_constraint(sketch, "Radius", int(geo), float(value))


def add_diameter(sketch, geo: int, value: float) -> int:
    if value <= 0:
        raise SketchError(f"diameter must be positive, got {value}")
    return add_constraint(sketch, "Diameter", int(geo), float(value))


def add_symmetric(sketch, geo1: int, pos1: int, geo2: int, pos2: int, axis_geo: int) -> int:
    return add_constraint(
        sketch, "Symmetric", int(geo1), int(pos1), int(geo2), int(pos2), int(axis_geo)
    )


# ---------------------------------------------------------------- inspection


def constraint_count(sketch) -> int:
    return len(sketch.Constraints)


def is_fully_constrained(sketch) -> bool | None:
    """True/False when FreeCAD exposes it; None when undeterminable."""
    value = getattr(sketch, "FullyConstrained", None)
    return bool(value) if value is not None else None


def degrees_of_freedom(sketch) -> int | None:
    """Remaining sketch DoF when the solver exposes it, else None."""
    try:
        sketch.solve()
        return int(sketch.getDoF())
    except Exception:
        return None
