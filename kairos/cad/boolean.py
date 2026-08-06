"""Boolean operations between solids at the Part-workbench level.

PartDesign features cover single-body modeling; these helpers combine shapes
across bodies (multi-part designs, imported geometry). They operate on
``Part.Shape`` values and return new shapes, leaving inputs untouched.
"""

from __future__ import annotations

from kairos.cad.errors import GeometryInvalidError


def _check_inputs(a, b, op: str) -> None:
    for label, shape in (("first", a), ("second", b)):
        if shape is None or shape.isNull():
            raise GeometryInvalidError(f"{op}: {label} shape is null")
        if not shape.Solids:
            raise GeometryInvalidError(f"{op}: {label} shape contains no solid")


def _check_result(shape, op: str):
    if shape is None or shape.isNull():
        raise GeometryInvalidError(f"{op} produced a null shape")
    if not shape.isValid():
        raise GeometryInvalidError(f"{op} produced an invalid shape")
    return shape


def union(shape_a, shape_b):
    """Fuse two solids into one shape."""
    _check_inputs(shape_a, shape_b, "union")
    try:
        result = shape_a.fuse(shape_b)
        result = result.removeSplitter()
    except Exception as err:
        raise GeometryInvalidError(f"union failed: {err}") from err
    return _check_result(result, "union")


def cut(shape_a, shape_b):
    """Subtract ``shape_b`` from ``shape_a``."""
    _check_inputs(shape_a, shape_b, "cut")
    try:
        result = shape_a.cut(shape_b)
        result = result.removeSplitter()
    except Exception as err:
        raise GeometryInvalidError(f"cut failed: {err}") from err
    return _check_result(result, "cut")


def intersection(shape_a, shape_b):
    """Intersect two solids; raises if the intersection is empty."""
    _check_inputs(shape_a, shape_b, "intersection")
    try:
        result = shape_a.common(shape_b)
    except Exception as err:
        raise GeometryInvalidError(f"intersection failed: {err}") from err
    if result.isNull() or not result.Solids:
        raise GeometryInvalidError("intersection is empty")
    return _check_result(result, "intersection")
