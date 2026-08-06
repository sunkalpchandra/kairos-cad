"""Sketch creation and 2D geometry API.

Sketches are attached to body origin planes (XY / XZ / YZ) with an optional
normal offset. Geometry functions return the indices of created sketch
geometry so callers (and recorded trajectories) can reference them for
constraints and edits.
"""

from __future__ import annotations

import math

from kairos.cad.backend import load_freecad, load_module
from kairos.cad.document import CADDocument
from kairos.cad.errors import SketchError


def create_sketch(
    cad_doc: CADDocument,
    plane: str = "XY",
    offset: float = 0.0,
    name: str | None = None,
):
    """Create a sketch attached to a body origin plane.

    Args:
        cad_doc: managed document.
        plane: 'XY', 'XZ', or 'YZ'.
        offset: displacement along the plane normal, in mm.
        name: optional object name; FreeCAD may uniquify it.

    Returns:
        The Sketcher::SketchObject.
    """
    app = load_freecad()
    plane_obj = cad_doc.origin_plane(plane)
    sketch = cad_doc.body.newObject("Sketcher::SketchObject", name or "Sketch")
    try:
        # FreeCAD >= 1.0 renamed Support -> AttachmentSupport.
        if hasattr(sketch, "AttachmentSupport"):
            sketch.AttachmentSupport = [(plane_obj, "")]
        else:
            sketch.Support = [(plane_obj, "")]
        sketch.MapMode = "FlatFace"
        if offset:
            sketch.AttachmentOffset = app.Placement(
                app.Vector(0, 0, float(offset)), app.Rotation()
            )
    except Exception as err:
        cad_doc.doc.removeObject(sketch.Name)
        raise SketchError(f"could not attach sketch to plane {plane}: {err}") from err
    errors = cad_doc.recompute()
    if errors:
        cad_doc.doc.removeObject(sketch.Name)
        raise SketchError(f"sketch attachment failed: {errors}")
    return sketch


def add_line(sketch, x1: float, y1: float, x2: float, y2: float) -> int:
    """Add a line segment; returns the geometry index."""
    app = load_freecad()
    part = load_module("Part")
    if (x1, y1) == (x2, y2):
        raise SketchError("degenerate line: endpoints coincide")
    try:
        return sketch.addGeometry(
            part.LineSegment(app.Vector(x1, y1, 0), app.Vector(x2, y2, 0)), False
        )
    except Exception as err:
        raise SketchError(f"could not add line: {err}") from err


def add_circle(sketch, cx: float, cy: float, radius: float) -> int:
    """Add a full circle; returns the geometry index."""
    app = load_freecad()
    part = load_module("Part")
    if radius <= 0:
        raise SketchError(f"circle radius must be positive, got {radius}")
    try:
        return sketch.addGeometry(
            part.Circle(app.Vector(cx, cy, 0), app.Vector(0, 0, 1), float(radius)),
            False,
        )
    except Exception as err:
        raise SketchError(f"could not add circle: {err}") from err


def add_arc(
    sketch,
    cx: float,
    cy: float,
    radius: float,
    start_angle_deg: float,
    end_angle_deg: float,
) -> int:
    """Add a circular arc (angles in degrees, CCW from +x); returns the index."""
    app = load_freecad()
    part = load_module("Part")
    if radius <= 0:
        raise SketchError(f"arc radius must be positive, got {radius}")
    if math.isclose(start_angle_deg % 360.0, end_angle_deg % 360.0):
        raise SketchError("degenerate arc: start and end angles coincide")
    circle = part.Circle(app.Vector(cx, cy, 0), app.Vector(0, 0, 1), float(radius))
    try:
        return sketch.addGeometry(
            part.ArcOfCircle(
                circle, math.radians(start_angle_deg), math.radians(end_angle_deg)
            ),
            False,
        )
    except Exception as err:
        raise SketchError(f"could not add arc: {err}") from err


def add_rectangle(sketch, x: float, y: float, width: float, height: float) -> list[int]:
    """Add an axis-aligned rectangle with (x, y) as its lower-left corner.

    Creates four lines closed with coincident constraints plus
    horizontal/vertical constraints, matching interactive CAD practice.

    Returns:
        The four line geometry indices [bottom, right, top, left].
    """
    sketcher = load_module("Sketcher")
    if width <= 0 or height <= 0:
        raise SketchError(f"rectangle sides must be positive, got {width}x{height}")
    x2, y2 = x + width, y + height
    bottom = add_line(sketch, x, y, x2, y)
    right = add_line(sketch, x2, y, x2, y2)
    top = add_line(sketch, x2, y2, x, y2)
    left = add_line(sketch, x, y2, x, y)
    try:
        # Endpoint indices: 1 = start, 2 = end.
        sketch.addConstraint(sketcher.Constraint("Coincident", bottom, 2, right, 1))
        sketch.addConstraint(sketcher.Constraint("Coincident", right, 2, top, 1))
        sketch.addConstraint(sketcher.Constraint("Coincident", top, 2, left, 1))
        sketch.addConstraint(sketcher.Constraint("Coincident", left, 2, bottom, 1))
        sketch.addConstraint(sketcher.Constraint("Horizontal", bottom))
        sketch.addConstraint(sketcher.Constraint("Horizontal", top))
        sketch.addConstraint(sketcher.Constraint("Vertical", right))
        sketch.addConstraint(sketcher.Constraint("Vertical", left))
    except Exception as err:
        raise SketchError(f"could not close rectangle with constraints: {err}") from err
    return [bottom, right, top, left]


def add_polygon(sketch, points: list[tuple[float, float]], closed: bool = True) -> list[int]:
    """Add a polyline through ``points``; closes back to the first if ``closed``.

    Consecutive endpoints are joined with coincident constraints.

    Returns:
        Line geometry indices in order.
    """
    sketcher = load_module("Sketcher")
    if len(points) < (3 if closed else 2):
        raise SketchError(f"polygon needs at least {3 if closed else 2} points")
    segments = list(zip(points, points[1:] + ([points[0]] if closed else [])))
    indices = [add_line(sketch, x1, y1, x2, y2) for (x1, y1), (x2, y2) in segments]
    try:
        for a, b in zip(indices, indices[1:]):
            sketch.addConstraint(sketcher.Constraint("Coincident", a, 2, b, 1))
        if closed:
            sketch.addConstraint(
                sketcher.Constraint("Coincident", indices[-1], 2, indices[0], 1)
            )
    except Exception as err:
        raise SketchError(f"could not join polygon segments: {err}") from err
    return indices


def delete_geometry(sketch, index: int) -> None:
    """Delete sketch geometry by index (associated constraints go with it)."""
    try:
        sketch.delGeometry(int(index))
    except Exception as err:
        raise SketchError(f"could not delete geometry {index}: {err}") from err


def move_geometry(sketch, index: int, dx: float, dy: float) -> None:
    """Translate a geometry element by (dx, dy) in sketch coordinates."""
    app = load_freecad()
    try:
        geo = sketch.Geometry[int(index)]
        geo.translate(app.Vector(float(dx), float(dy), 0))
        sketch.Geometry = sketch.Geometry[: int(index)] + [geo] + sketch.Geometry[int(index) + 1 :]
    except Exception as err:
        raise SketchError(f"could not move geometry {index}: {err}") from err


def geometry_count(sketch) -> int:
    return len(sketch.Geometry)
