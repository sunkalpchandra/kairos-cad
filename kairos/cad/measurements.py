"""Geometric measurements over solids.

These readouts feed the numerical engineering state, constraint checking,
and the reward function: volume, mass, areas, bounding boxes, topology
census, edge/face inventories, and cylindrical-hole detection.

All lengths are mm, areas mm^2, volumes mm^3, masses grams.
"""

from __future__ import annotations

import math
from typing import Any

from kairos.cad.errors import MeasurementError

#: Densities in g/mm^3.
DENSITIES = {
    "aluminum": 2.70e-3,
    "steel": 7.85e-3,
    "abs": 1.05e-3,
    "pla": 1.24e-3,
    "titanium": 4.51e-3,
}


def _require_shape(shape):
    if shape is None or shape.isNull():
        raise MeasurementError("no geometry to measure")
    return shape


def volume(shape) -> float:
    return float(_require_shape(shape).Volume)


def surface_area(shape) -> float:
    return float(_require_shape(shape).Area)


def mass(shape, material: str = "aluminum") -> float:
    density = DENSITIES.get(material.lower())
    if density is None:
        raise MeasurementError(
            f"unknown material {material!r}; known: {sorted(DENSITIES)}"
        )
    return volume(shape) * density


def bounding_box(shape) -> dict[str, float]:
    bb = _require_shape(shape).BoundBox
    return {
        "x_min": bb.XMin,
        "y_min": bb.YMin,
        "z_min": bb.ZMin,
        "x_max": bb.XMax,
        "y_max": bb.YMax,
        "z_max": bb.ZMax,
        "x_len": bb.XLength,
        "y_len": bb.YLength,
        "z_len": bb.ZLength,
        "diagonal": bb.DiagonalLength,
    }


def center_of_mass(shape) -> tuple[float, float, float]:
    s = _require_shape(shape)
    # Compounds have no CenterOfMass; aggregate volume-weighted over solids.
    if hasattr(s, "CenterOfMass"):
        com = s.CenterOfMass
        return (com.x, com.y, com.z)
    solids = s.Solids
    if not solids:
        raise MeasurementError("shape has no solids for center of mass")
    total = sum(solid.Volume for solid in solids)
    if total <= 0:
        raise MeasurementError("shape has no volume for center of mass")
    weighted = [
        sum(solid.CenterOfMass[i] * solid.Volume for solid in solids) / total
        for i in range(3)
    ]
    return (weighted[0], weighted[1], weighted[2])


def topology_counts(shape) -> dict[str, int]:
    s = _require_shape(shape)
    return {
        "solids": len(s.Solids),
        "shells": len(s.Shells),
        "faces": len(s.Faces),
        "edges": len(s.Edges),
        "vertices": len(s.Vertexes),
    }


def distance(shape_a, shape_b) -> float:
    """Minimum distance between two shapes/subshapes."""
    try:
        dist, _, _ = _require_shape(shape_a).distToShape(_require_shape(shape_b))
        return float(dist)
    except Exception as err:
        raise MeasurementError(f"distance computation failed: {err}") from err


# ------------------------------------------------------------------ edges


def _edge_direction(edge):
    """Unit direction for line edges, axis for circular edges, else None."""
    curve = getattr(edge, "Curve", None)
    if curve is None:
        return None
    type_name = type(curve).__name__
    if type_name == "Line":
        d = curve.Direction
        return (d.x, d.y, d.z)
    if type_name == "Circle":
        a = curve.Axis
        return (a.x, a.y, a.z)
    return None


def list_edges(shape) -> list[dict[str, Any]]:
    """Inventory of edges as dicts with FreeCAD subelement names 'EdgeN'."""
    s = _require_shape(shape)
    inventory = []
    for i, edge in enumerate(s.Edges, start=1):
        curve = getattr(edge, "Curve", None)
        mid = edge.valueAt(0.5 * (edge.FirstParameter + edge.LastParameter))
        inventory.append(
            {
                "name": f"Edge{i}",
                "curve": type(curve).__name__ if curve else "Unknown",
                "length": float(edge.Length),
                "midpoint": (mid.x, mid.y, mid.z),
                "direction": _edge_direction(edge),
            }
        )
    return inventory


def find_edges(
    shape,
    curve: str | None = None,
    direction: tuple[float, float, float] | None = None,
    near: tuple[float, float, float] | None = None,
    tol: float = 1e-4,
    near_tol: float = 1.0,
) -> list[str]:
    """Find edge names matching a curve type, direction (up to sign), and/or
    a point the edge midpoint must lie near (within ``near_tol`` mm)."""
    matches = []
    for entry in list_edges(shape):
        if curve is not None and entry["curve"] != curve:
            continue
        if direction is not None:
            d = entry["direction"]
            if d is None:
                continue
            norm = math.sqrt(sum(c * c for c in direction))
            if norm == 0:
                raise MeasurementError("zero direction vector")
            want = tuple(c / norm for c in direction)
            dot = abs(sum(a * b for a, b in zip(d, want, strict=True)))
            if abs(dot - 1.0) > tol:
                continue
        if near is not None:
            mx, my, mz = entry["midpoint"]
            dist2 = (mx - near[0]) ** 2 + (my - near[1]) ** 2 + (mz - near[2]) ** 2
            if dist2 > near_tol**2:
                continue
        matches.append(entry["name"])
    return matches


# ------------------------------------------------------------------ faces


def _cylinder_is_concave(face, surface) -> bool:
    """True when the cylindrical face's material lies outside the axis
    (a hole/bore), False for convex surfaces (bosses, fillets)."""
    u0, u1, v0, v1 = face.ParameterRange
    point = face.valueAt((u0 + u1) / 2.0, (v0 + v1) / 2.0)
    normal = face.normalAt((u0 + u1) / 2.0, (v0 + v1) / 2.0)
    axis = surface.Axis
    center = surface.Center
    to_point = point - center
    along = to_point * axis / axis.Length**2
    radial = to_point - axis * along
    return (normal * radial) < 0


def list_faces(shape) -> list[dict[str, Any]]:
    """Inventory of faces as dicts with FreeCAD subelement names 'FaceN'."""
    s = _require_shape(shape)
    inventory = []
    for i, face in enumerate(s.Faces, start=1):
        surface = getattr(face, "Surface", None)
        entry: dict[str, Any] = {
            "name": f"Face{i}",
            "surface": type(surface).__name__ if surface else "Unknown",
            "area": float(face.Area),
        }
        if entry["surface"] == "Cylinder":
            entry["radius"] = float(surface.Radius)
            axis = surface.Axis
            center = surface.Center
            entry["axis"] = (axis.x, axis.y, axis.z)
            entry["axis_point"] = (center.x, center.y, center.z)
            entry["concave"] = _cylinder_is_concave(face, surface)
            u0, u1, _, _ = face.ParameterRange
            entry["angular_extent"] = float(abs(u1 - u0))
        elif entry["surface"] == "Plane":
            normal = surface.Axis
            entry["normal"] = (normal.x, normal.y, normal.z)
        inventory.append(entry)
    return inventory


def find_cylindrical_holes(
    shape, diameter: float | None = None, tol: float = 0.05
) -> list[dict[str, Any]]:
    """Detect cylindrical holes by grouping concave cylindrical faces on a
    shared axis line, keeping only groups that wrap (nearly) the full 360
    degrees — this excludes concave corner fillets (coves), which share the
    concavity but only sweep ~90 degrees. Optionally filter by diameter (mm).

    Returns one entry per distinct hole: ``{"diameter", "axis", "axis_point",
    "faces"}``. This is the readout used to check requirements like
    "4 x M5 holes".
    """
    groups: list[dict[str, Any]] = []
    for entry in list_faces(shape):
        if entry["surface"] != "Cylinder" or not entry.get("concave"):
            continue
        d = 2.0 * entry["radius"]
        if diameter is not None and abs(d - diameter) > tol:
            continue
        axis = entry["axis"]
        point = entry["axis_point"]
        norm = math.sqrt(sum(c * c for c in axis))
        axis = tuple(c / norm for c in axis)
        # Project the axis point onto the plane orthogonal to the axis so
        # faces on the same infinite axis line group together.
        along = sum(p * a for p, a in zip(point, axis, strict=True))
        foot = tuple(p - along * a for p, a in zip(point, axis, strict=True))
        for group in groups:
            same_dir = abs(sum(a * b for a, b in zip(axis, group["axis"], strict=True))) > 1 - 1e-4
            close = (
                sum((f - g) ** 2 for f, g in zip(foot, group["_foot"], strict=True)) < tol**2
                and abs(d - group["diameter"]) <= tol
            )
            if same_dir and close:
                group["faces"].append(entry["name"])
                group["_angle"] += entry["angular_extent"]
                break
        else:
            groups.append(
                {
                    "diameter": d,
                    "axis": axis,
                    "axis_point": point,
                    "_foot": foot,
                    "_angle": entry["angular_extent"],
                    "faces": [entry["name"]],
                }
            )
    holes = []
    for group in groups:
        group.pop("_foot")
        angle = group.pop("_angle")
        if angle >= 1.75 * math.pi:  # full bore, not a partial cove
            holes.append(group)
    return holes


def summary(shape, material: str = "aluminum") -> dict[str, Any]:
    """One-call numerical engineering state for observations and rewards."""
    s = _require_shape(shape)
    return {
        "volume_mm3": volume(s),
        "surface_area_mm2": surface_area(s),
        "mass_g": mass(s, material),
        "material": material,
        "bounding_box": bounding_box(s),
        "center_of_mass": center_of_mass(s),
        "topology": topology_counts(s),
        "hole_count": len(find_cylindrical_holes(s)),
    }
