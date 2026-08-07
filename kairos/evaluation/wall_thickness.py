"""Measure a solid's minimum wall thickness.

`min_wall_thickness` has been the project's standing example of an *unmeasured*
constraint: 283 designs declare one and the checker could only report
"unmeasured", so it earned no reward credit either way. This module measures it.

**Method: inward ray casting.** For each planar face, sample points across the
face, shoot a ray inward along the face normal, and intersect that ray with the
solid. The length of the intersection is the material thickness at that point;
the minimum over all samples is the wall thickness.

Why not a maximal-inscribed-sphere or medial-axis method, which is the textbook
answer: those need a distance field or a medial-axis transform that FreeCAD does
not expose, and both cost far more than the whole rest of an episode. Ray
casting is exact along each ray and its error is purely one of *sampling* — it
can only ever over-estimate, by missing a thin spot between samples. That is the
safe direction for a manufacturing check to be wrong in only if you know it, so
the sample density is explicit and the result is reported with the ray count.

Sampling is on each face's parameter grid rather than uniformly in space, so
small faces (exactly where thin walls appear) are not under-represented.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

#: The single manufacturing tolerance for a thickness check, mm. Near zero
#: because the measurement over-estimates: see kairos/evaluation/constraints.
THICKNESS_TOLERANCE_MM = 1e-6

#: Rays whose intersection is shorter than this are treated as grazing hits at
#: a face boundary rather than real material, in mm.
_MIN_CREDIBLE = 1e-3

#: How far a probe ray travels before giving up, as a multiple of the solid's
#: bounding-box diagonal. 1.1 clears any part while keeping the boolean cheap.
_RAY_REACH = 1.1


@dataclass
class ThicknessMeasurement:
    """The result of a wall-thickness measurement."""

    min_thickness_mm: float | None
    rays_cast: int
    rays_hit: int
    #: Where the thinnest wall was found, in model coordinates.
    location: tuple[float, float, float] | None = None

    @property
    def measured(self) -> bool:
        return self.min_thickness_mm is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_thickness_mm": (
                round(self.min_thickness_mm, 4) if self.min_thickness_mm is not None else None
            ),
            "rays_cast": self.rays_cast,
            "rays_hit": self.rays_hit,
            "location": (
                [round(v, 3) for v in self.location] if self.location is not None else None
            ),
        }


def _face_samples(face, per_axis: int):
    """Points and inward normals sampled across a face's parameter grid."""
    surface = face.ParameterRange  # (uMin, uMax, vMin, vMax)
    u_min, u_max, v_min, v_max = surface
    # Interior samples only: a point exactly on the boundary produces a ray
    # that grazes an adjacent face and reads as near-zero thickness.
    for i in range(per_axis):
        for j in range(per_axis):
            u = u_min + (u_max - u_min) * (i + 0.5) / per_axis
            v = v_min + (v_max - v_min) * (j + 0.5) / per_axis
            try:
                point = face.valueAt(u, v)
                normal = face.normalAt(u, v)
            except Exception:  # pragma: no cover - degenerate parameter spot
                continue
            # No orientation flip here: Face.normalAt() already returns the
            # outward normal for the face's own orientation. Flipping again
            # for Reversed faces aimed those rays out of the solid, and a
            # part whose faces are all Reversed measured nothing at all.
            # (Note `*` returns a new vector; Vector.multiply() mutates.)
            yield point, normal


def measure_min_wall_thickness(
    shape,
    samples_per_face: int = 3,
    max_faces: int = 40,
    planar_only: bool = True,
) -> ThicknessMeasurement:
    """Minimum material thickness of ``shape``, by inward ray casting.

    Args:
        shape: a FreeCAD ``Part.Shape`` with at least one solid.
        samples_per_face: grid resolution per face axis (3 → 9 rays per face).
        max_faces: cap on faces probed, largest first, to bound cost.
        planar_only: probe planar faces only. Curved walls (a spacer's bore)
            are measured from the planar faces that bound them, and probing
            cylinders as well roughly triples cost for these families.

    Returns a :class:`ThicknessMeasurement`; ``min_thickness_mm`` is None when
    no ray produced a credible hit, which the caller must treat as unmeasured
    rather than as a pass.
    """
    import Part

    if shape is None or not getattr(shape, "Faces", None):
        return ThicknessMeasurement(None, 0, 0)

    # PartDesign tips are Compounds, and a boolean against a compound
    # returns nothing — every ray silently missed. Work on the solid.
    solids = list(getattr(shape, "Solids", None) or [])
    if solids:
        shape = solids[0] if len(solids) == 1 else solids[0].multiFuse(solids[1:])

    try:
        diagonal = shape.BoundBox.DiagonalLength
    except Exception:  # pragma: no cover - malformed shape
        return ThicknessMeasurement(None, 0, 0)
    reach = max(diagonal * _RAY_REACH, 1.0)

    faces = [f for f in shape.Faces if not planar_only or _is_planar(f)]
    faces.sort(key=lambda f: -f.Area)
    faces = faces[:max_faces]

    best: float | None = None
    best_at: tuple[float, float, float] | None = None
    cast = hit = 0

    for face in faces:
        for point, normal in _face_samples(face, samples_per_face):
            inward = normal * -1.0
            start = point + inward * _MIN_CREDIBLE
            end = point + inward * reach
            cast += 1
            try:
                ray = Part.makeLine(start, end)
                inside = shape.common(ray)
            except Exception:  # pragma: no cover - boolean can fail on edges
                continue
            length = float(getattr(inside, "Length", 0.0) or 0.0)
            if length <= _MIN_CREDIBLE:
                continue
            # The ray starts _MIN_CREDIBLE inside the face, so the measured
            # segment is short by exactly that much.
            length += _MIN_CREDIBLE
            hit += 1
            if best is None or length < best:
                best = length
                best_at = (point.x, point.y, point.z)

    # A tube's wall is radial and has no planar face to probe, so the
    # coaxial-cylinder gap is folded in: for a spacer it IS the wall.
    radial = cylindrical_wall_thickness(shape)
    if radial is not None and (best is None or radial < best):
        best, best_at = radial, None
    return ThicknessMeasurement(best, cast, hit, best_at)


def _is_planar(face) -> bool:
    try:
        return face.Surface.__class__.__name__ == "Plane"
    except Exception:  # pragma: no cover - exotic surface types
        return False


def thinnest_of(measurements: list[ThicknessMeasurement]) -> float | None:
    """Smallest measured thickness across several measurements."""
    values = [m.min_thickness_mm for m in measurements if m.min_thickness_mm is not None]
    return min(values) if values else None


def is_manufacturable(
    measurement: ThicknessMeasurement, minimum_mm: float, tolerance_mm: float = 1e-6
) -> bool | None:
    """Whether a measured thickness clears a manufacturing minimum.

    Returns None when nothing was measured — the caller must not read that as
    a pass, which is the whole reason `unmeasured` exists as a status.
    """
    if not measurement.measured:
        return None
    return bool(measurement.min_thickness_mm >= minimum_mm - tolerance_mm)


def sampling_error_bound(measurement: ThicknessMeasurement) -> str:
    """Human-readable statement of what the measurement can and cannot say."""
    if not measurement.measured:
        return "no credible ray hit: thickness unmeasured"
    return (
        f"{measurement.min_thickness_mm:.3f} mm from {measurement.rays_hit}/"
        f"{measurement.rays_cast} rays; sampling can only over-estimate "
        "(a thin spot between samples would be missed)"
    )


def cylindrical_wall_thickness(shape, tolerance: float = 1e-6) -> float | None:
    """Thickness of a tube wall, from concentric cylinder radii.

    Ray casting along a cylinder's own normal is unreliable near the axis, and
    a spacer's wall — the gap between its bore and its outer surface — is the
    case that matters. Coaxial cylinder pairs give it exactly.
    """
    cylinders = []
    for face in getattr(shape, "Faces", []) or []:
        try:
            if face.Surface.__class__.__name__ != "Cylinder":
                continue
            axis = face.Surface.Axis
            centre = face.Surface.Center
            cylinders.append((face.Surface.Radius, (axis.x, axis.y, axis.z), (centre.x, centre.y)))
        except Exception:  # pragma: no cover
            continue

    best: float | None = None
    for i, (r1, a1, c1) in enumerate(cylinders):
        for r2, a2, c2 in cylinders[i + 1 :]:
            parallel = abs(sum(x * y for x, y in zip(a1, a2, strict=True))) > 1 - 1e-4
            coaxial = math.dist(c1, c2) < 1e-3
            if parallel and coaxial and abs(r1 - r2) > tolerance:
                gap = abs(r1 - r2)
                best = gap if best is None else min(best, gap)
    return best
