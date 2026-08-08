"""Constraint checking: EngineeringSpec vs. an observation snapshot.

Consumes the plain-dict observation from ``kairos.representation.observe``, never live
FreeCAD objects, so it is pure logic, unit-testable anywhere and
replayable over recorded trajectories.

Every constraint resolves to one of three statuses:

- ``satisfied`` / ``violated``: the checker measured it against geometry.
- ``unmeasured``: KAIROS cannot measure this kind yet (e.g. minimum wall
  thickness before Phase 6). Unmeasured constraints are *excluded* from the
  satisfaction rate rather than silently counted as satisfied, rewards and
  benchmark numbers must never claim credit for unchecked requirements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from kairos.language.spec import Constraint, EngineeringSpec

#: Default tolerance for diameter comparisons, mm.
_DIAMETER_TOL = 0.1
#: Default tolerance for dimension comparisons, mm.
_DIM_TOL = 1.0
#: Angle tolerance, degrees.
_ANGLE_TOL = 1.0
#: A part that bends through an angle leaves its bounding box mostly empty.
#: An L-bracket fills roughly half; a plate or block fills all of it.
_MAX_BENT_FILL = 0.95
#: Wall-thickness slack, mm. Deliberately ~0: ray sampling can only
#: OVER-estimate thickness (it misses thin spots between samples), so the
#: measurement is an upper bound on the true wall. A measured value below
#: the floor therefore means the real wall is below it too, a definitive
#: failure, not a near-miss. Slack here would pass parts that are provably
#: too thin; it was letting 6.983 mm clear a 7.0 mm floor.
_THICKNESS_TOL = 1e-6  # mirrors wall_thickness.THICKNESS_TOLERANCE_MM


@dataclass
class ConstraintResult:
    constraint: Constraint
    status: str  # 'satisfied' | 'violated' | 'unmeasured'
    measured: Any = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraint": self.constraint.to_dict(),
            "status": self.status,
            "measured": self.measured,
            "detail": self.detail,
        }


@dataclass
class ConstraintReport:
    results: list[ConstraintResult] = field(default_factory=list)

    @property
    def satisfied(self) -> list[ConstraintResult]:
        return [r for r in self.results if r.status == "satisfied"]

    @property
    def violated(self) -> list[ConstraintResult]:
        return [r for r in self.results if r.status == "violated"]

    @property
    def unmeasured(self) -> list[ConstraintResult]:
        return [r for r in self.results if r.status == "unmeasured"]

    @property
    def all_measured_satisfied(self) -> bool:
        """No violations, and at least one constraint actually verified.

        A spec with zero constraints is trivially satisfied; a spec whose
        constraints are all unmeasured is NOT (no credit for unchecked
        requirements).
        """
        if self.violated:
            return False
        if not self.results:
            return True
        return bool(self.satisfied)

    @property
    def satisfaction_rate(self) -> float:
        """Fraction of *measured* constraints satisfied (1.0 if none measured)."""
        measured = len(self.satisfied) + len(self.violated)
        if measured == 0:
            return 1.0
        return len(self.satisfied) / measured

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "satisfaction_rate": self.satisfaction_rate,
            "all_measured_satisfied": self.all_measured_satisfied,
            "counts": {
                "satisfied": len(self.satisfied),
                "violated": len(self.violated),
                "unmeasured": len(self.unmeasured),
            },
        }


# --------------------------------------------------------------- checkers


def _holes_matching(observation: dict, diameter: float | None, tol: float) -> list[dict]:
    holes = observation.get("holes", [])
    if diameter is None:
        return list(holes)
    return [h for h in holes if abs(h["diameter"] - diameter) <= tol]


def _check_hole_count(c: Constraint, observation: dict, spec: EngineeringSpec) -> ConstraintResult:
    """Check the part's total hole count against the requirement.

    Counts **every** hole, not only those matching the nominal diameter. A
    requirement's stated total spans groups of different sizes, a flange's
    "12 mm central bore, and 6 bolt holes of 5 mm diameter" is seven holes. So filtering
    by the nominal diameter would find six and report a correct
    part as violated. Whether the named diameter is actually present is
    :func:`_check_hole_diameter`'s job.
    """
    found = len(observation.get("holes", []))
    ok = found == int(c.value)
    return ConstraintResult(
        c,
        "satisfied" if ok else "violated",
        measured=found,
        detail=f"{found} holes, need {c.value}",
    )


def _check_hole_diameter(c: Constraint, observation: dict, spec: EngineeringSpec) -> ConstraintResult:
    """Check that the holes the requirement names exist at the stated diameter.

    A part legitimately carries holes at other diameters, a flange's central
    bore sits alongside its bolt holes. So extra sizes are not violations, and
    the requirement's total cannot be used as the number that must match: it
    spans those groups. This check therefore asks only that the named diameter
    is present; ``hole_count`` pins the total independently.
    """
    tol = c.tolerance if c.tolerance is not None else _DIAMETER_TOL
    matching = _holes_matching(observation, float(c.value), tol)
    holes = observation.get("holes", [])
    ok = len(matching) >= 1
    detail = f"{len(matching)}/{len(holes)} holes at d={c.value}±{tol}mm"
    return ConstraintResult(
        c,
        "satisfied" if ok else "violated",
        measured=sorted(round(h["diameter"], 3) for h in holes),
        detail=detail,
    )


def _check_min_wall_thickness(
    c: Constraint, observation: dict, spec: EngineeringSpec
) -> ConstraintResult:
    """Check the thinnest wall against the requirement's manufacturing floor.

    The measurement is expensive (ray casting against the solid), so it is not
    part of every observation: whoever built the observation puts it in
    ``summary["min_wall_thickness_mm"]``. Absent, the constraint stays
    ``unmeasured``, never satisfied by default.
    """
    measured = (observation.get("summary") or {}).get("min_wall_thickness_mm")
    if measured is None:
        return ConstraintResult(
            c, "unmeasured", detail="no wall-thickness measurement in this observation"
        )
    required = float(c.value)
    # measured is an UPPER bound on the true thickness, so failing it is
    # conclusive while passing it is only necessary, not sufficient.
    ok = float(measured) >= required - _THICKNESS_TOL
    return ConstraintResult(
        c,
        "satisfied" if ok else "violated",
        measured=round(float(measured), 3),
        detail=f"thinnest wall {float(measured):.3f} mm, need >= {required} mm",
    )


def _bbox_lens(observation: dict) -> list[float] | None:
    bbox = observation.get("summary", {}).get("bounding_box")
    if not bbox:
        return None
    return sorted([bbox["x_len"], bbox["y_len"], bbox["z_len"]])


def _check_bbox_exact(c: Constraint, observation: dict, spec: EngineeringSpec) -> ConstraintResult:
    lens = _bbox_lens(observation)
    if lens is None:
        return ConstraintResult(c, "violated", detail="no solid to measure")
    tol = c.tolerance if c.tolerance is not None else _DIM_TOL
    want = sorted(float(v) for v in c.value)
    ok = all(abs(a - b) <= tol for a, b in zip(lens, want, strict=True))
    return ConstraintResult(
        c,
        "satisfied" if ok else "violated",
        measured=[round(v, 2) for v in lens],
        detail=f"bbox {lens} vs {want} ±{tol} (orientation-invariant)",
    )


def _check_bbox_max(c: Constraint, observation: dict, spec: EngineeringSpec) -> ConstraintResult:
    lens = _bbox_lens(observation)
    if lens is None:
        return ConstraintResult(c, "violated", detail="no solid to measure")
    want = sorted(float(v) for v in c.value)
    ok = all(a <= b + 1e-6 for a, b in zip(lens, want, strict=True))
    return ConstraintResult(
        c,
        "satisfied" if ok else "violated",
        measured=[round(v, 2) for v in lens],
        detail=f"bbox {lens} within {want} (orientation-invariant)",
    )


def _check_mounting_angle(c: Constraint, observation: dict, spec: EngineeringSpec) -> ConstraintResult:
    """Check that two large planar faces meet at the required angle.

    Uses the six largest planar faces; the constraint is satisfied when any
    pair's normals subtend the target angle (or its supplement, a 90° wall
    pair reads as 90° or 270° depending on orientation).
    """
    faces = [f for f in observation.get("faces", []) if f.get("surface") == "Plane"]
    if len(faces) < 2:
        return ConstraintResult(c, "violated", detail="fewer than two planar faces")

    # Every prismatic solid has face pairs at 90 degrees, so a face-angle
    # test alone is satisfied by a plain plate. A part that genuinely bends
    # through the angle does not fill its bounding box; a box does.
    summary = observation.get("summary", {})
    bbox = summary.get("bounding_box") or {}
    volume = float(summary.get("volume_mm3") or 0.0)
    envelope = (
        float(bbox.get("x_len", 0.0))
        * float(bbox.get("y_len", 0.0))
        * float(bbox.get("z_len", 0.0))
    )
    fill = volume / envelope if envelope > 0 else 1.0
    if float(c.value) % 180.0 != 0.0 and fill > _MAX_BENT_FILL:
        return ConstraintResult(
            c,
            "violated",
            measured=round(fill, 3),
            detail=f"solid fills {fill:.0%} of its bounding box: not a bent part",
        )
    faces = sorted(faces, key=lambda f: -f["area"])[:6]
    target = float(c.value)
    tol = c.tolerance if c.tolerance is not None else _ANGLE_TOL
    best = None
    for i, fa in enumerate(faces):
        for fb in faces[i + 1 :]:
            na, nb = fa.get("normal"), fb.get("normal")
            if not na or not nb:
                continue
            dot = sum(a * b for a, b in zip(na, nb, strict=True))
            dot = max(-1.0, min(1.0, dot))
            angle = math.degrees(math.acos(abs(dot)))  # fold to [0, 90]
            folded_target = min(target % 180.0, 180.0 - (target % 180.0))
            err = abs(angle - folded_target)
            if best is None or err < best[0]:
                best = (err, angle)
            if err <= tol:
                return ConstraintResult(
                    c, "satisfied", measured=round(angle, 2),
                    detail=f"planar faces at {angle:.1f}° (target {target}°)",
                )
    return ConstraintResult(
        c,
        "violated",
        measured=round(best[1], 2) if best else None,
        detail=f"no large planar face pair at {target}° (closest {best[1]:.1f}°)" if best else "no measurable pair",
    )


def _check_cylindrical_interface(c: Constraint, observation: dict, spec: EngineeringSpec) -> ConstraintResult:
    """Count convex (external) cylindrical surfaces of the required diameter,
    grouping coaxial faces the same way hole detection does."""
    want = c.value or {}
    diameter = float(want.get("diameter", 0))
    count = int(want.get("count", 0))
    tol = c.tolerance if c.tolerance is not None else _DIAMETER_TOL
    groups: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = []
    for face in observation.get("faces", []):
        if face.get("surface") != "Cylinder" or face.get("concave"):
            continue
        if abs(2.0 * face["radius"] - diameter) > tol:
            continue
        axis = face["axis"]
        norm = math.sqrt(sum(a * a for a in axis)) or 1.0
        axis = tuple(a / norm for a in axis)
        point = face["axis_point"]
        along = sum(p * a for p, a in zip(point, axis, strict=True))
        foot = tuple(p - along * a for p, a in zip(point, axis, strict=True))
        for gaxis, gfoot in groups:
            same_dir = abs(sum(a * b for a, b in zip(axis, gaxis, strict=True))) > 1 - 1e-4
            close = sum((f - g) ** 2 for f, g in zip(foot, gfoot, strict=True)) < tol**2
            if same_dir and close:
                break
        else:
            groups.append((axis, foot))
    ok = len(groups) >= count
    return ConstraintResult(
        c,
        "satisfied" if ok else "violated",
        measured=len(groups),
        detail=f"{len(groups)} external cylindrical interfaces at d={diameter}±{tol}mm, need {count}",
    )


def _check_mass_reduction(c: Constraint, observation: dict, spec: EngineeringSpec,
                          context: dict | None) -> ConstraintResult:
    initial = (context or {}).get("initial_mass_g")
    if initial is None or initial <= 0:
        return ConstraintResult(c, "unmeasured", detail="no baseline mass in context")
    current = observation.get("summary", {}).get("mass_g")
    if current is None:
        return ConstraintResult(c, "violated", detail="no solid to weigh")
    reduction = 100.0 * (initial - current) / initial
    ok = reduction >= float(c.value)
    return ConstraintResult(
        c,
        "satisfied" if ok else "violated",
        measured=round(reduction, 2),
        detail=f"mass {initial:.1f}g → {current:.1f}g ({reduction:.1f}%, need >={c.value}%)",
    )


def _check_hole_positions_preserved(c: Constraint, observation: dict, spec: EngineeringSpec,
                                    context: dict | None) -> ConstraintResult:
    tol = c.tolerance if c.tolerance is not None else 0.5
    required = c.value or (context or {}).get("required_hole_positions")
    if not required:
        return ConstraintResult(c, "unmeasured", detail="no reference hole positions")
    holes = observation.get("holes", [])
    missing = []
    for pos in required:
        hit = any(
            sum((a - b) ** 2 for a, b in zip(h["axis_point"], pos, strict=True)) <= tol**2
            for h in holes
        )
        if not hit:
            missing.append([round(v, 2) for v in pos])
    return ConstraintResult(
        c,
        "satisfied" if not missing else "violated",
        measured=len(required) - len(missing),
        detail=f"{len(required) - len(missing)}/{len(required)} positions preserved"
        + (f"; missing {missing}" if missing else ""),
    )


def check_constraints(
    observation: dict,
    spec: EngineeringSpec,
    context: dict | None = None,
) -> ConstraintReport:
    """Evaluate every spec constraint against an observation snapshot.

    Args:
        observation: dict from ``kairos.representation.observe`` (or an
            equivalent recorded snapshot).
        spec: the parsed engineering requirement.
        context: optional episode context, e.g. ``{"initial_mass_g": ...,
            "required_hole_positions": [...]}`` for Task-E style requirements.
    """
    report = ConstraintReport()
    for constraint in spec.constraints:
        kind = constraint.kind
        if kind == "hole_count":
            result = _check_hole_count(constraint, observation, spec)
        elif kind == "hole_diameter":
            result = _check_hole_diameter(constraint, observation, spec)
        elif kind == "bounding_box_exact":
            result = _check_bbox_exact(constraint, observation, spec)
        elif kind == "bounding_box_max":
            result = _check_bbox_max(constraint, observation, spec)
        elif kind == "mounting_angle":
            result = _check_mounting_angle(constraint, observation, spec)
        elif kind == "cylindrical_interface":
            result = _check_cylindrical_interface(constraint, observation, spec)
        elif kind == "mass_reduction_pct":
            result = _check_mass_reduction(constraint, observation, spec, context)
        elif kind == "hole_positions_preserved":
            result = _check_hole_positions_preserved(constraint, observation, spec, context)
        elif kind == "min_wall_thickness":
            result = _check_min_wall_thickness(constraint, observation, spec)
        else:
            # symmetry and unknown kinds.
            result = ConstraintResult(
                constraint, "unmeasured", detail=f"no checker for {kind!r} yet"
            )
        report.results.append(result)
    return report
