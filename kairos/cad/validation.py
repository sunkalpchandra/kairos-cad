"""Geometry and document validation.

Produces structured reports rather than booleans so the reward function can
grade *how* a model is broken (null shape vs. open shell vs. feature error),
and so dataset generation can reject invalid designs with a recorded reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kairos.cad.document import CADDocument


@dataclass
class ValidationReport:
    """Outcome of validating a shape or document."""

    is_valid: bool
    checks: dict[str, bool] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"is_valid": self.is_valid, "checks": dict(self.checks), "issues": list(self.issues)}


def validate_shape(shape) -> ValidationReport:
    """Validate a Part shape: non-null, kernel-valid, closed solid, volume > 0."""
    checks: dict[str, bool] = {}
    issues: list[str] = []

    if shape is None or shape.isNull():
        return ValidationReport(False, {"non_null": False}, ["shape is null or missing"])
    checks["non_null"] = True

    try:
        checks["kernel_valid"] = bool(shape.isValid())
    except Exception as err:
        checks["kernel_valid"] = False
        issues.append(f"isValid() raised: {err}")
    if not checks["kernel_valid"]:
        issues.append("OCC kernel reports the shape invalid")

    solids = shape.Solids
    checks["has_solid"] = len(solids) > 0
    if not checks["has_solid"]:
        issues.append("shape contains no solid")

    checks["closed_shells"] = all(shell.isClosed() for shell in shape.Shells) and bool(
        shape.Shells
    )
    if not checks["closed_shells"]:
        issues.append("shape has open or missing shells")

    try:
        checks["positive_volume"] = shape.Volume > 1e-9
    except Exception:
        checks["positive_volume"] = False
    if not checks["positive_volume"]:
        issues.append("shape volume is not positive")

    return ValidationReport(all(checks.values()), checks, issues)


def validate_document(cad_doc: CADDocument) -> ValidationReport:
    """Validate the whole document: feature error states plus tip shape."""
    issues: list[str] = []
    checks: dict[str, bool] = {}

    recompute_errors = cad_doc.recompute()
    checks["features_ok"] = not recompute_errors
    issues.extend(recompute_errors)

    shape = cad_doc.tip_shape()
    if shape is None:
        checks["has_tip_shape"] = False
        issues.append("body has no tip shape")
        return ValidationReport(False, checks, issues)
    checks["has_tip_shape"] = True

    shape_report = validate_shape(shape)
    checks.update(shape_report.checks)
    issues.extend(shape_report.issues)
    return ValidationReport(all(checks.values()), checks, issues)
