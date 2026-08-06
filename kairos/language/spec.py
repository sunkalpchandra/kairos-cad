"""EngineeringSpec: the structured requirement the agent reasons against.

A spec is a bag of typed ``Constraint`` records plus optimization objectives.
Constraints carry a ``kind`` from a fixed vocabulary so downstream code
(constraint checker, reward function, benchmark metrics) can dispatch on them
without re-parsing text. Everything serializes to JSON for dataset storage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Constraint kinds understood by the checker (kairos.evaluation.constraints).
#: Kinds outside this set are carried through but reported as unmeasured.
CONSTRAINT_KINDS = (
    "hole_count",  # value: int
    "hole_diameter",  # value: mm
    "min_wall_thickness",  # value: mm (measured in Phase 6; recorded now)
    "mounting_angle",  # value: degrees
    "bounding_box_max",  # value: [x, y, z] mm upper bounds
    "bounding_box_exact",  # value: [x, y, z] mm with tolerance
    "symmetry",  # value: plane name "XZ" | "YZ" | "XY"
    "hole_positions_preserved",  # value: list of [x, y, z]
    "cylindrical_interface",  # value: {"diameter": mm, "count": int}
)

OBJECTIVES = ("minimize_mass", "minimize_volume", "minimize_actions")


@dataclass
class Constraint:
    """One structured requirement, e.g. hole_count == 4."""

    kind: str
    value: Any
    tolerance: float | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"kind": self.kind, "value": self.value}
        if self.tolerance is not None:
            data["tolerance"] = self.tolerance
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Constraint:
        return cls(data["kind"], data["value"], data.get("tolerance"))


@dataclass
class EngineeringSpec:
    """Structured engineering requirement for one design episode."""

    text: str = ""
    constraints: list[Constraint] = field(default_factory=list)
    objectives: list[str] = field(default_factory=list)
    material: str | None = None
    tolerance: float | None = None

    # ------------------------------------------------------------- accessors

    def get(self, kind: str) -> Constraint | None:
        """First constraint of the given kind, or None."""
        for constraint in self.constraints:
            if constraint.kind == kind:
                return constraint
        return None

    def value(self, kind: str, default: Any = None) -> Any:
        constraint = self.get(kind)
        return constraint.value if constraint is not None else default

    @property
    def hole_count(self) -> int | None:
        return self.value("hole_count")

    @property
    def hole_diameter(self) -> float | None:
        return self.value("hole_diameter")

    @property
    def min_wall_thickness(self) -> float | None:
        return self.value("min_wall_thickness")

    def has_objective(self, objective: str) -> bool:
        return objective in self.objectives

    # --------------------------------------------------------- serialization

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "constraints": [c.to_dict() for c in self.constraints],
            "objectives": list(self.objectives),
            "material": self.material,
            "tolerance": self.tolerance,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EngineeringSpec:
        return cls(
            text=data.get("text", ""),
            constraints=[Constraint.from_dict(c) for c in data.get("constraints", [])],
            objectives=list(data.get("objectives", [])),
            material=data.get("material"),
            tolerance=data.get("tolerance"),
        )
