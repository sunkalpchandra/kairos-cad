"""Flange family: disk with raised hub, central bore, and a bolt circle.

Geometry: radial half-section sketched on XZ (sketch x-coords are radii,
all > 0), revolved 360 degrees about the sketch V axis (global Z) into a
disk-plus-hub with a through-bore; one bolt hole is pocketed through the
disk from an offset XY sketch, then repeated around Z with a circular
pattern (pattern count = total occurrences including the original).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from kairos.actions.executor import ActionExecutor
from kairos.actions.schema import Action, Operation
from kairos.data.families.base import Family, register
from kairos.data.families.profiles import draw_profile


@dataclass
class FlangeParams:
    """Parameters of a hubbed disk flange with a bolt circle."""

    bore_radius: float = 6.0  # central bore radius, mm
    disk_radius: float = 32.0  # disk outer radius, mm
    disk_thickness: float = 6.0  # disk thickness along z, mm
    hub_radius: float = 12.0  # hub outer radius, mm
    hub_height: float = 10.0  # hub height above the disk, mm
    bolt_diameter: float = 5.0  # bolt hole diameter, mm
    bolt_circle_radius: float = 22.0  # bolt-center circle radius, mm
    n_bolts: int = 6  # bolt holes on the bolt circle

    @classmethod
    def sample(cls, rng: random.Random) -> FlangeParams:
        bore_radius = rng.uniform(3.0, 7.0)
        hub_radius = bore_radius + rng.uniform(3.0, 8.0)
        bolt_diameter = rng.choice([3.0, 4.0, 5.0, 6.0])
        bolt_circle_radius = hub_radius + bolt_diameter / 2.0 + 2.0 + rng.uniform(1.0, 6.0)
        return cls(
            bore_radius=bore_radius,
            disk_radius=bolt_circle_radius + bolt_diameter / 2.0 + 2.0 + rng.uniform(1.0, 8.0),
            disk_thickness=rng.uniform(4.0, 10.0),
            hub_radius=hub_radius,
            hub_height=rng.uniform(5.0, 15.0),
            bolt_diameter=bolt_diameter,
            bolt_circle_radius=bolt_circle_radius,
            n_bolts=rng.randint(3, 8),
        )

    def is_feasible(self) -> bool:
        """Reject parameter draws whose bore, hub, or bolt circle collide."""
        if self.bore_radius < 2.0:
            return False
        if self.bore_radius + 2.0 >= self.hub_radius:
            return False
        if self.hub_radius >= self.bolt_circle_radius - self.bolt_diameter / 2.0 - 2.0:
            return False
        if self.bolt_circle_radius + self.bolt_diameter / 2.0 + 2.0 >= self.disk_radius:
            return False
        if not 3 <= self.n_bolts <= 8:
            return False
        if self.disk_thickness < 3.0 or self.hub_height < 3.0:
            return False
        # Adjacent bolt holes must not merge along the bolt circle.
        chord = 2.0 * self.bolt_circle_radius * math.sin(math.pi / self.n_bolts)
        if chord <= self.bolt_diameter + 2.0:
            return False
        # The bore must stay distinguishable from the bolt holes: when the two
        # diameters land within measurement tolerance of each other, "6 mm
        # holes" no longer identifies which feature a requirement means, and
        # hole checks cannot tell the bore from the bolt circle.
        if abs(2.0 * self.bore_radius - self.bolt_diameter) <= 0.5:
            return False
        return True


def flange_profile_actions(p: FlangeParams) -> list[Action]:
    """Sketch the radial half-section on XZ and revolve it about global Z."""
    profile = [
        (p.bore_radius, 0.0),
        (p.disk_radius, 0.0),
        (p.disk_radius, p.disk_thickness),
        (p.hub_radius, p.disk_thickness),
        (p.hub_radius, p.disk_thickness + p.hub_height),
        (p.bore_radius, p.disk_thickness + p.hub_height),
    ]
    return [
        Action(Operation.CREATE_SKETCH, parameters={"plane": "XZ"}),
        *draw_profile(profile),
        Action(Operation.REVOLVE, parameters={"angle": 360.0, "axis": "V"}),
    ]


def flange_bolt_hole_actions(p: FlangeParams) -> list[Action]:
    """One bolt hole through the disk: offset XY sketch, through-all pocket."""
    return [
        Action(Operation.CREATE_SKETCH, parameters={"plane": "XY", "offset": p.disk_thickness}),
        Action(
            Operation.ADD_CIRCLE,
            parameters={"cx": p.bolt_circle_radius, "cy": 0.0, "radius": p.bolt_diameter / 2.0},
        ),
        Action(Operation.POCKET, parameters={"through_all": True}),
    ]


def build_flange(executor: ActionExecutor, p: FlangeParams) -> list[Action]:
    """Execute the full flange recipe; returns the actions performed.

    Raises RuntimeError on the first failed action (procedural recipes are
    expected to succeed; failures indicate infeasible parameters or bugs).
    """
    actions = flange_profile_actions(p) + flange_bolt_hole_actions(p)
    pocket_feature: str | None = None
    for action in actions:
        result = executor.execute(action)
        if not result.ok:
            raise RuntimeError(
                f"flange recipe failed at {action.operation.value}: {result.message}"
            )
        if action.operation is Operation.POCKET:
            pocket_feature = result.info["feature"]
    pattern = Action(
        Operation.CIRCULAR_PATTERN,
        target=pocket_feature,
        parameters={"axis": "Z", "angle": 360.0, "count": p.n_bolts},
    )
    result = executor.execute(pattern)
    if not result.ok:
        raise RuntimeError(f"flange bolt pattern failed: {result.message}")
    actions.append(pattern)
    finish = Action(Operation.FINISH_DESIGN)
    executor.execute(finish)
    actions.append(finish)
    return actions


def _requirements(p: FlangeParams) -> dict:
    text = (
        f"Design a circular flange of outer diameter {2 * p.disk_radius:.0f} mm and "
        f"thickness {p.disk_thickness:.1f} mm with a {2 * p.hub_radius:.0f} mm diameter "
        f"hub raised {p.hub_height:.0f} mm, a {2 * p.bore_radius:.0f} mm central bore, "
        f"and {p.n_bolts} bolt holes of {p.bolt_diameter:.0f} mm diameter on a "
        f"{2 * p.bolt_circle_radius:.0f} mm bolt circle. Minimize mass."
    )
    return {
        "text": text,
        "spec": {
            "kind": "flange",
            "hole_count": 1 + p.n_bolts,
            "hole_diameter": p.bolt_diameter,
            "bore_diameter": 2 * p.bore_radius,
            "bolt_count": p.n_bolts,
            "bolt_circle_diameter": 2 * p.bolt_circle_radius,
            "min_wall_thickness": p.disk_thickness,
            "objective": "minimize_mass",
        },
    }


FAMILY = register(
    Family(
        name="flange",
        params_cls=FlangeParams,
        build=build_flange,
        requirements=_requirements,
        expected_holes=lambda p: [(2 * p.bore_radius, 1), (p.bolt_diameter, p.n_bolts)],
    )
)
