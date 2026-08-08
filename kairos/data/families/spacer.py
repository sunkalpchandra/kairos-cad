"""Spacer family: annular cylinder with optional chamfered outer rims.

Geometry: rectangular radial section sketched on XZ (sketch x-coords are
radii, both > 0), revolved 360 degrees about the sketch V axis (global Z)
into a tube around the origin; the two outer rim circles at z = 0 and
z = ``height`` are then chamfered, located by geometric edge search (as an
agent would locate them).
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
class SpacerParams:
    """Parameters of an annular spacer with optional outer-rim chamfers."""

    inner_radius: float = 5.0  # bore radius, mm
    outer_radius: float = 11.0  # outer radius, mm
    height: float = 16.0  # tube height along z, mm
    chamfer: float = 1.5  # outer-rim chamfer leg size, mm; 0 disables

    @classmethod
    def sample(cls, rng: random.Random) -> SpacerParams:
        inner_radius = rng.uniform(2.0, 8.0)
        return cls(
            inner_radius=inner_radius,
            outer_radius=inner_radius + rng.uniform(2.5, 10.0),
            height=rng.uniform(6.0, 30.0),
            chamfer=rng.choice([0.0, rng.uniform(0.4, 1.2)]),
        )

    def is_feasible(self) -> bool:
        """Reject parameter draws that cannot produce a valid chamfered tube."""
        wall = self.outer_radius - self.inner_radius
        if self.inner_radius < 1.0:
            return False
        if wall < 2.0:
            return False
        if self.height < 4.0:
            return False
        if self.chamfer < 0.0:
            return False
        if self.chamfer > 0.0:
            if self.chamfer >= 0.5 * wall:
                return False
            if 2.0 * self.chamfer >= self.height - 1.0:
                return False
        return True


def spacer_profile_actions(p: SpacerParams) -> list[Action]:
    """Sketch the radial section on XZ and revolve it about global Z."""
    profile = [
        (p.inner_radius, 0.0),
        (p.outer_radius, 0.0),
        (p.outer_radius, p.height),
        (p.inner_radius, p.height),
    ]
    return [
        Action(Operation.CREATE_SKETCH, parameters={"plane": "XZ"}),
        *draw_profile(profile),
        Action(Operation.REVOLVE, parameters={"angle": 360.0, "axis": "V"}),
    ]


def _outer_rim_edges(executor: ActionExecutor, p: SpacerParams) -> list[str]:
    """Locate the outer rim circles at z=0 and z=height by geometric search.

    ``find_edges`` narrows to z-axis circles near each rim plane; the
    candidates are then filtered on ``list_edges`` metadata to keep only
    circles whose midpoint sits at radius ~``outer_radius`` (the outer rims,
    not the bore rims).
    """
    candidates: list[str] = []
    for z in (0.0, p.height):
        for name in executor.engine.find_edges(
            curve="Circle",
            direction=(0.0, 0.0, 1.0),
            near=(0.0, 0.0, z),
            near_tol=p.outer_radius + 1.0,
        ):
            if name not in candidates:
                candidates.append(name)
    by_name = {entry["name"]: entry for entry in executor.engine.list_edges()}
    rims: list[str] = []
    for name in candidates:
        mx, my, _ = by_name[name]["midpoint"]
        if abs(math.hypot(mx, my) - p.outer_radius) < 1e-3:
            rims.append(name)
    return rims


def build_spacer(executor: ActionExecutor, p: SpacerParams) -> list[Action]:
    """Execute the full spacer recipe; returns the actions performed.

    Raises RuntimeError on the first failed action (procedural recipes are
    expected to succeed; failures indicate infeasible parameters or bugs).
    """
    actions = spacer_profile_actions(p)
    for action in actions:
        result = executor.execute(action)
        if not result.ok:
            raise RuntimeError(
                f"spacer recipe failed at {action.operation.value}: {result.message}"
            )
    if p.chamfer > 0:
        # Inspect state to find the outer rim circles, as an agent would.
        rims = _outer_rim_edges(executor, p)
        if not rims:
            raise RuntimeError("spacer chamfer failed: no outer rim edges found")
        chamfer = Action(
            Operation.CHAMFER,
            target=",".join(rims),
            parameters={"size": p.chamfer},
        )
        result = executor.execute(chamfer)
        if not result.ok:
            raise RuntimeError(f"spacer chamfer failed: {result.message}")
        actions.append(chamfer)
    finish = Action(Operation.FINISH_DESIGN)
    executor.execute(finish)
    actions.append(finish)
    return actions


def _requirements(p: SpacerParams) -> dict:
    chamfer_text = (
        f", with {p.chamfer:.1f} mm chamfers on the outer rims" if p.chamfer > 0 else ""
    )
    text = (
        f"Design a cylindrical spacer of outer diameter {2 * p.outer_radius:.0f} mm and "
        # One decimal, not zero: the bore radius is sampled continuously, and
        # rounding the stated diameter to a whole number puts the requirement
        # up to 0.5 mm away from the geometry the recipe actually builds —
        # five times the 0.1 mm diameter tolerance, so the family would fail
        # its own requirement once the parser reads this number.
        f"height {p.height:.0f} mm with a {2 * p.inner_radius:.1f} mm diameter "
        f"through-bore{chamfer_text}. Minimize mass."
    )
    return {
        "text": text,
        "spec": {
            "kind": "spacer",
            "hole_count": 1,
            "hole_diameter": 2 * p.inner_radius,
            "outer_diameter": 2 * p.outer_radius,
            "height": p.height,
            "rim_chamfer": p.chamfer,
            "objective": "minimize_mass",
        },
    }


FAMILY = register(
    Family(
        name="spacer",
        params_cls=SpacerParams,
        build=build_spacer,
        requirements=_requirements,
        expected_holes=lambda p: [(2 * p.inner_radius, 1)],
    )
)
