"""Corner-bracket family: gusseted 90-degree bracket with per-leg holes.

Geometry: L profile sketched on XZ (legs along +x and +z, wall ``thickness``),
midplane-padded along y by ``width``; a triangular gusset rib at the inner
corner (XZ triangle of leg ``gusset``, midplane-padded by ``rib_width`` and
fused with the body); through-holes pocketed from offset XY / YZ sketches.
Hole centers are kept beyond the gusset extent so no bore ever crosses the
rib (a crossing would split the bore cylinder and break hole detection).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from kairos.actions.executor import ActionExecutor
from kairos.actions.schema import Action, Operation
from kairos.data.families.base import Family, register
from kairos.data.families.profiles import draw_profile
from kairos.data.families.wording import state_minimum, stated_minimum


@dataclass
class CornerBracketParams:
    """Parameters of a gusseted 90-degree corner bracket with per-leg holes."""

    leg1: float = 60.0  # base leg length along +x, mm
    leg2: float = 50.0  # vertical leg height along +z, mm
    width: float = 30.0  # extrusion width along y, mm
    thickness: float = 5.0  # wall thickness, mm
    gusset: float = 12.0  # gusset triangle leg along +x/+z from the inner corner, mm
    rib_width: float = 10.0  # gusset rib extrusion width along y (< width), mm
    hole_diameter: float = 5.0
    holes_per_leg: int = 2
    hole_margin: float = 8.0  # clearance from leg edges to hole centers, mm
    margin_from_rib: float = 4.0  # clearance from gusset tip to hole centers, mm

    @classmethod
    def sample(cls, rng: random.Random) -> CornerBracketParams:
        thickness = rng.uniform(3.0, 7.0)
        hole_diameter = rng.choice([3.0, 4.0, 5.0, 6.0])
        width = rng.uniform(20.0, 45.0)
        return cls(
            leg1=rng.uniform(50.0, 95.0),
            leg2=rng.uniform(45.0, 85.0),
            width=width,
            thickness=thickness,
            gusset=rng.uniform(1.5 * thickness, 3.0 * thickness),
            rib_width=rng.uniform(0.25 * width, 0.6 * width),
            hole_diameter=hole_diameter,
            holes_per_leg=rng.choice([1, 2]),
            hole_margin=rng.uniform(1.5 * hole_diameter, 2.5 * hole_diameter),
            margin_from_rib=rng.uniform(0.5 * hole_diameter + 1.5, hole_diameter + 2.0),
        )

    def rib_clearance(self) -> float:
        """Minimum admissible hole-center offset along a leg (past the rib)."""
        return self.thickness + self.gusset + self.margin_from_rib

    def hole_positions(self, leg_length: float) -> list[float]:
        """Hole-center offsets along a leg, clear of the rib and free end."""
        start = max(self.thickness + self.hole_margin, self.rib_clearance())
        end = leg_length - self.hole_margin
        n = self.holes_per_leg
        if n == 1:
            return [(start + end) / 2.0]
        step = (end - start) / (n - 1)
        return [start + i * step for i in range(n)]

    def is_feasible(self) -> bool:
        """Reject parameter draws that cannot produce the required geometry."""
        radius = self.hole_diameter / 2.0
        if self.gusset <= 0.0 or self.rib_width <= 0.0:
            return False
        if self.rib_width >= self.width:
            return False
        if self.margin_from_rib <= radius + 1.0:
            return False
        for leg in (self.leg1, self.leg2):
            start = max(self.thickness + self.hole_margin, self.rib_clearance())
            if start > leg - self.hole_margin:
                return False
            positions = self.hole_positions(leg)
            if positions[0] - radius <= self.thickness:
                return False
            if positions[0] < self.rib_clearance():  # bores must never cross the rib
                return False
            if positions[-1] + radius >= leg:
                return False
            if any(
                b - a <= self.hole_diameter + 2.0
                for a, b in zip(positions, positions[1:], strict=False)
            ):
                return False
        if self.hole_diameter + 2.0 >= self.width:
            return False
        return True


def corner_bracket_profile_actions(p: CornerBracketParams) -> list[Action]:
    """Sketch and pad the L profile plus the fused gusset rib (XZ, midplane)."""
    profile = [
        (0.0, 0.0),
        (p.leg1, 0.0),
        (p.leg1, p.thickness),
        (p.thickness, p.thickness),
        (p.thickness, p.leg2),
        (0.0, p.leg2),
    ]
    t, g = p.thickness, p.gusset
    gusset_triangle = [(t, t), (t + g, t), (t, t + g)]
    return [
        Action(Operation.CREATE_SKETCH, parameters={"plane": "XZ"}),
        *draw_profile(profile),
        Action(Operation.PAD, parameters={"length": p.width, "midplane": True}),
        Action(Operation.CREATE_SKETCH, parameters={"plane": "XZ"}),
        *draw_profile(gusset_triangle),
        Action(Operation.PAD, parameters={"length": p.rib_width, "midplane": True}),
    ]


def corner_bracket_hole_actions(p: CornerBracketParams) -> list[Action]:
    """Through-holes on both legs via offset sketches and through-all pockets."""
    actions: list[Action] = []
    radius = p.hole_diameter / 2.0
    # Base-leg holes: sketch on the top of the base leg (z = thickness), cut -z.
    actions.append(
        Action(Operation.CREATE_SKETCH, parameters={"plane": "XY", "offset": p.thickness})
    )
    for x in p.hole_positions(p.leg1):
        actions.append(
            Action(Operation.ADD_CIRCLE, parameters={"cx": x, "cy": 0.0, "radius": radius})
        )
    actions.append(Action(Operation.POCKET, parameters={"through_all": True}))
    # Vertical-leg holes: sketch on the inner face (x = thickness), cut -x.
    actions.append(
        Action(Operation.CREATE_SKETCH, parameters={"plane": "YZ", "offset": p.thickness})
    )
    for z in p.hole_positions(p.leg2):
        actions.append(
            Action(Operation.ADD_CIRCLE, parameters={"cx": 0.0, "cy": z, "radius": radius})
        )
    actions.append(Action(Operation.POCKET, parameters={"through_all": True}))
    return actions


def build_corner_bracket(executor: ActionExecutor, p: CornerBracketParams) -> list[Action]:
    """Execute the full corner-bracket recipe; returns the actions performed.

    Raises RuntimeError on the first failed action (procedural recipes are
    expected to succeed; failures indicate infeasible parameters or bugs).
    """
    actions = corner_bracket_profile_actions(p) + corner_bracket_hole_actions(p)
    for action in actions:
        result = executor.execute(action)
        if not result.ok:
            raise RuntimeError(
                f"corner-bracket recipe failed at {action.operation.value}: {result.message}"
            )
    # Inspect state to confirm no bore was split by the rib, as an agent would.
    holes = executor.engine.find_holes(diameter=p.hole_diameter)
    if len(holes) != 2 * p.holes_per_leg:
        raise RuntimeError(
            f"corner-bracket hole check failed: found {len(holes)}, "
            f"expected {2 * p.holes_per_leg}"
        )
    finish = Action(Operation.FINISH_DESIGN)
    executor.execute(finish)
    actions.append(finish)
    return actions


def _min_wall(p: CornerBracketParams) -> float:
    """The thinnest wall the part actually has.

    The gusset rib is a wall. Declaring only ``thickness`` overstated the
    minimum whenever ``rib_width`` was thinner, which the ray-cast measurement
    then found and reported as a violation -- 75 of 144 corner brackets, by up
    to 0.391 mm.
    """
    return min(p.thickness, p.rib_width)


def _requirements(p: CornerBracketParams) -> dict:
    holes = 2 * p.holes_per_leg
    text = (
        f"Design a 90-degree corner bracket with a {p.gusset:.0f} mm triangular gusset "
        f"rib ({p.rib_width:.0f} mm wide) at the inner corner and {holes} mounting holes "
        f"of {p.hole_diameter:.0f} mm diameter ({p.holes_per_leg} per leg), wall thickness "
        f"{state_minimum(_min_wall(p))} mm, legs {p.leg1:.0f} mm and {p.leg2:.0f} mm, width "
        f"{p.width:.0f} mm. Minimize mass."
    )
    return {
        "text": text,
        "spec": {
            "kind": "corner_bracket",
            "hole_count": holes,
            "hole_diameter": p.hole_diameter,
            "min_wall_thickness": stated_minimum(_min_wall(p)),
            "mounting_angle": 90,
            "objective": "minimize_mass",
        },
    }


FAMILY = register(
    Family(
        name="corner_bracket",
        params_cls=CornerBracketParams,
        build=build_corner_bracket,
        requirements=_requirements,
        expected_holes=lambda p: [(p.hole_diameter, 2 * p.holes_per_leg)],
    )
)
