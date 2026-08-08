"""L-bracket family: 90-degree bracket with per-leg mounting holes.

Geometry: L profile sketched on XZ (legs along +x and +z, wall ``thickness``),
midplane-padded along y by ``width``; through-holes pocketed from offset
XY / YZ sketches; optional inner-corner fillet located by geometric edge
search (as an agent would locate it).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from kairos.actions.executor import ActionExecutor
from kairos.actions.schema import Action, Operation
from kairos.data.families.base import Family, register
from kairos.data.families.profiles import draw_profile


@dataclass
class LBracketParams:
    """Parameters of a 90-degree L-bracket with per-leg mounting holes."""

    leg1: float = 60.0  # base leg length along +x, mm
    leg2: float = 50.0  # vertical leg height along +z, mm
    width: float = 30.0  # extrusion width along y, mm
    thickness: float = 5.0  # wall thickness, mm
    hole_diameter: float = 5.0
    holes_per_leg: int = 2
    hole_margin: float = 8.0  # clearance from leg edges to hole centers, mm
    fillet_radius: float = 0.0  # inner-corner fillet; 0 disables

    @classmethod
    def sample(cls, rng: random.Random) -> LBracketParams:
        thickness = rng.uniform(3.0, 8.0)
        hole_diameter = rng.choice([3.0, 4.0, 5.0, 6.0])
        return cls(
            leg1=rng.uniform(40.0, 90.0),
            leg2=rng.uniform(35.0, 80.0),
            width=rng.uniform(20.0, 45.0),
            thickness=thickness,
            hole_diameter=hole_diameter,
            holes_per_leg=rng.choice([1, 2, 3]),
            hole_margin=rng.uniform(1.5 * hole_diameter, 2.5 * hole_diameter),
            fillet_radius=rng.choice([0.0, rng.uniform(1.0, 0.6 * thickness)]),
        )

    def hole_positions(self, leg_length: float) -> list[float]:
        """Hole-center offsets along a leg, clear of the corner and free end."""
        start = self.thickness + self.hole_margin
        end = leg_length - self.hole_margin
        n = self.holes_per_leg
        if n == 1:
            return [(start + end) / 2.0]
        step = (end - start) / (n - 1)
        return [start + i * step for i in range(n)]

    def is_feasible(self) -> bool:
        """Reject parameter draws that cannot produce the required holes."""
        radius = self.hole_diameter / 2.0
        for leg in (self.leg1, self.leg2):
            positions = self.hole_positions(leg)
            if positions[0] - radius <= self.thickness:
                return False
            if positions[-1] + radius >= leg:
                return False
            # Adjacent bores on one leg must not merge into a slot.
            if any(
                b - a <= self.hole_diameter + 2.0
                for a, b in zip(positions, positions[1:], strict=False)
            ):
                return False
        if self.hole_diameter + 2.0 >= self.width:
            return False
        if self.fillet_radius >= self.thickness:
            return False
        return True


def l_bracket_profile_actions(p: LBracketParams) -> list[Action]:
    """Sketch and pad the L profile (XZ plane, midplane extrusion)."""
    profile = [
        (0.0, 0.0),
        (p.leg1, 0.0),
        (p.leg1, p.thickness),
        (p.thickness, p.thickness),
        (p.thickness, p.leg2),
        (0.0, p.leg2),
    ]
    return [
        Action(Operation.CREATE_SKETCH, parameters={"plane": "XZ"}),
        *draw_profile(profile),
        Action(Operation.PAD, parameters={"length": p.width, "midplane": True}),
    ]


def l_bracket_hole_actions(p: LBracketParams) -> list[Action]:
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


def build_l_bracket(executor: ActionExecutor, p: LBracketParams) -> list[Action]:
    """Execute the full L-bracket recipe; returns the actions performed.

    Raises RuntimeError on the first failed action (procedural recipes are
    expected to succeed; failures indicate infeasible parameters or bugs).
    """
    actions = l_bracket_profile_actions(p) + l_bracket_hole_actions(p)
    for action in actions:
        result = executor.execute(action)
        if not result.ok:
            raise RuntimeError(
                f"L-bracket recipe failed at {action.operation.value}: {result.message}"
            )
    if p.fillet_radius > 0:
        # Inspect state to find the inner-corner edge, as an agent would.
        edges = executor.engine.find_edges(
            curve="Line",
            direction=(0.0, 1.0, 0.0),
            near=(p.thickness, 0.0, p.thickness),
            near_tol=0.5,
        )
        if edges:
            fillet = Action(
                Operation.FILLET,
                target=",".join(edges[:1]),
                parameters={"radius": p.fillet_radius},
            )
            result = executor.execute(fillet)
            if not result.ok:
                raise RuntimeError(f"L-bracket fillet failed: {result.message}")
            actions.append(fillet)
    finish = Action(Operation.FINISH_DESIGN)
    executor.execute(finish)
    actions.append(finish)
    return actions


def _requirements(p: LBracketParams) -> dict:
    holes = 2 * p.holes_per_leg
    text = (
        f"Design a 90-degree L-bracket with {holes} mounting holes of "
        f"{p.hole_diameter:.0f} mm diameter ({p.holes_per_leg} per leg), wall thickness "
        f"{p.thickness:.1f} mm, legs {p.leg1:.0f} mm and "
        f"{p.leg2:.0f} mm, width {p.width:.0f} mm. Minimize mass."
    )
    return {
        "text": text,
        "spec": {
            "kind": "l_bracket",
            "hole_count": holes,
            "hole_diameter": p.hole_diameter,
            "min_wall_thickness": p.thickness,
            "mounting_angle": 90,
            "objective": "minimize_mass",
        },
    }


FAMILY = register(
    Family(
        name="l_bracket",
        params_cls=LBracketParams,
        build=build_l_bracket,
        requirements=_requirements,
        expected_holes=lambda p: [(p.hole_diameter, 2 * p.holes_per_leg)],
    )
)
