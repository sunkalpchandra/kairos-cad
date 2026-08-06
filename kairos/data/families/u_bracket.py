"""U-bracket family: U-channel with base holes and coaxial cross-wall holes.

Geometry: U profile sketched on XZ (outer width along +x, height along +z,
side walls ``wall_thickness``, floor ``base_thickness``), midplane-padded
along y by ``depth``. Base holes are pocketed through the floor from an
offset XY sketch; side holes are sketched on the outer +x wall face (YZ at
``outer_width``) and pocketed through-all along -x, piercing BOTH walls —
the two wall bores share one axis, so each position counts as one hole.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from kairos.actions.executor import ActionExecutor
from kairos.actions.schema import Action, Operation
from kairos.data.families.base import Family, register


@dataclass
class UBracketParams:
    """Parameters of a U-channel bracket with base and cross-wall holes."""

    outer_width: float = 60.0  # outer channel width along +x, mm
    height: float = 40.0  # channel height along +z, mm
    depth: float = 30.0  # extrusion depth along y, mm
    wall_thickness: float = 6.0  # side-wall thickness along x, mm
    base_thickness: float = 6.0  # floor thickness along z, mm
    hole_diameter: float = 5.0
    n_base: int = 2  # through-holes in the floor
    n_side: int = 2  # cross-wall hole positions (one coaxial hole each)
    hole_margin: float = 8.0  # clearance from walls/edges to hole centers, mm

    @classmethod
    def sample(cls, rng: random.Random) -> UBracketParams:
        hole_diameter = rng.choice([3.0, 4.0, 5.0, 6.0])
        return cls(
            outer_width=rng.uniform(45.0, 90.0),
            height=rng.uniform(30.0, 60.0),
            depth=rng.uniform(20.0, 45.0),
            wall_thickness=rng.uniform(4.0, 9.0),
            base_thickness=rng.uniform(4.0, 9.0),
            hole_diameter=hole_diameter,
            n_base=rng.choice([1, 2, 3]),
            n_side=rng.choice([1, 2]),
            hole_margin=rng.uniform(1.5 * hole_diameter, 2.5 * hole_diameter),
        )

    @staticmethod
    def _spread(n: int, start: float, end: float) -> list[float]:
        """``n`` centers evenly spread over [start, end] (midpoint when n=1)."""
        if n == 1:
            return [(start + end) / 2.0]
        step = (end - start) / (n - 1)
        return [start + i * step for i in range(n)]

    def base_hole_xs(self) -> list[float]:
        """Floor hole-center x positions, clear of both side walls."""
        return self._spread(
            self.n_base,
            self.wall_thickness + self.hole_margin,
            self.outer_width - self.wall_thickness - self.hole_margin,
        )

    def side_hole_zs(self) -> list[float]:
        """Cross-wall hole-center z heights, clear of the floor and the rim."""
        return self._spread(
            self.n_side,
            self.base_thickness + self.hole_margin,
            self.height - self.hole_margin,
        )

    def _spaced(self, positions: list[float]) -> bool:
        """Adjacent centers must leave >2 mm of web between bores."""
        return all(
            b - a > self.hole_diameter + 2.0
            for a, b in zip(positions, positions[1:], strict=False)
        )

    def is_feasible(self) -> bool:
        """Reject parameter draws that cannot produce the required holes."""
        radius = self.hole_diameter / 2.0
        if self.n_base < 1 or self.n_side < 1:
            return False
        if self.hole_margin <= radius + 1.0:
            return False
        if self.hole_diameter + 2.0 >= self.depth:
            return False
        xs = self.base_hole_xs()
        if xs[0] - radius <= self.wall_thickness + 1.0:
            return False
        if xs[-1] + radius >= self.outer_width - self.wall_thickness - 1.0:
            return False
        zs = self.side_hole_zs()
        if zs[0] - radius <= self.base_thickness + 1.0:
            return False
        if zs[-1] + radius >= self.height - 1.0:
            return False
        if not self._spaced(xs) or not self._spaced(zs):
            return False
        return True


def u_bracket_profile_actions(p: UBracketParams) -> list[Action]:
    """Sketch and pad the U profile (XZ plane, midplane extrusion)."""
    w, h = p.outer_width, p.height
    t, b = p.wall_thickness, p.base_thickness
    profile = [
        (0.0, 0.0),
        (w, 0.0),
        (w, h),
        (w - t, h),
        (w - t, b),
        (t, b),
        (t, h),
        (0.0, h),
    ]
    return [
        Action(Operation.CREATE_SKETCH, parameters={"plane": "XZ"}),
        Action(Operation.ADD_POLYGON, parameters={"points": [list(pt) for pt in profile]}),
        Action(Operation.PAD, parameters={"length": p.depth, "midplane": True}),
    ]


def u_bracket_hole_actions(p: UBracketParams) -> list[Action]:
    """Floor holes plus cross-wall holes via offset sketches and pockets."""
    actions: list[Action] = []
    radius = p.hole_diameter / 2.0
    # Floor holes: sketch on the channel floor (z = base_thickness), cut -z.
    actions.append(
        Action(Operation.CREATE_SKETCH, parameters={"plane": "XY", "offset": p.base_thickness})
    )
    for x in p.base_hole_xs():
        actions.append(
            Action(Operation.ADD_CIRCLE, parameters={"cx": x, "cy": 0.0, "radius": radius})
        )
    actions.append(Action(Operation.POCKET, parameters={"through_all": True}))
    # Cross-wall holes: sketch on the outer +x face (x = outer_width), cut -x
    # through both side walls; each position yields one coaxial hole.
    actions.append(
        Action(Operation.CREATE_SKETCH, parameters={"plane": "YZ", "offset": p.outer_width})
    )
    for z in p.side_hole_zs():
        actions.append(
            Action(Operation.ADD_CIRCLE, parameters={"cx": 0.0, "cy": z, "radius": radius})
        )
    actions.append(Action(Operation.POCKET, parameters={"through_all": True}))
    return actions


def build_u_bracket(executor: ActionExecutor, p: UBracketParams) -> list[Action]:
    """Execute the full U-bracket recipe; returns the actions performed.

    Raises RuntimeError on the first failed action (procedural recipes are
    expected to succeed; failures indicate infeasible parameters or bugs).
    """
    actions = u_bracket_profile_actions(p) + u_bracket_hole_actions(p)
    for action in actions:
        result = executor.execute(action)
        if not result.ok:
            raise RuntimeError(
                f"U-bracket recipe failed at {action.operation.value}: {result.message}"
            )
    # Inspect state to confirm the coaxial wall bores merged, as an agent would.
    holes = executor.engine.find_holes(diameter=p.hole_diameter)
    if len(holes) != p.n_base + p.n_side:
        raise RuntimeError(
            f"U-bracket hole check failed: found {len(holes)}, "
            f"expected {p.n_base + p.n_side}"
        )
    finish = Action(Operation.FINISH_DESIGN)
    executor.execute(finish)
    actions.append(finish)
    return actions


def _requirements(p: UBracketParams) -> dict:
    holes = p.n_base + p.n_side
    text = (
        f"Design a U-channel bracket {p.outer_width:.0f} mm wide, {p.height:.0f} mm tall "
        f"and {p.depth:.0f} mm deep, with {p.wall_thickness:.1f} mm side walls and a "
        f"{p.base_thickness:.1f} mm base. Provide {p.n_base} base mounting holes and "
        f"{p.n_side} cross-wall holes, all {p.hole_diameter:.0f} mm diameter. Minimize mass."
    )
    return {
        "text": text,
        "spec": {
            "kind": "u_bracket",
            "hole_count": holes,
            "hole_diameter": p.hole_diameter,
            "min_wall_thickness": min(p.wall_thickness, p.base_thickness),
            "objective": "minimize_mass",
        },
    }


FAMILY = register(
    Family(
        name="u_bracket",
        params_cls=UBracketParams,
        build=build_u_bracket,
        requirements=_requirements,
        expected_holes=lambda p: [(p.hole_diameter, p.n_base + p.n_side)],
    )
)
