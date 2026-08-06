"""Support-bracket family: two mounting surfaces braced by a triangular rib.

Geometry (benchmark Task C shape): a base plate padded +z from an XY sketch
(straddling y = 0), a vertical wall at the x = 0 end padded midplane from an
XZ sketch, and a right-triangle support rib fusing wall to plate; mounting
through-holes in the base plate (axis z) and the wall (axis x) via
offset-sketch through-all pockets.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from kairos.actions.executor import ActionExecutor
from kairos.actions.schema import Action, Operation
from kairos.data.families.base import Family, register


def _spread(start: float, end: float, count: int) -> list[float]:
    """Evenly spread ``count`` positions across [start, end] (midpoint if 1)."""
    if count == 1:
        return [(start + end) / 2.0]
    step = (end - start) / (count - 1)
    return [start + i * step for i in range(count)]


@dataclass
class SupportBracketParams:
    """Parameters of a support bracket: base plate, wall, and support rib."""

    base_length: float = 90.0  # base plate extent along +x, mm
    base_width: float = 50.0  # base plate extent along y (straddles y=0), mm
    base_thickness: float = 8.0  # base plate thickness along z, mm
    wall_thickness: float = 8.0  # vertical wall thickness along x, mm
    wall_height: float = 50.0  # wall extent above the base plate top, mm
    wall_width: float = 40.0  # wall extrusion width along y (<= base_width), mm
    rib_size: float = 20.0  # rib leg length along +x and +z, mm
    rib_width: float = 8.0  # rib extrusion width along y (< wall_width), mm
    hole_diameter: float = 6.0
    n_base_holes: int = 2  # through-holes in the base plate (axis z)
    n_wall_holes: int = 2  # through-holes in the wall (axis x)
    hole_margin: float = 10.0  # clearance from rib/edges to hole centers, mm

    @classmethod
    def sample(cls, rng: random.Random) -> SupportBracketParams:
        base_width = rng.uniform(40.0, 70.0)
        wall_height = rng.uniform(45.0, 75.0)
        hole_diameter = rng.choice([4.0, 5.0, 6.0])
        return cls(
            base_length=rng.uniform(70.0, 120.0),
            base_width=base_width,
            base_thickness=rng.uniform(5.0, 10.0),
            wall_thickness=rng.uniform(5.0, 10.0),
            wall_height=wall_height,
            wall_width=rng.uniform(0.55, 0.9) * base_width,
            rib_size=rng.uniform(8.0, 0.35 * wall_height),
            rib_width=rng.uniform(5.0, 12.0),
            hole_diameter=hole_diameter,
            n_base_holes=rng.choice([1, 2, 3]),
            n_wall_holes=rng.choice([1, 2]),
            hole_margin=rng.uniform(1.2 * hole_diameter, 1.9 * hole_diameter),
        )

    def base_hole_positions(self) -> list[float]:
        """Hole-center x offsets on the base plate centerline (y = 0)."""
        start = self.wall_thickness + self.rib_size + self.hole_margin
        end = self.base_length - self.hole_margin
        return _spread(start, end, self.n_base_holes)

    def wall_hole_positions(self) -> list[float]:
        """Hole-center z heights on the wall midline (y = 0)."""
        start = self.base_thickness + self.rib_size + self.hole_margin
        end = self.base_thickness + self.wall_height - self.hole_margin
        return _spread(start, end, self.n_wall_holes)

    def is_feasible(self) -> bool:
        """Reject parameter draws that cannot produce the required geometry."""
        radius = self.hole_diameter / 2.0
        if self.wall_width > self.base_width:
            return False
        if self.rib_width >= self.wall_width:
            return False
        if self.wall_width <= self.hole_diameter + 2.0:
            return False
        if self.hole_margin <= radius + 1.0:
            return False
        spans = (
            (
                self.wall_thickness + self.rib_size + self.hole_margin,
                self.base_length - self.hole_margin,
                self.n_base_holes,
            ),
            (
                self.base_thickness + self.rib_size + self.hole_margin,
                self.base_thickness + self.wall_height - self.hole_margin,
                self.n_wall_holes,
            ),
        )
        for start, end, count in spans:
            if end < start:
                return False
            if count > 1 and (end - start) / (count - 1) <= self.hole_diameter + 2.0:
                return False
        return True


def support_bracket_profile_actions(p: SupportBracketParams) -> list[Action]:
    """Sketch and pad the base plate, vertical wall, and triangular rib."""
    rib = [
        (p.wall_thickness, p.base_thickness),
        (p.wall_thickness + p.rib_size, p.base_thickness),
        (p.wall_thickness, p.base_thickness + p.rib_size),
    ]
    return [
        Action(Operation.CREATE_SKETCH, parameters={"plane": "XY"}),
        Action(
            Operation.ADD_RECTANGLE,
            parameters={
                "x": 0.0,
                "y": -p.base_width / 2.0,
                "width": p.base_length,
                "height": p.base_width,
            },
        ),
        Action(Operation.PAD, parameters={"length": p.base_thickness}),
        Action(Operation.CREATE_SKETCH, parameters={"plane": "XZ"}),
        Action(
            Operation.ADD_RECTANGLE,
            parameters={
                "x": 0.0,
                "y": p.base_thickness,
                "width": p.wall_thickness,
                "height": p.wall_height,
            },
        ),
        Action(Operation.PAD, parameters={"length": p.wall_width, "midplane": True}),
        Action(Operation.CREATE_SKETCH, parameters={"plane": "XZ"}),
        Action(Operation.ADD_POLYGON, parameters={"points": [list(pt) for pt in rib]}),
        Action(Operation.PAD, parameters={"length": p.rib_width, "midplane": True}),
    ]


def support_bracket_hole_actions(p: SupportBracketParams) -> list[Action]:
    """Through-holes: base plate (pocket cuts -z) and wall (pocket cuts -x)."""
    actions: list[Action] = []
    radius = p.hole_diameter / 2.0
    # Base holes: sketch on the plate top (z = base_thickness), cut -z.
    actions.append(
        Action(Operation.CREATE_SKETCH, parameters={"plane": "XY", "offset": p.base_thickness})
    )
    for x in p.base_hole_positions():
        actions.append(
            Action(Operation.ADD_CIRCLE, parameters={"cx": x, "cy": 0.0, "radius": radius})
        )
    actions.append(Action(Operation.POCKET, parameters={"through_all": True}))
    # Wall holes: sketch on the rib-side wall face (x = wall_thickness), cut -x.
    actions.append(
        Action(Operation.CREATE_SKETCH, parameters={"plane": "YZ", "offset": p.wall_thickness})
    )
    for z in p.wall_hole_positions():
        actions.append(
            Action(Operation.ADD_CIRCLE, parameters={"cx": 0.0, "cy": z, "radius": radius})
        )
    actions.append(Action(Operation.POCKET, parameters={"through_all": True}))
    return actions


def build_support_bracket(executor: ActionExecutor, p: SupportBracketParams) -> list[Action]:
    """Execute the full support-bracket recipe; returns the actions performed.

    Raises RuntimeError on the first failed action (procedural recipes are
    expected to succeed; failures indicate infeasible parameters or bugs).
    """
    actions: list[Action] = []
    for action in support_bracket_profile_actions(p):
        result = executor.execute(action)
        if not result.ok:
            raise RuntimeError(
                f"support-bracket recipe failed at {action.operation.value}: {result.message}"
            )
        actions.append(action)
    # Inspect state as an agent would: wall and rib must fuse onto the plate,
    # topping out at base_thickness + wall_height, before any drilling.
    bbox = executor.engine.measure_bounding_box()
    top = p.base_thickness + p.wall_height
    if abs(bbox["z_max"] - top) > 1e-4:
        raise RuntimeError(
            f"support-bracket wall tops out at z={bbox['z_max']:.3f}, expected {top:.3f}"
        )
    for action in support_bracket_hole_actions(p):
        result = executor.execute(action)
        if not result.ok:
            raise RuntimeError(
                f"support-bracket recipe failed at {action.operation.value}: {result.message}"
            )
        actions.append(action)
    finish = Action(Operation.FINISH_DESIGN)
    executor.execute(finish)
    actions.append(finish)
    return actions


def _requirements(p: SupportBracketParams) -> dict:
    holes = p.n_base_holes + p.n_wall_holes
    text = (
        f"Design a support bracket with a {p.base_length:.0f} x {p.base_width:.0f} x "
        f"{p.base_thickness:.1f} mm base plate and a {p.wall_thickness:.1f} mm thick "
        f"vertical wall {p.wall_height:.0f} mm tall and {p.wall_width:.0f} mm wide, "
        f"braced by a {p.rib_size:.0f} mm triangular rib. Provide {p.n_base_holes} base "
        f"holes and {p.n_wall_holes} wall holes of {p.hole_diameter:.0f} mm diameter. "
        f"Minimize mass."
    )
    return {
        "text": text,
        "spec": {
            "kind": "support_bracket",
            "hole_count": holes,
            "hole_diameter": p.hole_diameter,
            "min_wall_thickness": min(p.base_thickness, p.wall_thickness),
            "mounting_angle": 90,
            "objective": "minimize_mass",
        },
    }


FAMILY = register(
    Family(
        name="support_bracket",
        params_cls=SupportBracketParams,
        build=build_support_bracket,
        requirements=_requirements,
        expected_holes=lambda p: [(p.hole_diameter, p.n_base_holes + p.n_wall_holes)],
    )
)
