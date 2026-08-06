"""Mounting-plate family: rectangular plate with a grid of through-holes."""

from __future__ import annotations

import random
from dataclasses import dataclass

from kairos.actions.executor import ActionExecutor
from kairos.actions.schema import Action, Operation
from kairos.data.families.base import Family, register


@dataclass
class PlateParams:
    """Rectangular mounting plate with a grid of through-holes."""

    length: float = 80.0  # x, mm
    width: float = 50.0  # y, mm
    thickness: float = 6.0  # z, mm
    hole_diameter: float = 5.0
    holes_x: int = 2
    holes_y: int = 2
    hole_margin: float = 10.0  # edge-to-center clearance, mm

    @classmethod
    def sample(cls, rng: random.Random) -> PlateParams:
        hole_diameter = rng.choice([3.0, 4.0, 5.0, 6.0, 8.0])
        return cls(
            length=rng.uniform(50.0, 120.0),
            width=rng.uniform(35.0, 90.0),
            thickness=rng.uniform(3.0, 10.0),
            hole_diameter=hole_diameter,
            holes_x=rng.choice([2, 3, 4]),
            holes_y=rng.choice([1, 2]),
            hole_margin=rng.uniform(1.5 * hole_diameter, 3.0 * hole_diameter),
        )

    def hole_grid(self) -> list[tuple[float, float]]:
        def spread(count: int, extent: float) -> list[float]:
            lo, hi = self.hole_margin, extent - self.hole_margin
            if count == 1:
                return [extent / 2.0]
            step = (hi - lo) / (count - 1)
            return [lo + i * step for i in range(count)]

        return [
            (x, y)
            for x in spread(self.holes_x, self.length)
            for y in spread(self.holes_y, self.width)
        ]

    def is_feasible(self) -> bool:
        radius = self.hole_diameter / 2.0
        if self.hole_margin <= radius + 1.0:
            return False
        centers = self.hole_grid()
        for i, (x1, y1) in enumerate(centers):
            for x2, y2 in centers[i + 1 :]:
                if (x1 - x2) ** 2 + (y1 - y2) ** 2 <= (2 * radius + 2.0) ** 2:
                    return False
        return True


def plate_actions(p: PlateParams) -> list[Action]:
    """Full mounting-plate recipe as a fixed action sequence."""
    actions = [
        Action(Operation.CREATE_SKETCH, parameters={"plane": "XY"}),
        Action(
            Operation.ADD_RECTANGLE,
            parameters={"x": 0.0, "y": 0.0, "width": p.length, "height": p.width},
        ),
        Action(Operation.PAD, parameters={"length": p.thickness}),
        Action(Operation.CREATE_SKETCH, parameters={"plane": "XY", "offset": p.thickness}),
    ]
    for cx, cy in p.hole_grid():
        actions.append(
            Action(
                Operation.ADD_CIRCLE,
                parameters={"cx": cx, "cy": cy, "radius": p.hole_diameter / 2.0},
            )
        )
    actions.append(Action(Operation.POCKET, parameters={"through_all": True}))
    actions.append(Action(Operation.FINISH_DESIGN))
    return actions


def build_plate(executor: ActionExecutor, p: PlateParams) -> list[Action]:
    """Execute the plate recipe; raises RuntimeError on failure."""
    actions = plate_actions(p)
    for action in actions:
        result = executor.execute(action)
        if not result.ok:
            raise RuntimeError(
                f"plate recipe failed at {action.operation.value}: {result.message}"
            )
    return actions


def _requirements(p: PlateParams) -> dict:
    holes = p.holes_x * p.holes_y
    text = (
        f"Design a rectangular mounting plate {p.length:.0f} x "
        f"{p.width:.0f} x {p.thickness:.1f} mm with {holes} "
        f"through-holes of {p.hole_diameter:.0f} mm diameter in a "
        f"{p.holes_x}x{p.holes_y} grid. Minimize mass."
    )
    return {
        "text": text,
        "spec": {
            "kind": "plate",
            "hole_count": holes,
            "hole_diameter": p.hole_diameter,
            "min_wall_thickness": p.thickness,
            "objective": "minimize_mass",
        },
    }


FAMILY = register(
    Family(
        name="plate",
        params_cls=PlateParams,
        build=build_plate,
        requirements=_requirements,
        expected_holes=lambda p: [(p.hole_diameter, p.holes_x * p.holes_y)],
    )
)
