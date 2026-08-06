"""Reinforced-plate family: rectangular plate with full-length stiffening ribs.

Geometry: a base plate padded +z from an XY sketch, ``n_ribs`` narrow ribs
running the full length padded upward from a second XY sketch on the top
face (extrusion direction verified mid-build via bounding box, re-padded
``reversed`` if the backend extrudes offset sketches downward), and four
corner through-holes pocketed from the top face, positioned clear of every
rib centerline.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from kairos.actions.executor import ActionExecutor
from kairos.actions.schema import Action, Operation
from kairos.data.families.base import Family, register


@dataclass
class ReinforcedPlateParams:
    """Rectangular plate stiffened by top ribs, with corner mounting holes."""

    length: float = 100.0  # x, mm
    width: float = 60.0  # y, mm
    thickness: float = 6.0  # plate thickness along z, mm
    n_ribs: int = 2  # full-length ribs on the top face
    rib_width: float = 6.0  # rib extent along y, mm
    rib_height: float = 8.0  # rib extent above the plate top, mm
    hole_diameter: float = 5.0
    corner_margin: float = 10.0  # corner-to-hole-center clearance, mm

    @classmethod
    def sample(cls, rng: random.Random) -> ReinforcedPlateParams:
        hole_diameter = rng.choice([3.0, 4.0, 5.0, 6.0])
        return cls(
            length=rng.uniform(70.0, 130.0),
            width=rng.uniform(45.0, 90.0),
            thickness=rng.uniform(4.0, 9.0),
            n_ribs=rng.choice([1, 2, 3]),
            rib_width=rng.uniform(4.0, 9.0),
            rib_height=rng.uniform(5.0, 12.0),
            hole_diameter=hole_diameter,
            corner_margin=rng.uniform(1.4 * hole_diameter, 2.2 * hole_diameter),
        )

    def rib_centerlines(self) -> list[float]:
        """Rib centerline y positions, evenly spaced across the width."""
        pitch = self.width / (self.n_ribs + 1)
        return [pitch * (i + 1) for i in range(self.n_ribs)]

    def hole_centers(self) -> list[tuple[float, float]]:
        """The four corner hole centers, ``corner_margin`` in from each corner."""
        m = self.corner_margin
        return [
            (m, m),
            (self.length - m, m),
            (m, self.width - m),
            (self.length - m, self.width - m),
        ]

    def is_feasible(self) -> bool:
        """Reject parameter draws whose holes leave the plate or touch a rib."""
        radius = self.hole_diameter / 2.0
        if self.corner_margin <= radius + 1.0:
            return False
        if self.length - 2.0 * self.corner_margin <= self.hole_diameter + 2.0:
            return False
        if self.width - 2.0 * self.corner_margin <= self.hole_diameter + 2.0:
            return False
        if self.width / (self.n_ribs + 1) <= self.rib_width + 1.0:
            return False
        # Hole bores must never touch a rib: keep centers clear in y.
        clearance = self.rib_width / 2.0 + radius + 2.0
        hole_rows = (self.corner_margin, self.width - self.corner_margin)
        for center in self.rib_centerlines():
            for row in hole_rows:
                if abs(row - center) < clearance:
                    return False
        return True


def reinforced_plate_rib_sketch_actions(p: ReinforcedPlateParams) -> list[Action]:
    """Sketch the rib rectangles on the plate top face (z = thickness)."""
    actions = [
        Action(Operation.CREATE_SKETCH, parameters={"plane": "XY", "offset": p.thickness})
    ]
    for center in p.rib_centerlines():
        actions.append(
            Action(
                Operation.ADD_RECTANGLE,
                parameters={
                    "x": 0.0,
                    "y": center - p.rib_width / 2.0,
                    "width": p.length,
                    "height": p.rib_width,
                },
            )
        )
    return actions


def reinforced_plate_hole_actions(p: ReinforcedPlateParams) -> list[Action]:
    """Corner through-holes from the top face; through-all cuts -z only."""
    actions = [
        Action(Operation.CREATE_SKETCH, parameters={"plane": "XY", "offset": p.thickness})
    ]
    for cx, cy in p.hole_centers():
        actions.append(
            Action(
                Operation.ADD_CIRCLE,
                parameters={"cx": cx, "cy": cy, "radius": p.hole_diameter / 2.0},
            )
        )
    actions.append(Action(Operation.POCKET, parameters={"through_all": True}))
    return actions


def build_reinforced_plate(executor: ActionExecutor, p: ReinforcedPlateParams) -> list[Action]:
    """Execute the full reinforced-plate recipe; returns the actions performed.

    Raises RuntimeError on the first failed action (procedural recipes are
    expected to succeed; failures indicate infeasible parameters or bugs).
    """
    actions: list[Action] = []

    def run(action: Action) -> None:
        result = executor.execute(action)
        if not result.ok:
            raise RuntimeError(
                f"reinforced-plate recipe failed at {action.operation.value}: {result.message}"
            )
        actions.append(action)

    run(Action(Operation.CREATE_SKETCH, parameters={"plane": "XY"}))
    run(
        Action(
            Operation.ADD_RECTANGLE,
            parameters={"x": 0.0, "y": 0.0, "width": p.length, "height": p.width},
        )
    )
    run(Action(Operation.PAD, parameters={"length": p.thickness}))
    for action in reinforced_plate_rib_sketch_actions(p):
        run(action)
    run(Action(Operation.PAD, parameters={"length": p.rib_height}))
    # Pads on offset sketches may extrude toward -z on some backends; verify
    # the ribs actually rose above the plate top, as an agent would.
    top = p.thickness + p.rib_height
    bbox = executor.engine.measure_bounding_box()
    if abs(bbox["z_max"] - top) > 1e-4:
        # Ribs went downward (absorbed by the plate); re-pad them upward.
        for action in reinforced_plate_rib_sketch_actions(p):
            run(action)
        run(Action(Operation.PAD, parameters={"length": p.rib_height, "reversed": True}))
        bbox = executor.engine.measure_bounding_box()
        if abs(bbox["z_max"] - top) > 1e-4 or bbox["z_min"] < -1e-4:
            raise RuntimeError(
                f"reinforced-plate ribs failed to extrude upward: bbox z "
                f"[{bbox['z_min']:.3f}, {bbox['z_max']:.3f}], expected top {top:.3f}"
            )
    for action in reinforced_plate_hole_actions(p):
        run(action)
    finish = Action(Operation.FINISH_DESIGN)
    executor.execute(finish)
    actions.append(finish)
    return actions


def _requirements(p: ReinforcedPlateParams) -> dict:
    text = (
        f"Design a reinforced rectangular plate {p.length:.0f} x {p.width:.0f} x "
        f"{p.thickness:.1f} mm stiffened by {p.n_ribs} full-length ribs "
        f"{p.rib_width:.0f} mm wide and {p.rib_height:.0f} mm tall, with 4 corner "
        f"through-holes of {p.hole_diameter:.0f} mm diameter. Minimize mass."
    )
    return {
        "text": text,
        "spec": {
            "kind": "reinforced_plate",
            "hole_count": 4,
            "hole_diameter": p.hole_diameter,
            "min_wall_thickness": p.thickness,
            "rib_count": p.n_ribs,
            "objective": "minimize_mass",
        },
    }


FAMILY = register(
    Family(
        name="reinforced_plate",
        params_cls=ReinforcedPlateParams,
        build=build_reinforced_plate,
        requirements=_requirements,
        expected_holes=lambda p: [(p.hole_diameter, 4)],
    )
)
