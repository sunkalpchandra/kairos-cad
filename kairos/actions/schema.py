"""Action vocabulary and dataclasses.

An ``Action`` is the atomic unit of agent behavior and of recorded
trajectories. It must serialize losslessly to JSON: operation name, optional
target (sketch/edge/face/feature name strings), a flat parameter dict of
JSON scalars/lists, and the policy's confidence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Operation(str, Enum):
    """Every CAD operation the agent may request."""

    # Sketch
    CREATE_SKETCH = "CREATE_SKETCH"
    ADD_LINE = "ADD_LINE"
    ADD_RECTANGLE = "ADD_RECTANGLE"
    ADD_CIRCLE = "ADD_CIRCLE"
    ADD_ARC = "ADD_ARC"
    ADD_POLYGON = "ADD_POLYGON"
    DELETE_GEOMETRY = "DELETE_GEOMETRY"
    MOVE_GEOMETRY = "MOVE_GEOMETRY"
    # Constraints
    ADD_HORIZONTAL = "ADD_HORIZONTAL"
    ADD_VERTICAL = "ADD_VERTICAL"
    ADD_PARALLEL = "ADD_PARALLEL"
    ADD_PERPENDICULAR = "ADD_PERPENDICULAR"
    ADD_TANGENT = "ADD_TANGENT"
    ADD_EQUAL = "ADD_EQUAL"
    ADD_DISTANCE = "ADD_DISTANCE"
    ADD_RADIUS = "ADD_RADIUS"
    ADD_DIAMETER = "ADD_DIAMETER"
    ADD_COINCIDENT = "ADD_COINCIDENT"
    ADD_SYMMETRY = "ADD_SYMMETRY"
    # Features
    PAD = "PAD"
    POCKET = "POCKET"
    REVOLVE = "REVOLVE"
    FILLET = "FILLET"
    CHAMFER = "CHAMFER"
    SHELL = "SHELL"
    MIRROR = "MIRROR"
    LINEAR_PATTERN = "LINEAR_PATTERN"
    CIRCULAR_PATTERN = "CIRCULAR_PATTERN"
    # Boolean (multi-body, Phase 2+ in the executor)
    UNION = "UNION"
    CUT = "CUT"
    INTERSECTION = "INTERSECTION"
    # Inspection
    MEASURE_DISTANCE = "MEASURE_DISTANCE"
    MEASURE_VOLUME = "MEASURE_VOLUME"
    MEASURE_AREA = "MEASURE_AREA"
    MEASURE_BOUNDING_BOX = "MEASURE_BOUNDING_BOX"
    CHECK_VALIDITY = "CHECK_VALIDITY"
    RENDER_VIEW = "RENDER_VIEW"
    # Termination
    FINISH_DESIGN = "FINISH_DESIGN"


#: Operation groups used by the hierarchical policy and masking.
SKETCH_OPS = {
    Operation.CREATE_SKETCH,
    Operation.ADD_LINE,
    Operation.ADD_RECTANGLE,
    Operation.ADD_CIRCLE,
    Operation.ADD_ARC,
    Operation.ADD_POLYGON,
    Operation.DELETE_GEOMETRY,
    Operation.MOVE_GEOMETRY,
}
CONSTRAINT_OPS = {
    Operation.ADD_HORIZONTAL,
    Operation.ADD_VERTICAL,
    Operation.ADD_PARALLEL,
    Operation.ADD_PERPENDICULAR,
    Operation.ADD_TANGENT,
    Operation.ADD_EQUAL,
    Operation.ADD_DISTANCE,
    Operation.ADD_RADIUS,
    Operation.ADD_DIAMETER,
    Operation.ADD_COINCIDENT,
    Operation.ADD_SYMMETRY,
}
FEATURE_OPS = {
    Operation.PAD,
    Operation.POCKET,
    Operation.REVOLVE,
    Operation.FILLET,
    Operation.CHAMFER,
    Operation.SHELL,
    Operation.MIRROR,
    Operation.LINEAR_PATTERN,
    Operation.CIRCULAR_PATTERN,
}
BOOLEAN_OPS = {Operation.UNION, Operation.CUT, Operation.INTERSECTION}
INSPECTION_OPS = {
    Operation.MEASURE_DISTANCE,
    Operation.MEASURE_VOLUME,
    Operation.MEASURE_AREA,
    Operation.MEASURE_BOUNDING_BOX,
    Operation.CHECK_VALIDITY,
    Operation.RENDER_VIEW,
}


@dataclass
class Action:
    """One structured CAD action emitted by a policy or expert."""

    operation: Operation
    target: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.operation, Operation):
            self.operation = Operation(str(self.operation))
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        self.confidence = float(self.confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation.value,
            "target": self.target,
            "parameters": dict(self.parameters),
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Action:
        return cls(
            operation=Operation(data["operation"]),
            target=data.get("target"),
            parameters=dict(data.get("parameters", {})),
            confidence=float(data.get("confidence", 1.0)),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> Action:
        return cls.from_dict(json.loads(text))


@dataclass
class ActionResult:
    """Outcome of executing one action against the engine."""

    ok: bool
    operation: Operation
    message: str = ""
    info: dict[str, Any] = field(default_factory=dict)
    done: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "operation": self.operation.value,
            "message": self.message,
            "info": dict(self.info),
            "done": self.done,
        }
