"""Per-operation parameter specifications and validation.

This is the gate that makes the action interface *controlled*: every action
is checked against a typed spec (required params, types, ranges, choices)
before it may touch the CAD engine. Validation failures are cheap, typed,
and never mutate the document, the RL environment maps them to invalid
action penalties.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kairos.actions.schema import Action, Operation


class ActionValidationError(ValueError):
    """The action's target or parameters do not satisfy the operation spec."""


@dataclass(frozen=True)
class ParamSpec:
    """Specification of one parameter of one operation."""

    name: str
    type: type
    required: bool = True
    default: Any = None
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple | None = None


@dataclass(frozen=True)
class OperationSpec:
    """Full calling convention of an operation."""

    params: tuple[ParamSpec, ...] = ()
    #: What the ``target`` field must reference (None = target forbidden).
    target: str | None = None  # 'sketch' | 'edges' | 'faces' | 'features' | 'geometry'
    target_required: bool = False


_PLANES = ("XY", "XZ", "YZ")
_AXES = ("X", "Y", "Z")

OPERATION_SPECS: dict[Operation, OperationSpec] = {
    # ------------------------------------------------------------- sketching
    Operation.CREATE_SKETCH: OperationSpec(
        params=(
            ParamSpec("plane", str, required=False, default="XY", choices=_PLANES),
            ParamSpec("offset", float, required=False, default=0.0, minimum=-1e4, maximum=1e4),
        )
    ),
    Operation.ADD_LINE: OperationSpec(
        params=(
            ParamSpec("x1", float),
            ParamSpec("y1", float),
            ParamSpec("x2", float),
            ParamSpec("y2", float),
        ),
        target="sketch",
    ),
    Operation.ADD_RECTANGLE: OperationSpec(
        params=(
            ParamSpec("x", float),
            ParamSpec("y", float),
            ParamSpec("width", float, minimum=1e-3, maximum=1e4),
            ParamSpec("height", float, minimum=1e-3, maximum=1e4),
        ),
        target="sketch",
    ),
    Operation.ADD_CIRCLE: OperationSpec(
        params=(
            ParamSpec("cx", float),
            ParamSpec("cy", float),
            ParamSpec("radius", float, minimum=1e-3, maximum=1e4),
        ),
        target="sketch",
    ),
    Operation.ADD_ARC: OperationSpec(
        params=(
            ParamSpec("cx", float),
            ParamSpec("cy", float),
            ParamSpec("radius", float, minimum=1e-3, maximum=1e4),
            ParamSpec("start_deg", float, minimum=-360.0, maximum=360.0),
            ParamSpec("end_deg", float, minimum=-360.0, maximum=360.0),
        ),
        target="sketch",
    ),
    Operation.ADD_POLYGON: OperationSpec(
        params=(
            ParamSpec("points", list),
            ParamSpec("closed", bool, required=False, default=True),
        ),
        target="sketch",
    ),
    Operation.DELETE_GEOMETRY: OperationSpec(
        params=(ParamSpec("index", int, minimum=0),), target="sketch"
    ),
    Operation.MOVE_GEOMETRY: OperationSpec(
        params=(
            ParamSpec("index", int, minimum=0),
            ParamSpec("dx", float, minimum=-1e4, maximum=1e4),
            ParamSpec("dy", float, minimum=-1e4, maximum=1e4),
        ),
        target="sketch",
    ),
    # ----------------------------------------------------------- constraints
    Operation.ADD_HORIZONTAL: OperationSpec(
        params=(ParamSpec("geo", int, minimum=0),), target="sketch"
    ),
    Operation.ADD_VERTICAL: OperationSpec(
        params=(ParamSpec("geo", int, minimum=0),), target="sketch"
    ),
    Operation.ADD_PARALLEL: OperationSpec(
        params=(ParamSpec("geo1", int, minimum=0), ParamSpec("geo2", int, minimum=0)),
        target="sketch",
    ),
    Operation.ADD_PERPENDICULAR: OperationSpec(
        params=(ParamSpec("geo1", int, minimum=0), ParamSpec("geo2", int, minimum=0)),
        target="sketch",
    ),
    Operation.ADD_TANGENT: OperationSpec(
        params=(ParamSpec("geo1", int, minimum=0), ParamSpec("geo2", int, minimum=0)),
        target="sketch",
    ),
    Operation.ADD_EQUAL: OperationSpec(
        params=(ParamSpec("geo1", int, minimum=0), ParamSpec("geo2", int, minimum=0)),
        target="sketch",
    ),
    Operation.ADD_DISTANCE: OperationSpec(
        params=(
            ParamSpec("geo1", int, minimum=0),
            ParamSpec("pos1", int, minimum=0, maximum=3),
            ParamSpec("geo2", int, minimum=0),
            ParamSpec("pos2", int, minimum=0, maximum=3),
            ParamSpec("value", float, minimum=1e-3, maximum=1e4),
        ),
        target="sketch",
    ),
    Operation.ADD_RADIUS: OperationSpec(
        params=(
            ParamSpec("geo", int, minimum=0),
            ParamSpec("value", float, minimum=1e-3, maximum=1e4),
        ),
        target="sketch",
    ),
    Operation.ADD_DIAMETER: OperationSpec(
        params=(
            ParamSpec("geo", int, minimum=0),
            ParamSpec("value", float, minimum=1e-3, maximum=1e4),
        ),
        target="sketch",
    ),
    Operation.ADD_COINCIDENT: OperationSpec(
        params=(
            ParamSpec("geo1", int, minimum=0),
            ParamSpec("pos1", int, minimum=1, maximum=3),
            ParamSpec("geo2", int, minimum=0),
            ParamSpec("pos2", int, minimum=1, maximum=3),
        ),
        target="sketch",
    ),
    Operation.ADD_SYMMETRY: OperationSpec(
        params=(
            ParamSpec("geo1", int, minimum=0),
            ParamSpec("pos1", int, minimum=1, maximum=3),
            ParamSpec("geo2", int, minimum=0),
            ParamSpec("pos2", int, minimum=1, maximum=3),
            ParamSpec("axis_geo", int, minimum=0),
        ),
        target="sketch",
    ),
    # -------------------------------------------------------------- features
    Operation.PAD: OperationSpec(
        params=(
            ParamSpec("length", float, minimum=1e-3, maximum=1e4),
            ParamSpec("reversed", bool, required=False, default=False),
            ParamSpec("midplane", bool, required=False, default=False),
        ),
        target="sketch",
    ),
    Operation.POCKET: OperationSpec(
        params=(
            ParamSpec("depth", float, required=False, minimum=1e-3, maximum=1e4),
            ParamSpec("through_all", bool, required=False, default=False),
            ParamSpec("reversed", bool, required=False, default=False),
        ),
        target="sketch",
    ),
    Operation.REVOLVE: OperationSpec(
        params=(
            ParamSpec("angle", float, required=False, default=360.0, minimum=1e-3, maximum=360.0),
            ParamSpec("axis", str, required=False, default="V", choices=("V", "H")),
        ),
        target="sketch",
    ),
    Operation.FILLET: OperationSpec(
        params=(ParamSpec("radius", float, minimum=1e-3, maximum=1e3),),
        target="edges",
        target_required=True,
    ),
    Operation.CHAMFER: OperationSpec(
        params=(ParamSpec("size", float, minimum=1e-3, maximum=1e3),),
        target="edges",
        target_required=True,
    ),
    Operation.SHELL: OperationSpec(
        params=(ParamSpec("thickness", float, minimum=1e-3, maximum=1e3),),
        target="faces",
        target_required=True,
    ),
    Operation.MIRROR: OperationSpec(
        params=(ParamSpec("plane", str, required=False, default="XZ", choices=_PLANES),),
        target="features",
        target_required=True,
    ),
    Operation.LINEAR_PATTERN: OperationSpec(
        params=(
            ParamSpec("axis", str, choices=_AXES),
            ParamSpec("length", float, minimum=1e-3, maximum=1e4),
            ParamSpec("count", int, minimum=2, maximum=100),
        ),
        target="features",
        target_required=True,
    ),
    Operation.CIRCULAR_PATTERN: OperationSpec(
        params=(
            ParamSpec("axis", str, choices=_AXES),
            ParamSpec("angle", float, required=False, default=360.0, minimum=1e-3, maximum=360.0),
            ParamSpec("count", int, minimum=2, maximum=100),
        ),
        target="features",
        target_required=True,
    ),
    # --------------------------------------------------------------- boolean
    Operation.UNION: OperationSpec(target="features", target_required=True),
    Operation.CUT: OperationSpec(target="features", target_required=True),
    Operation.INTERSECTION: OperationSpec(target="features", target_required=True),
    # ------------------------------------------------------------ inspection
    Operation.MEASURE_DISTANCE: OperationSpec(
        params=(ParamSpec("sub_a", str), ParamSpec("sub_b", str))
    ),
    Operation.MEASURE_VOLUME: OperationSpec(),
    Operation.MEASURE_AREA: OperationSpec(),
    Operation.MEASURE_BOUNDING_BOX: OperationSpec(),
    Operation.CHECK_VALIDITY: OperationSpec(),
    Operation.RENDER_VIEW: OperationSpec(
        params=(
            ParamSpec("view", str, required=False, default="iso", choices=("iso", "front", "top", "right")),
            ParamSpec("size", int, required=False, default=512, minimum=64, maximum=2048),
        )
    ),
    # ----------------------------------------------------------- termination
    Operation.FINISH_DESIGN: OperationSpec(),
}


def _coerce(spec: ParamSpec, value: Any) -> Any:
    if spec.type is float and isinstance(value, (int, float)) and not isinstance(value, bool):
        value = float(value)
    if spec.type is int and isinstance(value, float) and value.is_integer():
        value = int(value)
    if not isinstance(value, spec.type):
        raise ActionValidationError(
            f"parameter {spec.name!r} must be {spec.type.__name__}, "
            f"got {type(value).__name__} ({value!r})"
        )
    if spec.minimum is not None and value < spec.minimum:
        raise ActionValidationError(
            f"parameter {spec.name!r}={value} below minimum {spec.minimum}"
        )
    if spec.maximum is not None and value > spec.maximum:
        raise ActionValidationError(
            f"parameter {spec.name!r}={value} above maximum {spec.maximum}"
        )
    if spec.choices is not None and value not in spec.choices:
        raise ActionValidationError(
            f"parameter {spec.name!r}={value!r} not in {spec.choices}"
        )
    return value


def validate_action(action: Action) -> dict[str, Any]:
    """Validate an action; returns normalized parameters (defaults applied).

    Raises:
        ActionValidationError: unknown operation, bad target, missing/unknown
            parameters, wrong types, or out-of-range values.
    """
    spec = OPERATION_SPECS.get(action.operation)
    if spec is None:
        raise ActionValidationError(f"unknown operation {action.operation!r}")

    if spec.target is None and action.target is not None:
        raise ActionValidationError(
            f"{action.operation.value} takes no target, got {action.target!r}"
        )
    if spec.target_required and not action.target:
        raise ActionValidationError(
            f"{action.operation.value} requires a target referencing {spec.target}"
        )

    known = {p.name for p in spec.params}
    unknown = set(action.parameters) - known
    if unknown:
        raise ActionValidationError(
            f"{action.operation.value} got unknown parameters {sorted(unknown)}; "
            f"accepts {sorted(known)}"
        )

    normalized: dict[str, Any] = {}
    for param in spec.params:
        if param.name in action.parameters:
            normalized[param.name] = _coerce(param, action.parameters[param.name])
        elif param.required:
            raise ActionValidationError(
                f"{action.operation.value} missing required parameter {param.name!r}"
            )
        elif param.default is not None or param.name in ("depth",):
            if param.default is not None:
                normalized[param.name] = param.default
    return normalized
