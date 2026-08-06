"""Action-space codec: policy outputs ↔ structured CAD actions.

The RL policy emits ``(operation_index, params ∈ [0,1]^PARAM_SLOTS,
target_index)``. This module deterministically decodes that into a validated
``Action`` — denormalizing each slot into the operation's documented range —
and encodes expert Actions back into the same representation for behavioral
cloning. The workspace envelope is ±100 mm.

Slot ranges are frozen here; policies and BC datasets must agree on them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kairos.actions.schema import Action, Operation

#: Number of continuous parameter slots the policy emits.
PARAM_SLOTS = 6
#: Cap on enumerable targets (edges/faces/features) the codec will index.
MAX_TARGETS = 64

OPERATIONS: tuple[Operation, ...] = tuple(Operation)
NUM_OPERATIONS = len(OPERATIONS)

_PLANES = ("XY", "XZ", "YZ")
_AXES = ("X", "Y", "Z")
_VIEWS = ("iso", "front", "top", "right")


def _lin(x: float, lo: float, hi: float) -> float:
    return lo + float(np.clip(x, 0.0, 1.0)) * (hi - lo)


def _inv(v: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.0
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def _choice(x: float, options: tuple) -> object:
    index = min(int(np.clip(x, 0.0, 0.9999) * len(options)), len(options) - 1)
    return options[index]


def _choice_inv(value: object, options: tuple) -> float:
    index = options.index(value)
    return (index + 0.5) / len(options)


@dataclass(frozen=True)
class DecodedTarget:
    """What kind of target list an operation indexes into."""

    kind: str | None  # 'edges' | 'faces' | 'features' | None


TARGET_KIND: dict[Operation, str] = {
    Operation.FILLET: "edges",
    Operation.CHAMFER: "edges",
    Operation.SHELL: "faces",
    Operation.MIRROR: "features",
    Operation.LINEAR_PATTERN: "features",
    Operation.CIRCULAR_PATTERN: "features",
    # Booleans are multi-body (Phase 6); decoding still resolves a feature
    # target so the action validates and the executor's "not executable"
    # failure — not a validation crash — is what the policy experiences.
    Operation.UNION: "features",
    Operation.CUT: "features",
    Operation.INTERSECTION: "features",
}


def decode(
    operation_index: int,
    params: np.ndarray,
    target_index: int = 0,
    targets: dict[str, list[str]] | None = None,
) -> Action:
    """Decode policy outputs into a structured Action.

    Args:
        operation_index: index into ``OPERATIONS``.
        params: [PARAM_SLOTS] floats in [0, 1].
        target_index: index into the operation's target list.
        targets: available targets by kind, e.g. ``{"edges": [...],
            "faces": [...], "features": [...]}`` from the live engine. If an
            operation needs a target and none are available, the Action is
            emitted with target=None and will fail validation downstream —
            that is the correct penalty path, not an exception here.
    """
    op = OPERATIONS[int(operation_index) % NUM_OPERATIONS]
    p = np.asarray(params, dtype=np.float64).reshape(-1)
    if p.size < PARAM_SLOTS:
        p = np.pad(p, (0, PARAM_SLOTS - p.size))

    target: str | None = None
    kind = TARGET_KIND.get(op)
    if kind is not None:
        pool = (targets or {}).get(kind, [])
        if pool:
            target = pool[int(target_index) % min(len(pool), MAX_TARGETS)]

    if op is Operation.CREATE_SKETCH:
        parameters = {"plane": _choice(p[0], _PLANES), "offset": round(_lin(p[1], -50, 50), 3)}
    elif op is Operation.ADD_LINE:
        parameters = {
            "x1": round(_lin(p[0], -100, 100), 3), "y1": round(_lin(p[1], -100, 100), 3),
            "x2": round(_lin(p[2], -100, 100), 3), "y2": round(_lin(p[3], -100, 100), 3),
        }
    elif op is Operation.ADD_RECTANGLE:
        parameters = {
            "x": round(_lin(p[0], -100, 100), 3), "y": round(_lin(p[1], -100, 100), 3),
            "width": round(_lin(p[2], 1, 150), 3), "height": round(_lin(p[3], 1, 150), 3),
        }
    elif op is Operation.ADD_CIRCLE:
        parameters = {
            "cx": round(_lin(p[0], -100, 100), 3), "cy": round(_lin(p[1], -100, 100), 3),
            "radius": round(_lin(p[2], 0.5, 25), 3),
        }
    elif op is Operation.ADD_ARC:
        parameters = {
            "cx": round(_lin(p[0], -100, 100), 3), "cy": round(_lin(p[1], -100, 100), 3),
            "radius": round(_lin(p[2], 0.5, 25), 3),
            "start_deg": round(_lin(p[3], 0, 360), 2), "end_deg": round(_lin(p[4], 0, 360), 2),
        }
    elif op is Operation.ADD_POLYGON:
        # v0: policies build polygons via ADD_LINE; expert data may still use it.
        parameters = {"points": []}
    elif op is Operation.DELETE_GEOMETRY:
        parameters = {"index": int(_lin(p[0], 0, 11.999))}
    elif op is Operation.MOVE_GEOMETRY:
        parameters = {
            "index": int(_lin(p[0], 0, 11.999)),
            "dx": round(_lin(p[1], -20, 20), 3), "dy": round(_lin(p[2], -20, 20), 3),
        }
    elif op in (Operation.ADD_HORIZONTAL, Operation.ADD_VERTICAL):
        parameters = {"geo": int(_lin(p[0], 0, 11.999))}
    elif op in (
        Operation.ADD_PARALLEL, Operation.ADD_PERPENDICULAR,
        Operation.ADD_TANGENT, Operation.ADD_EQUAL,
    ):
        parameters = {"geo1": int(_lin(p[0], 0, 11.999)), "geo2": int(_lin(p[1], 0, 11.999))}
    elif op is Operation.ADD_DISTANCE:
        parameters = {
            "geo1": int(_lin(p[0], 0, 11.999)), "pos1": int(_lin(p[1], 0, 3.999)),
            "geo2": int(_lin(p[2], 0, 11.999)), "pos2": int(_lin(p[3], 0, 3.999)),
            "value": round(_lin(p[4], 0.5, 150), 3),
        }
    elif op in (Operation.ADD_RADIUS, Operation.ADD_DIAMETER):
        parameters = {"geo": int(_lin(p[0], 0, 11.999)), "value": round(_lin(p[1], 0.5, 50), 3)}
    elif op is Operation.ADD_COINCIDENT:
        parameters = {
            "geo1": int(_lin(p[0], 0, 11.999)), "pos1": 1 + int(_lin(p[1], 0, 2.999)),
            "geo2": int(_lin(p[2], 0, 11.999)), "pos2": 1 + int(_lin(p[3], 0, 2.999)),
        }
    elif op is Operation.ADD_SYMMETRY:
        parameters = {
            "geo1": int(_lin(p[0], 0, 11.999)), "pos1": 1 + int(_lin(p[1], 0, 2.999)),
            "geo2": int(_lin(p[2], 0, 11.999)), "pos2": 1 + int(_lin(p[3], 0, 2.999)),
            "axis_geo": int(_lin(p[4], 0, 11.999)),
        }
    elif op is Operation.PAD:
        parameters = {
            "length": round(_lin(p[0], 1, 100), 3),
            "reversed": bool(p[1] > 0.5), "midplane": bool(p[2] > 0.5),
        }
    elif op is Operation.POCKET:
        through_all = bool(p[0] > 0.5)
        parameters = {"through_all": through_all, "reversed": bool(p[2] > 0.5)}
        if not through_all:
            parameters["depth"] = round(_lin(p[1], 1, 50), 3)
    elif op is Operation.REVOLVE:
        parameters = {
            "angle": round(_lin(p[0], 10, 360), 2),
            "axis": _choice(p[1], ("V", "H")),
        }
    elif op is Operation.FILLET:
        parameters = {"radius": round(_lin(p[0], 0.5, 10), 3)}
    elif op is Operation.CHAMFER:
        parameters = {"size": round(_lin(p[0], 0.5, 10), 3)}
    elif op is Operation.SHELL:
        parameters = {"thickness": round(_lin(p[0], 0.5, 10), 3)}
    elif op is Operation.MIRROR:
        parameters = {"plane": _choice(p[0], _PLANES)}
    elif op is Operation.LINEAR_PATTERN:
        parameters = {
            "axis": _choice(p[0], _AXES),
            "length": round(_lin(p[1], 5, 150), 3),
            "count": 2 + int(_lin(p[2], 0, 6.999)),
        }
    elif op is Operation.CIRCULAR_PATTERN:
        parameters = {
            "axis": _choice(p[0], _AXES),
            "angle": round(_lin(p[1], 30, 360), 2),
            "count": 2 + int(_lin(p[2], 0, 6.999)),
        }
    elif op is Operation.MEASURE_DISTANCE:
        parameters = {"sub_a": "Face1", "sub_b": "Face2"}  # v0 placeholder pair
    elif op is Operation.RENDER_VIEW:
        parameters = {"view": _choice(p[0], _VIEWS)}
    else:
        # UNION/CUT/INTERSECTION (multi-body, Phase 6) and no-param inspection.
        parameters = {}

    return Action(operation=op, target=target, parameters=parameters)


def encode(action: Action) -> tuple[int, np.ndarray, int]:
    """Best-effort inverse of ``decode`` for behavioral cloning targets.

    Returns (operation_index, params, target_index). Continuous values are
    normalized back into their slot ranges; values outside a range clip (BC
    loss then sees the boundary, which is the closest representable action).
    The target index cannot be recovered without the live target list; 0 is
    returned and BC treats target selection as unsupervised in v0.
    """
    op = action.operation
    p = np.zeros(PARAM_SLOTS, dtype=np.float64)
    prm = action.parameters

    if op is Operation.CREATE_SKETCH:
        p[0] = _choice_inv(prm.get("plane", "XY"), _PLANES)
        p[1] = _inv(prm.get("offset", 0.0), -50, 50)
    elif op is Operation.ADD_LINE:
        p[0] = _inv(prm["x1"], -100, 100)
        p[1] = _inv(prm["y1"], -100, 100)
        p[2] = _inv(prm["x2"], -100, 100)
        p[3] = _inv(prm["y2"], -100, 100)
    elif op is Operation.ADD_RECTANGLE:
        p[0] = _inv(prm["x"], -100, 100)
        p[1] = _inv(prm["y"], -100, 100)
        p[2] = _inv(prm["width"], 1, 150)
        p[3] = _inv(prm["height"], 1, 150)
    elif op is Operation.ADD_CIRCLE:
        p[0] = _inv(prm["cx"], -100, 100)
        p[1] = _inv(prm["cy"], -100, 100)
        p[2] = _inv(prm["radius"], 0.5, 25)
    elif op is Operation.PAD:
        p[0] = _inv(prm["length"], 1, 100)
        p[1] = 1.0 if prm.get("reversed") else 0.0
        p[2] = 1.0 if prm.get("midplane") else 0.0
    elif op is Operation.POCKET:
        p[0] = 1.0 if prm.get("through_all") else 0.0
        if "depth" in prm and prm["depth"] is not None:
            p[1] = _inv(prm["depth"], 1, 50)
        p[2] = 1.0 if prm.get("reversed") else 0.0
    elif op is Operation.REVOLVE:
        p[0] = _inv(prm.get("angle", 360.0), 10, 360)
        p[1] = _choice_inv(prm.get("axis", "V"), ("V", "H"))
    elif op is Operation.FILLET:
        p[0] = _inv(prm["radius"], 0.5, 10)
    elif op is Operation.CHAMFER:
        p[0] = _inv(prm["size"], 0.5, 10)
    elif op is Operation.SHELL:
        p[0] = _inv(prm["thickness"], 0.5, 10)
    elif op is Operation.MIRROR:
        p[0] = _choice_inv(prm.get("plane", "XZ"), _PLANES)
    elif op is Operation.CIRCULAR_PATTERN:
        p[0] = _choice_inv(prm.get("axis", "Z"), _AXES)
        p[1] = _inv(prm.get("angle", 360.0), 30, 360)
        p[2] = _inv(prm["count"] - 2, 0, 6.999)
    elif op is Operation.LINEAR_PATTERN:
        p[0] = _choice_inv(prm.get("axis", "X"), _AXES)
        p[1] = _inv(prm.get("length", 50.0), 5, 150)
        p[2] = _inv(prm["count"] - 2, 0, 6.999)
    # Remaining ops carry no continuous parameters worth inverting in v0.

    return OPERATIONS.index(op), p, 0
