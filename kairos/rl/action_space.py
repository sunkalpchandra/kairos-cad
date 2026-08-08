"""Action-space codec: policy outputs <-> structured CAD actions.

The policy emits (operation_index, params in [0,1]^PARAM_SLOTS, target_index).
decode() denormalizes each slot into the operation's range and returns a
validated Action; encode() maps expert Actions back for behavioral cloning.

Slot ranges are frozen here and policies and BC datasets must agree on them.
Changing one invalidates every trained checkpoint, since the parameter head is
calibrated to the span it trained against.

Encoding a value outside its range raises rather than clipping: a clipped value
is a different action that still executes cleanly and reports ok, so it corrupts
geometry with nothing recording that it happened.
"""

from __future__ import annotations

import math

import numpy as np

from kairos.actions.schema import Action, Operation

#: Decimal places decode keeps on a millimetre value.
#:
#: Was 3, which is 5e-4 mm of quantization: five thousand times FreeCAD's
#: Precision::Confusion (1e-7 mm). That is invisible on a standalone build,
#: where every coordinate is rounded the same way, and fatal across a
#: COMPLETE(k) seam: the benchmark replays the expert prefix at RAW precision
#: and the policy supplies the suffix through the codec, so a line chain does
#: not close and the following REVOLVE fails. All 8 oracle failures in the
#: 76-task run were ADD_LINE chains followed by a failed REVOLVE.
#:
#: 6 would leave 5e-7, still above Confusion. 9 leaves 5e-10, and float64 over
#: a 300 mm span resolves ~1e-13, so this costs nothing.
MM_DECIMALS = 9
#: Angles in degrees; a coarser quantum is harmless and keeps them readable.
DEG_DECIMALS = 6

#: Number of continuous parameter slots the policy emits.
PARAM_SLOTS = 6
#: Cap on enumerable targets (edges/faces/features) the codec will index.
MAX_TARGETS = 64

OPERATIONS: tuple[Operation, ...] = tuple(Operation)
NUM_OPERATIONS = len(OPERATIONS)

_PLANES = ("XY", "XZ", "YZ")
_AXES = ("X", "Y", "Z")
_VIEWS = ("iso", "front", "top", "right")

# Slot ranges, shared by encode and decode. Keeping them as literals in both
# directions is how they drifted below the data in the first place.
#
# Widen only what the data proves is too narrow. Generous headroom is not free:
# widening every range (_SMALL to 0.1-50, _RADIUS to 0.1-100) put a mid-range
# fillet at 25 mm instead of 5 mm, and PPO's invalid-action rate went 0.001 ->
# 0.422 with BC validity 0.957 -> 0.485. Normalized parameter MAE *improved*
# over the same change (0.035 -> 0.023) while absolute mm error got worse.
#
# Anything outside a range raises in `_inv` rather than clipping, so the next
# overflow shows up as a failed audit. A per-operation or relative encoding
# would remove the range/precision tradeoff entirely.

#: Sketch-plane coordinates (mm). Data spans -34.8 .. 122.9; was +-100, which
#: clipped 142 circle centres.
_COORD = (-150.0, 150.0)
#: Sketch-plane offset along the normal (mm). Data 3.0 .. 89.9; was +-50, which
#: clipped 110 sketches.
_OFFSET = (-120.0, 120.0)
#: Rectangle side lengths (mm). Data 4.1 .. 129.9. Never clipped; unchanged.
_SIDE = (1.0, 150.0)
#: Circle and arc radii (mm). Data 1.5 .. 4.0, i.e. 3-8 mm fastener holes.
#: A (0.5, 25) span put ten times more resolution than needed behind a slot the
#: benchmark shows is decisive: every policy dies at the holes -> constraints
#: rung, and hole diameter is what that check reads. 2-16 mm covers every
#: family with headroom; anything larger raises and gets widened deliberately.
_RADIUS = (1.0, 8.0)
#: Regular-polygon circumradius (mm). Never clipped; unchanged.
_POLY_RADIUS = (1.0, 100.0)
#: Pad / revolve extrusion length (mm). Data 3.1 .. 62.5. Never clipped.
_LENGTH = (1.0, 100.0)
#: Pocket depth (mm). Never clipped; unchanged.
_DEPTH = (1.0, 50.0)
#: Fillet radius, chamfer leg, shell thickness (mm). Data 0.404 .. 3.96; the old
#: 0.5 floor clipped 8 chamfers upward, so only the floor moved.
_SMALL = (0.25, 10.0)
#: Sketch-constraint dimension value (mm). Never clipped; unchanged.
_DIMENSION = (0.5, 150.0)
#: Linear-pattern span (mm). Never clipped; unchanged.
_SPAN = (5.0, 150.0)


def _lin(x: float, lo: float, hi: float) -> float:
    return lo + float(np.clip(x, 0.0, 1.0)) * (hi - lo)


def _inv(v: float, lo: float, hi: float) -> float:
    """Normalize a value into its slot range, raising rather than clipping.

    A sketch offset of 89.9 mm clipped to a 50 mm range decodes back as 50 mm,
    so replaying the expert builds the feature 40 mm off with ok=True on every
    step. Raising makes audit_codec.py count it and BC drop the step.
    """
    if hi == lo:
        return 0.0
    # Tolerance covers float noise at the boundary, not genuine overflow.
    if not (lo - 1e-9) <= v <= (hi + 1e-9):
        raise UnrepresentableAction(
            f"value {v:.4f} is outside the codec's [{lo}, {hi}] slot range; "
            "clipping it would encode a different action, so widen the range "
            "in action_space.py if this geometry is legitimate"
        )
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def _int_slot(x: float, count: int) -> int:
    """Decode a slot into one of `count` integer values, 0..count-1.

    `int(_lin(x, 0, n - 0.001))` looks equivalent and is not: the inverse
    `_inv(v, 0, n - 0.001)` maps v to v/(n - 0.001), and truncation then lands
    a value short. Index 7 of 12 came back as 6, executed, and reported ok
    while referring to different geometry. This is the bucketing `_choice`
    already gets right, so it uses the same scheme.
    """
    return min(int(np.clip(x, 0.0, 0.9999) * count), count - 1)


def _int_slot_inv(value: int, count: int) -> float:
    """Midpoint of the bucket, so the round trip is exact."""
    index = min(max(int(value), 0), count - 1)
    return (index + 0.5) / count


def _choice(x: float, options: tuple) -> object:
    index = min(int(np.clip(x, 0.0, 0.9999) * len(options)), len(options) - 1)
    return options[index]


def _choice_inv(value: object, options: tuple) -> float:
    index = options.index(value)
    return (index + 0.5) / len(options)


class UnrepresentableAction(ValueError):
    """An expert Action the codec's fixed parameter slots cannot express."""


def _fit_regular_polygon(points: list) -> tuple[float, float, float, int, float]:
    """Recover (cx, cy, radius, sides, rotation_deg) from a regular polygon.

    Raises ``UnrepresentableAction`` for irregular profiles, BC pipelines must
    expand those into ADD_LINE actions rather than train toward a target the
    codec would decode into a different shape.
    """
    n = len(points)
    if not 3 <= n <= 8:
        raise UnrepresentableAction(
            f"ADD_POLYGON with {n} vertices is outside the codec's 3-8 regular "
            "n-gon range; expand it into ADD_LINE actions"
        )
    cx = sum(pt[0] for pt in points) / n
    cy = sum(pt[1] for pt in points) / n
    radii = [math.hypot(pt[0] - cx, pt[1] - cy) for pt in points]
    angles = [math.atan2(pt[1] - cy, pt[0] - cx) for pt in points]
    radius = sum(radii) / n
    step = 2 * math.pi / n
    regular = radius > 0 and all(abs(r - radius) <= 1e-3 * radius for r in radii)
    if regular:
        for i, angle in enumerate(angles):
            drift = (angle - angles[0] - i * step + math.pi) % (2 * math.pi) - math.pi
            if abs(drift) > 1e-3:
                regular = False
                break
    if not regular:
        raise UnrepresentableAction(
            "ADD_POLYGON is only representable as a regular n-gon; expand this "
            "irregular profile into ADD_LINE actions"
        )
    return cx, cy, radius, n, math.degrees(angles[0]) % 360.0


TARGET_KIND: dict[Operation, str] = {
    Operation.FILLET: "edges",
    Operation.CHAMFER: "edges",
    Operation.SHELL: "faces",
    Operation.MIRROR: "features",
    Operation.LINEAR_PATTERN: "features",
    Operation.CIRCULAR_PATTERN: "features",
    # Booleans are multi-body (Phase 6); decoding still resolves a feature
    # target so the action validates and the executor's "not executable"
    # failure, not a validation crash, is what the policy experiences.
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
            emitted with target=None and will fail validation downstream. That is the
            correct penalty path, not an exception here.
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
        parameters = {"plane": _choice(p[0], _PLANES), "offset": round(_lin(p[1], *_OFFSET), MM_DECIMALS)}
    elif op is Operation.ADD_LINE:
        parameters = {
            "x1": round(_lin(p[0], *_COORD), MM_DECIMALS), "y1": round(_lin(p[1], *_COORD), MM_DECIMALS),
            "x2": round(_lin(p[2], *_COORD), MM_DECIMALS), "y2": round(_lin(p[3], *_COORD), MM_DECIMALS),
        }
    elif op is Operation.ADD_RECTANGLE:
        parameters = {
            "x": round(_lin(p[0], *_COORD), MM_DECIMALS), "y": round(_lin(p[1], *_COORD), MM_DECIMALS),
            "width": round(_lin(p[2], *_SIDE), MM_DECIMALS), "height": round(_lin(p[3], *_SIDE), MM_DECIMALS),
        }
    elif op is Operation.ADD_CIRCLE:
        parameters = {
            "cx": round(_lin(p[0], *_COORD), MM_DECIMALS), "cy": round(_lin(p[1], *_COORD), MM_DECIMALS),
            "radius": round(_lin(p[2], *_RADIUS), MM_DECIMALS),
        }
    elif op is Operation.ADD_ARC:
        parameters = {
            "cx": round(_lin(p[0], *_COORD), MM_DECIMALS), "cy": round(_lin(p[1], *_COORD), MM_DECIMALS),
            "radius": round(_lin(p[2], *_RADIUS), MM_DECIMALS),
            "start_deg": round(_lin(p[3], 0, 360), DEG_DECIMALS), "end_deg": round(_lin(p[4], 0, 360), DEG_DECIMALS),
        }
    elif op is Operation.ADD_POLYGON:
        # A regular n-gon: the codec's continuous slots cannot express an
        # arbitrary vertex list, and emitting an empty one made this op a
        # guaranteed executor failure. Irregular profiles (the L and U family
        # recipes) are built from ADD_LINE instead.
        cx, cy = _lin(p[0], *_COORD), _lin(p[1], *_COORD)
        radius = _lin(p[2], *_POLY_RADIUS)
        sides = int(_lin(p[3], 3, 8.999))
        rotation = math.radians(_lin(p[4], 0, 360))
        parameters = {
            "points": [
                [
                    round(cx + radius * math.cos(rotation + 2 * math.pi * i / sides),
                          MM_DECIMALS),
                    round(cy + radius * math.sin(rotation + 2 * math.pi * i / sides),
                          MM_DECIMALS),
                ]
                for i in range(sides)
            ]
        }
    elif op is Operation.DELETE_GEOMETRY:
        parameters = {"index": _int_slot(p[0], 12)}
    elif op is Operation.MOVE_GEOMETRY:
        parameters = {
            "index": _int_slot(p[0], 12),
            "dx": round(_lin(p[1], -20, 20), MM_DECIMALS), "dy": round(_lin(p[2], -20, 20), MM_DECIMALS),
        }
    elif op in (Operation.ADD_HORIZONTAL, Operation.ADD_VERTICAL):
        parameters = {"geo": _int_slot(p[0], 12)}
    elif op in (
        Operation.ADD_PARALLEL, Operation.ADD_PERPENDICULAR,
        Operation.ADD_TANGENT, Operation.ADD_EQUAL,
    ):
        parameters = {"geo1": _int_slot(p[0], 12), "geo2": _int_slot(p[1], 12)}
    elif op is Operation.ADD_DISTANCE:
        parameters = {
            "geo1": _int_slot(p[0], 12), "pos1": _int_slot(p[1], 4),
            "geo2": _int_slot(p[2], 12), "pos2": _int_slot(p[3], 4),
            "value": round(_lin(p[4], *_DIMENSION), MM_DECIMALS),
        }
    elif op in (Operation.ADD_RADIUS, Operation.ADD_DIAMETER):
        parameters = {"geo": _int_slot(p[0], 12), "value": round(_lin(p[1], *_SMALL), MM_DECIMALS)}
    elif op is Operation.ADD_COINCIDENT:
        parameters = {
            "geo1": _int_slot(p[0], 12), "pos1": 1 + _int_slot(p[1], 3),
            "geo2": _int_slot(p[2], 12), "pos2": 1 + _int_slot(p[3], 3),
        }
    elif op is Operation.ADD_SYMMETRY:
        parameters = {
            "geo1": _int_slot(p[0], 12), "pos1": 1 + _int_slot(p[1], 3),
            "geo2": _int_slot(p[2], 12), "pos2": 1 + _int_slot(p[3], 3),
            "axis_geo": _int_slot(p[4], 12),
        }
    elif op is Operation.PAD:
        parameters = {
            "length": round(_lin(p[0], *_LENGTH), MM_DECIMALS),
            "reversed": bool(p[1] > 0.5), "midplane": bool(p[2] > 0.5),
        }
    elif op is Operation.POCKET:
        through_all = bool(p[0] > 0.5)
        parameters = {"through_all": through_all, "reversed": bool(p[2] > 0.5)}
        if not through_all:
            parameters["depth"] = round(_lin(p[1], *_DEPTH), MM_DECIMALS)
    elif op is Operation.REVOLVE:
        parameters = {
            "angle": round(_lin(p[0], 10, 360), DEG_DECIMALS),
            "axis": _choice(p[1], ("V", "H")),
        }
    elif op is Operation.FILLET:
        parameters = {"radius": round(_lin(p[0], *_SMALL), MM_DECIMALS)}
    elif op is Operation.CHAMFER:
        parameters = {"size": round(_lin(p[0], *_SMALL), MM_DECIMALS)}
    elif op is Operation.SHELL:
        parameters = {"thickness": round(_lin(p[0], *_SMALL), MM_DECIMALS)}
    elif op is Operation.MIRROR:
        parameters = {"plane": _choice(p[0], _PLANES)}
    elif op is Operation.LINEAR_PATTERN:
        parameters = {
            "axis": _choice(p[0], _AXES),
            "length": round(_lin(p[1], *_SPAN), MM_DECIMALS),
            "count": 2 + _int_slot(p[2], 7),
        }
    elif op is Operation.CIRCULAR_PATTERN:
        parameters = {
            "axis": _choice(p[0], _AXES),
            "angle": round(_lin(p[1], 30, 360), 2),
            "count": 2 + _int_slot(p[2], 7),
        }
    elif op is Operation.MEASURE_DISTANCE:
        parameters = {"sub_a": "Face1", "sub_b": "Face2"}  # v0 placeholder pair
    elif op is Operation.RENDER_VIEW:
        parameters = {"view": _choice(p[0], _VIEWS)}
    else:
        # UNION/CUT/INTERSECTION (multi-body, Phase 6) and no-param inspection.
        parameters = {}

    return Action(operation=op, target=target, parameters=parameters)


def _target_index(action: Action, targets: dict[str, list[str]] | None) -> int:
    """Resolve a recorded target name to its index in the live pool.

    Without `targets` there is nothing to resolve against and this returns 0,
    which is what BC dataset building gets: target selection stays unsupervised
    because trajectories do not record the pool the expert chose from.

    Given a pool, an index of 0 is a real answer rather than a placeholder, and
    for the oracle it is the difference between chamfering the rim the expert
    chamfered and chamfering whichever edge happens to be listed first.
    """
    if not targets or not action.target:
        return 0
    kind = TARGET_KIND.get(action.operation)
    if kind is None:
        return 0
    pool = targets.get(kind) or []
    if not pool:
        return 0

    # Multi-target actions record their names comma-joined ("Edge3,Edge7"), and
    # the codec indexes exactly one. Taking the first at least applies the
    # feature to an edge the expert named; the rest are lost, which is a codec
    # limit worth seeing in the ceiling rather than hiding behind index 0.
    first = str(action.target).split(",")[0].strip()
    try:
        index = pool.index(first)
    except ValueError:
        return 0
    if index >= MAX_TARGETS:
        raise UnrepresentableAction(
            f"target {first!r} sits at index {index}, beyond the codec's "
            f"{MAX_TARGETS}-target limit; decode would wrap to a different one"
        )
    return index


def encode(
    action: Action, targets: dict[str, list[str]] | None = None
) -> tuple[int, np.ndarray, int]:
    """Inverse of ``decode``, for behavioral cloning targets and oracle replay.

    Returns (operation_index, params, target_index). Continuous values are
    normalized back into their slot ranges; anything outside one raises.

    `targets` is the live pool by kind, as ``decode`` takes it. Pass it when the
    caller has an engine to ask, and the recorded target name resolves to its
    index. Omit it and the index is 0.
    """
    op = action.operation
    p = np.zeros(PARAM_SLOTS, dtype=np.float64)
    prm = action.parameters

    if op is Operation.CREATE_SKETCH:
        p[0] = _choice_inv(prm.get("plane", "XY"), _PLANES)
        p[1] = _inv(prm.get("offset", 0.0), *_OFFSET)
    elif op is Operation.ADD_LINE:
        p[0] = _inv(prm["x1"], *_COORD)
        p[1] = _inv(prm["y1"], *_COORD)
        p[2] = _inv(prm["x2"], *_COORD)
        p[3] = _inv(prm["y2"], *_COORD)
    elif op is Operation.ADD_RECTANGLE:
        p[0] = _inv(prm["x"], *_COORD)
        p[1] = _inv(prm["y"], *_COORD)
        p[2] = _inv(prm["width"], *_SIDE)
        p[3] = _inv(prm["height"], *_SIDE)
    elif op is Operation.ADD_CIRCLE:
        p[0] = _inv(prm["cx"], *_COORD)
        p[1] = _inv(prm["cy"], *_COORD)
        p[2] = _inv(prm["radius"], *_RADIUS)
    elif op is Operation.ADD_POLYGON:
        cx, cy, radius, sides, rotation = _fit_regular_polygon(prm.get("points") or [])
        p[0] = _inv(cx, *_COORD)
        p[1] = _inv(cy, *_COORD)
        p[2] = _inv(radius, *_POLY_RADIUS)
        p[3] = _inv(sides + 0.5, 3, 8.999)
        p[4] = _inv(rotation, 0, 360)
    elif op is Operation.ADD_ARC:
        p[0] = _inv(prm["cx"], *_COORD)
        p[1] = _inv(prm["cy"], *_COORD)
        p[2] = _inv(prm["radius"], *_RADIUS)
        p[3] = _inv(prm.get("start_deg", 0.0), 0, 360)
        p[4] = _inv(prm.get("end_deg", 360.0), 0, 360)
    elif op is Operation.DELETE_GEOMETRY:
        p[0] = _int_slot_inv(prm["index"], 12)
    elif op is Operation.MOVE_GEOMETRY:
        p[0] = _int_slot_inv(prm["index"], 12)
        p[1] = _inv(prm.get("dx", 0.0), -20, 20)
        p[2] = _inv(prm.get("dy", 0.0), -20, 20)
    elif op in (Operation.ADD_HORIZONTAL, Operation.ADD_VERTICAL):
        p[0] = _int_slot_inv(prm["geo"], 12)
    elif op in (
        Operation.ADD_PARALLEL, Operation.ADD_PERPENDICULAR,
        Operation.ADD_TANGENT, Operation.ADD_EQUAL,
    ):
        p[0] = _int_slot_inv(prm["geo1"], 12)
        p[1] = _int_slot_inv(prm["geo2"], 12)
    elif op is Operation.ADD_DISTANCE:
        p[0] = _int_slot_inv(prm["geo1"], 12)
        p[1] = _int_slot_inv(prm.get("pos1", 0), 4)
        p[2] = _int_slot_inv(prm["geo2"], 12)
        p[3] = _int_slot_inv(prm.get("pos2", 0), 4)
        p[4] = _inv(prm["value"], *_DIMENSION)
    elif op in (Operation.ADD_RADIUS, Operation.ADD_DIAMETER):
        p[0] = _int_slot_inv(prm["geo"], 12)
        p[1] = _inv(prm["value"], *_SMALL)
    elif op is Operation.ADD_COINCIDENT:
        p[0] = _int_slot_inv(prm["geo1"], 12)
        p[1] = _int_slot_inv(prm.get("pos1", 1) - 1, 3)
        p[2] = _int_slot_inv(prm["geo2"], 12)
        p[3] = _int_slot_inv(prm.get("pos2", 1) - 1, 3)
    elif op is Operation.ADD_SYMMETRY:
        p[0] = _int_slot_inv(prm["geo1"], 12)
        p[1] = _int_slot_inv(prm.get("pos1", 1) - 1, 3)
        p[2] = _int_slot_inv(prm["geo2"], 12)
        p[3] = _int_slot_inv(prm.get("pos2", 1) - 1, 3)
        p[4] = _int_slot_inv(prm["axis_geo"], 12)
    elif op is Operation.RENDER_VIEW:
        p[0] = _choice_inv(prm.get("view", "iso"), _VIEWS)
    elif op is Operation.PAD:
        p[0] = _inv(prm["length"], *_LENGTH)
        p[1] = 1.0 if prm.get("reversed") else 0.0
        p[2] = 1.0 if prm.get("midplane") else 0.0
    elif op is Operation.POCKET:
        p[0] = 1.0 if prm.get("through_all") else 0.0
        if "depth" in prm and prm["depth"] is not None:
            p[1] = _inv(prm["depth"], *_DEPTH)
        p[2] = 1.0 if prm.get("reversed") else 0.0
    elif op is Operation.REVOLVE:
        p[0] = _inv(prm.get("angle", 360.0), 10, 360)
        p[1] = _choice_inv(prm.get("axis", "V"), ("V", "H"))
    elif op is Operation.FILLET:
        p[0] = _inv(prm["radius"], *_SMALL)
    elif op is Operation.CHAMFER:
        p[0] = _inv(prm["size"], *_SMALL)
    elif op is Operation.SHELL:
        p[0] = _inv(prm["thickness"], *_SMALL)
    elif op is Operation.MIRROR:
        p[0] = _choice_inv(prm.get("plane", "XZ"), _PLANES)
    elif op is Operation.CIRCULAR_PATTERN:
        p[0] = _choice_inv(prm.get("axis", "Z"), _AXES)
        p[1] = _inv(prm.get("angle", 360.0), 30, 360)
        p[2] = _int_slot_inv(prm["count"] - 2, 7)
    elif op is Operation.LINEAR_PATTERN:
        p[0] = _choice_inv(prm.get("axis", "X"), _AXES)
        p[1] = _inv(prm.get("length", 50.0), *_SPAN)
        p[2] = _int_slot_inv(prm["count"] - 2, 7)
    # Every remaining operation genuinely carries no continuous parameters.
    # This used to be a catch-all that silently returned zeros for sixteen
    # operations decode() fully supports, and _slots_used_by_operation()
    # probes the *decoder*, so BC would have trained those slots toward the
    # all-zero encoding rather than the expert's value. Latent only because
    # no family emits them yet.

    return OPERATIONS.index(op), p, _target_index(action, targets)
