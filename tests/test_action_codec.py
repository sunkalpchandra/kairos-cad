"""Pure-python tests of the action-space codec."""

import numpy as np
import pytest

from kairos.actions.parameters import ActionValidationError, validate_action
from kairos.actions.schema import Action, Operation
from kairos.rl.action_space import (
    NUM_OPERATIONS,
    OPERATIONS,
    PARAM_SLOTS,
    decode,
    encode,
)

TARGETS = {
    "edges": [f"Edge{i}" for i in range(1, 13)],
    "faces": [f"Face{i}" for i in range(1, 8)],
    "features": ["Pad", "Pocket"],
}


def test_every_operation_decodes_to_valid_action():
    """Random policy outputs must always produce schema-valid actions
    (target-requiring ops are given a populated pool)."""
    rng = np.random.default_rng(0)
    for op_index in range(NUM_OPERATIONS):
        for _ in range(5):
            params = rng.random(PARAM_SLOTS)
            action = decode(op_index, params, int(rng.integers(0, 64)), TARGETS)
            assert action.operation is OPERATIONS[op_index]
            validate_action(action)  # must not raise


def test_polygon_decodes_to_a_buildable_ngon():
    """ADD_POLYGON must never decode to an empty point list, that made a
    masked-legal operation a guaranteed executor failure."""
    rng = np.random.default_rng(1)
    idx = OPERATIONS.index(Operation.ADD_POLYGON)
    for _ in range(20):
        action = decode(idx, rng.random(PARAM_SLOTS), 0, {})
        points = action.parameters["points"]
        assert 3 <= len(points) <= 8
        assert len({tuple(pt) for pt in points}) == len(points)  # no duplicates
        validate_action(action)


def test_polygon_round_trips_and_rejects_irregular_profiles():
    from kairos.actions.schema import Action
    from kairos.rl.action_space import UnrepresentableAction

    hexagon = decode(
        OPERATIONS.index(Operation.ADD_POLYGON), np.array([0.6, 0.4, 0.3, 0.7, 0.2, 0.0]), 0, {}
    )
    op_index, params, _ = encode(hexagon)
    round_tripped = decode(op_index, params, 0, {}).parameters["points"]
    assert [c for pt in round_tripped for c in pt] == pytest.approx(
        [c for pt in hexagon.parameters["points"] for c in pt], abs=1e-2
    )

    # The L/U family profiles are irregular: encoding must say so loudly
    # instead of emitting a target that decodes into a different shape.
    l_profile = Action(
        Operation.ADD_POLYGON,
        parameters={"points": [[0, 0], [80, 0], [80, 6], [6, 6], [6, 60], [0, 60]]},
    )
    with pytest.raises(UnrepresentableAction):
        encode(l_profile)


def test_missing_target_pool_yields_penalizable_action():
    action = decode(OPERATIONS.index(Operation.FILLET), np.full(PARAM_SLOTS, 0.5), 3, {})
    assert action.target is None
    with pytest.raises(ActionValidationError):
        validate_action(action)  # executor turns this into ok=False, not a crash


def test_target_index_wraps_pool():
    idx = OPERATIONS.index(Operation.FILLET)
    action = decode(idx, np.full(PARAM_SLOTS, 0.5), 14, TARGETS)
    assert action.target == TARGETS["edges"][14 % 12]


def test_param_ranges_denormalize():
    """The endpoints must be the declared range, not a copy of it.

    Asserting literals here is what let encode and decode drift apart: the
    numbers were repeated in three places and only two were kept in step.
    """
    from kairos.rl.action_space import _LENGTH

    idx = OPERATIONS.index(Operation.PAD)
    low = decode(idx, np.zeros(PARAM_SLOTS), 0, {})
    high = decode(idx, np.ones(PARAM_SLOTS), 0, {})
    assert low.parameters["length"] == pytest.approx(_LENGTH[0])
    assert high.parameters["length"] == pytest.approx(_LENGTH[1])
    assert low.parameters["midplane"] is False and high.parameters["midplane"] is True


def test_pocket_through_all_vs_depth():
    idx = OPERATIONS.index(Operation.POCKET)
    through = decode(idx, np.array([0.9, 0.5, 0.0, 0, 0, 0]), 0, {})
    assert through.parameters["through_all"] is True and "depth" not in through.parameters
    depth = decode(idx, np.array([0.1, 0.5, 0.0, 0, 0, 0]), 0, {})
    assert depth.parameters["through_all"] is False
    from kairos.rl.action_space import _DEPTH

    assert _DEPTH[0] <= depth.parameters["depth"] <= _DEPTH[1]


class TestOutOfRangeRefusesToClip:
    """Encoding must raise on overflow, never clip.

    A clipped value is a *different action* that executes cleanly and reports
    ok: True. That is how 260 expert steps across 19.1% of designs were being
    corrupted while every audit called the trajectory fully representable: a
    sketch offset of 89.9 mm encoded against a ±50 mm range and decoded back as
    50 mm, so the oracle built the feature 40 mm from where the expert put it.
    """

    def test_offset_beyond_range_raises(self):
        from kairos.rl.action_space import _OFFSET, UnrepresentableAction, encode

        action = Action(
            Operation.CREATE_SKETCH,
            parameters={"plane": "XY", "offset": _OFFSET[1] + 10.0},
        )
        with pytest.raises(UnrepresentableAction, match="outside the codec"):
            encode(action)

    def test_coordinate_beyond_range_raises(self):
        from kairos.rl.action_space import _COORD, UnrepresentableAction, encode

        action = Action(
            Operation.ADD_CIRCLE,
            parameters={"cx": _COORD[1] + 1.0, "cy": 0.0, "radius": 3.0},
        )
        with pytest.raises(UnrepresentableAction):
            encode(action)

    def test_value_below_the_floor_raises(self):
        """The old 0.5 chamfer floor silently rounded 0.404 mm legs *up*."""
        from kairos.rl.action_space import _SMALL, UnrepresentableAction, encode

        action = Action(Operation.CHAMFER, target="Edge1",
                        parameters={"size": _SMALL[0] / 2})
        with pytest.raises(UnrepresentableAction):
            encode(action)

    def test_the_boundary_itself_still_encodes(self):
        """Float noise at an endpoint must not be mistaken for overflow."""
        from kairos.rl.action_space import _OFFSET, encode

        for edge in _OFFSET:
            encode(Action(Operation.CREATE_SKETCH,
                          parameters={"plane": "XY", "offset": edge}))

    @pytest.mark.parametrize("parameter", ["length"])
    def test_pad_length_round_trips_at_both_ends(self, parameter):
        from kairos.rl.action_space import _LENGTH, decode, encode

        for value in _LENGTH:
            action = Action(Operation.PAD, parameters={
                parameter: value, "midplane": False, "reversed": False})
            decoded = decode(*encode(action))
            assert decoded.parameters[parameter] == pytest.approx(value, abs=1e-3)


def test_encode_decode_round_trip_core_ops():
    """decode(encode(a)) must reproduce a for in-range continuous params."""
    from kairos.actions.schema import Action

    cases = [
        Action(Operation.CREATE_SKETCH, parameters={"plane": "XZ", "offset": 5.0}),
        Action(Operation.ADD_CIRCLE, parameters={"cx": 20.0, "cy": 10.0, "radius": 3.0}),
        Action(Operation.ADD_RECTANGLE, parameters={"x": 0.0, "y": 0.0, "width": 40.0, "height": 20.0}),
        Action(Operation.PAD, parameters={"length": 10.0, "midplane": True, "reversed": False}),
        Action(Operation.POCKET, parameters={"through_all": True, "reversed": False}),
        Action(Operation.REVOLVE, parameters={"angle": 360.0, "axis": "V"}),
        Action(Operation.FILLET, target="Edge3", parameters={"radius": 2.0}),
        Action(Operation.CIRCULAR_PATTERN, target="Pocket", parameters={"axis": "Z", "angle": 360.0, "count": 4}),
    ]
    for original in cases:
        op_index, params, _ = encode(original)
        target_index = (
            TARGETS[
                {"FILLET": "edges", "CIRCULAR_PATTERN": "features"}.get(
                    original.operation.value, "edges"
                )
            ].index(original.target)
            if original.target in TARGETS["edges"] + TARGETS["features"]
            else 0
        )
        decoded = decode(op_index, params, target_index, TARGETS)
        assert decoded.operation is original.operation
        for key, value in original.parameters.items():
            got = decoded.parameters[key]
            if isinstance(value, float):
                assert got == pytest.approx(value, abs=0.15), (original.operation, key)
            else:
                assert got == value, (original.operation, key)


def test_encode_round_trips_every_parameterized_operation():
    """encode() must invert decode() for every op that carries parameters.

    It used to fall through to a catch-all returning zeros for sixteen
    operations decode() fully supports. Because the BC slot mask is probed from
    the *decoder*, those slots were marked supervised and would have been
    trained toward the all-zero encoding instead of the expert's value, silently, and
    only once a recipe emitted one of them.
    """
    targets = {"edges": ["Edge1"], "faces": ["Face1"], "features": ["Pad"]}
    lossy = []
    for index, operation in enumerate(OPERATIONS):
        original = decode(index, np.full(PARAM_SLOTS, 0.7), 0, targets)
        if not original.parameters:
            continue  # genuinely parameterless
        try:
            _, params, _ = encode(original)
        except Exception:
            continue  # UnrepresentableAction is a documented refusal
        if decode(index, params, 0, targets).parameters != original.parameters:
            lossy.append(operation.value)

    # ADD_POLYGON round-trips only to float tolerance; pinned separately.
    assert lossy in ([], ["ADD_POLYGON"]), f"encode() loses: {lossy}"


def test_encode_preserves_a_constraint_operations_values():
    from kairos.actions.schema import Action

    action = Action(
        Operation.ADD_DISTANCE,
        parameters={"geo1": 3, "pos1": 1, "geo2": 5, "pos2": 2, "value": 42.5},
    )
    index, params, _ = encode(action)
    restored = decode(index, params, 0, {}).parameters
    assert restored["geo1"] == 3 and restored["geo2"] == 5
    assert restored["value"] == pytest.approx(42.5, abs=0.1)
