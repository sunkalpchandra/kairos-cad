"""Pure-python tests of the action-space codec."""

import numpy as np
import pytest

from kairos.actions.parameters import ActionValidationError, validate_action
from kairos.actions.schema import Operation
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
    (except ADD_POLYGON, whose empty v0 point list validates but is unusable —
    and target-requiring ops when a target pool exists)."""
    rng = np.random.default_rng(0)
    for op_index in range(NUM_OPERATIONS):
        for _ in range(5):
            params = rng.random(PARAM_SLOTS)
            action = decode(op_index, params, int(rng.integers(0, 64)), TARGETS)
            assert action.operation is OPERATIONS[op_index]
            validate_action(action)  # must not raise


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
    idx = OPERATIONS.index(Operation.PAD)
    low = decode(idx, np.zeros(PARAM_SLOTS), 0, {})
    high = decode(idx, np.ones(PARAM_SLOTS), 0, {})
    assert low.parameters["length"] == pytest.approx(1.0)
    assert high.parameters["length"] == pytest.approx(100.0)
    assert low.parameters["midplane"] is False and high.parameters["midplane"] is True


def test_pocket_through_all_vs_depth():
    idx = OPERATIONS.index(Operation.POCKET)
    through = decode(idx, np.array([0.9, 0.5, 0.0, 0, 0, 0]), 0, {})
    assert through.parameters["through_all"] is True and "depth" not in through.parameters
    depth = decode(idx, np.array([0.1, 0.5, 0.0, 0, 0, 0]), 0, {})
    assert depth.parameters["through_all"] is False
    assert 1.0 <= depth.parameters["depth"] <= 50.0


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
