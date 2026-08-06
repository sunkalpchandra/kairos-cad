"""Pure-python tests of the Action schema (no FreeCAD required)."""

import pytest

from kairos.actions.schema import (
    BOOLEAN_OPS,
    CONSTRAINT_OPS,
    FEATURE_OPS,
    INSPECTION_OPS,
    SKETCH_OPS,
    Action,
    ActionResult,
    Operation,
)


def test_action_json_round_trip():
    action = Action(
        Operation.FILLET,
        target="Edge17",
        parameters={"radius": 2.0},
        confidence=0.87,
    )
    restored = Action.from_json(action.to_json())
    assert restored == action
    assert restored.operation is Operation.FILLET
    assert restored.parameters["radius"] == 2.0


def test_action_accepts_operation_strings():
    action = Action("POCKET", parameters={"through_all": True})
    assert action.operation is Operation.POCKET


def test_action_rejects_unknown_operation():
    with pytest.raises(ValueError):
        Action("EXTRUDE_MAGICALLY")


def test_confidence_bounds_enforced():
    with pytest.raises(ValueError):
        Action(Operation.PAD, parameters={"length": 5}, confidence=1.5)
    with pytest.raises(ValueError):
        Action(Operation.PAD, parameters={"length": 5}, confidence=-0.1)


def test_operation_groups_are_disjoint_and_cover_vocabulary():
    groups = [SKETCH_OPS, CONSTRAINT_OPS, FEATURE_OPS, BOOLEAN_OPS, INSPECTION_OPS]
    seen = set()
    for group in groups:
        assert not (seen & group), "operation groups overlap"
        seen |= group
    seen.add(Operation.FINISH_DESIGN)
    assert seen == set(Operation)


def test_action_result_serializes():
    result = ActionResult(True, Operation.PAD, "ok", {"volume_mm3": 8000.0})
    data = result.to_dict()
    assert data["ok"] is True
    assert data["operation"] == "PAD"
    assert data["info"]["volume_mm3"] == 8000.0
