"""Pure-python tests of the parameter validation gate (no FreeCAD)."""

import pytest

from kairos.actions.parameters import (
    OPERATION_SPECS,
    ActionValidationError,
    validate_action,
)
from kairos.actions.schema import Action, Operation


def test_every_operation_has_a_spec():
    assert set(OPERATION_SPECS) == set(Operation)


def test_defaults_applied():
    params = validate_action(Action(Operation.CREATE_SKETCH))
    assert params == {"plane": "XY", "offset": 0.0}


def test_missing_required_parameter_rejected():
    with pytest.raises(ActionValidationError, match="missing required parameter"):
        validate_action(Action(Operation.ADD_CIRCLE, parameters={"cx": 0, "cy": 0}))


def test_unknown_parameter_rejected():
    with pytest.raises(ActionValidationError, match="unknown parameters"):
        validate_action(Action(Operation.PAD, parameters={"length": 5, "speed": 9}))


def test_type_mismatch_rejected():
    with pytest.raises(ActionValidationError, match="must be"):
        validate_action(Action(Operation.PAD, parameters={"length": "tall"}))


def test_range_enforced():
    with pytest.raises(ActionValidationError, match="below minimum"):
        validate_action(Action(Operation.PAD, parameters={"length": -3.0}))
    with pytest.raises(ActionValidationError, match="above maximum"):
        validate_action(Action(Operation.FILLET, target="Edge1", parameters={"radius": 1e6}))


def test_choices_enforced():
    with pytest.raises(ActionValidationError, match="not in"):
        validate_action(Action(Operation.CREATE_SKETCH, parameters={"plane": "QQ"}))


def test_int_accepted_for_float_param():
    params = validate_action(Action(Operation.PAD, parameters={"length": 10}))
    assert params["length"] == 10.0
    assert isinstance(params["length"], float)


def test_bool_not_accepted_as_number():
    with pytest.raises(ActionValidationError):
        validate_action(Action(Operation.PAD, parameters={"length": True}))


def test_target_required_for_fillet():
    with pytest.raises(ActionValidationError, match="requires a target"):
        validate_action(Action(Operation.FILLET, parameters={"radius": 1.0}))


def test_target_forbidden_when_spec_has_none():
    with pytest.raises(ActionValidationError, match="takes no target"):
        validate_action(Action(Operation.CREATE_SKETCH, target="Sketch"))


def test_finish_design_validates_empty():
    assert validate_action(Action(Operation.FINISH_DESIGN)) == {}
