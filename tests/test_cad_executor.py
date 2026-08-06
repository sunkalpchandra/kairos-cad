"""CAD integration tests: the structured action executor end-to-end."""

import pytest

from kairos.actions.executor import ActionExecutor
from kairos.actions.masking import flags_from_engine, legal_operations
from kairos.actions.schema import Action, Operation

pytestmark = pytest.mark.cad


@pytest.fixture
def executor(engine):
    return ActionExecutor(engine)


def _run_ok(executor, action):
    result = executor.execute(action)
    assert result.ok, result.message
    return result


def test_plate_via_actions(executor):
    _run_ok(executor, Action(Operation.CREATE_SKETCH, parameters={"plane": "XY"}))
    _run_ok(
        executor,
        Action(
            Operation.ADD_RECTANGLE,
            parameters={"x": 0, "y": 0, "width": 40, "height": 20},
        ),
    )
    result = _run_ok(executor, Action(Operation.PAD, parameters={"length": 10}))
    assert result.info["volume_mm3"] == pytest.approx(8000.0)
    result = _run_ok(executor, Action(Operation.MEASURE_VOLUME))
    assert result.info["volume_mm3"] == pytest.approx(8000.0)
    result = _run_ok(executor, Action(Operation.FINISH_DESIGN))
    assert result.done
    assert result.info["summary"]["valid"] is True


def test_invalid_action_returns_failure_not_exception(executor):
    result = executor.execute(Action(Operation.PAD, parameters={"length": 10}))
    assert not result.ok
    assert "no sketch" in result.message


def test_validation_failure_reported_before_engine_touched(executor):
    result = executor.execute(Action(Operation.PAD, parameters={"length": -1}))
    assert not result.ok
    assert result.message.startswith("validation:")


def test_failed_feature_keeps_document_usable(executor):
    _run_ok(executor, Action(Operation.CREATE_SKETCH, parameters={"plane": "XY"}))
    result = executor.execute(Action(Operation.PAD, parameters={"length": 5}))
    assert not result.ok  # empty profile
    # Recovery: add geometry to a fresh sketch and pad successfully.
    _run_ok(executor, Action(Operation.CREATE_SKETCH, parameters={"plane": "XY"}))
    _run_ok(
        executor,
        Action(Operation.ADD_CIRCLE, parameters={"cx": 0, "cy": 0, "radius": 5}),
    )
    _run_ok(executor, Action(Operation.PAD, parameters={"length": 4}))
    assert executor.engine.has_solid()


def test_actions_after_finish_rejected(executor):
    _run_ok(executor, Action(Operation.FINISH_DESIGN))
    result = executor.execute(Action(Operation.CREATE_SKETCH))
    assert not result.ok and result.done


def test_trajectory_records_failures_and_successes(executor):
    executor.execute(Action(Operation.PAD, parameters={"length": 10}))
    executor.execute(Action(Operation.CREATE_SKETCH))
    trajectory = executor.trajectory()
    assert len(trajectory) == 2
    assert trajectory[0]["result"]["ok"] is False
    assert trajectory[1]["result"]["ok"] is True
    assert trajectory[1]["action"]["operation"] == "CREATE_SKETCH"


def test_masking_flags_track_live_engine(executor):
    flags = flags_from_engine(executor.engine)
    assert not flags.has_sketch and not flags.has_solid
    _run_ok(executor, Action(Operation.CREATE_SKETCH))
    flags = flags_from_engine(executor.engine)
    assert flags.has_sketch and not flags.sketch_has_geometry
    _run_ok(
        executor,
        Action(Operation.ADD_CIRCLE, parameters={"cx": 0, "cy": 0, "radius": 6}),
    )
    _run_ok(executor, Action(Operation.PAD, parameters={"length": 5}))
    flags = flags_from_engine(executor.engine)
    assert flags.has_solid and flags.has_edges and flags.has_features
    assert Operation.FILLET in legal_operations(flags)


def test_fillet_via_action_with_edge_target(executor):
    _run_ok(executor, Action(Operation.CREATE_SKETCH))
    _run_ok(
        executor,
        Action(
            Operation.ADD_RECTANGLE,
            parameters={"x": 0, "y": 0, "width": 20, "height": 20},
        ),
    )
    _run_ok(executor, Action(Operation.PAD, parameters={"length": 10}))
    edges = executor.engine.find_edges(curve="Line", direction=(0, 0, 1))
    result = _run_ok(
        executor,
        Action(Operation.FILLET, target=edges[0], parameters={"radius": 2.0}),
    )
    assert result.info["volume_mm3"] < 4000.0
