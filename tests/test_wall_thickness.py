"""Wall-thickness measurement tests.

The pure-logic half runs anywhere; the measurement itself needs FreeCAD and is
marked `cad`.
"""

import pytest

from kairos.evaluation.constraints import check_constraints
from kairos.evaluation.wall_thickness import (
    ThicknessMeasurement,
    is_manufacturable,
    sampling_error_bound,
    thinnest_of,
)
from kairos.language import parse_requirement


def _obs(thickness=None):
    summary = {"has_solid": True, "valid": True}
    if thickness is not None:
        summary["min_wall_thickness_mm"] = thickness
    return {"summary": summary, "holes": [], "faces": [], "sketch": None}


def test_unmeasured_thickness_is_never_a_pass():
    """The whole point of the `unmeasured` status: no credit for not checking."""
    spec = parse_requirement("Bracket with wall thickness 3 mm.")
    result = check_constraints(_obs(None), spec).results[0]
    assert result.status == "unmeasured"
    assert not check_constraints(_obs(None), spec).all_measured_satisfied


def test_thickness_above_the_floor_is_satisfied():
    spec = parse_requirement("Bracket with wall thickness 3 mm.")
    assert check_constraints(_obs(5.0), spec).results[0].status == "satisfied"


def test_thickness_below_the_floor_is_violated():
    spec = parse_requirement("Bracket with wall thickness 3 mm.")
    result = check_constraints(_obs(2.0), spec).results[0]
    assert result.status == "violated"
    assert result.measured == pytest.approx(2.0)


def test_a_measurement_under_the_floor_is_a_definitive_failure():
    """Sampling only ever OVER-estimates, so measured is an upper bound.

    A measured 2.99 mm means the true wall is at most 2.99 mm, which is below a
    3 mm floor, conclusively. Slack here would pass parts that are provably
    too thin, and it was letting 6.983 mm clear a 7.0 mm floor.
    """
    spec = parse_requirement("Bracket with wall thickness 3 mm.")
    assert check_constraints(_obs(2.99), spec).results[0].status == "violated"
    assert check_constraints(_obs(3.0), spec).results[0].status == "satisfied"
    assert check_constraints(_obs(3.01), spec).results[0].status == "satisfied"


def test_is_manufacturable_distinguishes_unmeasured_from_false():
    assert is_manufacturable(ThicknessMeasurement(None, 0, 0), 3.0) is None
    assert is_manufacturable(ThicknessMeasurement(5.0, 10, 8), 3.0) is True
    assert is_manufacturable(ThicknessMeasurement(1.0, 10, 8), 3.0) is False


def test_thinnest_of_ignores_unmeasured():
    measurements = [
        ThicknessMeasurement(None, 0, 0),
        ThicknessMeasurement(4.0, 9, 9),
        ThicknessMeasurement(2.5, 9, 9),
    ]
    assert thinnest_of(measurements) == pytest.approx(2.5)
    assert thinnest_of([ThicknessMeasurement(None, 0, 0)]) is None


def test_error_bound_states_the_direction_of_the_error():
    text = sampling_error_bound(ThicknessMeasurement(3.25, 40, 30))
    assert "over-estimate" in text and "30/40" in text
    assert "unmeasured" in sampling_error_bound(ThicknessMeasurement(None, 4, 0))


@pytest.mark.cad
def test_measures_known_geometry_exactly():
    from kairos.cad.backend import load_freecad, load_module
    from kairos.evaluation.wall_thickness import measure_min_wall_thickness

    load_freecad()
    Part = load_module("Part")

    for shape, true in ((Part.makeBox(40, 30, 5), 5.0), (Part.makeBox(20, 20, 20), 20.0)):
        measurement = measure_min_wall_thickness(shape)
        assert measurement.measured
        assert measurement.min_thickness_mm == pytest.approx(true, abs=0.02)


@pytest.mark.cad
def test_measures_a_tube_wall_radially():
    """A tube's wall has no planar face to probe; coaxial cylinders give it."""
    from kairos.cad.backend import load_freecad, load_module
    from kairos.evaluation.wall_thickness import measure_min_wall_thickness

    load_freecad()
    Part = load_module("Part")
    tube = Part.makeCylinder(15, 40).cut(Part.makeCylinder(10, 40))
    assert measure_min_wall_thickness(tube).min_thickness_mm == pytest.approx(5.0, abs=0.02)


@pytest.mark.cad
def test_every_family_measures_its_declared_wall_thickness():
    """The families are the ground truth: each declares what it builds."""
    from kairos.actions.executor import ActionExecutor
    from kairos.cad.engine import CADEngine
    from kairos.data.families import family_names, get_family
    from kairos.representation import observe

    for name in family_names():
        family = get_family(name)
        params = family.params_cls()
        engine = CADEngine(f"wt_test_{name}")
        try:
            family.build(ActionExecutor(engine), params)
            measured = observe(engine, wall_thickness=True)["summary"].get(
                "min_wall_thickness_mm"
            )
            assert measured is not None, f"{name}: no wall thickness measured"
            declared = family.requirements(params)["spec"].get("min_wall_thickness")
            if declared is not None:
                assert measured == pytest.approx(float(declared), abs=0.05), name
        finally:
            engine.close()


@pytest.mark.cad
def test_measurement_is_off_by_default():
    """It ray-casts against the solid; every observation must not pay that."""
    from kairos.actions.executor import ActionExecutor
    from kairos.cad.engine import CADEngine
    from kairos.data.families import get_family
    from kairos.representation import observe

    family = get_family("plate")
    engine = CADEngine("wt_default")
    try:
        family.build(ActionExecutor(engine), family.params_cls())
        assert "min_wall_thickness_mm" not in observe(engine)["summary"]
        assert "min_wall_thickness_mm" in observe(engine, wall_thickness=True)["summary"]
    finally:
        engine.close()


@pytest.mark.cad
def test_terminal_step_of_a_trajectory_measures_thickness():
    """min_wall_thickness must be a real check in recorded data, not 'unmeasured'."""
    from kairos.actions.executor import ActionExecutor
    from kairos.cad.engine import CADEngine
    from kairos.data.families import get_family
    from kairos.data.trajectories import TrajectoryRecorder

    family = get_family("l_bracket")
    params = family.params_cls()
    engine = CADEngine("wt_trajectory")
    try:
        executor = ActionExecutor(engine)
        recorder = TrajectoryRecorder(executor, family.requirements(params)["text"])
        family.build(executor, params)
        results = recorder.to_dict()["final_metrics"]["constraints"]["results"]
        wall = [r for r in results if r["constraint"]["kind"] == "min_wall_thickness"]
        assert wall, "the l_bracket requirement declares a wall thickness"
        assert wall[0]["status"] == "satisfied", wall[0]
    finally:
        engine.close()


@pytest.mark.cad
def test_only_the_terminal_step_pays_for_the_measurement():
    """Ray casting every step would dominate the cost of generating a design."""
    from kairos.actions.executor import ActionExecutor
    from kairos.cad.engine import CADEngine
    from kairos.data.families import get_family
    from kairos.data.trajectories import TrajectoryRecorder

    family = get_family("plate")
    params = family.params_cls()
    engine = CADEngine("wt_cost")
    try:
        executor = ActionExecutor(engine)
        recorder = TrajectoryRecorder(executor, family.requirements(params)["text"])
        family.build(executor, params)
        measured = [
            "min_wall_thickness_mm" in (step.get("summary") or {})
            for step in recorder.steps
        ]
        assert measured[-1] is True or not any(measured[:-1]), (
            "intermediate steps must not carry the expensive measurement"
        )
        assert sum(measured) <= 1
    finally:
        engine.close()
