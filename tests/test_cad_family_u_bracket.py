"""CAD integration tests for the u_bracket design family."""

import math
import random

import pytest

from kairos.actions.executor import ActionExecutor
from kairos.data.families.u_bracket import FAMILY, UBracketParams, build_u_bracket

pytestmark = pytest.mark.cad


def test_u_bracket_recipe_end_to_end(engine):
    params = UBracketParams()
    assert params.is_feasible()
    actions = build_u_bracket(ActionExecutor(engine), params)
    assert engine.check_validity().is_valid
    ((diameter, count),) = FAMILY.expected_holes(params)
    assert len(engine.find_holes(diameter=diameter)) == count
    ops = [a.operation.value for a in actions]
    assert ops[0] == "CREATE_SKETCH" and ops[-1] == "FINISH_DESIGN"
    assert ops.count("POCKET") == 2


def test_u_bracket_volume_matches_analytic(engine):
    p = UBracketParams()
    build_u_bracket(ActionExecutor(engine), p)
    profile_area = p.outer_width * p.height - (p.outer_width - 2 * p.wall_thickness) * (
        p.height - p.base_thickness
    )
    radius = p.hole_diameter / 2.0
    # Each base hole bores the floor; each side hole bores both walls.
    hole_volume = (
        math.pi
        * radius**2
        * (p.n_base * p.base_thickness + p.n_side * 2.0 * p.wall_thickness)
    )
    expected = profile_area * p.depth - hole_volume
    assert engine.measure_volume() == pytest.approx(expected, rel=1e-6)


def test_u_bracket_sampled_builds_validly():
    """Property test: every feasible sampled draw must build and validate."""
    from kairos.cad.engine import CADEngine

    rng = random.Random(3)
    built = 0
    for _ in range(8):
        params = UBracketParams.sample(rng)
        if not params.is_feasible():
            continue
        engine = CADEngine("prop")
        try:
            build_u_bracket(ActionExecutor(engine), params)
            assert engine.check_validity().is_valid
            ((diameter, count),) = FAMILY.expected_holes(params)
            assert len(engine.find_holes(diameter=diameter)) == count
            built += 1
        finally:
            engine.close()
    assert built >= 4, "too few feasible samples exercised"


def test_u_bracket_infeasible_rejections():
    # Walls so thick the base holes would bite into them.
    assert not UBracketParams(outer_width=30.0, wall_thickness=10.0).is_feasible()
    # Channel too shallow for the wall bores to clear the floor and rim.
    assert not UBracketParams(height=18.0).is_feasible()
    # Extrusion too shallow for the hole diameter.
    assert not UBracketParams(depth=6.0).is_feasible()
