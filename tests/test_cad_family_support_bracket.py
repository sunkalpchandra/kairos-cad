"""CAD integration tests for the support_bracket design family."""

import math
import random

import pytest

from kairos.actions.executor import ActionExecutor
from kairos.data.families.support_bracket import (
    FAMILY,
    SupportBracketParams,
    build_support_bracket,
)

pytestmark = pytest.mark.cad


def test_support_bracket_end_to_end(engine):
    p = SupportBracketParams()
    actions = build_support_bracket(ActionExecutor(engine), p)
    assert engine.check_validity().is_valid
    for diameter, count in FAMILY.expected_holes(p):
        assert len(engine.find_holes(diameter=diameter)) == count
    assert len(engine.find_holes(diameter=p.hole_diameter)) == (
        p.n_base_holes + p.n_wall_holes
    )
    ops = [a.operation.value for a in actions]
    assert ops[0] == "CREATE_SKETCH" and ops[-1] == "FINISH_DESIGN"


def test_support_bracket_volume_matches_analytic(engine):
    p = SupportBracketParams()
    build_support_bracket(ActionExecutor(engine), p)
    radius = p.hole_diameter / 2.0
    expected = (
        p.base_length * p.base_width * p.base_thickness
        + p.wall_thickness * p.wall_height * p.wall_width
        + p.rib_size * p.rib_size / 2.0 * p.rib_width
        - p.n_base_holes * math.pi * radius**2 * p.base_thickness
        - p.n_wall_holes * math.pi * radius**2 * p.wall_thickness
    )
    assert engine.measure_volume() == pytest.approx(expected, rel=1e-6)


def test_support_bracket_sampled_recipes_build_validly():
    """Property test: every feasible sampled design must build and validate."""
    from kairos.cad.engine import CADEngine

    rng = random.Random(19)
    built = 0
    for _ in range(8):
        p = SupportBracketParams.sample(rng)
        if not p.is_feasible():
            continue
        engine = CADEngine("prop_support_bracket")
        try:
            build_support_bracket(ActionExecutor(engine), p)
            assert engine.check_validity().is_valid
            holes = engine.find_holes(diameter=p.hole_diameter)
            assert len(holes) == p.n_base_holes + p.n_wall_holes
            built += 1
        finally:
            engine.close()
    assert built >= 4, "too few feasible samples exercised"


def test_support_bracket_infeasible_when_margin_too_small():
    p = SupportBracketParams(hole_margin=3.0, hole_diameter=6.0)
    assert not p.is_feasible()


def test_support_bracket_infeasible_when_wall_wider_than_base():
    assert not SupportBracketParams(wall_width=60.0, base_width=50.0).is_feasible()
