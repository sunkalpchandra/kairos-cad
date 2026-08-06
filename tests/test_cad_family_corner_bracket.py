"""CAD integration tests for the corner_bracket design family."""

import math
import random

import pytest

from kairos.actions.executor import ActionExecutor
from kairos.data.families.corner_bracket import (
    FAMILY,
    CornerBracketParams,
    build_corner_bracket,
)

pytestmark = pytest.mark.cad


def test_corner_bracket_recipe_end_to_end(engine):
    params = CornerBracketParams()
    assert params.is_feasible()
    actions = build_corner_bracket(ActionExecutor(engine), params)
    assert engine.check_validity().is_valid
    ((diameter, count),) = FAMILY.expected_holes(params)
    assert len(engine.find_holes(diameter=diameter)) == count
    ops = [a.operation.value for a in actions]
    assert ops[0] == "CREATE_SKETCH" and ops[-1] == "FINISH_DESIGN"
    assert ops.count("PAD") == 2  # L profile plus fused gusset rib


def test_corner_bracket_volume_matches_analytic(engine):
    p = CornerBracketParams()
    build_corner_bracket(ActionExecutor(engine), p)
    profile_area = p.leg1 * p.thickness + (p.leg2 - p.thickness) * p.thickness
    gusset_volume = 0.5 * p.gusset**2 * p.rib_width
    radius = p.hole_diameter / 2.0
    hole_volume = math.pi * radius**2 * p.thickness * 2 * p.holes_per_leg
    expected = profile_area * p.width + gusset_volume - hole_volume
    assert engine.measure_volume() == pytest.approx(expected, rel=1e-6)


def test_corner_bracket_sampled_builds_validly():
    """Property test: every feasible sampled draw must build and validate."""
    from kairos.cad.engine import CADEngine

    rng = random.Random(11)
    built = 0
    for _ in range(8):
        params = CornerBracketParams.sample(rng)
        if not params.is_feasible():
            continue
        engine = CADEngine("prop")
        try:
            build_corner_bracket(ActionExecutor(engine), params)
            assert engine.check_validity().is_valid
            ((diameter, count),) = FAMILY.expected_holes(params)
            assert len(engine.find_holes(diameter=diameter)) == count
            built += 1
        finally:
            engine.close()
    assert built >= 4, "too few feasible samples exercised"


def test_corner_bracket_infeasible_rejections():
    # Gusset so large the holes cannot clear it on the short leg.
    assert not CornerBracketParams(gusset=40.0).is_feasible()
    # Rib as wide as the body is rejected.
    assert not CornerBracketParams(rib_width=30.0).is_feasible()
    # Rib clearance smaller than the hole radius is rejected.
    assert not CornerBracketParams(margin_from_rib=1.0).is_feasible()
