"""CAD integration tests for the spacer design family."""

import math
import random

import pytest

import kairos.data.families.spacer as spacer
from kairos.actions.executor import ActionExecutor

pytestmark = pytest.mark.cad


def spacer_volume(p: spacer.SpacerParams) -> float:
    """Closed-form spacer volume, derived by Pappus's centroid theorem.

    Ring: pi * (r_out^2 - r_in^2) * h. Each rim chamfer removes a ring whose
    cross-section is a right triangle with legs c x c at the outer corner
    (vertices (r_out - c, z_rim), (r_out, z_rim), (r_out, z_rim -/+ c)):
    area c^2 / 2 at centroid radius r_out - c/3, so each removed ring is
    2*pi * (r_out - c/3) * c^2/2 = pi * c^2 * (r_out - c/3).
    """
    ring = math.pi * (p.outer_radius**2 - p.inner_radius**2) * p.height
    c = p.chamfer
    rim = math.pi * c**2 * (p.outer_radius - c / 3.0)
    return ring - 2.0 * rim


def test_spacer_recipe_end_to_end(engine):
    params = spacer.SpacerParams()
    assert params.chamfer > 0, "defaults must exercise the chamfer feature"
    executor = ActionExecutor(engine)
    actions = spacer.build_spacer(executor, params)
    assert engine.check_validity().is_valid
    for diameter, count in spacer.FAMILY.expected_holes(params):
        assert len(engine.find_holes(diameter=diameter)) == count
    ops = [a.operation.value for a in actions]
    assert ops[0] == "CREATE_SKETCH" and ops[-1] == "FINISH_DESIGN"
    assert "REVOLVE" in ops and "CHAMFER" in ops


def test_spacer_volume_matches_analytic(engine):
    p = spacer.SpacerParams()
    spacer.build_spacer(ActionExecutor(engine), p)
    assert engine.measure_volume() == pytest.approx(spacer_volume(p), rel=1e-6)


def test_spacer_volume_no_chamfer_matches_analytic(engine):
    p = spacer.SpacerParams(chamfer=0.0)
    spacer.build_spacer(ActionExecutor(engine), p)
    expected = math.pi * (p.outer_radius**2 - p.inner_radius**2) * p.height
    assert engine.measure_volume() == pytest.approx(expected, rel=1e-6)
    # c = 0 limit of the chamfered formula must recover the plain ring.
    assert spacer_volume(p) == pytest.approx(expected)


def test_sampled_spacers_build_validly():
    """Property test: every feasible sampled spacer must build and validate."""
    from kairos.cad.engine import CADEngine

    rng = random.Random(19)
    built = 0
    for _ in range(8):
        params = spacer.SpacerParams.sample(rng)
        if not params.is_feasible():
            continue
        engine = CADEngine("prop")
        try:
            spacer.build_spacer(ActionExecutor(engine), params)
            assert engine.check_validity().is_valid
            for diameter, count in spacer.FAMILY.expected_holes(params):
                assert len(engine.find_holes(diameter=diameter)) == count
            assert engine.measure_volume() == pytest.approx(spacer_volume(params), rel=1e-6)
            built += 1
        finally:
            engine.close()
    assert built >= 5, "too few feasible samples exercised"


def test_spacer_infeasible_rejections():
    assert spacer.SpacerParams().is_feasible()
    # Chamfer at least half the wall would eat through to the bore.
    assert not spacer.SpacerParams(chamfer=4.0).is_feasible()
    # Wall thinner than 2 mm.
    assert not spacer.SpacerParams(outer_radius=6.5).is_feasible()
    # Too short for the two rim chamfers.
    assert not spacer.SpacerParams(height=3.0).is_feasible()
