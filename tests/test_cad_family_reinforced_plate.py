"""CAD integration tests for the reinforced_plate design family."""

import math
import random

import pytest

from kairos.actions.executor import ActionExecutor
from kairos.data.families.reinforced_plate import (
    FAMILY,
    ReinforcedPlateParams,
    build_reinforced_plate,
)

pytestmark = pytest.mark.cad


def test_reinforced_plate_end_to_end(engine):
    p = ReinforcedPlateParams()
    actions = build_reinforced_plate(ActionExecutor(engine), p)
    assert engine.check_validity().is_valid
    for diameter, count in FAMILY.expected_holes(p):
        assert len(engine.find_holes(diameter=diameter)) == count
    # Ribs must rise above the plate top, and holes must not pierce them.
    bbox = engine.measure_bounding_box()
    assert bbox["z_max"] == pytest.approx(p.thickness + p.rib_height)
    assert bbox["z_min"] == pytest.approx(0.0, abs=1e-9)
    ops = [a.operation.value for a in actions]
    assert ops[0] == "CREATE_SKETCH" and ops[-1] == "FINISH_DESIGN"


def test_reinforced_plate_volume_matches_analytic(engine):
    p = ReinforcedPlateParams()
    build_reinforced_plate(ActionExecutor(engine), p)
    radius = p.hole_diameter / 2.0
    expected = (
        p.length * p.width * p.thickness
        + p.n_ribs * p.rib_width * p.rib_height * p.length
        - 4.0 * math.pi * radius**2 * p.thickness
    )
    assert engine.measure_volume() == pytest.approx(expected, rel=1e-6)


def test_reinforced_plate_sampled_recipes_build_validly():
    """Property test: every feasible sampled design must build and validate."""
    from kairos.cad.engine import CADEngine

    rng = random.Random(23)
    built = 0
    for _ in range(8):
        p = ReinforcedPlateParams.sample(rng)
        if not p.is_feasible():
            continue
        engine = CADEngine("prop_reinforced_plate")
        try:
            build_reinforced_plate(ActionExecutor(engine), p)
            assert engine.check_validity().is_valid
            assert len(engine.find_holes(diameter=p.hole_diameter)) == 4
            built += 1
        finally:
            engine.close()
    assert built >= 4, "too few feasible samples exercised"


def test_reinforced_plate_infeasible_when_hole_row_hits_rib():
    # width=45 with 2 ribs puts a rib centerline at y=15; a 15 mm corner
    # margin lands the hole row exactly on it.
    p = ReinforcedPlateParams(width=45.0, corner_margin=15.0, hole_diameter=6.0)
    assert not p.is_feasible()


def test_reinforced_plate_infeasible_when_margin_too_small():
    assert not ReinforcedPlateParams(corner_margin=3.0, hole_diameter=6.0).is_feasible()
