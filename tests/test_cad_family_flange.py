"""CAD integration tests for the flange design family."""

import math
import random

import pytest

import kairos.data.families.flange as flange
from kairos.actions.executor import ActionExecutor

pytestmark = pytest.mark.cad


def flange_volume(p: flange.FlangeParams) -> float:
    """Closed-form flange volume: bored disk + bored hub - bolt holes.

    The revolved section is a disk annulus (r_bore..R_disk, thickness t)
    stacked with a hub annulus (r_bore..R_hub, height h_hub); each of the
    n bolt holes removes a d_bolt cylinder through the disk only, since the
    bolt circle clears the hub.
    """
    disk = math.pi * (p.disk_radius**2 - p.bore_radius**2) * p.disk_thickness
    hub = math.pi * (p.hub_radius**2 - p.bore_radius**2) * p.hub_height
    bolts = p.n_bolts * math.pi * (p.bolt_diameter / 2.0) ** 2 * p.disk_thickness
    return disk + hub - bolts


def test_flange_recipe_end_to_end(engine):
    params = flange.FlangeParams()
    executor = ActionExecutor(engine)
    actions = flange.build_flange(executor, params)
    assert engine.check_validity().is_valid
    for diameter, count in flange.FAMILY.expected_holes(params):
        assert len(engine.find_holes(diameter=diameter)) == count
    ops = [a.operation.value for a in actions]
    assert ops[0] == "CREATE_SKETCH" and ops[-1] == "FINISH_DESIGN"
    assert "REVOLVE" in ops and "POCKET" in ops and "CIRCULAR_PATTERN" in ops


def test_flange_volume_matches_analytic(engine):
    p = flange.FlangeParams()
    flange.build_flange(ActionExecutor(engine), p)
    assert engine.measure_volume() == pytest.approx(flange_volume(p), rel=1e-6)


def test_sampled_flanges_build_validly():
    """Property test: every feasible sampled flange must build and validate."""
    from kairos.cad.engine import CADEngine

    rng = random.Random(23)
    built = 0
    for _ in range(8):
        params = flange.FlangeParams.sample(rng)
        if not params.is_feasible():
            continue
        engine = CADEngine("prop")
        try:
            flange.build_flange(ActionExecutor(engine), params)
            assert engine.check_validity().is_valid
            for diameter, count in flange.FAMILY.expected_holes(params):
                assert len(engine.find_holes(diameter=diameter)) == count
            assert engine.measure_volume() == pytest.approx(flange_volume(params), rel=1e-6)
            built += 1
        finally:
            engine.close()
    assert built >= 5, "too few feasible samples exercised"


def test_flange_infeasible_rejections():
    assert flange.FlangeParams().is_feasible()
    # Hub collides with the bolt circle.
    assert not flange.FlangeParams(hub_radius=18.0).is_feasible()
    # Bolt circle too close to the disk rim.
    assert not flange.FlangeParams(disk_radius=26.0).is_feasible()
    # Bore leaves no hub wall.
    assert not flange.FlangeParams(bore_radius=10.5).is_feasible()
    # Bolt count out of range.
    assert not flange.FlangeParams(n_bolts=2).is_feasible()
    # Bore and bolt holes indistinguishable: "6 mm holes" would name both, and
    # the generator's own hole-count validation rejects the built solid.
    assert not flange.FlangeParams(bore_radius=2.6, bolt_diameter=5.0).is_feasible()
