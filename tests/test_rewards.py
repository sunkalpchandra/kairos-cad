"""Pure-python tests of the shaped reward tracker on synthetic episodes."""

import pytest

from kairos.actions.schema import ActionResult, Operation
from kairos.language import parse_requirement
from kairos.rl.rewards import RewardTracker, RewardWeights


def _ok(op):
    return ActionResult(True, op, "ok")


def _fail(op):
    return ActionResult(False, op, "nope")


def _obs(has_solid=False, valid=None, mass=None, holes=(), sketch=None):
    return {
        "summary": {
            "has_solid": has_solid,
            "valid": has_solid if valid is None else valid,
            "mass_g": mass,
        },
        "holes": list(holes),
        "faces": [],
        "sketch": sketch,
    }


def _holes(n, d=5.0):
    return [
        {"diameter": d, "axis": (0, 0, 1), "axis_point": (10 * i, 0, 0), "faces": []}
        for i in range(n)
    ]


SPEC = parse_requirement("Plate with 2 M5 holes. Minimize mass.")


def test_invalid_action_penalized():
    tracker = RewardTracker(SPEC)
    r = tracker.step(_fail(Operation.PAD), _obs())
    assert r.components["invalid_action"] == pytest.approx(-0.5)
    assert r.total < 0


def test_action_cost_always_applied():
    tracker = RewardTracker(SPEC)
    r = tracker.step(_ok(Operation.CHECK_VALIDITY), _obs())
    assert r.components["action_cost"] == pytest.approx(-0.01)


def test_sketch_and_solid_bonuses_awarded_once():
    tracker = RewardTracker(SPEC)
    sketch = {"geometry_count": 1, "fully_constrained": False}
    r1 = tracker.step(_ok(Operation.ADD_CIRCLE), _obs(sketch=sketch))
    assert r1.components["valid_sketch"] == pytest.approx(0.2)
    r2 = tracker.step(_ok(Operation.ADD_CIRCLE), _obs(sketch=sketch))
    assert "valid_sketch" not in r2.components
    r3 = tracker.step(_ok(Operation.PAD), _obs(has_solid=True, mass=100.0))
    assert r3.components["first_solid"] == pytest.approx(0.5)
    r4 = tracker.step(_ok(Operation.PAD), _obs(has_solid=True, mass=100.0))
    assert "first_solid" not in r4.components


def test_constraint_bonus_then_all_constraints_bonus():
    tracker = RewardTracker(SPEC)
    obs = _obs(has_solid=True, mass=80.0, holes=_holes(2))
    r = tracker.step(_ok(Operation.POCKET), obs)
    # hole_count and hole_diameter both satisfied -> two constraint bonuses + all bonus.
    constraint_bonuses = [v for k, v in r.components.items() if k.startswith("constraint:")]
    assert len(constraint_bonuses) == 2
    assert r.components["all_constraints"] == pytest.approx(2.0)
    # Repeat step: no double award.
    r2 = tracker.step(_ok(Operation.CHECK_VALIDITY), obs)
    assert not any(k.startswith("constraint:") for k in r2.components)
    assert "all_constraints" not in r2.components


def test_mass_progress_only_while_constraints_satisfied():
    tracker = RewardTracker(SPEC)
    # Mass drops while holes are missing: no progress reward.
    tracker.step(_ok(Operation.PAD), _obs(has_solid=True, mass=100.0))
    r = tracker.step(_ok(Operation.POCKET), _obs(has_solid=True, mass=90.0))
    assert "mass_progress" not in r.components
    # Constraints become satisfied at mass 80 (baseline), then improve to 72.
    tracker.step(_ok(Operation.POCKET), _obs(has_solid=True, mass=80.0, holes=_holes(2)))
    r = tracker.step(_ok(Operation.POCKET), _obs(has_solid=True, mass=72.0, holes=_holes(2)))
    assert r.components["mass_progress"] == pytest.approx(1.0 * (80 - 72) / 80)
    # Mass increase yields no reward (and no penalty here; complexity covers spam).
    r = tracker.step(_ok(Operation.PAD), _obs(has_solid=True, mass=75.0, holes=_holes(2)))
    assert "mass_progress" not in r.components


def test_finish_success_and_failure():
    good = RewardTracker(SPEC)
    obs_ok = _obs(has_solid=True, mass=50.0, holes=_holes(2))
    good.step(_ok(Operation.POCKET), obs_ok)
    r = good.step(_ok(Operation.FINISH_DESIGN), obs_ok)
    assert r.components["finish"] == pytest.approx(5.0)

    bad = RewardTracker(SPEC)
    r = bad.step(_ok(Operation.FINISH_DESIGN), _obs(has_solid=True, mass=50.0))
    assert r.components["finish"] == pytest.approx(-1.0)


def test_validity_regression_penalized():
    tracker = RewardTracker(SPEC)
    tracker.step(_ok(Operation.PAD), _obs(has_solid=True, valid=True, mass=10.0))
    r = tracker.step(_ok(Operation.MIRROR), _obs(has_solid=True, valid=False, mass=20.0))
    assert r.components["validity_broken"] == pytest.approx(-1.0)


def test_complexity_charged_per_feature():
    tracker = RewardTracker(SPEC)
    r = tracker.step(_ok(Operation.FILLET), _obs(has_solid=True, mass=10.0))
    assert r.components["complexity"] == pytest.approx(-0.02)
    r2 = tracker.step(_ok(Operation.ADD_CIRCLE), _obs(has_solid=True, mass=10.0))
    assert "complexity" not in r2.components


def test_custom_weights_respected():
    weights = RewardWeights(invalid_action=-2.0, action_cost=0.0)
    tracker = RewardTracker(SPEC, weights=weights)
    r = tracker.step(_fail(Operation.PAD), _obs())
    assert r.total == pytest.approx(-2.0)


def test_breakdown_serializes():
    tracker = RewardTracker(SPEC)
    r = tracker.step(_ok(Operation.CHECK_VALIDITY), _obs())
    data = r.to_dict()
    assert set(data) == {"total", "components"}
    assert isinstance(data["components"], dict)
