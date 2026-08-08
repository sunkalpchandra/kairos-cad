"""Milestone absorption from what the environment reported.

Credit is prefix-scored, so a milestone that reads False by mistake does not
just lose its own weight, it zeroes every later rung too. That is not
hypothetical: reading hole_count from a payload field the RESET and REPLAY
paths never filled took the oracle from 0.938 to 0.111 while it still finished
94% of its tasks.
"""

from __future__ import annotations

from kairos.actions.schema import Operation
from kairos.benchmark.metrics import EpisodeOutcome
from kairos.benchmark.runner import _absorb
from kairos.rl.action_space import OPERATIONS


def _index(operation: Operation) -> int:
    return OPERATIONS.index(operation)


def _solid(**extra) -> dict:
    return {"has_solid": True, "valid": True, "mass_g": 100.0, **extra}


def test_holes_come_from_counted_geometry():
    outcome = EpisodeOutcome(requirement="r", family="plate")
    _absorb(outcome, before=None, after=_solid(hole_count=4),
            info={"ok": True}, operation_index=_index(Operation.POCKET))
    assert outcome.has_any_hole


def test_no_holes_means_no_credit_however_satisfied():
    """A satisfied mounting_angle is true of every prismatic solid.

    Awarding has_any_hole from satisfaction_rate > 0 let that buy credit for
    holes the policy never drilled.
    """
    outcome = EpisodeOutcome(requirement="r", family="plate")
    _absorb(
        outcome, before=None, after=_solid(hole_count=0),
        info={"ok": True, "satisfaction_rate": 0.5},
        operation_index=_index(Operation.PAD),
    )
    assert not outcome.has_any_hole


def test_all_constraints_met_does_not_imply_holes():
    outcome = EpisodeOutcome(requirement="r", family="spacer")
    _absorb(
        outcome, before=None, after=_solid(hole_count=0),
        info={"ok": True, "all_satisfied": True, "satisfaction_rate": 1.0},
        operation_index=_index(Operation.PAD),
    )
    assert outcome.all_constraints_met
    assert not outcome.has_any_hole


def test_a_solid_backfills_the_milestones_that_must_have_preceded_it():
    """COMPLETE(k) replays the prefix, so the policy never emits those steps."""
    outcome = EpisodeOutcome(requirement="r", family="plate")
    _absorb(outcome, before=None, after=_solid(hole_count=2),
            info={"ok": True}, operation_index=_index(Operation.PAD))
    assert outcome.opened_a_sketch
    assert outcome.drew_geometry
    assert outcome.made_a_solid
    assert outcome.solid_is_valid


def test_a_missing_hole_count_does_not_silently_zero_the_ladder():
    """The payload field must exist; absent it, this reads 0 and prefix-scoring
    would discard every later milestone."""
    outcome = EpisodeOutcome(requirement="r", family="plate")
    state = _solid()  # no hole_count key at all
    _absorb(outcome, before=None, after=state,
            info={"ok": True}, operation_index=_index(Operation.PAD))
    assert not outcome.has_any_hole
    assert outcome.solid_is_valid  # earlier rungs still credited


def test_env_server_payload_reports_hole_count():
    """Pins the field name across the bridge; a rename here reads as zero holes."""
    from kairos.rl.env_server import _observation_payload

    class FakeEngine:
        pass

    class FakeEnv:
        engine = FakeEngine()

    info = {"observation": {"summary": {
        "has_solid": True, "valid": True, "mass_g": 12.0,
        "hole_count": 6, "feature_history": [],
    }}}
    payload = _observation_payload(
        FakeEnv(), {"numeric": [0.0], "action_mask": [1], "targets": {}}, info
    )
    assert payload["hole_count"] == 6


def test_a_rejected_action_does_not_freeze_the_observation():
    """Why cutting an episode on repeated rejections is not safe.

    A rejected action leaves the geometry untouched, which looks absorbing, but
    step_fraction advances every step, so the policy's input keeps drifting and
    the argmax eventually flips. Cutting after 8 consecutive rejections cost BC
    0.435 -> 0.321 progress and 0.342 -> 0.237 success, measured on the same 76
    tasks: the policies genuinely recover.
    """
    import numpy as np

    from kairos.language import parse_requirement
    from kairos.representation.numerical_encoder import FEATURE_NAMES, encode_numeric

    assert "step_fraction" in FEATURE_NAMES
    observation = {"summary": {"has_solid": False, "valid": False}, "holes": [],
                   "faces": [], "sketch": None, "edge_count": 0}
    spec = parse_requirement("Design a rectangular plate 40 x 40 x 5 mm")
    early = encode_numeric(observation, spec, step=1, max_steps=40)
    late = encode_numeric(observation, spec, step=20, max_steps=40)
    assert not np.allclose(early, late), (
        "identical geometry must still yield a changing observation, or the "
        "episode really would be absorbing"
    )
