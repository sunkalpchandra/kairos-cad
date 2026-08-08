"""Benchmark metric tests, pure python, no torch, no FreeCAD."""

import pytest

from kairos.benchmark import (
    MAX_PROGRESS,
    MILESTONES,
    EpisodeOutcome,
    format_scores,
    outcome_from_episode,
    score_policy,
)


def _outcome(**kwargs):
    return EpisodeOutcome(**kwargs)


def test_milestone_weights_are_strictly_dominating():
    """Each milestone must outweigh every earlier one combined.

    Otherwise the ranking of two policies would depend on the exact weights,
    and a policy could out-score a strictly better one by collecting several
    cheap milestones.
    """
    running = 0.0
    for _, weight in MILESTONES:
        assert weight > running, "a later milestone must dominate all earlier ones"
        running += weight
    assert MAX_PROGRESS == pytest.approx(running)


def test_progress_is_prefix_scored():
    """Credit stops at the first milestone missed.

    A constraint check can pass vacuously on geometry that was never built;
    awarding it would rank an empty document above a real imperfect part.
    """
    skipped = _outcome(opened_a_sketch=True, all_constraints_met=True)
    assert skipped.milestones_reached() == ["opened_a_sketch"]
    weights = dict(MILESTONES)
    assert skipped.progress_score() == pytest.approx(
        weights["opened_a_sketch"] / MAX_PROGRESS
    )


def test_progress_discriminates_when_every_policy_scores_zero_success():
    """The reason this metric exists: BC, PPO and random all succeed 0.000."""
    strong = _outcome(steps=12, opened_a_sketch=True, drew_geometry=True,
                      made_a_solid=True, solid_is_valid=True, has_any_hole=True)
    weak = _outcome(steps=9, opened_a_sketch=True, drew_geometry=True)
    nothing = _outcome(steps=6)

    assert all(not o.finished_successfully for o in (strong, weak, nothing))
    assert strong.progress_score() > weak.progress_score() > nothing.progress_score()
    assert nothing.progress_score() == 0.0


def test_a_finished_design_scores_one():
    complete = _outcome(
        steps=12, opened_a_sketch=True, drew_geometry=True, made_a_solid=True,
        solid_is_valid=True, has_any_hole=True, all_constraints_met=True,
        finished_successfully=True,
    )
    assert complete.progress_score() == pytest.approx(1.0)


def test_validity_rate_reflects_rejected_actions():
    assert _outcome(steps=10, invalid_actions=0).validity_rate() == 1.0
    assert _outcome(steps=10, invalid_actions=2).validity_rate() == pytest.approx(0.8)
    assert _outcome(steps=0).validity_rate() == 0.0  # no actions, no credit


def test_efficiency_needs_an_expert_reference():
    assert _outcome(steps=10, solid_is_valid=True).efficiency() is None
    assert _outcome(steps=10, expert_steps=5,
                    solid_is_valid=True).efficiency() == pytest.approx(0.5)
    # Beating the expert is capped: fewer steps is not unboundedly better.
    assert _outcome(steps=4, expert_steps=8,
                    solid_is_valid=True).efficiency() == pytest.approx(1.0)


def test_efficiency_is_none_when_nothing_was_built():
    """immediate-finish quit in one step and scored a perfect 1.000 here.

    A ratio of steps is not an efficiency when no solid exists, and the
    baseline's contract is that it must lose every column.
    """
    assert _outcome(steps=1, expert_steps=13, solid_is_valid=False).efficiency() is None


def test_score_policy_aggregates_and_reports_milestone_rates():
    outcomes = [
        _outcome(steps=10, opened_a_sketch=True, drew_geometry=True, made_a_solid=True,
                 solid_is_valid=True),
        _outcome(steps=10, opened_a_sketch=True),
    ]
    score = score_policy("bc", outcomes)
    assert score.episodes == 2
    assert score.milestone_rates["opened_a_sketch"] == pytest.approx(1.0)
    assert score.milestone_rates["made_a_solid"] == pytest.approx(0.5)
    assert score.milestone_rates["finished_successfully"] == 0.0
    assert 0.0 < score.progress_score < 1.0


def test_empty_policy_scores_zero_rather_than_crashing():
    score = score_policy("nothing", [])
    assert score.episodes == 0 and score.progress_score == 0.0
    assert score.milestone_rates == {}


def test_leaderboard_sorts_by_progress_not_success():
    strong = score_policy("strong", [
        _outcome(steps=10, opened_a_sketch=True, drew_geometry=True, made_a_solid=True)
    ])
    weak = score_policy("weak", [_outcome(steps=10)])
    text = format_scores([weak, strong])
    assert text.index("strong") < text.index("weak")
    assert "progress" in text and "milestone reach rates" in text


def test_outcome_reads_an_rl_episode_summary():
    # EpisodeSummary lives on the torch side; the metrics module itself
    # stays pure so it can score results under FreeCAD's interpreter too.
    pytest.importorskip("torch", reason="EpisodeSummary needs the learn extra")
    from kairos.rl.collect import EpisodeSummary

    episode = EpisodeSummary(
        requirement="Design a plate", steps=11, invalid_actions=1,
        has_solid=True, satisfaction_rate=1.0, finished_successfully=False, mass_g=42.0,
    )
    outcome = outcome_from_episode(episode, expert_steps=10)
    assert outcome.steps == 11 and outcome.made_a_solid is True
    assert outcome.validity_rate() == pytest.approx(1 - 1 / 11)
    assert outcome.efficiency() == pytest.approx(10 / 11)
    assert outcome.finished_successfully is False


def test_outcome_does_not_guess_fields_the_collector_lacks():
    """Unrecorded milestones stay False rather than being inferred."""
    pytest.importorskip("torch", reason="EpisodeSummary needs the learn extra")
    from kairos.rl.collect import EpisodeSummary

    outcome = outcome_from_episode(EpisodeSummary(requirement="x", steps=3))
    assert outcome.opened_a_sketch is False
    assert outcome.drew_geometry is False


def test_outcome_serializes_with_derived_fields():
    import json

    payload = _outcome(steps=8, expert_steps=8, opened_a_sketch=True).to_dict()
    assert payload["progress_score"] > 0
    assert payload["milestones_reached"] == ["opened_a_sketch"]
    json.dumps(payload)
