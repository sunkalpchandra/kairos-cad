"""Policy comparison tests using a scripted environment (no FreeCAD)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="requires the 'learn' extra")

from kairos.models.actor_critic import ActorCritic  # noqa: E402
from kairos.models.vla import KairosVLA, VLAConfig  # noqa: E402
from kairos.rl.action_space import NUM_OPERATIONS  # noqa: E402
from kairos.rl.evaluate import (  # noqa: E402
    RandomPolicy,
    bootstrap_interval,
    compare_policies,
    evaluate_policy,
    format_comparison,
)

TINY = VLAConfig(
    embed_dim=16, language_depth=1, language_heads=2, max_text_length=16,
    vision_widths=(8,), fusion_depth=1, fusion_heads=2, hidden_dim=16, dropout=0.0,
)
REQUIREMENTS = ["Design a plate 60 x 40 x 5 mm", "Design a spacer 20 mm tall"]


class ScriptedEnv:
    def __init__(self, episode_length=3, success=True, mask=None):
        self.episode_length = episode_length
        self.success = success
        self.mask = mask
        self.step_count = 0
        self.operations_taken = []
        self.requirements_seen = []

    def _obs(self):
        mask = self.mask if self.mask is not None else np.ones(NUM_OPERATIONS, dtype=np.int64)
        return {
            "numeric": np.zeros(24, dtype=np.float32),
            "action_mask": mask,
            "feature_history": [],
            "has_solid": True,
            "valid": True,
            "mass_g": 3.0,
        }

    def reset(self, requirement=None, seed=None):
        self.requirements_seen.append(requirement)
        self.step_count = 0
        return self._obs()

    def step(self, operation, params, target=0):
        self.operations_taken.append(int(operation))
        self.step_count += 1
        last = self.step_count >= self.episode_length
        info = {
            "operation": "PAD", "ok": True, "satisfaction_rate": 1.0,
            "all_satisfied": self.success,
            "reward_components": {"finish": 5.0 if self.success else -1.0} if last else {},
        }
        return self._obs(), 1.0, last, False, info


def _model():
    return ActorCritic(KairosVLA(TINY))


def test_evaluate_reports_success_and_episode_counts():
    result = evaluate_policy(
        ScriptedEnv(), _model(), REQUIREMENTS, episodes=3, max_episode_steps=5
    )
    assert result["episodes"] == 3
    assert result["success_rate"] == pytest.approx(1.0)
    assert result["deterministic"] is True


def test_failed_finishes_are_not_success():
    result = evaluate_policy(
        ScriptedEnv(success=False), _model(), REQUIREMENTS, episodes=2, max_episode_steps=5
    )
    assert result["success_rate"] == pytest.approx(0.0)


def test_random_policy_only_picks_legal_operations():
    """The baseline must be legal-random, or it is not a fair floor."""
    mask = np.zeros(NUM_OPERATIONS, dtype=np.int64)
    mask[4] = mask[9] = 1
    env = ScriptedEnv(episode_length=4, mask=mask)
    evaluate_policy(env, RandomPolicy(max_text_length=16), REQUIREMENTS, episodes=3,
                    max_episode_steps=4, deterministic=False)
    assert set(env.operations_taken) <= {4, 9}
    assert len(set(env.operations_taken)) == 2  # actually varies


def test_random_policy_drives_the_same_collector_as_a_real_one():
    result = evaluate_policy(
        ScriptedEnv(), RandomPolicy(max_text_length=16), REQUIREMENTS,
        episodes=2, max_episode_steps=4, deterministic=False,
    )
    assert result["episodes"] == 2


def test_comparison_uses_identical_requirements_for_every_policy():
    """Different requirement draws would make the comparison meaningless."""
    env = ScriptedEnv(episode_length=2)
    compare_policies(
        env,
        {"a": _model(), "b": _model()},
        REQUIREMENTS, episodes=3, max_episode_steps=4, seed=7,
    )
    half = len(env.requirements_seen) // 2
    assert env.requirements_seen[:half] == env.requirements_seen[half:]


def test_comparison_returns_a_row_per_policy():
    results = compare_policies(
        ScriptedEnv(), {"ppo": _model(), "random": RandomPolicy(max_text_length=16)},
        REQUIREMENTS, episodes=2, max_episode_steps=4,
    )
    assert set(results) == {"ppo", "random"}
    text = format_comparison(results)
    assert "ppo" in text and "random" in text and "success" in text


def test_format_handles_a_policy_with_no_episodes():
    assert "no episodes" in format_comparison({"broken": {"episodes": 0}})


def test_bootstrap_interval_brackets_the_mean():
    values = [1.0] * 5 + [0.0] * 5
    low, high = bootstrap_interval(values, samples=500, seed=0)
    assert low <= 0.5 <= high
    assert low < high


def test_bootstrap_of_a_constant_is_degenerate():
    low, high = bootstrap_interval([1.0] * 8, samples=200)
    assert low == pytest.approx(1.0) and high == pytest.approx(1.0)


def test_bootstrap_of_nothing_is_nan():
    low, high = bootstrap_interval([])
    assert np.isnan(low) and np.isnan(high)


def test_evaluation_reports_a_confidence_interval():
    """A dozen episodes cannot support a bare point estimate."""
    result = evaluate_policy(
        ScriptedEnv(), _model(), REQUIREMENTS, episodes=4, max_episode_steps=5
    )
    low, high = result["success_ci"]
    assert low <= result["success_rate"] <= high


def test_per_requirement_breakdown_covers_every_episode():
    """One dominant requirement could otherwise carry the aggregate."""
    result = evaluate_policy(
        ScriptedEnv(), _model(), REQUIREMENTS, episodes=6, max_episode_steps=4
    )
    breakdown = result["per_requirement"]
    assert breakdown
    assert sum(row["episodes"] for row in breakdown.values()) == result["episodes"]
    assert all(0.0 <= row["success_rate"] <= 1.0 for row in breakdown.values())


def test_comparison_table_shows_the_interval():
    results = compare_policies(
        ScriptedEnv(), {"ppo": _model()}, REQUIREMENTS, episodes=2, max_episode_steps=4
    )
    assert "95% CI" in format_comparison(results)
