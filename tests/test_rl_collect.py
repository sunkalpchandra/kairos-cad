"""Rollout collection tests using a scripted environment (no FreeCAD)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="requires the 'learn' extra")

from kairos.models.actor_critic import ActorCritic  # noqa: E402
from kairos.models.vla import KairosVLA, VLAConfig  # noqa: E402
from kairos.rl.action_space import NUM_OPERATIONS  # noqa: E402
from kairos.rl.buffer import RolloutBuffer  # noqa: E402
from kairos.rl.collect import (  # noqa: E402
    RolloutCollector,
    build_inputs,
    summarize_episodes,
)

TINY = VLAConfig(
    embed_dim=16, language_depth=1, language_heads=2, max_text_length=16,
    vision_widths=(8,), fusion_depth=1, fusion_heads=2, hidden_dim=16, dropout=0.0,
)
REQUIREMENTS = ["Design a plate 60 x 40 x 5 mm", "Design a spacer of 20 mm height"]


class ScriptedEnv:
    """Ends every episode after `episode_length` steps in a chosen way."""

    def __init__(self, episode_length=3, ending="terminated", reset_error=None, mask=None):
        self.episode_length = episode_length
        self.ending = ending
        self.reset_error = reset_error
        self.mask = mask
        self.step_count = 0
        self.requirements_seen = []

    def _obs(self):
        mask = self.mask if self.mask is not None else np.ones(NUM_OPERATIONS, dtype=np.int64)
        return {
            "numeric": np.zeros(24, dtype=np.float32),
            "action_mask": mask,
            "feature_history": ["Pad"] if self.step_count else [],
            "has_solid": self.step_count > 1,
            "valid": True,
            "mass_g": 10.0 * self.step_count,
        }

    def reset(self, requirement=None, seed=None):
        if self.reset_error:
            raise self.reset_error
        self.requirements_seen.append(requirement)
        self.step_count = 0
        return self._obs()

    def step(self, operation, params, target=0):
        self.step_count += 1
        last = self.step_count >= self.episode_length
        info = {
            "operation": "PAD",
            "ok": True,
            "satisfaction_rate": 1.0,
            "all_satisfied": True,
            "reward_components": {"finish": 5.0} if last else {},
        }
        if last and self.ending == "crashed":
            return None, 0.0, False, True, {"ok": False, "crashed": True, "message": "died"}
        return (
            self._obs(),
            1.0,
            last and self.ending == "terminated",
            last and self.ending == "truncated",
            info,
        )


def _model():
    return ActorCritic(KairosVLA(TINY))


def _collector(env, **kwargs):
    return RolloutCollector(env, _model(), REQUIREMENTS, max_episode_steps=10, **kwargs)


def test_build_inputs_shapes_match_the_model():
    env = ScriptedEnv()
    inputs = build_inputs(
        REQUIREMENTS[0], env.reset(), max_text_length=TINY.max_text_length
    )
    assert inputs["numeric"].shape == (1, 24)
    assert inputs["operation_mask"].shape == (1, NUM_OPERATIONS)
    assert inputs["token_ids"].shape == (1, TINY.max_text_length)
    _model().act(inputs)  # must not raise


def test_text_longer_than_the_encoder_fails_with_a_readable_error():
    """A silent IndexError from the position embedding wastes hours."""
    inputs = build_inputs(REQUIREMENTS[0], ScriptedEnv().reset(), max_text_length=64)
    with pytest.raises(ValueError, match="max_length=16"):
        _model().act(inputs)


def test_collector_takes_the_text_length_from_the_model():
    collector = _collector(ScriptedEnv())
    assert collector.max_text_length == TINY.max_text_length


def test_collect_fills_the_buffer_and_records_episodes():
    buffer = RolloutBuffer()
    episodes = _collector(ScriptedEnv(episode_length=3)).collect(buffer, n_steps=9)
    assert len(buffer) == 9
    assert len(episodes) == 3
    assert all(e.steps == 3 for e in episodes)
    assert all(e.terminated for e in episodes)


def test_requirements_are_sampled_per_episode():
    """One fixed requirement would give the language encoder no signal."""
    env = ScriptedEnv(episode_length=1)
    _collector(env, seed=0).collect(RolloutBuffer(), n_steps=40)
    assert set(env.requirements_seen) == set(REQUIREMENTS)


def test_collection_stops_at_the_step_budget():
    buffer = RolloutBuffer()
    _collector(ScriptedEnv(episode_length=10)).collect(buffer, n_steps=4)
    assert len(buffer) == 4


def test_successful_finish_is_recorded_from_the_environment_verdict():
    episodes = _collector(ScriptedEnv(episode_length=2, ending="terminated")).collect(
        RolloutBuffer(), n_steps=2
    )
    assert episodes[0].finished_successfully is True


def test_truncation_is_not_counted_as_success():
    episodes = _collector(ScriptedEnv(episode_length=2, ending="truncated")).collect(
        RolloutBuffer(), n_steps=2
    )
    assert episodes[0].terminated is False
    assert episodes[0].finished_successfully is False


def test_truncated_transitions_carry_a_bootstrap_value():
    """Otherwise GAE would treat a step-budget cutoff as catastrophe."""
    buffer = RolloutBuffer()
    _collector(ScriptedEnv(episode_length=2, ending="truncated")).collect(buffer, n_steps=2)
    assert buffer._bootstrap, "no bootstrap recorded for a truncated ending"


def test_a_crashed_environment_truncates_the_episode():
    episodes = _collector(ScriptedEnv(episode_length=2, ending="crashed")).collect(
        RolloutBuffer(), n_steps=2
    )
    assert episodes[0].crashed is True
    assert episodes[0].truncated is True
    assert episodes[0].terminated is False


def test_a_failed_reset_does_not_hang_collection():
    episodes = _collector(ScriptedEnv(reset_error=RuntimeError("no FreeCAD"))).collect(
        RolloutBuffer(), n_steps=50
    )
    assert len(episodes) == 1 and episodes[0].steps == 0 and episodes[0].crashed


def test_masked_operations_are_never_taken():
    mask = np.zeros(NUM_OPERATIONS, dtype=np.int64)
    mask[7] = 1
    env = ScriptedEnv(episode_length=5, mask=mask)
    buffer = RolloutBuffer()
    _collector(env).collect(buffer, n_steps=10)
    operations = torch.stack([t.action["operation"].reshape(()) for t in buffer.transitions])
    assert (operations == 7).all()


def test_summary_reports_rates_over_scored_episodes():
    episodes = _collector(ScriptedEnv(episode_length=2)).collect(RolloutBuffer(), n_steps=6)
    stats = summarize_episodes(episodes)
    assert stats["episodes"] == 3 and stats["steps"] == 6
    assert stats["success_rate"] == pytest.approx(1.0)
    assert stats["invalid_action_rate"] == pytest.approx(0.0)
    assert summarize_episodes([]) == {"episodes": 0}


def test_collector_requires_at_least_one_requirement():
    with pytest.raises(ValueError, match="at least one requirement"):
        RolloutCollector(ScriptedEnv(), _model(), [])
