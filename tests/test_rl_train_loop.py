"""PPO training loop tests using a scripted environment (no FreeCAD)."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="requires the 'learn' extra")

from kairos.models.actor_critic import ActorCritic  # noqa: E402
from kairos.models.vla import KairosVLA, VLAConfig  # noqa: E402
from kairos.rl.action_space import NUM_OPERATIONS  # noqa: E402
from kairos.rl.collect import RolloutCollector  # noqa: E402
from kairos.rl.ppo import PPOConfig, PPOTrainer  # noqa: E402
from kairos.rl.train_loop import LoopConfig, PPOTrainingLoop  # noqa: E402

TINY = VLAConfig(
    embed_dim=16, language_depth=1, language_heads=2, max_text_length=16,
    vision_widths=(8,), fusion_depth=1, fusion_heads=2, hidden_dim=16, dropout=0.0,
)
TRAIN_REQUIREMENTS = ["Design a plate 60 x 40 x 5 mm", "Design a spacer 20 mm tall"]
EVAL_REQUIREMENTS = ["Design a flange of outer diameter 80 mm"]


class ScriptedEnv:
    def __init__(self, episode_length=4, success=True):
        self.episode_length = episode_length
        self.success = success
        self.step_count = 0
        self.requirements_seen = []

    def _obs(self):
        return {
            "numeric": np.zeros(24, dtype=np.float32),
            "action_mask": np.ones(NUM_OPERATIONS, dtype=np.int64),
            "feature_history": [],
            "has_solid": True,
            "valid": True,
            "mass_g": 5.0,
        }

    def reset(self, requirement=None, seed=None):
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
            "all_satisfied": self.success,
            "reward_components": {"finish": 5.0 if self.success else -1.0} if last else {},
        }
        return self._obs(), 1.0, last, False, info


def _loop(tmp_path, env=None, **loop_kwargs):
    env = env or ScriptedEnv()
    model = ActorCritic(KairosVLA(TINY))
    trainer = PPOTrainer(model, PPOConfig(epochs_per_update=1, minibatch_size=8, target_kl=None))
    collector = RolloutCollector(env, model, TRAIN_REQUIREMENTS, max_episode_steps=8)
    settings = {
        "iterations": 2, "steps_per_iteration": 8, "max_episode_steps": 8,
        "eval_every": 1, "eval_episodes": 2, "out_dir": str(tmp_path / "run"),
    }
    settings.update(loop_kwargs)
    config = LoopConfig(**settings)
    return PPOTrainingLoop(trainer, collector, config, eval_requirements=EVAL_REQUIREMENTS), env


def test_run_produces_one_record_per_iteration(tmp_path):
    loop, _ = _loop(tmp_path)
    history = loop.run()
    assert [r.iteration for r in history] == [1, 2]
    assert all(r.rollout["steps"] > 0 for r in history)
    assert all("policy_loss" in r.update for r in history)
    assert all(r.seconds >= 0 for r in history)


def test_evaluation_uses_the_held_out_requirements(tmp_path):
    loop, env = _loop(tmp_path)
    loop.run()
    seen = set(env.requirements_seen)
    assert EVAL_REQUIREMENTS[0] in seen  # evaluation happened
    assert seen & set(TRAIN_REQUIREMENTS)  # training happened too


def test_best_checkpoint_tracks_the_best_evaluation(tmp_path):
    loop, _ = _loop(tmp_path)
    loop.run()
    assert loop.best_iteration in (1, 2)
    assert loop.best_success_rate == pytest.approx(1.0)
    assert (tmp_path / "run" / "best.pt").exists()


def test_a_worse_later_iteration_does_not_overwrite_best(tmp_path):
    """RL is not monotone; the last policy is often not the best one."""
    loop, _ = _loop(tmp_path)
    loop.run()
    best_before = loop.best_success_rate
    loop.best_success_rate = 2.0  # pretend an earlier iteration did better
    loop.best_iteration = 99
    loop.config.iterations = 1
    loop.run()
    assert loop.best_iteration == 99
    assert loop.best_success_rate > best_before


def test_last_checkpoint_and_history_are_written(tmp_path):
    loop, _ = _loop(tmp_path)
    loop.run()
    assert (tmp_path / "run" / "last.pt").exists()
    history = json.loads((tmp_path / "run" / "history.json").read_text())
    assert len(history) == 2 and history[0]["iteration"] == 1


def test_failed_finishes_are_not_counted_as_success(tmp_path):
    loop, _ = _loop(tmp_path, env=ScriptedEnv(success=False))
    loop.run()
    assert loop.best_success_rate == pytest.approx(0.0)


def test_eval_every_zero_skips_evaluation(tmp_path):
    loop, _ = _loop(tmp_path, eval_every=0)
    history = loop.run()
    assert all(r.evaluation is None for r in history)
    assert not (tmp_path / "run" / "best.pt").exists()


def test_report_summarizes_the_run(tmp_path):
    loop, _ = _loop(tmp_path)
    loop.run()
    report = loop.report()
    assert report["iterations"] == 2
    assert report["best_iteration"] in (1, 2)
    assert report["final_evaluation"]["success_rate"] == pytest.approx(1.0)
    assert "clip_range" in report["ppo_config"]
    json.dumps(report)  # must stay serializable


def test_on_iteration_callback_fires(tmp_path):
    loop, _ = _loop(tmp_path)
    seen = []
    loop.run(on_iteration=seen.append)
    assert [r.iteration for r in seen] == [1, 2]


def test_resume_restores_history_and_best(tmp_path):
    """A multi-hour run that cannot restart is one crash from worthless."""
    loop, _ = _loop(tmp_path)
    loop.run()
    first_best = loop.best_success_rate
    assert len(loop.history) == 2

    resumed, _ = _loop(tmp_path)
    last_iteration = resumed.resume_from(tmp_path / "run")
    assert last_iteration == 2
    assert len(resumed.history) == 2
    assert resumed.best_success_rate == pytest.approx(first_best)


def test_resumed_iterations_continue_the_numbering(tmp_path):
    loop, _ = _loop(tmp_path)
    loop.run()

    resumed, _ = _loop(tmp_path)
    start = resumed.resume_from(tmp_path / "run")
    resumed.config.iterations = 2
    history = resumed.run(start_iteration=start)
    assert [r.iteration for r in history] == [1, 2, 3, 4]


def test_resume_from_an_empty_directory_starts_at_zero(tmp_path):
    loop, _ = _loop(tmp_path)
    assert loop.resume_from(tmp_path / "nothing_here") == 0
    assert loop.history == []
