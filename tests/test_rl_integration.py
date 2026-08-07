"""Full Phase 5 stack against a real FreeCAD environment.

Runs under the **torch** interpreter (the one that can import torch) and drives
FreeCAD over the bridge, so it is the only test that exercises the whole loop
together: policy → action → subprocess → FreeCAD recompute → reward → GAE →
PPO update. Skipped when FreeCAD is not installed.
"""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="requires the 'learn' extra")

from kairos.models.actor_critic import ActorCritic  # noqa: E402
from kairos.models.vla import KairosVLA, VLAConfig  # noqa: E402
from kairos.rl.buffer import RolloutBuffer  # noqa: E402
from kairos.rl.collect import RolloutCollector  # noqa: E402
from kairos.rl.env_client import RemoteCADEnv, resolve_freecad_python  # noqa: E402
from kairos.rl.ppo import PPOConfig, PPOTrainer  # noqa: E402
from kairos.rl.requirements import FALLBACK_REQUIREMENTS  # noqa: E402
from kairos.rl.train_loop import LoopConfig, PPOTrainingLoop  # noqa: E402

try:
    FREECAD_PYTHON = resolve_freecad_python()
except FileNotFoundError:
    FREECAD_PYTHON = None

pytestmark = pytest.mark.skipif(
    FREECAD_PYTHON is None or not Path(FREECAD_PYTHON).exists(),
    reason="FreeCAD interpreter not found",
)

TINY = VLAConfig(
    embed_dim=16, language_depth=1, language_heads=2, max_text_length=32,
    vision_widths=(8,), fusion_depth=1, fusion_heads=2, hidden_dim=16, dropout=0.0,
)
REQUIREMENTS = list(FALLBACK_REQUIREMENTS[:2])


@pytest.fixture
def env():
    environment = RemoteCADEnv(max_steps=6)
    yield environment
    environment.close()


def test_policy_drives_a_real_cad_environment(env):
    model = ActorCritic(KairosVLA(TINY))
    collector = RolloutCollector(env, model, REQUIREMENTS, max_episode_steps=6)
    buffer = RolloutBuffer()
    episodes = collector.collect(buffer, n_steps=12)

    assert len(buffer) > 0
    assert episodes and all(e.steps > 0 for e in episodes)
    # Every action reached FreeCAD and came back with a verdict.
    assert all("ok" in t.info for t in buffer.transitions)


def test_ppo_updates_on_real_rollouts(env):
    model = ActorCritic(KairosVLA(TINY))
    trainer = PPOTrainer(
        model, PPOConfig(epochs_per_update=1, minibatch_size=8, target_kl=None)
    )
    collector = RolloutCollector(env, model, REQUIREMENTS, max_episode_steps=6)
    buffer = RolloutBuffer()
    collector.collect(buffer, n_steps=12)

    before = model.parameter_log_std.detach().clone()
    metrics = trainer.update(buffer)
    assert metrics.epochs_run == 1
    assert not torch.allclose(model.parameter_log_std.detach(), before)
    assert torch.isfinite(torch.tensor(metrics.policy_loss))


def test_training_loop_runs_end_to_end(env, tmp_path):
    model = ActorCritic(KairosVLA(TINY))
    trainer = PPOTrainer(
        model, PPOConfig(epochs_per_update=1, minibatch_size=8, target_kl=None)
    )
    collector = RolloutCollector(env, model, REQUIREMENTS, max_episode_steps=6)
    loop = PPOTrainingLoop(
        trainer,
        collector,
        LoopConfig(
            iterations=1, steps_per_iteration=8, max_episode_steps=6,
            eval_every=1, eval_episodes=1, out_dir=str(tmp_path / "run"),
        ),
        eval_requirements=REQUIREMENTS[:1],
    )
    history = loop.run()

    assert len(history) == 1
    assert history[0].evaluation is not None
    assert (tmp_path / "run" / "last.pt").exists()
    assert (tmp_path / "run" / "history.json").exists()
    assert env.restarts == 0, "the environment should survive a short run"


def test_invalid_actions_are_scored_not_crashed(env):
    """A policy will emit nonsense early; that must be a penalty, not a stop."""
    from kairos.actions.schema import Operation
    from kairos.rl.action_space import OPERATIONS

    env.reset(requirement=REQUIREMENTS[0])
    # PAD with no sketch is illegal in an empty document.
    _, reward, terminated, truncated, info = env.step(
        OPERATIONS.index(Operation.PAD), [0.5] * 6
    )
    assert info.get("ok") is False
    assert reward < 0  # penalized
    assert not terminated and not truncated  # the episode continues
