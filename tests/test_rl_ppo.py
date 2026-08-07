"""PPO tests (skipped without the optional torch extra)."""

import pytest

torch = pytest.importorskip("torch", reason="requires the 'learn' extra")

from kairos.models.actor_critic import ActorCritic  # noqa: E402
from kairos.models.vla import KairosVLA, VLAConfig  # noqa: E402
from kairos.rl.buffer import RolloutBuffer  # noqa: E402
from kairos.rl.ppo import PPOConfig, PPOTrainer  # noqa: E402

TINY = VLAConfig(
    embed_dim=16, language_depth=1, language_heads=2, max_text_length=8,
    vision_widths=(8,), fusion_depth=1, fusion_heads=2, hidden_dim=16, dropout=0.0,
)


def _inputs():
    return {
        "token_ids": torch.ones(1, 8, dtype=torch.long),
        "token_values": torch.zeros(1, 8),
        "token_mask": torch.ones(1, 8, dtype=torch.long),
        "numeric": torch.rand(1, 24),
        "history": torch.zeros(1, 4, dtype=torch.long),
    }


def _rollout(model, n=24, reward=1.0, seed=0):
    torch.manual_seed(seed)
    buffer = RolloutBuffer()
    for i in range(n):
        inputs = _inputs()
        out = model.act(inputs)
        terminated = (i + 1) % 8 == 0
        buffer.add(
            inputs=inputs,
            action=out["action"],
            log_prob=out["log_prob"],
            value=out["value"],
            reward=reward,
            terminated=terminated,
            truncated=False,
            info={"ok": True},
        )
    return buffer


def test_update_on_an_empty_buffer_is_a_no_op():
    model = ActorCritic(KairosVLA(TINY))
    metrics = PPOTrainer(model, PPOConfig()).update(RolloutBuffer())
    assert metrics.epochs_run == 0 and metrics.policy_loss == 0.0


def test_update_changes_parameters_and_reports_metrics():
    model = ActorCritic(KairosVLA(TINY))
    trainer = PPOTrainer(model, PPOConfig(epochs_per_update=2, minibatch_size=8, target_kl=None))
    before = model.parameter_log_std.detach().clone()
    metrics = trainer.update(_rollout(model))

    assert metrics.epochs_run == 2
    assert not torch.allclose(model.parameter_log_std.detach(), before)
    for value in (metrics.policy_loss, metrics.value_loss, metrics.entropy, metrics.approx_kl):
        assert torch.isfinite(torch.tensor(value))
    assert 0.0 <= metrics.clip_fraction <= 1.0


def test_value_function_learns_a_constant_return():
    """With a fixed reward the critic should explain most of the variance."""
    model = ActorCritic(KairosVLA(TINY))
    trainer = PPOTrainer(
        model,
        PPOConfig(
            learning_rate=3e-3, epochs_per_update=8, minibatch_size=8,
            entropy_coef=0.0, bc_kl_coef=0.0, target_kl=None,
        ),
    )
    losses = []
    for _ in range(6):
        losses.append(trainer.update(_rollout(model, n=32, reward=1.0)).value_loss)
    assert losses[-1] < losses[0]


def test_bc_anchor_keeps_the_policy_near_its_initialization():
    """Without an anchor the policy is free to drift off the BC manifold."""
    torch.manual_seed(0)
    anchored = ActorCritic(KairosVLA(TINY))
    free = ActorCritic(KairosVLA(TINY))
    free.load_state_dict(anchored.state_dict())
    start = {k: v.detach().clone() for k, v in anchored.state_dict().items()}

    shared = PPOConfig(learning_rate=3e-3, epochs_per_update=4, minibatch_size=8, target_kl=None)
    anchored_trainer = PPOTrainer(anchored, PPOConfig(**{**shared.to_dict(), "bc_kl_coef": 5.0}))
    free_trainer = PPOTrainer(free, PPOConfig(**{**shared.to_dict(), "bc_kl_coef": 0.0}))

    for _ in range(4):
        anchored_trainer.update(_rollout(anchored, n=24, reward=2.0, seed=1))
        free_trainer.update(_rollout(free, n=24, reward=2.0, seed=1))

    def drift(model):
        return sum(
            float((v.detach() - start[k]).abs().sum())
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        )

    assert drift(anchored) < drift(free)


def test_no_reference_model_is_kept_when_the_anchor_is_disabled():
    model = ActorCritic(KairosVLA(TINY))
    assert PPOTrainer(model, PPOConfig(bc_kl_coef=0.0)).reference is None
    assert PPOTrainer(model, PPOConfig(bc_kl_coef=0.1)).reference is not None


def test_reference_policy_is_frozen():
    trainer = PPOTrainer(ActorCritic(KairosVLA(TINY)), PPOConfig(bc_kl_coef=0.1))
    assert all(not p.requires_grad for p in trainer.reference.parameters())
    before = [p.detach().clone() for p in trainer.reference.parameters()]
    trainer.update(_rollout(trainer.model))
    for old, new in zip(before, trainer.reference.parameters(), strict=True):
        assert torch.allclose(old, new)


def test_target_kl_stops_the_update_early():
    model = ActorCritic(KairosVLA(TINY))
    trainer = PPOTrainer(
        model,
        PPOConfig(learning_rate=1e-1, epochs_per_update=8, minibatch_size=8, target_kl=1e-6),
    )
    metrics = trainer.update(_rollout(model, n=32, reward=5.0))
    assert metrics.stopped_early is True
    assert metrics.epochs_run < 8


def test_gradients_are_clipped():
    model = ActorCritic(KairosVLA(TINY))
    trainer = PPOTrainer(
        model, PPOConfig(max_grad_norm=1e-4, learning_rate=1e-2, epochs_per_update=1)
    )
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    trainer.update(_rollout(model, n=16, reward=100.0))
    moved = max(
        float((v.detach() - before[k]).abs().max())
        for k, v in model.state_dict().items()
        if v.dtype.is_floating_point
    )
    assert moved < 0.05  # a huge reward cannot produce a huge step


def test_dropout_is_off_during_the_update():
    """Stored log-probs come from the eval-mode policy; re-scoring must match.

    With dropout active the ratio compares two different functions and reads as
    a large spurious KL from the very first minibatch.
    """
    config = VLAConfig(**{**TINY.to_dict(), "dropout": 0.5})
    model = ActorCritic(KairosVLA(config))
    trainer = PPOTrainer(model, PPOConfig(epochs_per_update=1, minibatch_size=32, target_kl=None))
    buffer = _rollout(model, n=16, reward=1.0)

    # A single epoch at lr=0 must leave the policy exactly where it was, so any
    # measured KL is dropout noise rather than a real update.
    trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)
    metrics = trainer.update(buffer)
    assert abs(metrics.approx_kl) < 1e-5
