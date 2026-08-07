"""Actor-critic tests (skipped without the optional torch extra)."""

import pytest

torch = pytest.importorskip("torch", reason="requires the 'learn' extra")

from kairos.models.actor_critic import ActorCritic, load_actor_critic  # noqa: E402
from kairos.models.vla import KairosVLA, VLAConfig  # noqa: E402
from kairos.rl.action_space import NUM_OPERATIONS, PARAM_SLOTS  # noqa: E402

TINY = VLAConfig(
    embed_dim=16, language_depth=1, language_heads=2, max_text_length=16,
    vision_widths=(8,), fusion_depth=1, fusion_heads=2, hidden_dim=16, dropout=0.0,
)


def _inputs(batch=4, mask=None):
    inputs = {
        "token_ids": torch.ones(batch, 16, dtype=torch.long),
        "token_values": torch.zeros(batch, 16),
        "token_mask": torch.ones(batch, 16, dtype=torch.long),
        "numeric": torch.rand(batch, 24),
        "history": torch.zeros(batch, 8, dtype=torch.long),
    }
    if mask is not None:
        inputs["operation_mask"] = mask
    return inputs


def _model(**kwargs):
    return ActorCritic(KairosVLA(TINY), **kwargs).eval()


def test_act_returns_action_log_prob_and_value():
    out = _model().act(_inputs())
    assert out["action"]["operation"].shape == (4,)
    assert out["action"]["parameters"].shape == (4, PARAM_SLOTS)
    assert out["log_prob"].shape == (4,) and torch.isfinite(out["log_prob"]).all()
    assert out["value"].shape == (4,) and torch.isfinite(out["value"]).all()


def test_sampled_parameters_are_inside_the_codec_range():
    parameters = _model().act(_inputs(batch=64))["action"]["parameters"]
    assert (parameters > 0.0).all() and (parameters < 1.0).all()


def test_act_never_selects_a_masked_operation():
    mask = torch.zeros(4, NUM_OPERATIONS, dtype=torch.long)
    mask[:, 3] = 1
    model = _model()
    for _ in range(10):
        assert (model.act(_inputs(mask=mask))["action"]["operation"] == 3).all()


def test_value_head_starts_near_zero():
    """A near-zero critic makes early advantages reflect reward, not noise."""
    values = _model().act(_inputs(batch=16))["value"]
    assert values.abs().max() < 0.1


def test_evaluate_actions_reproduces_the_sampling_log_prob():
    """PPO's ratio is 1.0 at the start of an epoch only if these agree."""
    model = _model()
    inputs = _inputs()
    sampled = model.act(inputs)
    with torch.no_grad():
        rescored = model.evaluate_actions(inputs, sampled["action"])
    assert torch.allclose(rescored["log_prob"], sampled["log_prob"], atol=1e-5)


def test_deterministic_action_is_repeatable():
    model = _model()
    inputs = _inputs()
    first = model.act(inputs, deterministic=True)["action"]
    second = model.act(inputs, deterministic=True)["action"]
    assert torch.equal(first["operation"], second["operation"])
    assert torch.allclose(first["parameters"], second["parameters"])


def test_sampling_actually_varies():
    """A collapsed policy cannot explore; sampling must not be deterministic."""
    model = _model()
    inputs = _inputs(batch=32)
    a = model.act(inputs)["action"]["parameters"]
    b = model.act(inputs)["action"]["parameters"]
    assert not torch.allclose(a, b)


def test_log_std_is_learnable_and_shared_across_states():
    model = _model()
    assert model.parameter_log_std.requires_grad
    assert model.parameter_log_std.shape == (PARAM_SLOTS,)


def test_gradients_flow_to_trunk_value_and_log_std():
    model = ActorCritic(KairosVLA(TINY))
    inputs = _inputs()
    distribution, value = model.distribution(inputs)
    action = distribution.sample()
    loss = distribution.log_prob(action).sum() + value.sum()
    loss.backward()

    assert model.parameter_log_std.grad is not None
    assert model.parameter_log_std.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.value_head.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.vla.parameters())


def test_bc_checkpoint_initializes_the_policy(tmp_path):
    """PPO from scratch is hopeless here; the BC policy must carry over."""
    vla = KairosVLA(TINY)
    path = tmp_path / "bc.pt"
    torch.save({"model_state": vla.state_dict(), "model_config": TINY.to_dict()}, path)

    model = ActorCritic.from_bc_checkpoint(path)
    for (name, before), after in zip(
        vla.state_dict().items(), model.vla.state_dict().values(), strict=True
    ):
        assert torch.allclose(before, after), name


def test_checkpoint_round_trips(tmp_path):
    model = _model()
    inputs = _inputs()
    with torch.no_grad():
        before = model.distribution(inputs)[1]
    path = model.save(tmp_path / "ppo.pt")
    restored = load_actor_critic(path).eval()
    with torch.no_grad():
        after = restored.distribution(inputs)[1]
    assert torch.allclose(before, after, atol=1e-6)
