"""End-to-end VLA tests (skipped without the optional torch extra)."""

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch", reason="requires the 'learn' extra")

from kairos.language import tokenizer as tk  # noqa: E402
from kairos.models.vla import KairosVLA, VLAConfig, load_model_state  # noqa: E402
from kairos.representation.numerical_encoder import ENCODING_DIM  # noqa: E402
from kairos.rl.action_space import MAX_TARGETS, NUM_OPERATIONS, PARAM_SLOTS  # noqa: E402

CONFIG = VLAConfig(
    embed_dim=32,
    language_depth=1,
    language_heads=2,
    max_text_length=24,
    vision_widths=(8, 16),
    fusion_depth=1,
    fusion_heads=2,
    hidden_dim=32,
)


def _inputs(batch=2, with_views=False, text="Plate with 4 M5 holes, minimize mass"):
    encoded = [tk.encode(text, max_length=CONFIG.max_text_length) for _ in range(batch)]
    inputs = {
        "token_ids": torch.tensor([e[0] for e in encoded], dtype=torch.long),
        "token_values": torch.tensor([e[1] for e in encoded], dtype=torch.float32),
        "token_mask": torch.tensor([e[2] for e in encoded], dtype=torch.long),
        "numeric": torch.rand(batch, ENCODING_DIM),
        "history": torch.randint(0, 5, (batch, 8)),
    }
    if with_views:
        inputs["views"] = torch.rand(batch, 4, 3, 32, 32)
    return inputs


def test_forward_produces_the_full_action_distribution():
    model = KairosVLA(CONFIG).eval()
    out = model(**_inputs())
    assert out["operation_logits"].shape == (2, NUM_OPERATIONS)
    assert out["parameters"].shape == (2, PARAM_SLOTS)
    assert out["target_logits"].shape == (2, MAX_TARGETS)
    assert all(torch.isfinite(v).all() for v in out.values())


def test_runs_with_and_without_views():
    """Per-step renders do not exist yet; one checkpoint must serve both."""
    model = KairosVLA(replace(CONFIG, use_vision=True)).eval()
    with torch.no_grad():
        blind = model(**_inputs())["operation_logits"]
        seeing = model(**_inputs(with_views=True))["operation_logits"]
    assert blind.shape == seeing.shape
    assert not torch.allclose(blind, seeing, atol=1e-5)


def test_requirement_text_changes_the_prediction():
    """A language-conditioned policy must react to the requirement."""
    model = KairosVLA(CONFIG).eval()
    with torch.no_grad():
        plate = model(**_inputs(batch=1, text="Design a rectangular plate 100 x 60 x 6 mm"))
        spacer = model(**_inputs(batch=1, text="Design a cylindrical spacer, minimize mass"))
    assert not torch.allclose(
        plate["operation_logits"], spacer["operation_logits"], atol=1e-5
    )


def test_gradients_reach_every_submodule():
    """Passing views here is what hid the dead vision encoder for so long.

    Gradients reach `vision` in this test and never in training, because no
    training or rollout path passes `views` at all. See
    test_vision_is_off_unless_asked_for below, which pins the real default.
    """
    model = KairosVLA(replace(CONFIG, use_vision=True))
    out = model(**_inputs(with_views=True))
    (out["operation_logits"].sum() + out["parameters"].sum()).backward()
    for name in ("language", "vision", "state", "fusion", "heads"):
        module = getattr(model, name)
        grads = [p.grad for p in module.parameters() if p.requires_grad]
        assert grads, f"{name} has no parameters"
        assert any(g is not None and g.abs().sum() > 0 for g in grads), f"{name} got no gradient"


def test_vision_is_off_unless_asked_for():
    """The default model has no vision encoder, and says so when handed views."""
    model = KairosVLA(CONFIG)
    assert model.vision is None
    with pytest.raises(ValueError, match="vision encoder is disabled"):
        model(**_inputs(with_views=True))


def test_disabling_vision_drops_its_parameters():
    blind = sum(p.numel() for p in KairosVLA(CONFIG).parameters())
    seeing = sum(p.numel() for p in KairosVLA(replace(CONFIG, use_vision=True)).parameters())
    assert blind < seeing


def test_a_checkpoint_with_vision_loads_into_a_blind_model():
    """Checkpoints predate the flag and carry vision.* tensors."""
    seeing = KairosVLA(replace(CONFIG, use_vision=True))
    blind = KairosVLA(CONFIG)
    load_model_state(blind, seeing.state_dict())  # must not raise


def test_a_genuinely_wrong_checkpoint_still_fails():
    """strict=False would swallow this, and a half-loaded policy would score."""
    model = KairosVLA(CONFIG)
    state = dict(model.state_dict())
    state["nonsense.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="unexpected tensors"):
        load_model_state(model, state)


def test_config_round_trips_through_a_dict():
    restored = VLAConfig.from_dict(CONFIG.to_dict())
    assert restored == CONFIG
    assert isinstance(restored.vision_widths, tuple)  # JSON turns tuples into lists


def test_state_dict_round_trips():
    """Checkpoint reload must reproduce predictions exactly."""
    model = KairosVLA(CONFIG).eval()
    inputs = _inputs(batch=1)
    with torch.no_grad():
        before = model(**inputs)["operation_logits"]
    clone = KairosVLA(VLAConfig.from_dict(CONFIG.to_dict())).eval()
    clone.load_state_dict(model.state_dict())
    with torch.no_grad():
        after = clone(**inputs)["operation_logits"]
    assert torch.allclose(before, after, atol=1e-6)


def test_parameter_count_is_reported():
    assert KairosVLA(CONFIG).parameter_count() > 0
