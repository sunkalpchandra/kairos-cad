"""Policy head tests (skipped without the optional torch extra)."""

import pytest

torch = pytest.importorskip("torch", reason="requires the 'learn' extra")

from kairos.models.policy import MASK_FILL, PolicyHeads  # noqa: E402
from kairos.rl.action_space import MAX_TARGETS, NUM_OPERATIONS, PARAM_SLOTS  # noqa: E402

DIM = 32


def _fused(batch=2, seed=0):
    return torch.rand(batch, DIM, generator=torch.Generator().manual_seed(seed))


def test_output_shapes_match_the_action_space():
    heads = PolicyHeads(embed_dim=DIM).eval()
    out = heads(_fused())
    assert out["operation_logits"].shape == (2, NUM_OPERATIONS)
    assert out["parameters"].shape == (2, PARAM_SLOTS)
    assert out["target_logits"].shape == (2, MAX_TARGETS)


def test_parameters_stay_in_the_codec_range():
    """The codec denormalizes from [0, 1]; anything else silently clips."""
    heads = PolicyHeads(embed_dim=DIM).eval()
    params = heads(_fused(batch=8, seed=3))["parameters"]
    assert (params >= 0.0).all() and (params <= 1.0).all()


def test_illegal_operations_cannot_be_selected():
    heads = PolicyHeads(embed_dim=DIM).eval()
    mask = torch.zeros(2, NUM_OPERATIONS, dtype=torch.long)
    mask[:, 5] = 1  # exactly one legal operation
    out = heads(_fused(), operation_mask=mask)
    assert (out["operation_logits"].argmax(dim=-1) == 5).all()
    illegal = out["operation_logits"][mask == 0]
    assert (illegal == MASK_FILL).all()


def test_masked_logits_survive_softmax_without_nan():
    """A fully masked row must not poison the loss with NaNs."""
    heads = PolicyHeads(embed_dim=DIM).eval()
    mask = torch.zeros(1, NUM_OPERATIONS, dtype=torch.long)
    logits = heads(_fused(batch=1), operation_mask=mask)["operation_logits"]
    probabilities = torch.softmax(logits, dim=-1)
    assert torch.isfinite(probabilities).all()
    assert probabilities.sum().item() == pytest.approx(1.0)


def test_parameters_are_conditioned_on_the_operation():
    """PAD's length and FILLET's radius share a slot but are different things."""
    heads = PolicyHeads(embed_dim=DIM).eval()
    fused = _fused(batch=1)
    with torch.no_grad():
        as_pad = heads(fused, operation=torch.tensor([0]))["parameters"]
        as_fillet = heads(fused, operation=torch.tensor([7]))["parameters"]
    assert not torch.allclose(as_pad, as_fillet, atol=1e-5)


def test_target_mask_restricts_selection():
    heads = PolicyHeads(embed_dim=DIM).eval()
    mask = torch.zeros(1, MAX_TARGETS, dtype=torch.long)
    mask[:, 3] = 1
    _, _, target = heads.act(_fused(batch=1), target_mask=mask)
    assert target.item() == 3


def test_act_returns_a_decodable_triple():
    from kairos.rl.action_space import decode

    heads = PolicyHeads(embed_dim=DIM).eval()
    operation, parameters, target = heads.act(_fused(batch=1))
    action = decode(
        int(operation[0]), parameters[0].numpy(), int(target[0]),
        {"edges": ["Edge1"], "faces": ["Face1"], "features": ["Pad"]},
    )
    assert action.operation is not None
    assert isinstance(action.parameters, dict)


def test_act_is_deterministic_when_asked():
    heads = PolicyHeads(embed_dim=DIM).eval()
    fused = _fused(batch=4)
    first = heads.act(fused, deterministic=True)[0]
    second = heads.act(fused, deterministic=True)[0]
    assert torch.equal(first, second)
