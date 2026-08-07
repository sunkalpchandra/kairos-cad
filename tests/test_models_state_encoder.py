"""State encoder tests (skipped without the optional torch extra)."""

import pytest

torch = pytest.importorskip("torch", reason="requires the 'learn' extra")

from kairos.models.state_encoder import StateEncoder  # noqa: E402
from kairos.representation.feature_encoder import encode_history  # noqa: E402
from kairos.representation.numerical_encoder import ENCODING_DIM  # noqa: E402


def _history(names, length=8):
    ids, _mask = encode_history(names, max_length=length)
    return torch.from_numpy(ids).long().unsqueeze(0)


def test_output_shape_and_finiteness():
    encoder = StateEncoder(embed_dim=32).eval()
    out = encoder(torch.rand(3, ENCODING_DIM), torch.zeros(3, 8, dtype=torch.long))
    assert out.shape == (3, 32)
    assert torch.isfinite(out).all()


def test_rejects_wrong_numeric_width():
    """The 24-dim layout is frozen; a mismatch means the loader drifted."""
    encoder = StateEncoder(embed_dim=32)
    with pytest.raises(ValueError, match="24-dim"):
        encoder(torch.rand(2, ENCODING_DIM - 1), torch.zeros(2, 8, dtype=torch.long))


def test_history_order_changes_the_embedding():
    """Pad-then-pocket is a different build from pocket-then-pad."""
    encoder = StateEncoder(embed_dim=32).eval()
    numeric = torch.rand(1, ENCODING_DIM)
    with torch.no_grad():
        forward = encoder(numeric, _history(["Pad", "Pocket"]))
        reversed_ = encoder(numeric, _history(["Pocket", "Pad"]))
    assert not torch.allclose(forward, reversed_, atol=1e-5)


def test_numeric_state_changes_the_embedding():
    encoder = StateEncoder(embed_dim=32).eval()
    history = _history(["Pad"])
    with torch.no_grad():
        a = encoder(torch.zeros(1, ENCODING_DIM), history)
        b = encoder(torch.ones(1, ENCODING_DIM), history)
    assert not torch.allclose(a, b, atol=1e-5)


def test_padding_length_does_not_change_the_embedding():
    """The same build must encode identically however far it is padded."""
    encoder = StateEncoder(embed_dim=32).eval()
    numeric = torch.rand(1, ENCODING_DIM)
    with torch.no_grad():
        short = encoder(numeric, _history(["Pad", "Pocket"], length=4))
        long = encoder(numeric, _history(["Pad", "Pocket"], length=32))
    assert torch.allclose(short, long, atol=1e-6)


def test_empty_history_contributes_nothing():
    """Before the first feature there is no build to summarize."""
    encoder = StateEncoder(embed_dim=32).eval()
    numeric = torch.rand(1, ENCODING_DIM)
    with torch.no_grad():
        empty = encoder(numeric, _history([], length=8))
        merged = encoder.merge(
            torch.cat([encoder.numeric_mlp(numeric), torch.zeros(1, 32)], dim=-1)
        )
    assert torch.allclose(empty, encoder.norm(merged), atol=1e-6)


def test_gradients_reach_both_streams():
    encoder = StateEncoder(embed_dim=32)
    encoder(torch.rand(2, ENCODING_DIM), _history(["Pad", "Fillet"]).repeat(2, 1)).sum().backward()
    for module in (encoder.numeric_mlp, encoder.history_rnn):
        grads = [p.grad for p in module.parameters()]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
