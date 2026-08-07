"""Language encoder tests (skipped without the optional torch extra)."""

import pytest

torch = pytest.importorskip("torch", reason="requires the 'learn' extra")

from kairos.language import tokenizer as tk  # noqa: E402
from kairos.models.language_encoder import LanguageEncoder  # noqa: E402


def _batch(texts, max_length=32):
    encoded = [tk.encode(t, max_length=max_length) for t in texts]
    ids = torch.tensor([e[0] for e in encoded], dtype=torch.long)
    values = torch.tensor([e[1] for e in encoded], dtype=torch.float32)
    mask = torch.tensor([e[2] for e in encoded], dtype=torch.long)
    return ids, values, mask


def test_output_shape_and_finiteness():
    encoder = LanguageEncoder(embed_dim=32, depth=1, heads=2, max_length=32).eval()
    out = encoder(*_batch(["Plate with 4 M5 holes", "Design a spacer"]))
    assert out.shape == (2, 32)
    assert torch.isfinite(out).all()


def test_padding_does_not_change_the_embedding():
    """A short requirement must encode identically at any padded length."""
    encoder = LanguageEncoder(embed_dim=32, depth=1, heads=2, max_length=64).eval()
    text = "Design a rectangular plate with 4 holes of 5 mm diameter"
    with torch.no_grad():
        short = encoder(*_batch([text], max_length=24))
        long = encoder(*_batch([text], max_length=64))
    assert torch.allclose(short, long, atol=1e-5)


def test_magnitude_changes_the_embedding():
    """6 mm and 60 mm holes are different designs, so they must differ here."""
    encoder = LanguageEncoder(embed_dim=32, depth=1, heads=2, max_length=32).eval()
    with torch.no_grad():
        small = encoder(*_batch(["4 holes of 6 mm diameter"]))
        large = encoder(*_batch(["4 holes of 60 mm diameter"]))
    assert not torch.allclose(small, large, atol=1e-4)


def test_gradients_reach_every_parameter():
    encoder = LanguageEncoder(embed_dim=32, depth=1, heads=2, max_length=32)
    encoder(*_batch(["Plate with 4 M5 holes, minimize mass"])).sum().backward()
    missing = [
        name
        for name, p in encoder.named_parameters()
        if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())
    ]
    # The padding row of an embedding table never receives gradient by design.
    assert missing == [], missing
