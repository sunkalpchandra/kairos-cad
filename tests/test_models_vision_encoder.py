"""Vision encoder tests (skipped without the optional torch extra)."""

import pytest

torch = pytest.importorskip("torch", reason="requires the 'learn' extra")

from kairos.models.vision_encoder import VIEWS, VisionEncoder  # noqa: E402


def _views(batch=2, n_views=len(VIEWS), size=64, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(batch, n_views, 3, size, size, generator=generator)


def test_output_shape_and_finiteness():
    encoder = VisionEncoder(embed_dim=32, widths=(8, 16, 32)).eval()
    out = encoder(_views())
    assert out.shape == (2, 32)
    assert torch.isfinite(out).all()


def test_rejects_wrongly_shaped_input():
    encoder = VisionEncoder(embed_dim=32, widths=(8, 16))
    with pytest.raises(ValueError, match=r"\[B, V, 3, H, W\]"):
        encoder(torch.rand(2, 3, 64, 64))  # missing the view axis


def test_view_order_changes_the_embedding():
    """View identity is signal: a hole seen from the top differs from the front."""
    encoder = VisionEncoder(embed_dim=32, widths=(8, 16, 32)).eval()
    views = _views(batch=1)
    with torch.no_grad():
        forward = encoder(views)
        shuffled = encoder(views.flip(dims=[1]))
    assert not torch.allclose(forward, shuffled, atol=1e-5)


def test_trunk_weights_are_shared_across_views():
    """One trunk, not four: parameter count must not scale with view count."""
    four = VisionEncoder(embed_dim=32, widths=(8, 16), n_views=4)
    eight = VisionEncoder(embed_dim=32, widths=(8, 16), n_views=8)
    trunk_params = sum(p.numel() for p in four.trunk.parameters())
    assert trunk_params == sum(p.numel() for p in eight.trunk.parameters())
    # Only the view-identity table grows.
    assert sum(p.numel() for p in eight.parameters()) - sum(
        p.numel() for p in four.parameters()
    ) == 4 * 32


def test_resolution_is_flexible():
    """Adaptive pooling must accept whatever the loader downsamples to."""
    encoder = VisionEncoder(embed_dim=32, widths=(8, 16, 32)).eval()
    with torch.no_grad():
        assert encoder(_views(size=32)).shape == (2, 32)
        assert encoder(_views(size=128)).shape == (2, 32)


def test_gradients_reach_the_trunk():
    encoder = VisionEncoder(embed_dim=32, widths=(8, 16))
    encoder(_views()).sum().backward()
    grads = [p.grad for p in encoder.trunk.parameters()]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
