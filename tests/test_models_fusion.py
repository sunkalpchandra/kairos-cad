"""Fusion tests (skipped without the optional torch extra)."""

import pytest

torch = pytest.importorskip("torch", reason="requires the 'learn' extra")

from kairos.models.fusion import MODALITIES, FusionEncoder  # noqa: E402

DIM = 32


def _modalities(batch=2, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.rand(batch, DIM, generator=generator),
        torch.rand(batch, DIM, generator=generator),
        torch.rand(batch, DIM, generator=generator),
    )


def test_output_shape_and_finiteness():
    fusion = FusionEncoder(embed_dim=DIM, heads=2, depth=1).eval()
    language, state, vision = _modalities()
    out = fusion(language, state, vision)
    assert out.shape == (2, DIM)
    assert torch.isfinite(out).all()


def test_vision_is_optional():
    """BC trains without per-step renders; one checkpoint must serve both."""
    fusion = FusionEncoder(embed_dim=DIM, heads=2, depth=1).eval()
    language, state, _ = _modalities()
    with torch.no_grad():
        assert fusion(language, state).shape == (2, DIM)


def test_missing_vision_placeholder_is_used_when_absent():
    fusion = FusionEncoder(embed_dim=DIM, heads=2, depth=1).eval()
    language, state, _ = _modalities(batch=1)
    with torch.no_grad():
        implicit = fusion(language, state)
        explicit = fusion(language, state, fusion.missing_vision.expand(1, -1))
    assert torch.allclose(implicit, explicit, atol=1e-6)


def test_each_modality_changes_the_result():
    """If a modality were ignored, fusion would be doing nothing useful."""
    fusion = FusionEncoder(embed_dim=DIM, heads=2, depth=1).eval()
    language, state, vision = _modalities(batch=1)
    other = torch.rand(1, DIM, generator=torch.Generator().manual_seed(99))
    with torch.no_grad():
        base = fusion(language, state, vision)
        assert not torch.allclose(base, fusion(other, state, vision), atol=1e-5)
        assert not torch.allclose(base, fusion(language, other, vision), atol=1e-5)
        assert not torch.allclose(base, fusion(language, state, other), atol=1e-5)


def test_modality_identity_is_learned_not_positional():
    """Swapping language and state must change the output: they are not
    interchangeable slots in a bag."""
    fusion = FusionEncoder(embed_dim=DIM, heads=2, depth=1).eval()
    language, state, vision = _modalities(batch=1)
    with torch.no_grad():
        assert not torch.allclose(
            fusion(language, state, vision), fusion(state, language, vision), atol=1e-5
        )


def test_modality_count_matches_token_count():
    assert len(MODALITIES) == 3
