"""Action distribution tests (skipped without the optional torch extra)."""

import math

import pytest

torch = pytest.importorskip("torch", reason="requires the 'learn' extra")

from kairos.models.distributions import (  # noqa: E402
    ActionDistribution,
    SquashedGaussian,
    categorical_kl,
    explained_variance,
    gaussian_kl,
    masked_categorical,
)

N_OPS, N_SLOTS, N_TARGETS = 8, 6, 4


def _distribution(seed=0, operation_mask=None, target_mask=None):
    generator = torch.Generator().manual_seed(seed)
    return ActionDistribution(
        operation_logits=torch.randn(3, N_OPS, generator=generator),
        parameter_mean=torch.randn(3, N_SLOTS, generator=generator),
        parameter_log_std=torch.full((3, N_SLOTS), -1.0),
        target_logits=torch.randn(3, N_TARGETS, generator=generator),
        operation_mask=operation_mask,
        target_mask=target_mask,
    )


def test_illegal_operations_get_no_probability():
    mask = torch.zeros(3, N_OPS, dtype=torch.long)
    mask[:, 2] = 1
    distribution = _distribution(operation_mask=mask)
    probabilities = distribution.operation_probs()
    assert probabilities[:, 2].min() > 0.999
    assert probabilities[mask == 0].max() < 1e-6
    # Sampling can never pick a forbidden operation.
    for _ in range(20):
        assert (distribution.sample()["operation"] == 2).all()


def test_entropy_is_measured_over_legal_choices_only():
    """A one-legal-choice policy is certain, not merely constrained."""
    mask = torch.zeros(3, N_OPS, dtype=torch.long)
    mask[:, 1] = 1
    assert masked_categorical(torch.randn(3, N_OPS), mask).entropy().max() < 1e-4

    two_legal = torch.zeros(3, N_OPS, dtype=torch.long)
    two_legal[:, :2] = 1
    entropy = masked_categorical(torch.zeros(3, N_OPS), two_legal).entropy()
    assert entropy.mean().item() == pytest.approx(math.log(2), abs=1e-4)


def test_a_fully_masked_row_stays_finite():
    """No legal action is a state bug, but it must not NaN the batch."""
    mask = torch.zeros(2, N_OPS, dtype=torch.long)
    distribution = masked_categorical(torch.randn(2, N_OPS), mask)
    assert torch.isfinite(distribution.probs).all()
    assert distribution.probs.sum(dim=-1).allclose(torch.ones(2), atol=1e-5)


def test_parameters_stay_inside_the_codec_range():
    gaussian = SquashedGaussian(torch.full((256, N_SLOTS), 4.0), torch.zeros(256, N_SLOTS))
    sample = gaussian.sample()
    assert (sample > 0.0).all() and (sample < 1.0).all()
    assert gaussian.mode().mean().item() > 0.9  # a large mean squashes near 1


def test_squashed_log_prob_is_finite_at_the_boundaries():
    """BC initialization parks many expert parameters at exactly 0 or 1."""
    gaussian = SquashedGaussian(torch.zeros(2, N_SLOTS), torch.zeros(2, N_SLOTS))
    boundary = torch.cat([torch.zeros(1, N_SLOTS), torch.ones(1, N_SLOTS)])
    assert torch.isfinite(gaussian.log_prob(boundary)).all()


def test_squashed_density_integrates_to_one():
    """Numerically verify the change-of-variables correction."""
    gaussian = SquashedGaussian(torch.tensor([[0.3]]), torch.tensor([[0.0]]))
    grid = torch.linspace(1e-4, 1 - 1e-4, 20000).unsqueeze(-1)
    density = gaussian.log_prob(grid).exp()
    integral = torch.trapz(density, grid.squeeze(-1))
    assert float(integral) == pytest.approx(1.0, abs=1e-2)


def test_joint_log_prob_is_the_sum_of_its_factors():
    distribution = _distribution()
    action = distribution.sample()
    expected = (
        distribution.operation.log_prob(action["operation"])
        + distribution.parameters.log_prob(action["parameters"])
        + distribution.target.log_prob(action["target"])
    )
    assert torch.allclose(distribution.log_prob(action), expected, atol=1e-6)


def test_mode_is_deterministic_and_legal():
    mask = torch.zeros(3, N_OPS, dtype=torch.long)
    mask[:, 5] = 1
    distribution = _distribution(operation_mask=mask)
    first, second = distribution.mode(), distribution.mode()
    assert torch.equal(first["operation"], second["operation"])
    assert (first["operation"] == 5).all()


def test_kl_is_zero_between_identical_distributions():
    mean, log_std = torch.randn(4, N_SLOTS), torch.zeros(4, N_SLOTS)
    assert gaussian_kl(mean, log_std, mean, log_std).abs().max() < 1e-6
    logits = torch.randn(4, N_OPS)
    assert categorical_kl(logits, logits).abs().max() < 1e-6


def test_kl_grows_as_distributions_separate():
    mean, log_std = torch.zeros(1, N_SLOTS), torch.zeros(1, N_SLOTS)
    near = gaussian_kl(mean, log_std, mean + 0.1, log_std)
    far = gaussian_kl(mean, log_std, mean + 2.0, log_std)
    assert 0 < float(near) < float(far)


def test_explained_variance_reads_as_expected():
    actual = torch.randn(256)
    assert explained_variance(actual, actual) == pytest.approx(1.0, abs=1e-5)
    # Predicting the mean explains nothing.
    assert explained_variance(torch.full_like(actual, float(actual.mean())), actual) == (
        pytest.approx(0.0, abs=0.05)
    )
    assert math.isnan(explained_variance(torch.zeros(8), torch.zeros(8)))
