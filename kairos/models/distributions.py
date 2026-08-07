"""Action distribution over the structured CAD action space.

One action is a triple — a discrete operation, six continuous parameters, and a
discrete target — so the distribution is a product of three factors, and the
log-probability of an action is the sum of their log-probabilities.

Two decisions matter for correctness:

**Illegal operations are removed from the distribution, not penalized.** Their
logits are driven to a large negative value before the softmax, so they carry
essentially zero probability and cannot be sampled. Entropy is then computed
over the surviving support; including the masked-out tail would report a policy
as uncertain when it is merely forbidden from acting.

**Parameters use a squashed Gaussian, not a Beta.** The codec's slots are
bounded to [0, 1], and a Gaussian squashed through a sigmoid keeps a simple
reparameterized sample with a closed-form log-det correction, while remaining
well behaved when BC initialization puts the mean near a boundary — which it
does constantly, since many expert parameters sit at 0 or 1.

Parameters for slots an operation ignores still receive a log-probability; that
is intentional and harmless, because the same slots are scored identically
under old and new policies and so cancel in the PPO ratio.
"""

from __future__ import annotations

import math

import torch
from torch import distributions as td

#: Matches kairos.models.policy: finite so a fully masked row stays defined.
MASK_FILL = -1e9

#: Keeps log-probabilities finite when a squashed sample lands on a boundary.
_EPS = 1e-6


def masked_categorical(logits: torch.Tensor, mask: torch.Tensor | None) -> td.Categorical:
    """Categorical over legal choices only.

    ``mask`` is 1 where a choice is legal. Rows with nothing legal fall back to
    a uniform distribution rather than producing NaNs — that row's gradient is
    meaningless either way, and a NaN would poison the whole batch.
    """
    if mask is not None:
        empty = mask.sum(dim=-1, keepdim=True) == 0
        logits = logits.masked_fill((mask == 0) & ~empty, MASK_FILL)
    return td.Categorical(logits=logits)


class SquashedGaussian:
    """Gaussian pushed through a sigmoid onto the codec's [0, 1] slots."""

    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor) -> None:
        self.mean = mean
        self.log_std = log_std.clamp(-5.0, 2.0)
        self.std = self.log_std.exp()
        self.normal = td.Normal(self.mean, self.std)

    def sample(self) -> torch.Tensor:
        return torch.sigmoid(self.normal.rsample())

    def mode(self) -> torch.Tensor:
        return torch.sigmoid(self.mean)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Log-density of an already-squashed ``value``, summed over slots."""
        value = value.clamp(_EPS, 1.0 - _EPS)
        pre_squash = torch.log(value) - torch.log1p(-value)  # logit
        # Change of variables: d/dx sigmoid(x) = s(1 - s).
        log_det = torch.log(value) + torch.log1p(-value)
        return (self.normal.log_prob(pre_squash) - log_det).sum(dim=-1)

    def entropy(self) -> torch.Tensor:
        """Entropy of the underlying Gaussian, summed over slots.

        The squashing correction is state-dependent and has no closed form
        here; the Gaussian's entropy is the standard surrogate and is what the
        bonus needs to keep exploration alive.
        """
        return self.normal.entropy().sum(dim=-1)


class ActionDistribution:
    """Joint distribution over (operation, parameters, target)."""

    def __init__(
        self,
        operation_logits: torch.Tensor,
        parameter_mean: torch.Tensor,
        parameter_log_std: torch.Tensor,
        target_logits: torch.Tensor | None = None,
        operation_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> None:
        self.operation = masked_categorical(operation_logits, operation_mask)
        self.parameters = SquashedGaussian(parameter_mean, parameter_log_std)
        self.target = (
            masked_categorical(target_logits, target_mask) if target_logits is not None else None
        )

    def sample(self) -> dict[str, torch.Tensor]:
        action = {
            "operation": self.operation.sample(),
            "parameters": self.parameters.sample(),
        }
        if self.target is not None:
            action["target"] = self.target.sample()
        return action

    def mode(self) -> dict[str, torch.Tensor]:
        """The greedy action, for evaluation rather than exploration."""
        action = {
            "operation": self.operation.probs.argmax(dim=-1),
            "parameters": self.parameters.mode(),
        }
        if self.target is not None:
            action["target"] = self.target.probs.argmax(dim=-1)
        return action

    def log_prob(self, action: dict[str, torch.Tensor]) -> torch.Tensor:
        """Joint log-probability: the factors are independent, so they add."""
        total = self.operation.log_prob(action["operation"])
        total = total + self.parameters.log_prob(action["parameters"])
        if self.target is not None and "target" in action:
            total = total + self.target.log_prob(action["target"])
        return total

    def entropy(self) -> torch.Tensor:
        total = self.operation.entropy() + self.parameters.entropy()
        if self.target is not None:
            total = total + self.target.entropy()
        return total

    def operation_probs(self) -> torch.Tensor:
        return self.operation.probs


def gaussian_kl(
    mean_a: torch.Tensor,
    log_std_a: torch.Tensor,
    mean_b: torch.Tensor,
    log_std_b: torch.Tensor,
) -> torch.Tensor:
    """KL(a ‖ b) between two diagonal Gaussians, summed over slots."""
    var_ratio = (2.0 * (log_std_a - log_std_b)).exp()
    mean_diff = ((mean_a - mean_b) / log_std_b.exp()) ** 2
    return 0.5 * (var_ratio + mean_diff - 1.0 - 2.0 * (log_std_a - log_std_b)).sum(dim=-1)


def categorical_kl(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    """KL(a ‖ b) between two categorical distributions given logits."""
    log_p = torch.log_softmax(logits_a, dim=-1)
    log_q = torch.log_softmax(logits_b, dim=-1)
    # Masked-out entries carry ~zero probability, so their (log_p - log_q)
    # term is weighted away rather than contributing an infinity.
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1)


def explained_variance(predicted: torch.Tensor, actual: torch.Tensor) -> float:
    """1 - Var(actual - predicted) / Var(actual); 0 means "no better than the mean"."""
    variance = actual.var()
    if float(variance) < 1e-8:
        return float("nan")
    return float(1.0 - (actual - predicted).var() / variance)


def log_prob_of_uniform(n: int) -> float:
    """Log-probability of one choice under a uniform categorical of size n."""
    return -math.log(max(n, 1))
