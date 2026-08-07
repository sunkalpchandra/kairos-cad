"""PPO with a behavioral-cloning anchor.

Standard clipped-surrogate PPO, plus one term this domain needs: a **KL penalty
against the frozen BC policy**.

Why the anchor. A valid CAD build is a long, precisely ordered action sequence,
and the reward for finishing one is sparse. Early PPO updates chase whatever
shaping reward is reachable — opening sketches, adding circles — and it is very
easy for the policy to drift off the BC manifold and lose the ability to
produce a coherent build at all. Once that happens it cannot recover, because
random exploration essentially never rediscovers a 12-step valid sequence. The
anchor keeps updates near the demonstrations while still letting reward shape
behavior; ``bc_kl_coef: 0.0`` disables it for comparison runs.

Advantages are normalized per minibatch, and the value function is clipped the
same way the policy is — with rollouts this small (a few hundred transitions
per iteration, since every step is a FreeCAD recompute) an unclipped critic
update is a reliable way to destabilize training.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from kairos.models.distributions import categorical_kl, explained_variance, gaussian_kl


@dataclass
class PPOConfig:
    """Optimization hyperparameters for a PPO run."""

    learning_rate: float = 1e-4
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    #: Weight on KL(current ‖ frozen BC policy). 0 disables the anchor.
    bc_kl_coef: float = 0.05
    max_grad_norm: float = 0.5
    epochs_per_update: int = 4
    minibatch_size: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    normalize_advantages: bool = True
    #: Stop an update early if the policy has already moved this far.
    target_kl: float | None = 0.05

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UpdateMetrics:
    """What one PPO update did, for logging and early-stopping decisions."""

    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    bc_kl: float = 0.0
    approx_kl: float = 0.0
    clip_fraction: float = 0.0
    explained_variance: float = 0.0
    epochs_run: int = 0
    stopped_early: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in asdict(self).items()}


class PPOTrainer:
    """Clipped-surrogate PPO over an :class:`ActorCritic`."""

    def __init__(
        self,
        model,
        config: PPOConfig | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.config = config or PPOConfig()
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.learning_rate
        )
        self.reference = self._freeze_reference() if self.config.bc_kl_coef > 0 else None

    def _freeze_reference(self):
        """A frozen copy of the starting policy for the KL anchor."""
        reference = copy.deepcopy(self.model).to(self.device)
        reference.eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
        return reference

    # ------------------------------------------------------------ objective

    def _bc_kl(self, inputs, action, outputs) -> torch.Tensor:
        """KL from the current policy to the frozen BC policy."""
        if self.reference is None:
            return torch.zeros((), device=self.device)
        with torch.no_grad():
            frozen = self.reference.evaluate_actions(inputs, action)
        operation_kl = categorical_kl(outputs["operation_logits"], frozen["operation_logits"])
        parameter_kl = gaussian_kl(
            outputs["parameter_mean"],
            outputs["parameter_log_std"],
            frozen["parameter_mean"],
            frozen["parameter_log_std"],
        )
        return (operation_kl + parameter_kl).mean()

    def update(self, buffer) -> UpdateMetrics:
        """Run the configured epochs of PPO over one rollout buffer."""
        c = self.config
        metrics = UpdateMetrics()
        if len(buffer) == 0:
            return metrics

        collected: dict[str, list[float]] = {
            "policy": [], "value": [], "entropy": [], "bc_kl": [], "kl": [], "clip": []
        }
        predicted_values: list[np.ndarray] = []
        actual_returns: list[np.ndarray] = []

        # eval(), not train(): dropout must stay off during the update. The
        # stored log-probs were computed by the deterministic (eval) policy, so
        # re-scoring under dropout compares two different functions and the PPO
        # ratio stops meaning "how much did the policy change" — it reads as a
        # large spurious KL from step one. Only dropout differs between modes
        # here; the norm layers are LayerNorm/GroupNorm, which do not.
        self.model.eval()
        for epoch in range(c.epochs_per_update):
            metrics.epochs_run = epoch + 1
            epoch_kl: list[float] = []

            for batch in buffer.batches(c.minibatch_size, seed=epoch):
                inputs = {k: v.to(self.device) for k, v in batch["inputs"].items()}
                action = {k: v.to(self.device) for k, v in batch["action"].items()}
                old_log_prob = batch["log_prob"].to(self.device)
                old_value = batch["value"].to(self.device)
                advantage = batch["advantage"].to(self.device)
                returns = batch["return"].to(self.device)

                if c.normalize_advantages and advantage.numel() > 1:
                    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

                outputs = self.model.evaluate_actions(inputs, action)
                ratio = (outputs["log_prob"] - old_log_prob).exp()

                unclipped = ratio * advantage
                clipped = ratio.clamp(1.0 - c.clip_range, 1.0 + c.clip_range) * advantage
                policy_loss = -torch.min(unclipped, clipped).mean()

                # Clipped value loss: tiny rollouts make an unclipped critic
                # update a reliable way to destabilize training.
                value = outputs["value"]
                value_clipped = old_value + (value - old_value).clamp(
                    -c.value_clip_range, c.value_clip_range
                )
                value_loss = torch.max(
                    (value - returns) ** 2, (value_clipped - returns) ** 2
                ).mean()

                entropy = outputs["entropy"].mean()
                bc_kl = self._bc_kl(inputs, action, outputs)

                loss = (
                    policy_loss
                    + c.value_coef * value_loss
                    - c.entropy_coef * entropy
                    + c.bc_kl_coef * bc_kl
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), c.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    # Schulman's low-variance approximate KL between the old
                    # and new policies; negative values mean the estimate is
                    # noisy, not that KL is negative.
                    log_ratio = outputs["log_prob"] - old_log_prob
                    approx_kl = float(((log_ratio.exp() - 1) - log_ratio).mean())
                    clip_fraction = float(
                        ((ratio - 1.0).abs() > c.clip_range).float().mean()
                    )

                collected["policy"].append(float(policy_loss.detach()))
                collected["value"].append(float(value_loss.detach()))
                collected["entropy"].append(float(entropy.detach()))
                collected["bc_kl"].append(float(bc_kl.detach()))
                collected["kl"].append(approx_kl)
                collected["clip"].append(clip_fraction)
                epoch_kl.append(approx_kl)
                if epoch == 0:
                    predicted_values.append(value.detach().cpu().numpy())
                    actual_returns.append(returns.detach().cpu().numpy())

            if c.target_kl is not None and epoch_kl and float(np.mean(epoch_kl)) > c.target_kl:
                # The policy has moved far enough on this data; more epochs
                # would be optimizing against a stale advantage estimate.
                metrics.stopped_early = True
                break

        metrics.policy_loss = float(np.mean(collected["policy"])) if collected["policy"] else 0.0
        metrics.value_loss = float(np.mean(collected["value"])) if collected["value"] else 0.0
        metrics.entropy = float(np.mean(collected["entropy"])) if collected["entropy"] else 0.0
        metrics.bc_kl = float(np.mean(collected["bc_kl"])) if collected["bc_kl"] else 0.0
        metrics.approx_kl = float(np.mean(collected["kl"])) if collected["kl"] else 0.0
        metrics.clip_fraction = float(np.mean(collected["clip"])) if collected["clip"] else 0.0
        if predicted_values:
            metrics.explained_variance = explained_variance(
                torch.tensor(np.concatenate(predicted_values)),
                torch.tensor(np.concatenate(actual_returns)),
            )
        return metrics
