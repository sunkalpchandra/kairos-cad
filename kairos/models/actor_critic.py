"""Actor-critic wrapper around :class:`KairosVLA` for PPO.

The BC policy already maps a fused multimodal embedding to an action; PPO needs
two more things from the same trunk: a state value, and a *distribution* rather
than a point estimate.

The parameter spread (``log_std``) is a **state-independent learned vector**,
one entry per codec slot. A state-conditioned spread is the more expressive
choice and the wrong one here: a FreeCAD rollout step is orders of magnitude
more expensive than a gradient step, so samples are far too scarce to fit a
per-state variance without it collapsing early and killing exploration.

Sampling is **hierarchical, matching the action space**: the operation is drawn
first, then the parameter distribution is conditioned on the operation actually
drawn. That keeps `log p(action) = log p(op) + log p(params | op)` exact, which
is what the PPO ratio depends on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from kairos.models.distributions import ActionDistribution
from kairos.models.value_head import ValueHead
from kairos.models.vla import KairosVLA, VLAConfig, load_model_state
from kairos.rl.action_space import PARAM_SLOTS


class ActorCritic(nn.Module):
    """Policy and value function sharing the VLA encoder trunk."""

    def __init__(
        self,
        vla: KairosVLA | None = None,
        init_log_std: float = -1.0,
        supervise_target: bool = False,
    ) -> None:
        super().__init__()
        self.vla = vla if vla is not None else KairosVLA()
        self.value_head = ValueHead(
            embed_dim=self.vla.config.embed_dim,
            hidden_dim=self.vla.config.hidden_dim,
            dropout=self.vla.config.dropout,
        )
        self.parameter_log_std = nn.Parameter(torch.full((PARAM_SLOTS,), float(init_log_std)))
        # BC could not supervise targets (trajectories record no edge list), so
        # the target factor is off unless a caller opts in.
        self.supervise_target = supervise_target

    # ------------------------------------------------------------- forward

    def distribution(
        self,
        inputs: dict[str, torch.Tensor],
        operation: torch.Tensor | None = None,
    ) -> tuple[ActionDistribution, torch.Tensor]:
        """Build the action distribution and state value for a batch.

        ``operation`` conditions the parameter factor. Pass the taken operation
        when re-scoring stored transitions; leave it None to let the policy
        pick greedily for conditioning while sampling.
        """
        fused = self.vla.encode(
            inputs["token_ids"],
            inputs["token_values"],
            inputs["token_mask"],
            inputs["numeric"],
            inputs["history"],
            inputs.get("views"),
        )
        value = self.value_head(fused)
        operation_mask = inputs.get("operation_mask")
        target_mask = inputs.get("target_mask")

        heads = self.vla.heads(
            fused, operation_mask=operation_mask, operation=operation, target_mask=target_mask
        )
        log_std = self.parameter_log_std.expand_as(heads["parameter_mean"])
        distribution = ActionDistribution(
            operation_logits=heads["operation_logits"],
            parameter_mean=heads["parameter_mean"],
            parameter_log_std=log_std,
            target_logits=heads["target_logits"] if self.supervise_target else None,
            operation_mask=operation_mask,
            target_mask=target_mask,
        )
        return distribution, value

    @torch.no_grad()
    def act(
        self, inputs: dict[str, torch.Tensor], deterministic: bool = False
    ) -> dict[str, torch.Tensor]:
        """Sample one action per row, with its log-probability and value.

        Dropout is forced off here regardless of the module's mode. Exploration
        must come from the action distribution, not from network noise: PPO's
        ratio compares the stored log-probability against a later re-scoring,
        and if dropout perturbed either one the ratio would measure noise
        instead of how far the policy moved.
        """
        was_training = self.training
        self.eval()
        try:
            return self._act(inputs, deterministic)
        finally:
            self.train(was_training)

    def _act(
        self, inputs: dict[str, torch.Tensor], deterministic: bool
    ) -> dict[str, torch.Tensor]:
        distribution, value = self.distribution(inputs)
        operation = (
            distribution.operation.probs.argmax(dim=-1)
            if deterministic
            else distribution.operation.sample()
        )
        # Re-condition the parameter factor on the operation actually chosen.
        conditioned, _ = self.distribution(inputs, operation=operation)
        parameters = (
            conditioned.parameters.mode() if deterministic else conditioned.parameters.sample()
        )

        action = {"operation": operation, "parameters": parameters}
        if conditioned.target is not None:
            action["target"] = (
                conditioned.target.probs.argmax(dim=-1)
                if deterministic
                else conditioned.target.sample()
            )
        return {
            "action": action,
            "log_prob": conditioned.log_prob(action),
            "value": value,
            "entropy": conditioned.entropy(),
        }

    def evaluate_actions(
        self, inputs: dict[str, torch.Tensor], action: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Re-score stored transitions under the current parameters."""
        distribution, value = self.distribution(inputs, operation=action["operation"])
        return {
            "log_prob": distribution.log_prob(action),
            "entropy": distribution.entropy(),
            "value": value,
            "operation_logits": distribution.operation.logits,
            "parameter_mean": distribution.parameters.mean,
            "parameter_log_std": distribution.parameters.log_std,
        }

    # --------------------------------------------------------- persistence

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_bc_checkpoint(
        cls,
        path: str | Path,
        device: torch.device | str = "cpu",
        init_log_std: float = -1.0,
        **kwargs: Any,
    ) -> ActorCritic:
        """Initialize from a behavioral-cloning checkpoint.

        Starting PPO from a random policy would be hopeless here: a valid CAD
        build is a long, precisely ordered action sequence, and random
        exploration essentially never produces one to learn from. The value
        head is new and starts at zero, only the policy is inherited.
        """
        payload = torch.load(Path(path), map_location=device, weights_only=False)
        vla = KairosVLA(VLAConfig.from_dict(payload["model_config"]))
        load_model_state(vla, payload["model_state"])
        model = cls(vla=vla, init_log_std=init_log_std, **kwargs)
        return model.to(device)

    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor_critic_state": self.state_dict(),
                "model_config": self.vla.config.to_dict(),
                "supervise_target": self.supervise_target,
                **(extra or {}),
            },
            path,
        )
        return path


def load_actor_critic(path: str | Path, device: torch.device | str = "cpu") -> ActorCritic:
    """Rebuild an :class:`ActorCritic` saved by :meth:`ActorCritic.save`."""
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    vla = KairosVLA(VLAConfig.from_dict(payload["model_config"]))
    model = ActorCritic(vla=vla, supervise_target=payload.get("supervise_target", False))
    model.load_state_dict(payload["actor_critic_state"])
    return model.to(device)
