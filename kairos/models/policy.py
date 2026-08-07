"""Hierarchical action heads: fused embedding → a structured CAD action.

The action space is not flat — an operation, its continuous parameters, and its
target are chosen at different levels. ``PAD``'s length and ``FILLET``'s radius
occupy the same slot but mean different things, so the parameter and target
heads are **conditioned on the operation**: the chosen operation's embedding is
concatenated to the fused state before they run. A flat head would have to
learn one parameter distribution averaged over every operation.

Legality is enforced at the logit level. ``kairos.actions.masking`` already
knows which operations the live document permits (no pad without a sketch, no
fillet without an edge); masked operations get ``-inf`` logits so they cannot be
sampled and contribute no gradient, rather than being sampled and penalized.
"""

from __future__ import annotations

import torch
from torch import nn

from kairos.rl.action_space import MAX_TARGETS, NUM_OPERATIONS, PARAM_SLOTS

#: Logit value for illegal choices — large negative, but finite so that a row
#: with every choice masked yields a uniform distribution instead of NaNs.
MASK_FILL = -1e9


class PolicyHeads(nn.Module):
    """Operation / parameter / target heads over a fused ``[B, D]`` embedding."""

    def __init__(
        self,
        embed_dim: int = 128,
        operation_embed_dim: int = 32,
        hidden_dim: int = 128,
        n_operations: int = NUM_OPERATIONS,
        n_param_slots: int = PARAM_SLOTS,
        max_targets: int = MAX_TARGETS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_operations = n_operations
        self.n_param_slots = n_param_slots
        self.max_targets = max_targets

        self.operation_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_operations),
        )
        self.operation_embedding = nn.Embedding(n_operations, operation_embed_dim)

        conditioned_dim = embed_dim + operation_embed_dim
        self.parameter_head = nn.Sequential(
            nn.Linear(conditioned_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_param_slots),
        )
        self.target_head = nn.Sequential(
            nn.Linear(conditioned_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, max_targets),
        )

    def forward(
        self,
        fused: torch.Tensor,
        operation_mask: torch.Tensor | None = None,
        operation: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Score one step.

        Args:
            fused: ``[B, D]`` multimodal embedding.
            operation_mask: ``[B, n_operations]`` 1 where the operation is
                legal. Illegal operations are driven to ``MASK_FILL``.
            operation: ``[B]`` operation ids to condition the parameter and
                target heads on. Training passes the expert's operation
                (teacher forcing); inference passes the head's own choice.
            target_mask: ``[B, max_targets]`` 1 where a target slot exists.

        Returns:
            ``operation_logits`` ``[B, n_ops]``, ``parameters`` ``[B, slots]``
            in [0, 1], ``target_logits`` ``[B, max_targets]``.
        """
        operation_logits = self.operation_head(fused)
        if operation_mask is not None:
            operation_logits = operation_logits.masked_fill(operation_mask == 0, MASK_FILL)

        if operation is None:
            operation = operation_logits.argmax(dim=-1)

        conditioned = torch.cat([fused, self.operation_embedding(operation)], dim=-1)
        # Sigmoid because the codec's slots are normalized to [0, 1]; emitting
        # unbounded values would silently clip at decode time.
        parameters = torch.sigmoid(self.parameter_head(conditioned))

        target_logits = self.target_head(conditioned)
        if target_mask is not None:
            target_logits = target_logits.masked_fill(target_mask == 0, MASK_FILL)

        return {
            "operation_logits": operation_logits,
            "parameters": parameters,
            "target_logits": target_logits,
        }

    @torch.no_grad()
    def act(
        self,
        fused: torch.Tensor,
        operation_mask: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
        deterministic: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Choose ``(operation, parameters, target)`` for a batch of states."""
        out = self.forward(fused, operation_mask=operation_mask, target_mask=target_mask)
        if deterministic:
            operation = out["operation_logits"].argmax(dim=-1)
        else:
            operation = torch.distributions.Categorical(
                logits=out["operation_logits"]
            ).sample()

        # Re-run the conditioned heads under the operation actually chosen.
        out = self.forward(
            fused,
            operation_mask=operation_mask,
            operation=operation,
            target_mask=target_mask,
        )
        target = out["target_logits"].argmax(dim=-1)
        return operation, out["parameters"], target
