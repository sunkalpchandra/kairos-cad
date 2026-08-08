"""State-value head for PPO.

Predicts the expected discounted return from a fused multimodal embedding.

The head is **deliberately separate from the policy heads but shares the
trunk**. Sharing the encoders is what makes value learning cheap here, a
FreeCAD rollout step costs orders of magnitude more than a forward pass, so
samples are scarce and the critic cannot afford its own encoder stack. Keeping
the final layers separate stops the value loss from dominating the shared
features, which is the usual failure of a single fused head.

Returns are unnormalized: the shaped reward already lives on a small, bounded
scale (roughly [-1, +5] per step), so the usual return-scaling machinery would
add tuning surface without fixing a real problem.
"""

from __future__ import annotations

import torch
from torch import nn


class ValueHead(nn.Module):
    """Map a fused ``[B, D]`` embedding to a scalar state value ``[B]``."""

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # A near-zero final layer starts the critic at V~0 rather than at some
        # arbitrary offset, so early advantages reflect rewards instead of
        # initialization noise.
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, std=0.01)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.net(fused).squeeze(-1)
