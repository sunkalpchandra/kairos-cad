"""Vision encoder: rendered CAD views → a fixed-width embedding.

Four canonical views (iso, front, top, right) are encoded by one **shared**
convolutional trunk and then pooled. Sharing weights is deliberate: an
orthographic silhouette means the same thing whichever axis produced it, and a
per-view trunk would quadruple parameters to relearn edges four times.

Which view showed a feature still matters, a hole visible from the top is not
the same design decision as one visible from the front, so each view's
embedding gets a learned view-identity vector added before pooling, and pooling
is attention-weighted rather than a plain mean.

The renders are flat-shaded solids on a uniform background, so a small trunk
(4 stages, ~0.3M parameters) is enough; there is no texture to model.
"""

from __future__ import annotations

import torch
from torch import nn

#: Canonical view order. The dataset loader must emit views in this order.
VIEWS: tuple[str, ...] = ("iso", "front", "top", "right")


class _ConvBlock(nn.Module):
    """Conv → GroupNorm → GELU, halving resolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            # GroupNorm, not BatchNorm: BC batches are small and the RL rollout
            # runs single-sample, where batch statistics are meaningless.
            # Largest divisor of out_channels that is <= 8: a fixed 8 would
            # raise for any width not divisible by it (12, 20, ...).
            nn.GroupNorm(
                num_groups=next(g for g in range(min(8, out_channels), 0, -1)
                                if out_channels % g == 0),
                num_channels=out_channels,
            ),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VisionEncoder(nn.Module):
    """Encode ``[B, V, 3, H, W]`` renders into ``[B, embed_dim]``."""

    def __init__(
        self,
        embed_dim: int = 128,
        widths: tuple[int, ...] = (16, 32, 64, 128),
        n_views: int = len(VIEWS),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.n_views = n_views

        channels = (3,) + widths
        self.trunk = nn.Sequential(
            *[_ConvBlock(channels[i], channels[i + 1]) for i in range(len(widths))]
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.project = nn.Linear(widths[-1], embed_dim)
        self.view_embedding = nn.Embedding(n_views, embed_dim)
        self.attention = nn.Linear(embed_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        """Args: ``views`` ``[B, V, 3, H, W]`` in [0, 1]. Returns ``[B, D]``."""
        if views.ndim != 5:
            raise ValueError(f"expected [B, V, 3, H, W], got {tuple(views.shape)}")
        batch, n_views = views.shape[:2]

        # Fold views into the batch so the trunk is applied once, not V times.
        flat = views.reshape(batch * n_views, *views.shape[2:])
        features = self.pool(self.trunk(flat)).flatten(1)
        embedded = self.project(features).reshape(batch, n_views, self.embed_dim)

        ids = torch.arange(n_views, device=views.device) % self.n_views
        embedded = embedded + self.view_embedding(ids)[None]
        embedded = self.dropout(embedded)

        weights = torch.softmax(self.attention(embedded), dim=1)
        return self.norm((embedded * weights).sum(dim=1))
