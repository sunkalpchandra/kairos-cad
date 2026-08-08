"""Multimodal fusion: requirement + views + geometry state → one embedding.

The three encoders answer different questions, what was asked, what the part
looks like, what it measures, and the next action depends on their
*relationship* ("the requirement wants 4 holes, the state has 2"). Concatenation
alone cannot express that, so the modality embeddings are treated as a
three-token sequence and passed through self-attention, letting each modality
read the others before pooling.

Vision is optional. Per-step renders do not exist in the Phase-2 dataset (only
final-design views), so BC currently trains on language + state; the visual
path is exercised by tests and by any run that supplies views. A learned
``missing view`` embedding stands in when they are absent, which keeps one
checkpoint valid whether or not renders are available.
"""

from __future__ import annotations

import torch
from torch import nn

MODALITIES: tuple[str, ...] = ("language", "vision", "state")


class FusionEncoder(nn.Module):
    """Fuse modality embeddings ``[B, D]`` each into a single ``[B, D]``."""

    def __init__(
        self,
        embed_dim: int = 128,
        heads: int = 4,
        depth: int = 2,
        ff_mult: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim

        self.modality_embedding = nn.Embedding(len(MODALITIES), embed_dim)
        #: Stands in for vision when a batch carries no renders.
        self.missing_vision = nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.missing_vision, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        language: torch.Tensor,
        state: torch.Tensor,
        vision: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the fused ``[B, embed_dim]`` embedding."""
        batch = language.shape[0]
        if vision is None:
            vision = self.missing_vision.expand(batch, -1)

        tokens = torch.stack([language, vision, state], dim=1)
        ids = torch.arange(len(MODALITIES), device=language.device)
        tokens = tokens + self.modality_embedding(ids)[None]

        fused = self.transformer(tokens)
        return self.norm(fused.mean(dim=1))
