"""Language encoder: tokenized requirement → a fixed-width embedding.

A small pre-norm transformer encoder over the frozen vocabulary in
``kairos.language.tokenizer``. Two details are specific to engineering text:

- **Magnitudes are embedded, not tokenized.** Each ``<num>`` position adds a
  learned projection of the literal's scaled value to its token embedding, so
  "6 mm" and "60 mm" differ continuously rather than sharing one token.
- **Masked mean pooling.** Requirements are short and every clause matters
  (holes, thickness, envelope, objective), so pooling averages real tokens
  instead of reading a single ``<bos>`` slot.
"""

from __future__ import annotations

import torch
from torch import nn

from kairos.language.tokenizer import PAD_ID, VOCAB_SIZE


class LanguageEncoder(nn.Module):
    """Encode a batch of tokenized requirements into ``[B, embed_dim]``."""

    def __init__(
        self,
        embed_dim: int = 128,
        depth: int = 2,
        heads: int = 4,
        ff_mult: int = 4,
        max_length: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_length = max_length

        self.token_embedding = nn.Embedding(VOCAB_SIZE, embed_dim, padding_idx=PAD_ID)
        self.position_embedding = nn.Embedding(max_length, embed_dim)
        # Scalar magnitude -> embedding space. Two layers because a linear map
        # of a single scalar can only scale one direction in embedding space.
        self.value_projection = nn.Sequential(
            nn.Linear(1, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * ff_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # Pre-norm layers cannot use the nested-tensor fast path; saying so
        # explicitly keeps torch from warning on every construction.
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=depth, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        token_ids: torch.Tensor,  # [B, L] long
        values: torch.Tensor,  # [B, L] float, scaled literals (0 where not <num>)
        mask: torch.Tensor,  # [B, L] 1 for real tokens
    ) -> torch.Tensor:
        """Return one ``[B, embed_dim]`` requirement embedding per row."""
        length = token_ids.shape[1]
        positions = torch.arange(length, device=token_ids.device)

        x = self.token_embedding(token_ids) + self.position_embedding(positions)[None]
        x = x + self.value_projection(values.unsqueeze(-1).to(x.dtype))
        x = self.dropout(x)

        # TransformerEncoder wants True where a token must be ignored.
        x = self.transformer(x, src_key_padding_mask=mask == 0)

        weights = mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1.0)
        return self.norm(pooled)
