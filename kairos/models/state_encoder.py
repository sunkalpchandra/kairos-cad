"""State encoder: geometry numerics + build history → an embedding.

Two streams, because they carry different kinds of information:

- The frozen 24-dim numeric vector from ``kairos.representation`` (volume,
  bounding box, hole and face counts, sketch status) — *what the part is now*.
- The feature history (``Pad``, ``Pocket``, ``Fillet``, …) as a token sequence
  — *how it got there*. Order matters: a pocket before a pad is a different
  build than a pad before a pocket, so this runs through a GRU rather than
  being bag-of-features pooled.
"""

from __future__ import annotations

import torch
from torch import nn

from kairos.representation.feature_encoder import VOCAB_SIZE as FEATURE_VOCAB_SIZE
from kairos.representation.numerical_encoder import ENCODING_DIM

#: PAD occupies index 0 of the feature vocabulary.
FEATURE_PAD_ID = 0


class StateEncoder(nn.Module):
    """Encode ``(numeric [B, 24], history [B, T])`` into ``[B, embed_dim]``."""

    def __init__(
        self,
        embed_dim: int = 128,
        numeric_dim: int = ENCODING_DIM,
        history_embed_dim: int = 32,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.numeric_dim = numeric_dim

        self.numeric_mlp = nn.Sequential(
            nn.Linear(numeric_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.feature_embedding = nn.Embedding(
            FEATURE_VOCAB_SIZE, history_embed_dim, padding_idx=FEATURE_PAD_ID
        )
        self.history_rnn = nn.GRU(
            history_embed_dim, embed_dim, batch_first=True, bidirectional=False
        )
        self.merge = nn.Linear(embed_dim * 2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, numeric: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        """Args: ``numeric`` ``[B, 24]``, ``history`` ``[B, T]`` long ids.

        Histories are right-padded with ``PAD``. The GRU still advances its
        state over those slots, so the summary is read at each row's last real
        feature rather than at the end of the padded sequence — otherwise a
        two-feature build and a ten-feature build with the same prefix would
        converge to whatever the trailing pads produce.
        """
        if numeric.shape[-1] != self.numeric_dim:
            raise ValueError(
                f"expected {self.numeric_dim}-dim numeric state, got {numeric.shape[-1]}"
            )
        numeric_embedding = self.numeric_mlp(numeric)

        embedded = self.feature_embedding(history)
        outputs, _ = self.history_rnn(embedded)

        lengths = (history != FEATURE_PAD_ID).sum(dim=1)
        last_index = (lengths - 1).clamp(min=0)
        rows = torch.arange(history.shape[0], device=history.device)
        history_embedding = outputs[rows, last_index]
        # An empty history (nothing built yet) contributes nothing.
        history_embedding = history_embedding * (lengths > 0).unsqueeze(-1).to(outputs.dtype)

        merged = self.merge(torch.cat([numeric_embedding, history_embedding], dim=-1))
        return self.norm(merged)
