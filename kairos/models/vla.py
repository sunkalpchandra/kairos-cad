"""KairosVLA: the vision-language-action policy.

Wires the four encoders and the action heads into one module:

    requirement tokens ─→ LanguageEncoder ─┐
    rendered views ─────→ VisionEncoder ───┼─→ FusionEncoder ─→ PolicyHeads
    numeric state + history → StateEncoder ┘

The model never emits code, only an operation id, ``PARAM_SLOTS`` normalized
floats, and a target index, which ``kairos.rl.action_space.decode`` turns into
a schema-validated :class:`~kairos.actions.schema.Action`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from kairos.models.fusion import FusionEncoder
from kairos.models.language_encoder import LanguageEncoder
from kairos.models.policy import PolicyHeads
from kairos.models.state_encoder import StateEncoder
from kairos.models.vision_encoder import VisionEncoder
from kairos.representation.numerical_encoder import ENCODING_DIM


@dataclass
class VLAConfig:
    """Architecture hyperparameters, saved alongside every checkpoint."""

    embed_dim: int = 128
    language_depth: int = 2
    language_heads: int = 4
    max_text_length: int = 64
    vision_widths: tuple[int, ...] = (16, 32, 64, 128)
    #: Build the vision encoder at all. Off by default because no training or
    #: rollout path passes `views`: grep for `views=` and every call site is
    #: rendering, not inference. Building it anyway cost 115,089 parameters
    #: (10.1% of the model) that never see a gradient, plus their Adam moments
    #: and their bytes in every checkpoint. Turn this on when something
    #: actually feeds images.
    use_vision: bool = False
    numeric_dim: int = ENCODING_DIM
    history_embed_dim: int = 32
    fusion_depth: int = 2
    fusion_heads: int = 4
    operation_embed_dim: int = 32
    hidden_dim: int = 128
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> VLAConfig:
        fields = {f for f in cls.__dataclass_fields__}
        config = cls(**{k: v for k, v in data.items() if k in fields})
        # tuple() because JSON round-trips tuples into lists, and nn.Sequential
        # construction depends on the width sequence being iterable-stable.
        config.vision_widths = tuple(config.vision_widths)
        return config


def load_model_state(model: nn.Module, state: dict) -> None:
    """Load weights, tolerating only the vision encoder's absence.

    Checkpoints written before `use_vision` defaulted off carry `vision.*`
    tensors the model no longer builds. `strict=False` would accept those, but
    it also silently accepts *missing* keys, which would let a genuinely
    mismatched checkpoint load as a partly random policy and score as a result.
    So the extra keys are dropped explicitly and everything else stays strict.
    """
    own = set(model.state_dict())
    extra = [k for k in state if k not in own]
    unexpected = [k for k in extra if not k.startswith("vision.")]
    if unexpected:
        raise RuntimeError(
            f"checkpoint has {len(unexpected)} unexpected tensors: {unexpected[:5]}"
        )
    model.load_state_dict({k: v for k, v in state.items() if k in own}, strict=True)


class KairosVLA(nn.Module):
    """Multimodal policy over structured CAD actions."""

    def __init__(self, config: VLAConfig | None = None) -> None:
        super().__init__()
        self.config = config or VLAConfig()
        c = self.config

        self.language = LanguageEncoder(
            embed_dim=c.embed_dim,
            depth=c.language_depth,
            heads=c.language_heads,
            max_length=c.max_text_length,
            dropout=c.dropout,
        )
        self.vision = (
            VisionEncoder(embed_dim=c.embed_dim, widths=c.vision_widths, dropout=c.dropout)
            if c.use_vision
            else None
        )
        self.state = StateEncoder(
            embed_dim=c.embed_dim,
            numeric_dim=c.numeric_dim,
            history_embed_dim=c.history_embed_dim,
            hidden_dim=c.hidden_dim,
            dropout=c.dropout,
        )
        self.fusion = FusionEncoder(
            embed_dim=c.embed_dim,
            heads=c.fusion_heads,
            depth=c.fusion_depth,
            dropout=c.dropout,
        )
        self.heads = PolicyHeads(
            embed_dim=c.embed_dim,
            operation_embed_dim=c.operation_embed_dim,
            hidden_dim=c.hidden_dim,
            dropout=c.dropout,
        )

    def encode(
        self,
        token_ids: torch.Tensor,
        token_values: torch.Tensor,
        token_mask: torch.Tensor,
        numeric: torch.Tensor,
        history: torch.Tensor,
        views: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the encoders and fuse them into ``[B, embed_dim]``."""
        language = self.language(token_ids, token_values, token_mask)
        state = self.state(numeric, history)
        if views is not None and self.vision is None:
            raise ValueError(
                "views were passed but the vision encoder is disabled; set "
                "VLAConfig.use_vision=True to build it"
            )
        vision = self.vision(views) if views is not None else None
        return self.fusion(language, state, vision)

    def forward(
        self,
        token_ids: torch.Tensor,
        token_values: torch.Tensor,
        token_mask: torch.Tensor,
        numeric: torch.Tensor,
        history: torch.Tensor,
        views: torch.Tensor | None = None,
        operation_mask: torch.Tensor | None = None,
        operation: torch.Tensor | None = None,
        target_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Score one step; see :meth:`PolicyHeads.forward` for the outputs."""
        fused = self.encode(token_ids, token_values, token_mask, numeric, history, views)
        return self.heads(
            fused,
            operation_mask=operation_mask,
            operation=operation,
            target_mask=target_mask,
        )

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
