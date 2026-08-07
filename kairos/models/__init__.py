"""Phase 4 learning stack: multimodal encoders, fusion, and the VLA policy.

Everything in this package requires torch (``pip install -e ".[learn]"``).
Nothing outside it may import torch: `kairos.cad`, `kairos.data`, and
`kairos.rl` must stay importable under FreeCAD's bundled interpreter, which
has no torch and cannot get one.
"""

from __future__ import annotations

__all__ = [
    "ActorCritic",
    "FusionEncoder",
    "KairosVLA",
    "LanguageEncoder",
    "PolicyHeads",
    "StateEncoder",
    "VLAConfig",
    "ValueHead",
    "VisionEncoder",
    "load_actor_critic",
]


def __getattr__(name: str):
    """Import submodules lazily so a missing torch fails loudly but late."""
    if name in __all__:
        from kairos.models import (
            actor_critic,
            fusion,
            language_encoder,
            policy,
            state_encoder,
            value_head,
            vision_encoder,
            vla,
        )

        for module in (language_encoder, vision_encoder, state_encoder,
                       fusion, policy, vla, value_head, actor_critic):
            if hasattr(module, name):
                return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
