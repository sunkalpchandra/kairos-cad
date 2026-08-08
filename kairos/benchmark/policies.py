"""Adapters that let trained checkpoints enter the benchmark.

The baselines in :mod:`kairos.benchmark.baselines` read the task and the raw
observation. A trained policy needs tensors, and building them is exactly what
:func:`kairos.rl.collect.build_inputs` already does for PPO rollouts, so this
module reuses that rather than growing a second encoder that could silently
drift from the one training used.

Imports torch, so it is optional: the benchmark runs its baselines and reports
the harness invariants without it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from kairos.benchmark.baselines import BenchmarkPolicy
from kairos.rl.collect import build_inputs


class TorchPolicy(BenchmarkPolicy):
    """Wraps an :class:`ActorCritic` so the benchmark runner can drive it."""

    def __init__(self, model, name: str, deterministic: bool = True) -> None:
        self.model = model.eval()
        self.name = name
        self.deterministic = deterministic
        self._max_text_length = int(model.vla.config.max_text_length)

    def begin_episode(self, task, seed: int = 0) -> None:
        # Sampling policies must still be reproducible per (task, policy).
        torch.manual_seed(seed)

    @torch.no_grad()
    def act(self, observation: dict, task, step: int) -> tuple[int, np.ndarray, int]:
        inputs = build_inputs(
            task.requirement, observation, max_text_length=self._max_text_length
        )
        out = self.model.act(inputs, deterministic=self.deterministic)
        action = out["action"]
        target = action.get("target")
        return (
            int(action["operation"][0]),
            action["parameters"][0].cpu().numpy(),
            int(target[0]) if target is not None else 0,
        )


def load_bc_policy(path: str | Path, name: str = "bc") -> TorchPolicy:
    """Load a behavioral-cloning checkpoint as a benchmark policy.

    Wrapped in an ActorCritic because the benchmark drives everything through
    one interface; the value head is unused here and its random weights never
    influence a decision.
    """
    from kairos.models.actor_critic import ActorCritic

    return TorchPolicy(ActorCritic.from_bc_checkpoint(Path(path)), name=name)


def load_ppo_policy(path: str | Path, name: str = "ppo") -> TorchPolicy:
    """Load a PPO checkpoint as a benchmark policy."""
    from kairos.models.actor_critic import load_actor_critic

    return TorchPolicy(load_actor_critic(Path(path)), name=name)
