"""Rollout storage and advantage estimation for PPO.

Transitions arrive one at a time from a single CAD environment, FreeCAD
recomputes are serial and slow, so the buffer is a plain append-and-stack
structure rather than a vectorized ring.

The subtlety that decides whether the advantages are correct is the difference
between the two ways an episode ends:

- **terminated**: the agent chose FINISH_DESIGN. There is no future beyond it,
  so the value of the next state is genuinely 0.
- **truncated**: the step budget ran out, or the CAD process died. The episode
  would have continued, so cutting the return at 0 would teach the policy that
  running long is catastrophic. The bootstrap value of the final state is used
  instead.

Conflating the two is the classic silent GAE bug: with a 40-step cap and
episodes that mostly hit it, every trajectory would carry a fabricated terminal
penalty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class Transition:
    """One environment step, kept as tensors on the training device."""

    inputs: dict[str, torch.Tensor]
    action: dict[str, torch.Tensor]
    log_prob: torch.Tensor
    value: torch.Tensor
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)


class RolloutBuffer:
    """Collects transitions and computes GAE advantages and returns."""

    def __init__(self, gamma: float = 0.99, gae_lambda: float = 0.95) -> None:
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.transitions: list[Transition] = []
        #: Bootstrap value per transition index, set for truncated endings.
        self._bootstrap: dict[int, float] = {}

    def __len__(self) -> int:
        return len(self.transitions)

    def clear(self) -> None:
        self.transitions.clear()
        self._bootstrap.clear()

    def add(
        self,
        inputs: dict[str, torch.Tensor],
        action: dict[str, torch.Tensor],
        log_prob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any] | None = None,
        bootstrap_value: float | None = None,
    ) -> None:
        """Append one transition.

        ``bootstrap_value`` is the critic's estimate of the state the episode
        was cut off in; it is required for truncation and ignored otherwise.
        """
        self.transitions.append(
            Transition(
                inputs={k: v.detach() for k, v in inputs.items()},
                action={k: v.detach() for k, v in action.items()},
                log_prob=log_prob.detach(),
                value=value.detach(),
                reward=float(reward),
                terminated=bool(terminated),
                truncated=bool(truncated),
                info=info or {},
            )
        )
        if truncated and not terminated:
            self._bootstrap[len(self.transitions) - 1] = float(bootstrap_value or 0.0)

    def mark_last_truncated(self, bootstrap_value: float = 0.0) -> bool:
        """Flag the most recent transition as a truncated episode ending.

        The collector can stop an episode for reasons the environment never
        reports, its own per-episode cap, or the rollout step budget running
        out. Those transitions arrive flagged as ordinary mid-episode steps, so
        GAE would chain the advantage into the *next* episode (or invent a
        terminal at the end of the buffer). Marking them here is what keeps the
        boundary real. Returns False if there is nothing to mark.
        """
        if not self.transitions:
            return False
        last = self.transitions[-1]
        if last.terminated or last.truncated:
            return False  # the environment already ended it
        last.truncated = True
        self._bootstrap[len(self.transitions) - 1] = float(bootstrap_value)
        return True

    # ---------------------------------------------------------- advantages

    def compute_advantages(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Generalized advantage estimation over the stored transitions."""
        n = len(self.transitions)
        if n == 0:
            return torch.zeros(0), torch.zeros(0)

        values = np.array([float(t.value) for t in self.transitions], dtype=np.float64)
        rewards = np.array([t.reward for t in self.transitions], dtype=np.float64)
        advantages = np.zeros(n, dtype=np.float64)

        next_advantage = 0.0
        for i in reversed(range(n)):
            transition = self.transitions[i]
            episode_over = transition.terminated or transition.truncated
            if transition.terminated:
                # Chosen ending: nothing follows, so the future is worth 0.
                next_value = 0.0
            elif transition.truncated:
                # Forced ending: the episode would have continued, so use the
                # critic's estimate rather than pretending the world stopped.
                next_value = self._bootstrap.get(i, 0.0)
            else:
                next_value = values[i + 1] if i + 1 < n else 0.0

            delta = rewards[i] + self.gamma * next_value - values[i]
            # An episode boundary cuts the GAE recursion: advantage must not
            # leak backwards from the next episode's first step.
            next_advantage = delta + (
                0.0 if episode_over else self.gamma * self.gae_lambda * next_advantage
            )
            advantages[i] = next_advantage

        returns = advantages + values
        return (
            torch.tensor(advantages, dtype=torch.float32),
            torch.tensor(returns, dtype=torch.float32),
        )

    # ------------------------------------------------------------ batching

    def batches(self, batch_size: int, shuffle: bool = True, seed: int | None = None):
        """Yield minibatches of stacked transitions with advantages attached."""
        advantages, returns = self.compute_advantages()
        n = len(self.transitions)
        order = np.arange(n)
        if shuffle:
            np.random.default_rng(seed).shuffle(order)

        for start in range(0, n, batch_size):
            index = order[start : start + batch_size]
            if len(index) == 0:
                continue
            chunk = [self.transitions[i] for i in index]
            yield {
                "inputs": _stack_dicts([t.inputs for t in chunk]),
                "action": _stack_dicts([t.action for t in chunk]),
                "log_prob": torch.stack([t.log_prob.reshape(()) for t in chunk]),
                "value": torch.stack([t.value.reshape(()) for t in chunk]),
                "advantage": advantages[index],
                "return": returns[index],
            }

    def statistics(self) -> dict[str, float]:
        """Summary of what was collected, for logging."""
        if not self.transitions:
            return {"transitions": 0}
        rewards = [t.reward for t in self.transitions]
        episodes = sum(1 for t in self.transitions if t.terminated or t.truncated)
        return {
            "transitions": len(self.transitions),
            "episodes": episodes,
            "reward_mean": float(np.mean(rewards)),
            "reward_sum": float(np.sum(rewards)),
            "terminated": sum(1 for t in self.transitions if t.terminated),
            "truncated": sum(1 for t in self.transitions if t.truncated),
            "invalid_actions": sum(1 for t in self.transitions if not t.info.get("ok", True)),
        }


def _stack_dicts(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Stack a list of same-keyed tensor dicts along a new batch axis."""
    if not items:
        return {}
    stacked = {}
    for key in items[0]:
        tensors = [item[key] for item in items]
        # Stored rows may carry a leading batch axis of 1 from collection.
        squeezed = [t[0] if t.dim() > 0 and t.shape[0] == 1 else t for t in tensors]
        stacked[key] = torch.stack(squeezed)
    return stacked
