"""Compare policies in the live CAD environment.

Behavioral cloning is scored on next-action agreement, which says nothing about
whether a policy can *drive* a build: teacher forcing hands it the expert's
state at every step, so per-step errors never compound. Closed-loop evaluation
is the number that actually answers the research question, and it is
consistently lower.

Every policy is measured the same way — same requirements, same seed, same step
budget — so the comparison is like-for-like. The random baseline exists to make
the others legible: it acts uniformly over *legal* operations, so it is already
stronger than pure noise, and anything not beating it has learned nothing.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from kairos.rl.action_space import NUM_OPERATIONS
from kairos.rl.buffer import RolloutBuffer
from kairos.rl.collect import RolloutCollector, summarize_episodes


class RandomPolicy:
    """Uniform over legal operations, uniform parameters.

    Shaped like an :class:`ActorCritic` so the same collector can drive it.
    """

    def __init__(self, n_param_slots: int = 6, seed: int = 0, max_text_length: int = 64) -> None:
        self.generator = torch.Generator().manual_seed(seed)
        self.n_param_slots = n_param_slots
        # Mimics the attribute the collector reads off a real policy.
        self.vla = type("Config", (), {"config": type("C", (), {
            "max_text_length": max_text_length
        })()})()

    def eval(self):
        return self

    def train(self, mode=True):
        return self

    def act(self, inputs, deterministic: bool = False) -> dict[str, Any]:
        mask = inputs.get("operation_mask")
        rows = inputs["numeric"].shape[0]
        if mask is None:
            operation = torch.randint(
                0, NUM_OPERATIONS, (rows,), generator=self.generator
            )
        else:
            probabilities = mask.float()
            # An all-illegal row would divide by zero; fall back to uniform.
            probabilities[probabilities.sum(dim=-1) == 0] = 1.0
            operation = torch.multinomial(probabilities, 1, generator=self.generator).squeeze(-1)
        return {
            "action": {
                "operation": operation,
                "parameters": torch.rand(rows, self.n_param_slots, generator=self.generator),
            },
            "log_prob": torch.zeros(rows),
            "value": torch.zeros(rows),
            "entropy": torch.zeros(rows),
        }

    def distribution(self, inputs):
        rows = inputs["numeric"].shape[0]
        return None, torch.zeros(rows)


def evaluate_policy(
    env,
    model,
    requirements: list[str],
    episodes: int = 12,
    max_episode_steps: int = 40,
    deterministic: bool = True,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Roll a policy out and summarize what it achieved."""
    collector = RolloutCollector(
        env, model, requirements,
        max_episode_steps=max_episode_steps, device=device, seed=seed,
    )
    buffer = RolloutBuffer()
    collected = collector.collect(
        buffer,
        n_steps=episodes * max_episode_steps,
        deterministic=deterministic,
        max_episodes=episodes,
    )
    finished = [e for e in collected if (e.terminated or e.truncated) and e.steps > 0]
    scored = finished or collected
    summary = summarize_episodes(scored)
    summary["episodes_requested"] = episodes
    summary["deterministic"] = deterministic

    # A point estimate over a dozen episodes overstates what was measured.
    successes = [float(e.finished_successfully) for e in scored]
    low, high = bootstrap_interval(successes, seed=seed)
    summary["success_ci"] = [round(low, 4), round(high, 4)]
    summary["per_requirement"] = per_requirement_breakdown(scored)
    return summary


def per_requirement_breakdown(episodes) -> dict[str, dict[str, Any]]:
    """Success per requirement — an aggregate can hide one dominant family."""
    grouped: dict[str, list] = {}
    for episode in episodes:
        grouped.setdefault(episode.requirement, []).append(episode)
    return {
        requirement[:60]: {
            "episodes": len(group),
            "success_rate": round(float(np.mean([e.finished_successfully for e in group])), 4),
            "reward_mean": round(float(np.mean([e.reward for e in group])), 4),
        }
        for requirement, group in grouped.items()
    }


def compare_policies(
    env,
    policies: dict[str, Any],
    requirements: list[str],
    episodes: int = 12,
    max_episode_steps: int = 40,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> dict[str, dict[str, Any]]:
    """Score several policies under identical conditions."""
    results: dict[str, dict[str, Any]] = {}
    for name, model in policies.items():
        results[name] = evaluate_policy(
            env, model, requirements,
            episodes=episodes, max_episode_steps=max_episode_steps,
            # The same seed for every policy: they must face the same
            # requirements in the same order or the comparison is noise.
            seed=seed, device=device,
            deterministic=not isinstance(model, RandomPolicy),
        )
    return results


def format_comparison(results: dict[str, dict[str, Any]]) -> str:
    """Render a comparison table."""
    header = (
        f"{'policy':>12}  {'episodes':>8}  {'success':>8}  {'95% CI':>14}  {'solid':>7}  "
        f"{'reward':>8}  {'steps':>6}  {'invalid':>8}"
    )
    lines = [header, "-" * len(header)]
    for name, row in results.items():
        if not row.get("episodes"):
            lines.append(f"{name:>12}  {'no episodes':>8}")
            continue
        low, high = row.get("success_ci", [float("nan"), float("nan")])
        lines.append(
            f"{name:>12}  {row['episodes']:>8}  {row.get('success_rate', 0.0):>8.3f}  "
            f"{f'[{low:.2f}, {high:.2f}]':>14}  "
            f"{row.get('solid_rate', 0.0):>7.3f}  {row.get('reward_mean', 0.0):>8.2f}  "
            f"{row.get('episode_length_mean', 0.0):>6.1f}  "
            f"{row.get('invalid_action_rate', 0.0):>8.3f}"
        )
    return "\n".join(lines)


def bootstrap_interval(
    values: list[float], samples: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    """Percentile bootstrap CI — episode counts here are small enough that a
    point estimate alone would overstate what was measured."""
    if not values:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    means = rng.choice(array, size=(samples, len(array)), replace=True).mean(axis=1)
    return (
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )
