"""Collect PPO rollouts by driving the bridged CAD environment.

Turns the environment's observation (a numeric state, a legality mask, and the
build's feature history) into the tensor inputs the VLA expects, samples an
action, sends it across the bridge, and stores the transition.

Requirements are **sampled per episode** from a pool rather than fixed. A
policy trained against one requirement learns that requirement's build, not how
to read requirements at all — and the language encoder would receive no
training signal whatsoever.

Episode endings are recorded as terminated or truncated separately, because
GAE treats them differently (see :mod:`kairos.rl.buffer`), and a crashed CAD
process counts as truncation: the episode is unfinished, not a failure the
policy chose.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from kairos.language import tokenizer as tk
from kairos.representation.feature_encoder import encode_history


@dataclass
class EpisodeSummary:
    """One finished episode, for logging and evaluation."""

    requirement: str
    steps: int = 0
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    crashed: bool = False
    invalid_actions: int = 0
    finished_successfully: bool = False
    satisfaction_rate: float = 0.0
    has_solid: bool = False
    mass_g: float = 0.0
    operations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement": self.requirement[:120],
            "steps": self.steps,
            "reward": round(self.reward, 4),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "crashed": self.crashed,
            "invalid_actions": self.invalid_actions,
            "finished_successfully": self.finished_successfully,
            "satisfaction_rate": round(self.satisfaction_rate, 4),
            "has_solid": self.has_solid,
            "mass_g": round(self.mass_g, 2),
        }


def build_inputs(
    requirement: str,
    observation: dict[str, Any],
    device: torch.device | str = "cpu",
    max_text_length: int = 64,
    history_length: int = 16,
) -> dict[str, torch.Tensor]:
    """Turn one bridged observation into a batch-of-one model input."""
    ids, values, mask = tk.encode(requirement, max_length=max_text_length)
    history_ids, _ = encode_history(
        list(observation.get("feature_history", [])), max_length=history_length
    )
    return {
        "token_ids": torch.tensor(np.asarray([ids]), dtype=torch.long, device=device),
        "token_values": torch.tensor(np.asarray([values]), dtype=torch.float32, device=device),
        "token_mask": torch.tensor(np.asarray([mask]), dtype=torch.long, device=device),
        "numeric": torch.tensor(
            np.asarray(observation["numeric"])[None], dtype=torch.float32, device=device
        ),
        "history": torch.tensor(history_ids[None], dtype=torch.long, device=device),
        "operation_mask": torch.tensor(
            np.asarray(observation["action_mask"])[None], dtype=torch.long, device=device
        ),
    }


class RolloutCollector:
    """Runs episodes against a bridged environment and fills a buffer."""

    def __init__(
        self,
        env,
        model,
        requirements: list[str],
        max_episode_steps: int = 40,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> None:
        if not requirements:
            raise ValueError("at least one requirement is needed to sample episodes")
        self.env = env
        self.model = model
        self.requirements = list(requirements)
        self.max_episode_steps = int(max_episode_steps)
        self.device = torch.device(device)
        self.rng = random.Random(seed)
        self.episodes: list[EpisodeSummary] = []
        # Read the text length off the model rather than assuming the default:
        # a shorter-configured policy has a shorter position embedding, and
        # over-long tokenization would index past it.
        self.max_text_length = int(
            getattr(getattr(model, "vla", None), "config", None).max_text_length
            if getattr(model, "vla", None) is not None
            else 64
        )

    def collect(
        self,
        buffer,
        n_steps: int,
        deterministic: bool = False,
        max_episodes: int | None = None,
    ) -> list[EpisodeSummary]:
        """Fill ``buffer`` until ``n_steps`` transitions or ``max_episodes``.

        Training wants a step budget (a fixed amount of experience per update);
        evaluation wants an episode count (N complete attempts). Whichever
        limit is reached first ends collection.
        """
        collected: list[EpisodeSummary] = []
        steps_taken = 0

        while steps_taken < n_steps:
            if max_episodes is not None and len(collected) >= max_episodes:
                break
            summary = self._run_episode(buffer, n_steps - steps_taken, deterministic)
            steps_taken += summary.steps
            collected.append(summary)
            self.episodes.append(summary)
            if summary.steps == 0:
                # The environment cannot even reset; stop rather than spin.
                break
        return collected

    def _run_episode(self, buffer, budget: int, deterministic: bool) -> EpisodeSummary:
        requirement = self.rng.choice(self.requirements)
        summary = EpisodeSummary(requirement=requirement)

        try:
            observation = self.env.reset(requirement=requirement)
        except Exception:
            summary.crashed = True
            summary.truncated = True
            return summary

        limit = min(budget, self.max_episode_steps)
        for _ in range(limit):
            inputs = build_inputs(
                requirement, observation, device=self.device,
                max_text_length=self.max_text_length,
            )
            self.model.eval()
            out = self.model.act(inputs, deterministic=deterministic)

            operation = int(out["action"]["operation"][0])
            parameters = out["action"]["parameters"][0].detach().cpu().numpy()
            target = int(out["action"].get("target", torch.zeros(1))[0])

            next_observation, reward, terminated, truncated, info = self.env.step(
                operation, parameters, target
            )
            crashed = bool(info.get("crashed", False))

            # A crash truncates: the episode is unfinished, not a policy
            # failure, so GAE must bootstrap rather than assume a terminal 0.
            bootstrap = 0.0
            if truncated and not terminated and next_observation is not None:
                with torch.no_grad():
                    next_inputs = build_inputs(
                        requirement, next_observation, device=self.device,
                        max_text_length=self.max_text_length,
                    )
                    bootstrap = float(self.model.distribution(next_inputs)[1][0])

            buffer.add(
                inputs=inputs,
                action={k: v for k, v in out["action"].items()},
                log_prob=out["log_prob"],
                value=out["value"],
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
                bootstrap_value=bootstrap,
            )

            summary.steps += 1
            summary.reward += float(reward)
            summary.operations.append(str(info.get("operation", "?")))
            if not info.get("ok", True):
                summary.invalid_actions += 1
            summary.satisfaction_rate = float(info.get("satisfaction_rate", 0.0))

            if next_observation is not None:
                observation = next_observation
                summary.has_solid = bool(next_observation.get("has_solid", False))
                summary.mass_g = float(next_observation.get("mass_g", 0.0))

            ended = terminated or truncated or crashed
            if ended:
                summary.terminated = bool(terminated)
                summary.truncated = bool(truncated or crashed)
                summary.crashed = crashed
                # A successful finish is FINISH_DESIGN on a valid solid whose
                # measured constraints all hold — the reward tracker's own
                # definition, read back rather than re-derived here.
                summary.finished_successfully = bool(
                    terminated
                    and info.get("all_satisfied", False)
                    and (info.get("reward_components", {}).get("finish", 0.0) > 0)
                )
                break
        else:
            # The loop ran out of budget rather than the environment ending the
            # episode, so the last transition is still flagged mid-episode.
            # Left that way, GAE chains its advantage into the next episode.
            summary.truncated = True
            bootstrap = 0.0
            if observation is not None:
                with torch.no_grad():
                    bootstrap = float(
                        self.model.distribution(
                            build_inputs(
                                requirement, observation, device=self.device,
                                max_text_length=self.max_text_length,
                            )
                        )[1][0]
                    )
            buffer.mark_last_truncated(bootstrap)

        return summary


def summarize_episodes(episodes: list[EpisodeSummary]) -> dict[str, Any]:
    """Aggregate episode summaries into the numbers worth logging."""
    if not episodes:
        return {"episodes": 0}
    scored = [e for e in episodes if e.steps > 0]
    if not scored:
        return {"episodes": len(episodes), "steps": 0}
    return {
        "episodes": len(scored),
        "steps": sum(e.steps for e in scored),
        "reward_mean": float(np.mean([e.reward for e in scored])),
        "reward_max": float(np.max([e.reward for e in scored])),
        "episode_length_mean": float(np.mean([e.steps for e in scored])),
        "success_rate": float(np.mean([e.finished_successfully for e in scored])),
        "solid_rate": float(np.mean([e.has_solid for e in scored])),
        "invalid_action_rate": float(
            np.sum([e.invalid_actions for e in scored]) / max(sum(e.steps for e in scored), 1)
        ),
        "crash_rate": float(np.mean([e.crashed for e in scored])),
        "satisfaction_rate_mean": float(np.mean([e.satisfaction_rate for e in scored])),
    }
