"""The PPO training loop: collect, update, evaluate, checkpoint.

Kept separate from :class:`PPOTrainer` (which owns the objective) and
:class:`RolloutCollector` (which owns the environment) so each is testable
without the others.

Two operational choices matter more here than the algorithm:

**Checkpoint the best evaluation, not the last iteration.** RL on a sparse,
shaped reward is not monotone; the final policy is routinely worse than one
from twenty iterations earlier. Keeping only the last would silently discard
the best result of a multi-hour run.

**Evaluate greedily on held-out requirements.** Training reward is measured
under a sampling policy on requirements the policy trains against, so it says
little about capability. Success rate is the number that matters, and it is
measured deterministically on requirements never trained against.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from kairos.rl.buffer import RolloutBuffer
from kairos.rl.collect import RolloutCollector, summarize_episodes


@dataclass
class LoopConfig:
    """How long to train and how often to look at the result."""

    iterations: int = 30
    steps_per_iteration: int = 256
    max_episode_steps: int = 40
    eval_every: int = 5
    eval_episodes: int = 8
    seed: int = 0
    out_dir: str = "runs/ppo"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class IterationRecord:
    """One iteration's collection, update, and (optional) evaluation."""

    iteration: int
    seconds: float = 0.0
    rollout: dict[str, Any] = field(default_factory=dict)
    update: dict[str, Any] = field(default_factory=dict)
    evaluation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PPOTrainingLoop:
    """Drives collection and updates, tracking the best policy seen."""

    def __init__(
        self,
        trainer,
        collector: RolloutCollector,
        config: LoopConfig | None = None,
        eval_requirements: list[str] | None = None,
    ) -> None:
        self.trainer = trainer
        self.collector = collector
        self.config = config or LoopConfig()
        self.eval_requirements = eval_requirements or collector.requirements
        self.history: list[IterationRecord] = []
        self.best_success_rate = -1.0
        self.best_iteration = -1

    # ------------------------------------------------------------ evaluation

    def evaluate(self, episodes: int | None = None) -> dict[str, Any]:
        """Greedy rollouts on held-out requirements."""
        episodes = episodes or self.config.eval_episodes
        evaluator = RolloutCollector(
            self.collector.env,
            self.trainer.model,
            self.eval_requirements,
            max_episode_steps=self.config.max_episode_steps,
            device=self.collector.device,
            seed=self.config.seed + 1000,
        )
        buffer = RolloutBuffer()
        collected = evaluator.collect(
            buffer,
            n_steps=episodes * self.config.max_episode_steps,
            deterministic=True,
            max_episodes=episodes,
        )
        # `collect` stops on the step budget, which can leave a partial final
        # episode; score only the ones that actually ended.
        finished = [e for e in collected if (e.terminated or e.truncated) and e.steps > 0]
        return summarize_episodes(finished or collected)

    # -------------------------------------------------------------- training

    def resume_from(self, out_dir: str | Path) -> int:
        """Restore history and best-so-far from a previous run's directory.

        Weights are loaded separately (by the caller, from ``last.pt``); this
        recovers the bookkeeping so a resumed run does not overwrite a better
        earlier checkpoint with a worse new one, and does not renumber its
        iterations from 1.
        """
        history_path = Path(out_dir) / "history.json"
        if not history_path.exists():
            return 0
        records = json.loads(history_path.read_text())
        self.history = [
            IterationRecord(
                iteration=r["iteration"],
                seconds=r.get("seconds", 0.0),
                rollout=r.get("rollout", {}),
                update=r.get("update", {}),
                evaluation=r.get("evaluation"),
            )
            for r in records
        ]
        for record in self.history:
            if record.evaluation:
                success = float(record.evaluation.get("success_rate", 0.0))
                if success > self.best_success_rate:
                    self.best_success_rate = success
                    self.best_iteration = record.iteration
        return self.history[-1].iteration if self.history else 0

    def run(self, on_iteration=None, start_iteration: int = 0) -> list[IterationRecord]:
        """Train for ``config.iterations``, returning the per-iteration log."""
        out_dir = Path(self.config.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        first = start_iteration + 1
        for iteration in range(first, first + self.config.iterations):
            started = time.perf_counter()
            buffer = RolloutBuffer(
                gamma=self.trainer.config.gamma, gae_lambda=self.trainer.config.gae_lambda
            )
            episodes = self.collector.collect(buffer, self.config.steps_per_iteration)
            update = self.trainer.update(buffer)

            record = IterationRecord(
                iteration=iteration,
                # buffer.statistics() also defines "episodes" (terminal transitions),
                # which would silently overwrite the episodes actually run.
                rollout={
                    **{f"buffer_{k}": v for k, v in buffer.statistics().items()},
                    **summarize_episodes(episodes),
                },
                update=update.to_dict(),
            )

            if self.config.eval_every and iteration % self.config.eval_every == 0:
                record.evaluation = self.evaluate()
                success = float(record.evaluation.get("success_rate", 0.0))
                if success > self.best_success_rate:
                    self.best_success_rate = success
                    self.best_iteration = iteration
                    self.trainer.model.save(
                        out_dir / "best.pt",
                        extra={"iteration": iteration, "evaluation": record.evaluation},
                    )

            record.seconds = time.perf_counter() - started
            self.history.append(record)
            if on_iteration is not None:
                on_iteration(record)

            self.trainer.model.save(out_dir / "last.pt", extra={"iteration": iteration})
            self._write_history(out_dir / "history.json")

        return self.history

    def _write_history(self, path: Path) -> None:
        path.write_text(
            json.dumps([record.to_dict() for record in self.history], indent=2) + "\n"
        )

    def report(self) -> dict[str, Any]:
        """Everything worth persisting about the run."""
        evaluations = [r for r in self.history if r.evaluation]
        return {
            "iterations": len(self.history),
            "best_success_rate": self.best_success_rate,
            "best_iteration": self.best_iteration,
            "final_evaluation": evaluations[-1].evaluation if evaluations else None,
            "loop_config": self.config.to_dict(),
            "ppo_config": self.trainer.config.to_dict(),
            "history": [record.to_dict() for record in self.history],
        }
