"""Trajectory recording: per-step states, actions, and shaped rewards.

A ``TrajectoryRecorder`` hooks an ``ActionExecutor`` and, after every action,
captures the observation snapshot, the numeric state vector, and the shaped
reward — single-pass, during the same execution that builds the design. The
result is the project trajectory format:

    {"requirement": ..., "states": [...], "actions": [...],
     "rewards": [...], "final_metrics": {...}}

which doubles as behavioral-cloning data (state → action pairs) and as
reward-curve ground truth for the dashboard.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kairos.actions.executor import ActionExecutor
from kairos.actions.schema import Action, ActionResult
from kairos.evaluation.constraints import check_constraints
from kairos.language import parse_requirement
from kairos.language.spec import EngineeringSpec
from kairos.representation.numerical_encoder import encode_numeric
from kairos.representation.observation import observe
from kairos.rl.rewards import RewardTracker, RewardWeights


class TrajectoryRecorder:
    """Attach to an executor; records (state, action, reward) per step."""

    def __init__(
        self,
        executor: ActionExecutor,
        requirement: str,
        max_steps: int = 64,
        weights: RewardWeights | None = None,
    ) -> None:
        self.executor = executor
        self.requirement = requirement
        self.spec: EngineeringSpec = parse_requirement(requirement)
        self.max_steps = max_steps
        self.tracker = RewardTracker(self.spec, weights)
        self.steps: list[dict[str, Any]] = []
        executor.callbacks.append(self._on_action)

    # ------------------------------------------------------------ recording

    def _on_action(self, action: Action, result: ActionResult) -> None:
        # Wall thickness is ray-cast against the solid, far too costly to run
        # every step — but the terminal step is the one whose constraint
        # report decides success, so it is measured exactly there. Without
        # this, min_wall_thickness stays 'unmeasured' on every design that
        # declares one, which is the state Phase 6 exists to end.
        observation = observe(self.executor.engine, wall_thickness=bool(result.done))
        breakdown = self.tracker.step(result, observation)
        report = self.tracker.last_report or check_constraints(observation, self.spec)
        numeric = encode_numeric(
            observation,
            self.spec,
            step=len(self.steps) + 1,
            max_steps=self.max_steps,
            satisfaction_rate=report.satisfaction_rate,
            all_satisfied=report.all_measured_satisfied,
        )
        self.steps.append(
            {
                "action": action.to_dict(),
                "ok": result.ok,
                "reward": breakdown.to_dict(),
                "numeric_state": [round(float(v), 6) for v in numeric.tolist()],
                "summary": observation["summary"],
                "constraints": report.to_dict(),
            }
        )

    # -------------------------------------------------------------- output

    def to_dict(self) -> dict[str, Any]:
        total_reward = sum(s["reward"]["total"] for s in self.steps)
        final = self.steps[-1] if self.steps else {}
        return {
            "requirement": self.requirement,
            "spec": self.spec.to_dict(),
            "states": [s["numeric_state"] for s in self.steps],
            "actions": [s["action"] for s in self.steps],
            "rewards": [s["reward"]["total"] for s in self.steps],
            "reward_breakdowns": [s["reward"] for s in self.steps],
            "step_summaries": [s["summary"] for s in self.steps],
            "final_metrics": {
                "total_reward": round(total_reward, 6),
                "steps": len(self.steps),
                "invalid_actions": sum(1 for s in self.steps if not s["ok"]),
                "constraints": final.get("constraints"),
                "summary": final.get("summary"),
            },
        }

    def write(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))
        return path
