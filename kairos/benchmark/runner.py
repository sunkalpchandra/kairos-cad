"""Drive policies over the benchmark suite and record what happened.

Separate from :class:`kairos.rl.collect.RolloutCollector` on purpose. The
collector *samples* requirements from a pool and speaks only tensors — both
right for PPO training, both wrong here. A benchmark must **enumerate** tasks so
every policy faces the same work in the same order, and it must hand the
requirement text to baselines that have no encoder at all.

``COMPLETE`` tasks replay the expert prefix through the ordinary step API
rather than a new protocol message: the environment already executes actions,
so replaying `n - k` of them and then handing over is the whole mechanism. A
prefix step that the environment rejects aborts the task rather than scoring it
— the policy never got the state the task promised, and grading that as a
policy failure would blame it for the harness.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from kairos.actions.schema import Operation
from kairos.benchmark.metrics import EpisodeOutcome
from kairos.benchmark.tasks import TaskSpec, TaskType, task_seed
from kairos.rl.action_space import OPERATIONS


@dataclass
class TaskResult:
    """One (task, policy, repeat) attempt."""

    task_id: str
    policy: str
    repeat: int
    outcome: EpisodeOutcome
    aborted: bool = False
    abort_reason: str = ""
    operations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "policy": self.policy,
            "repeat": self.repeat,
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
            "operations": self.operations,
            **self.outcome.to_dict(),
        }


def _replay_prefix(env, task: TaskSpec) -> tuple[dict | None, str]:
    """Execute the expert prefix verbatim. Returns ``(observation, error)``.

    Sent as raw actions rather than codec-encoded ones: a prefix is replayed,
    not predicted, so it need not fit the policy's action space. Encoding it
    dropped every irregular ADD_POLYGON — the profile step of six of eight
    families — which aborted the very tasks COMPLETE exists to pose.
    """
    env.reset(requirement=task.requirement)
    if not task.prefix_actions:
        return env.reset(requirement=task.requirement), ""
    observation = env.replay(task.prefix_actions)
    if observation is None:
        return None, "prefix replay was rejected by the environment"
    return observation, ""


def run_task(
    env,
    policy,
    task: TaskSpec,
    suite_version: str = "kairos-cad-v1",
    repeat: int = 0,
) -> TaskResult:
    """Run one policy on one task and score it."""
    seed = task_seed(suite_version, task.task_id, getattr(policy, "name", "policy"), repeat)
    outcome = EpisodeOutcome(
        requirement=task.requirement,
        family=task.family,
        expert_steps=task.expert_steps,
    )
    result = TaskResult(task.task_id, getattr(policy, "name", "policy"), repeat, outcome)

    try:
        if task.task_type is TaskType.COMPLETE:
            observation, error = _replay_prefix(env, task)
            if observation is None:
                result.aborted = True
                result.abort_reason = error
                return result
        else:
            observation = env.reset(requirement=task.requirement)
    except Exception as err:  # the bridge died before the policy acted
        result.aborted = True
        result.abort_reason = f"reset failed: {err}"
        return result

    policy.begin_episode(task, seed=seed)
    budget = min(task.max_steps, max(1, task.expert_steps * 3))

    for step in range(budget):
        try:
            operation, params, target = policy.act(observation, task, step)
        except Exception as err:  # a broken policy is a policy failure
            outcome.crashed = True
            result.abort_reason = f"policy raised: {err}"
            break

        next_observation, _reward, terminated, truncated, info = env.step(
            int(operation), np.asarray(params, dtype=np.float64), int(target)
        )
        outcome.steps += 1
        result.operations.append(str(info.get("operation", "?")))
        if not info.get("ok", True):
            outcome.invalid_actions += 1
        _absorb(outcome, observation, next_observation, info, int(operation))

        if next_observation is not None:
            observation = next_observation
        if info.get("crashed"):
            outcome.crashed = True
            break
        if terminated or truncated:
            outcome.finished_successfully = bool(
                terminated
                and info.get("all_satisfied", False)
                and (info.get("reward_components", {}).get("finish", 0.0) > 0)
            )
            break

    return result


def _absorb(
    outcome: EpisodeOutcome,
    before: dict | None,
    after: dict | None,
    info: dict,
    operation_index: int,
) -> None:
    """Update milestones from what the environment reported.

    Read from the environment rather than re-derived, so the benchmark cannot
    disagree with the simulator about what happened.
    """
    operation = OPERATIONS[operation_index % len(OPERATIONS)]
    if operation is Operation.CREATE_SKETCH and info.get("ok"):
        outcome.opened_a_sketch = True
    if operation in _SKETCH_GEOMETRY_OPS and info.get("ok"):
        outcome.drew_geometry = True

    state = after or before or {}
    if state.get("has_solid"):
        outcome.made_a_solid = True
        # A sketch and geometry necessarily preceded a solid, even when the
        # prefix replayed them before the policy took over.
        outcome.opened_a_sketch = True
        outcome.drew_geometry = True
    if state.get("valid") and state.get("has_solid"):
        outcome.solid_is_valid = True
    outcome.mass_g = float(state.get("mass_g", outcome.mass_g))

    rate = info.get("satisfaction_rate")
    if rate is not None:
        outcome.satisfaction_rate = float(rate)
    if info.get("all_satisfied") and outcome.solid_is_valid:
        outcome.all_constraints_met = True
        outcome.has_any_hole = True
    elif float(info.get("satisfaction_rate", 0.0)) > 0.0 and outcome.solid_is_valid:
        outcome.has_any_hole = True


_SKETCH_GEOMETRY_OPS = {
    Operation.ADD_LINE, Operation.ADD_RECTANGLE, Operation.ADD_CIRCLE,
    Operation.ADD_ARC, Operation.ADD_POLYGON,
}


def run_suite(
    env,
    policy,
    tasks: list[TaskSpec],
    suite_version: str = "kairos-cad-v1",
    repeats: int = 1,
    on_result=None,
) -> list[TaskResult]:
    """Run one policy over every task, ``repeats`` times each."""
    results: list[TaskResult] = []
    for repeat in range(repeats):
        for task in tasks:
            result = run_task(env, policy, task, suite_version, repeat)
            results.append(result)
            if on_result is not None:
                on_result(result)
    return results


def write_traces(results: list[TaskResult], path: str | Path) -> Path:
    """Append-only JSONL; every published table must be regenerable from it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for result in results:
            handle.write(json.dumps(result.to_dict()) + "\n")
    return path


def read_traces(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
