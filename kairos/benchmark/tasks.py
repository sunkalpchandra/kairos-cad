"""Benchmark tasks: an enumerable list, not a sampled pool.

Every policy faces the same tasks in the same order, so comparisons are paired.
The RL collector samples requirements from a pool, which is fine for training
and useless for a benchmark since two runs draw different work.

Two task types:

- BUILD: empty document, requirement text, full horizon. The Phase 5 setting.
- COMPLETE: the expert's first n-k actions are replayed and the policy supplies
  the last k. At k=1 a policy with high next-action accuracy should nearly
  always succeed; at k=8 it faces almost the full closed-loop problem. The curve
  between them measures compounding error, and unlike success rate it cannot be
  uniformly zero.

Difficulty tiers come from measured expert length and extracted constraint
count, not from intuition about which shapes look hard.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class TaskType(str, Enum):
    BUILD = "build"
    COMPLETE = "complete"


#: Family → difficulty tier, from measured expert step counts and the number of
#: constraints their requirement texts actually yield.
TIERS: dict[str, str] = {
    "spacer": "T1_single_feature",
    "plate": "T2_grid",
    "flange": "T2_grid",
    "l_bracket": "T3_bent",
    "corner_bracket": "T3_bent",
    "u_bracket": "T3_bent",
    "support_bracket": "T4_long_horizon",
    "reinforced_plate": "T4_long_horizon",
}

#: Suffix lengths for COMPLETE tasks.
COMPLETE_SUFFIXES: tuple[int, ...] = (1, 2, 4, 8)


@dataclass
class TaskSpec:
    """One benchmark task, fully determined by its own fields."""

    task_id: str
    task_type: TaskType
    design_id: str
    family: str
    tier: str
    requirement: str
    expert_actions: list[dict[str, Any]] = field(default_factory=list)
    #: How many trailing actions the policy must supply (COMPLETE only).
    suffix_length: int = 0
    max_steps: int = 40
    #: False when the expert prefix contains an action the codec cannot express,
    #: which caps what any policy can score. Reported, never silently averaged in.
    prefix_replayable: bool = True

    @property
    def prefix_actions(self) -> list[dict[str, Any]]:
        """Actions replayed before the policy takes over."""
        if self.task_type is not TaskType.COMPLETE:
            return []
        return self.expert_actions[: max(0, len(self.expert_actions) - self.suffix_length)]

    @property
    def expert_steps(self) -> int:
        """Actions the expert needed from the same starting point."""
        if self.task_type is TaskType.COMPLETE:
            return self.suffix_length
        return len(self.expert_actions)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["task_type"] = self.task_type.value
        return data

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TaskSpec:
        payload = dict(payload)
        payload["task_type"] = TaskType(payload["task_type"])
        return cls(**payload)


def make_task_id(task_type: TaskType, design_id: str, suffix_length: int = 0) -> str:
    """Stable id: adding a task never renumbers an existing one."""
    if task_type is TaskType.COMPLETE:
        return f"{task_type.value}-k{suffix_length}-{design_id}"
    return f"{task_type.value}-{design_id}"


def task_seed(suite_version: str, task_id: str, policy: str, repeat: int = 0) -> int:
    """Deterministic per-(task, policy, repeat) seed.

    Hashed rather than derived from an index, so adding a policy or a task never
    shifts the seeds of the others, which would silently change results that
    were supposed to be comparable.
    """
    key = f"{suite_version}|{task_id}|{policy}|{repeat}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def build_tasks(
    root: str | Path,
    design_ids: list[str],
    task_types: tuple[TaskType, ...] = (TaskType.BUILD, TaskType.COMPLETE),
    suffixes: tuple[int, ...] = COMPLETE_SUFFIXES,
    max_steps: int = 40,
) -> list[TaskSpec]:
    """Enumerate tasks from recorded trajectories of the given designs."""
    root = Path(root)
    tasks: list[TaskSpec] = []

    for design_id in sorted(design_ids):
        path = root / "designs" / design_id / "trajectory.json"
        if not path.exists():
            continue
        try:
            trajectory = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        actions = trajectory.get("actions") or []
        family = trajectory.get("family", "unknown")
        requirement = trajectory.get("requirement", "")
        if not actions or not requirement:
            continue

        tier = TIERS.get(family, "unknown")
        if TaskType.BUILD in task_types:
            tasks.append(
                TaskSpec(
                    task_id=make_task_id(TaskType.BUILD, design_id),
                    task_type=TaskType.BUILD,
                    design_id=design_id,
                    family=family,
                    tier=tier,
                    requirement=requirement,
                    expert_actions=actions,
                    max_steps=max_steps,
                )
            )

        if TaskType.COMPLETE in task_types:
            for k in suffixes:
                # A suffix longer than the build is the build; skip rather than
                # silently emitting a duplicate BUILD task under another id.
                if k >= len(actions):
                    continue
                tasks.append(
                    TaskSpec(
                        task_id=make_task_id(TaskType.COMPLETE, design_id, k),
                        task_type=TaskType.COMPLETE,
                        design_id=design_id,
                        family=family,
                        tier=tier,
                        requirement=requirement,
                        expert_actions=actions,
                        suffix_length=k,
                        max_steps=max_steps,
                        prefix_replayable=True,
                    )
                )
    return tasks


def select(
    tasks: list[TaskSpec],
    task_types: tuple[TaskType, ...] | None = None,
    tiers: tuple[str, ...] | None = None,
    limit_per_group: int | None = None,
) -> list[TaskSpec]:
    """Filter tasks, keeping at most ``limit_per_group`` per (type, tier).

    Capping per group rather than globally keeps every tier represented in a
    smoke run; a global head would return only spacers.
    """
    chosen: list[TaskSpec] = []
    counts: dict[tuple[str, str], int] = {}
    for task in tasks:
        if task_types and task.task_type not in task_types:
            continue
        if tiers and task.tier not in tiers:
            continue
        key = (task.task_type.value, task.tier)
        if limit_per_group is not None and counts.get(key, 0) >= limit_per_group:
            continue
        counts[key] = counts.get(key, 0) + 1
        chosen.append(task)
    return chosen


def save_tasks(tasks: list[TaskSpec], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([t.to_dict() for t in tasks], indent=2, sort_keys=True) + "\n"
    )
    return path


def load_tasks(path: str | Path) -> list[TaskSpec]:
    return [TaskSpec.from_dict(p) for p in json.loads(Path(path).read_text())]
