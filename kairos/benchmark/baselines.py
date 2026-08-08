"""Benchmark baselines, including two whose job is to audit the harness.

Most baselines exist to bound a learned policy from above and below. Two exist
to check that the benchmark itself is measuring what it claims:

- :class:`ExpertReplay` re-executes the recorded expert actions verbatim. It
  **must score 1.000**. If it does not, the fault is in the harness, the
  environment or the constraint checker, not in any policy, and every other
  number that run produces is meaningless.
- :class:`ImmediateFinish` calls FINISH_DESIGN at once. It **must score bottom
  on every metric**. This is not a hypothetical: PPO trained from scratch
  converged on exactly this degenerate policy, driving episode length to ~2
  steps because quitting stops the per-action cost. Any metric it can win is a
  broken metric.

:class:`ScriptedSpec` is the null hypothesis. It reads the parsed requirement
and emits a fixed rectangle → pad → circles → pocket → finish recipe, with no
learning at all. If a trained policy cannot beat it, that is the headline
result, not a footnote.
"""

from __future__ import annotations

import numpy as np

from kairos.actions.schema import Action, Operation
from kairos.language import parse_requirement
from kairos.rl.action_space import (
    NUM_OPERATIONS,
    OPERATIONS,
    PARAM_SLOTS,
    UnrepresentableAction,
    encode,
)

#: Slot values are normalized to [0, 1]; this is the codec's midpoint.
_MID = 0.5


def _blank() -> np.ndarray:
    return np.full(PARAM_SLOTS, _MID, dtype=np.float64)


def _index(operation: Operation) -> int:
    return OPERATIONS.index(operation)


class BenchmarkPolicy:
    """Interface the runner drives. Deliberately not a torch Module.

    A baseline needs the requirement text and the task, which the tensor-only
    interface the RL collector uses cannot carry.
    """

    name: str = "base"

    def begin_episode(self, task, seed: int = 0) -> None:  # pragma: no cover - trivial
        """Reset any per-episode state."""

    def act(self, observation: dict, task, step: int) -> tuple[int, np.ndarray, int]:
        """Return ``(operation_index, params[PARAM_SLOTS], target_index)``."""
        raise NotImplementedError


class ImmediateFinish(BenchmarkPolicy):
    """Finishes on the first step. The degenerate optimum, kept as a tripwire."""

    name = "immediate-finish"

    def act(self, observation: dict, task, step: int) -> tuple[int, np.ndarray, int]:
        return _index(Operation.FINISH_DESIGN), _blank(), 0


class LegalRandom(BenchmarkPolicy):
    """Uniform over *legal* operations, a floor stronger than pure noise."""

    name = "legal-random"

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def begin_episode(self, task, seed: int = 0) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, observation: dict, task, step: int) -> tuple[int, np.ndarray, int]:
        mask = np.asarray(observation.get("action_mask", []), dtype=np.int64)
        legal = np.flatnonzero(mask) if mask.size else np.arange(NUM_OPERATIONS)
        if legal.size == 0:
            legal = np.arange(NUM_OPERATIONS)
        operation = int(self._rng.choice(legal))
        return operation, self._rng.random(PARAM_SLOTS), int(self._rng.integers(0, 8))


class ExpertReplay(BenchmarkPolicy):
    """Replays the expert's remaining actions. The harness-validation oracle.

    Actions are pushed through ``encode()`` so the replay travels the same codec
    a policy would: the gap between this and a hypothetical direct replay is the
    **codec ceiling**, the real upper bound for any policy on this action space.
    Steps the codec cannot express are counted rather than skipped silently.
    """

    name = "oracle-replay"

    def __init__(self) -> None:
        self.unrepresentable = 0

    def begin_episode(self, task, seed: int = 0) -> None:
        self._remaining = list(task.expert_actions[len(task.prefix_actions):])
        self._cursor = 0

    def act(self, observation: dict, task, step: int) -> tuple[int, np.ndarray, int]:
        if self._cursor >= len(self._remaining):
            return _index(Operation.FINISH_DESIGN), _blank(), 0

        raw = self._remaining[self._cursor]
        self._cursor += 1
        try:
            operation = Operation(raw["operation"])
        except ValueError:
            return _index(Operation.CHECK_VALIDITY), _blank(), 0

        try:
            index, params, target = encode(
                Action(operation, target=raw.get("target"),
                       parameters=raw.get("parameters") or {}),
                # Resolve the expert's recorded target against the live pool.
                # Without this the oracle emitted index 0 for every fillet,
                # chamfer, shell and pattern, applying them to whichever
                # feature happened to be listed first, and the resulting
                # failures were charged to the codec rather than to the oracle.
                targets=observation.get("targets") if observation else None,
            )
            return int(index), np.asarray(params, dtype=np.float64), int(target)
        except UnrepresentableAction:
            # The codec cannot express this expert action. Emitting it anyway
            # would fabricate a different shape, so the oracle takes the
            # operation with default parameters and the shortfall is counted.
            self.unrepresentable += 1
            return _index(operation), _blank(), 0


class ScriptedSpec(BenchmarkPolicy):
    """A hand-written recipe driven by the parsed requirement. No learning.

    Builds the only shape a fixed script can build from a parsed spec: a
    rectangular plate padded to thickness, then one circle per required hole,
    pocketed through. It should nail the plate family and fail the bent ones. Which is
    exactly what makes it a useful null hypothesis.
    """

    name = "scripted-spec"

    def __init__(self) -> None:
        self.unrepresentable = 0

    def begin_episode(self, task, seed: int = 0) -> None:
        spec = parse_requirement(task.requirement)
        box = spec.get("bounding_box_exact")
        dims = sorted(float(v) for v in box.value) if box else [40.0, 60.0, 6.0]
        self._thickness, self._width, self._length = dims[0], dims[1], dims[2]
        self._holes = int(spec.hole_count or 0)
        self._diameter = float(spec.hole_diameter or 5.0)
        self._plan = self._make_plan()
        self._cursor = 0

    def _make_plan(self) -> list[Action]:
        """The recipe as real Actions, so the codec stays the only range table.

        This used to emit normalized slot values directly, with its own copy of
        every range and its own copy of decode's slot order. Both drifted:
        narrowing _RADIUS left `_norm(radius, 0.5, 25)` here, so a 5 mm hole
        requirement was drilled at 3.1 mm and every hole check failed for
        bookkeeping reasons rather than because a script cannot do the task.
        """
        half = self._length / 3.0
        plan: list[Action] = [
            Action(Operation.CREATE_SKETCH, parameters={"plane": "XY", "offset": 0.0}),
            Action(Operation.ADD_RECTANGLE, parameters={
                "x": 0.0, "y": 0.0, "width": self._length, "height": self._width,
            }),
            Action(Operation.PAD, parameters={
                "length": self._thickness, "reversed": False, "midplane": False,
            }),
        ]
        if self._holes:
            plan.append(
                Action(Operation.CREATE_SKETCH, parameters={"plane": "XY", "offset": 0.0})
            )
            for i in range(min(self._holes, 12)):
                # Spread hole centres across the plate rather than stacking them.
                fraction = (i + 1) / (self._holes + 1)
                plan.append(Action(Operation.ADD_CIRCLE, parameters={
                    "cx": -half + fraction * self._length * 2 / 3,
                    "cy": 0.0,
                    "radius": self._diameter / 2.0,
                }))
            plan.append(Action(Operation.POCKET, parameters={"through_all": True}))
        plan.append(Action(Operation.FINISH_DESIGN))
        return plan

    def act(self, observation: dict, task, step: int) -> tuple[int, np.ndarray, int]:
        if self._cursor >= len(self._plan):
            return _index(Operation.FINISH_DESIGN), _blank(), 0
        action = self._plan[self._cursor]
        self._cursor += 1
        try:
            index, params, _ = encode(action)
        except UnrepresentableAction:
            # The parsed requirement asks for geometry outside a slot range.
            # Emitting default parameters keeps the baseline running and keeps
            # the shortfall countable instead of crashing the suite.
            self.unrepresentable += 1
            return _index(action.operation), _blank(), 0
        return int(index), np.asarray(params, dtype=np.float64), 0





def registry(seed: int = 0) -> dict[str, BenchmarkPolicy]:
    """Baselines that need no checkpoint, by name."""
    return {
        p.name: p
        for p in (ExpertReplay(), ScriptedSpec(), LegalRandom(seed=seed), ImmediateFinish())
    }
