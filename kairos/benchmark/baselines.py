"""Benchmark baselines, including two whose job is to audit the harness.

Most baselines exist to bound a learned policy from above and below. Two exist
to check that the benchmark itself is measuring what it claims:

- :class:`ExpertReplay` re-executes the recorded expert actions verbatim. It
  **must score 1.000**. If it does not, the fault is in the harness, the
  environment or the constraint checker — not in any policy — and every other
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

from kairos.actions.schema import Operation
from kairos.language import parse_requirement
from kairos.rl.action_space import NUM_OPERATIONS, OPERATIONS, PARAM_SLOTS, encode

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
    """Uniform over *legal* operations — a floor stronger than pure noise."""

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
        from kairos.actions.schema import Action
        from kairos.rl.action_space import UnrepresentableAction

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
                       parameters=raw.get("parameters") or {})
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
    pocketed through. It should nail the plate family and fail the bent ones —
    which is exactly what makes it a useful null hypothesis.
    """

    name = "scripted-spec"

    def begin_episode(self, task, seed: int = 0) -> None:
        spec = parse_requirement(task.requirement)
        box = spec.get("bounding_box_exact")
        dims = sorted(float(v) for v in box.value) if box else [40.0, 60.0, 6.0]
        self._thickness, self._width, self._length = dims[0], dims[1], dims[2]
        self._holes = int(spec.hole_count or 0)
        self._diameter = float(spec.hole_diameter or 5.0)
        self._plan = self._make_plan()
        self._cursor = 0

    def _make_plan(self) -> list[tuple[Operation, dict]]:
        plan: list[tuple[Operation, dict]] = [
            (Operation.CREATE_SKETCH, {"plane": 0.0, "offset": _MID}),
            (Operation.ADD_RECTANGLE, {
                "x": _MID, "y": _MID,
                "width": _norm(self._length, 1, 150), "height": _norm(self._width, 1, 150),
            }),
            (Operation.PAD, {"length": _norm(self._thickness, 1, 100)}),
        ]
        if self._holes:
            plan.append((Operation.CREATE_SKETCH, {"plane": 0.0, "offset": _MID}))
            for i in range(min(self._holes, 12)):
                # Spread hole centres across the plate rather than stacking them.
                fraction = (i + 1) / (self._holes + 1)
                plan.append((Operation.ADD_CIRCLE, {
                    "cx": _norm(-self._length / 3 + fraction * self._length * 2 / 3, -100, 100),
                    "cy": _MID,
                    "radius": _norm(self._diameter / 2.0, 0.5, 25),
                }))
            plan.append((Operation.POCKET, {"through_all": 1.0}))
        plan.append((Operation.FINISH_DESIGN, {}))
        return plan

    def act(self, observation: dict, task, step: int) -> tuple[int, np.ndarray, int]:
        if self._cursor >= len(self._plan):
            return _index(Operation.FINISH_DESIGN), _blank(), 0
        operation, slots = self._plan[self._cursor]
        self._cursor += 1

        params = _blank()
        for position, key in enumerate(_SLOT_ORDER.get(operation, ())):
            if key in slots:
                params[position] = float(slots[key])
        return _index(operation), params, 0


#: Which named slot each operation's parameters occupy, mirroring `decode`.
_SLOT_ORDER: dict[Operation, tuple[str, ...]] = {
    Operation.CREATE_SKETCH: ("plane", "offset"),
    Operation.ADD_RECTANGLE: ("x", "y", "width", "height"),
    Operation.ADD_CIRCLE: ("cx", "cy", "radius"),
    Operation.PAD: ("length", "reversed", "midplane"),
    Operation.POCKET: ("through_all", "depth", "reversed"),
}


def _norm(value: float, low: float, high: float) -> float:
    """Map a real dimension into the codec's [0, 1] slot range."""
    if high == low:
        return 0.0
    return float(min(1.0, max(0.0, (value - low) / (high - low))))


def registry(seed: int = 0) -> dict[str, BenchmarkPolicy]:
    """Baselines that need no checkpoint, by name."""
    return {
        p.name: p
        for p in (ExpertReplay(), ScriptedSpec(), LegalRandom(seed=seed), ImmediateFinish())
    }
