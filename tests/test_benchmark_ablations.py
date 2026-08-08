"""Ablation wrapper tests — pure python, no torch, no FreeCAD."""

import numpy as np
import pytest

from kairos.benchmark.ablations import (
    BlankRequirement,
    NoActionMask,
    ShuffledRequirement,
    build_ablations,
)
from kairos.benchmark.baselines import BenchmarkPolicy
from kairos.benchmark.tasks import TaskSpec, TaskType


class Spy(BenchmarkPolicy):
    """Records exactly what the wrapper handed it."""

    name = "spy"

    def __init__(self) -> None:
        self.requirements: list[str] = []
        self.masks: list[np.ndarray] = []

    def begin_episode(self, task, seed: int = 0) -> None:
        self.requirements.append(task.requirement)

    def act(self, observation, task, step):
        self.requirements.append(task.requirement)
        self.masks.append(np.asarray(observation.get("action_mask", [])))
        return 0, np.zeros(6), 0


def _task(requirement="Design a plate 60 x 40 x 5 mm"):
    return TaskSpec(
        task_id="build-design_000000",
        task_type=TaskType.BUILD,
        design_id="design_000000",
        family="plate",
        tier="T2_grid",
        requirement=requirement,
        expert_actions=[{"operation": "FINISH_DESIGN", "target": None, "parameters": {}}],
    )


def _observation(mask=None):
    return {
        "numeric": np.zeros(24),
        "action_mask": np.zeros(38) if mask is None else mask,
    }


def test_shuffled_requirement_hands_over_a_different_one():
    spy = Spy()
    pool = ["Design a spacer 20 mm tall", "Design a flange of outer diameter 80 mm"]
    ablated = ShuffledRequirement(spy, pool, seed=0)
    task = _task()

    ablated.begin_episode(task, seed=1)
    ablated.act(_observation(), task, 0)
    assert spy.requirements, "the inner policy was never called"
    assert all(r != task.requirement for r in spy.requirements)
    assert all(r in pool for r in spy.requirements)


def test_shuffled_requirement_never_returns_the_real_one():
    """A swap that happened to pick the same text would measure nothing."""
    spy = Spy()
    task = _task()
    # Pool contains the task's own requirement plus one other.
    ablated = ShuffledRequirement(spy, [task.requirement, "Design a spacer"], seed=3)
    for seed in range(10):
        ablated.begin_episode(task, seed=seed)
    assert all(r == "Design a spacer" for r in spy.requirements)


def test_shuffled_requirement_falls_back_when_the_pool_has_only_the_real_one():
    spy = Spy()
    task = _task()
    ablated = ShuffledRequirement(spy, [task.requirement], seed=0)
    ablated.begin_episode(task, seed=0)  # must not raise or hang
    assert spy.requirements == [task.requirement]


def test_blank_requirement_removes_the_content():
    spy = Spy()
    task = _task()
    BlankRequirement(spy).begin_episode(task, seed=0)
    assert spy.requirements == [BlankRequirement.BLANK]
    assert "60" not in spy.requirements[0]


def test_no_action_mask_makes_every_operation_look_legal():
    spy = Spy()
    mask = np.zeros(38, dtype=np.int64)
    mask[3] = 1
    NoActionMask(spy).act(_observation(mask), _task(), 0)
    assert spy.masks[0].sum() == 38, "the mask was not stripped"


def test_no_action_mask_leaves_a_maskless_observation_alone():
    spy = Spy()
    NoActionMask(spy).act({"numeric": np.zeros(24)}, _task(), 0)
    assert spy.masks[0].size == 0


def test_ablations_do_not_mutate_the_original_task():
    """The runner scores against the real task; perturbing it would corrupt that."""
    spy = Spy()
    task = _task()
    original = task.requirement
    ShuffledRequirement(spy, ["other"], seed=0).begin_episode(task, seed=0)
    BlankRequirement(spy).begin_episode(task, seed=0)
    assert task.requirement == original


def test_ablations_do_not_mutate_the_original_observation():
    spy = Spy()
    observation = _observation(np.zeros(38, dtype=np.int64))
    NoActionMask(spy).act(observation, _task(), 0)
    assert np.asarray(observation["action_mask"]).sum() == 0


def test_wrapped_names_identify_the_ablation():
    spy = Spy()
    names = set(build_ablations(spy, ["a", "b"], seed=0))
    assert names == {"spy+shuffled-req", "spy+blank-req", "spy+no-mask"}


@pytest.mark.parametrize("wrapper", [BlankRequirement, NoActionMask])
def test_wrappers_delegate_the_action_through(wrapper):
    spy = Spy()
    operation, params, target = wrapper(spy).act(_observation(), _task(), 0)
    assert (operation, target) == (0, 0)
    assert params.shape == (6,)
