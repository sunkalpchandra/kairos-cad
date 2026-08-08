"""Task enumeration tests, pure python, no torch, no FreeCAD."""

import json

import pytest

from kairos.benchmark.tasks import (
    COMPLETE_SUFFIXES,
    TIERS,
    TaskSpec,
    TaskType,
    build_tasks,
    load_tasks,
    make_task_id,
    save_tasks,
    select,
    task_seed,
)


def _dataset(tmp_path, design_id="design_000000", n_actions=6, family="plate"):
    design = tmp_path / "designs" / design_id
    design.mkdir(parents=True)
    actions = [{"operation": "ADD_CIRCLE", "target": None, "parameters": {}}
               for _ in range(n_actions - 1)]
    actions.append({"operation": "FINISH_DESIGN", "target": None, "parameters": {}})
    (design / "trajectory.json").write_text(json.dumps({
        "requirement": f"Design a {family} 60 x 40 x 5 mm",
        "family": family,
        "actions": actions,
    }))
    return tmp_path


def test_build_task_carries_the_whole_expert_sequence(tmp_path):
    _dataset(tmp_path, n_actions=6)
    build = [t for t in build_tasks(tmp_path, ["design_000000"])
             if t.task_type is TaskType.BUILD][0]
    assert build.prefix_actions == []
    assert build.expert_steps == 6
    assert build.tier == TIERS["plate"]


def test_complete_task_splits_prefix_from_suffix(tmp_path):
    _dataset(tmp_path, n_actions=10)
    tasks = {t.suffix_length: t for t in build_tasks(tmp_path, ["design_000000"])
             if t.task_type is TaskType.COMPLETE}
    for k in COMPLETE_SUFFIXES:
        assert len(tasks[k].prefix_actions) == 10 - k
        assert tasks[k].expert_steps == k


def test_a_suffix_longer_than_the_build_is_skipped(tmp_path):
    """It would be the BUILD task under a different id."""
    _dataset(tmp_path, n_actions=3)
    suffixes = {t.suffix_length for t in build_tasks(tmp_path, ["design_000000"])
                if t.task_type is TaskType.COMPLETE}
    assert suffixes == {1, 2}  # 4 and 8 exceed the 3-action build


def test_task_ids_are_stable_and_unique(tmp_path):
    _dataset(tmp_path, n_actions=10)
    tasks = build_tasks(tmp_path, ["design_000000"])
    ids = [t.task_id for t in tasks]
    assert len(ids) == len(set(ids))
    assert make_task_id(TaskType.BUILD, "design_000001") == "build-design_000001"
    assert make_task_id(TaskType.COMPLETE, "design_000001", 4) == "complete-k4-design_000001"


def test_missing_or_unreadable_designs_are_skipped(tmp_path):
    _dataset(tmp_path)
    broken = tmp_path / "designs" / "design_000009"
    broken.mkdir(parents=True)
    (broken / "trajectory.json").write_text("{not json")
    tasks = build_tasks(tmp_path, ["design_000000", "design_000009", "design_999999"])
    assert {t.design_id for t in tasks} == {"design_000000"}


def test_select_keeps_every_tier_represented(tmp_path):
    """A global head would return only the first tier."""
    for i, family in enumerate(("spacer", "plate", "l_bracket", "support_bracket")):
        _dataset(tmp_path, design_id=f"design_00000{i}", family=family)
    tasks = build_tasks(tmp_path, [f"design_00000{i}" for i in range(4)])
    chosen = select(tasks, limit_per_group=1)
    assert len({t.tier for t in chosen}) == 4
    # One per (type, tier, k): COMPLETE fans out over suffix lengths, and each
    # k is its own group so every k draws the same tier mix.
    assert len(
        {(t.task_type.value, t.tier, t.suffix_length) for t in chosen}
    ) == len(chosen)


def test_select_filters_by_type_and_tier(tmp_path):
    _dataset(tmp_path, family="spacer")
    tasks = build_tasks(tmp_path, ["design_000000"])
    builds = select(tasks, task_types=(TaskType.BUILD,))
    assert builds and all(t.task_type is TaskType.BUILD for t in builds)
    assert select(tasks, tiers=("T4_long_horizon",)) == []


def test_seeds_are_stable_and_independent_across_policies():
    """Adding a policy or task must never shift another's seeds."""
    assert task_seed("v1", "t1", "bc") == task_seed("v1", "t1", "bc")
    assert task_seed("v1", "t1", "bc") != task_seed("v1", "t1", "ppo")
    assert task_seed("v1", "t1", "bc") != task_seed("v1", "t2", "bc")
    assert task_seed("v1", "t1", "bc", 0) != task_seed("v1", "t1", "bc", 1)
    assert task_seed("v2", "t1", "bc") != task_seed("v1", "t1", "bc")


def test_tasks_round_trip_through_json(tmp_path):
    _dataset(tmp_path, n_actions=8)
    tasks = build_tasks(tmp_path, ["design_000000"])
    restored = load_tasks(save_tasks(tasks, tmp_path / "tasks.json"))
    assert [t.task_id for t in restored] == [t.task_id for t in tasks]
    assert restored[0].task_type is tasks[0].task_type
    assert restored[-1].prefix_actions == tasks[-1].prefix_actions


def test_every_family_has_a_tier():
    from kairos.data.families import family_names

    assert set(family_names()) <= set(TIERS), "a family has no benchmark tier"


@pytest.mark.parametrize("preset_limit", [1, 4])
def test_select_is_deterministic(tmp_path, preset_limit):
    _dataset(tmp_path, n_actions=10)
    tasks = build_tasks(tmp_path, ["design_000000"])
    first = [t.task_id for t in select(tasks, limit_per_group=preset_limit)]
    second = [t.task_id for t in select(tasks, limit_per_group=preset_limit)]
    assert first == second


def test_selection_gives_every_k_the_same_tier_mix():
    """Without suffix_length in the key, the cap falls across a design's whole
    COMPLETE fan-out, so each k ends up averaging a different set of families
    and the success(k) curve reads their difficulty as compounding error.
    """
    tasks = []
    for tier, family in (("T1", "spacer"), ("T2", "plate")):
        for design in range(6):
            actions = [{"operation": "PAD", "parameters": {}} for _ in range(12)]
            tasks.append(TaskSpec(
                task_id=f"build-{family}{design}", task_type=TaskType.BUILD,
                design_id=f"{family}{design}", family=family, tier=tier,
                requirement="r", expert_actions=actions,
            ))
            for k in (1, 2, 4, 8):
                tasks.append(TaskSpec(
                    task_id=f"complete-k{k}-{family}{design}",
                    task_type=TaskType.COMPLETE, design_id=f"{family}{design}",
                    family=family, tier=tier, requirement="r",
                    expert_actions=actions, suffix_length=k,
                ))

    chosen = select(tasks, limit_per_group=2)
    mix: dict[int, set[str]] = {}
    for task in chosen:
        mix.setdefault(task.suffix_length, set()).add(task.tier)
    assert len(mix) == 5  # build plus four k values
    assert all(tiers == {"T1", "T2"} for tiers in mix.values())
    counts = {k: len([t for t in chosen if t.suffix_length == k]) for k in mix}
    assert len(set(counts.values())) == 1, f"uneven task counts per k: {counts}"
