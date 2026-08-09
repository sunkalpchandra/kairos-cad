"""Tests for the dashboard data bundle.

Every assertion here guards a *silent* failure mode. Nothing in this pipeline
raises when it reads a field that does not exist. It renders a dash, or drops
a series, or flattens a curve, and the page still looks finished. Two of these
tests were written after exactly that happened.
"""

import json

import pytest

from kairos.dashboard.bundle import (
    _bucket,
    _measured,
    _normalize_scores,
    collect_ablation_intervals,
    collect_ablations,
    collect_dataset,
    collect_designs,
    collect_effort,
    collect_failures,
    collect_families,
    collect_funnel,
    collect_jam,
    collect_matrix,
    collect_rollouts,
    collect_sources,
    collect_task_types,
    collect_training,
)


def _design(tmp_path, design_id="design_000000", **overrides):
    directory = tmp_path / "designs" / design_id
    directory.mkdir(parents=True)
    state = {
        "mass_g": 92.27,
        "volume_mm3": 34175.1,
        "surface_area_mm2": 12904.4,
        "material": "aluminum",
        "hole_count": 2,
        "bounding_box": {"x_len": 58.9, "y_len": 39.0, "z_len": 82.8},
        "topology": {"faces": 13},
    }
    state.update(overrides)
    (directory / "state.json").write_text(json.dumps(state))
    (directory / "requirements.json").write_text(json.dumps({
        "text": "Design a 90-degree corner bracket.",
        "spec": {"kind": "corner_bracket"},
    }))
    (directory / "trajectory.json").write_text(json.dumps({
        "actions": [{"operation": "CREATE_SKETCH"}, {"operation": "PAD"}],
        "final_metrics": {
            "steps": 13,
            "total_reward": 4.5,
            "constraints": {
                "satisfaction_rate": 1.0,
                "all_measured_satisfied": True,
                "results": [
                    {
                        "constraint": {"kind": "min_wall_thickness", "value": 6.2},
                        "status": "satisfied",
                        "measured": 6.187,
                        "detail": "thinnest wall 6.187 mm, need >= 6.2 mm",
                    }
                ],
            },
        },
    }))
    return directory


def test_extent_reads_the_len_fields_not_length_width_height(tmp_path):
    """The state has no `length`; reading it yields a silent null."""
    _design(tmp_path)
    design = collect_designs(tmp_path)[0]
    assert design["extent_mm"] == [58.9, 39.0, 82.8]


def test_wall_thickness_is_lifted_out_of_the_constraint_result(tmp_path):
    """It is measured by the checker and never stored on the state."""
    _design(tmp_path)
    assert collect_designs(tmp_path)[0]["min_wall_thickness_mm"] == 6.187


def test_unmeasured_constraint_reports_none_rather_than_zero(tmp_path):
    """Zero would render as a 0.00 mm wall, which reads as a defect."""
    assert _measured([{"constraint": {"kind": "hole_count"}, "status": "satisfied"}],
                     "min_wall_thickness") is None


def test_design_without_state_is_skipped_not_half_rendered(tmp_path):
    directory = tmp_path / "designs" / "design_000001"
    directory.mkdir(parents=True)
    (directory / "requirements.json").write_text("{}")
    assert collect_designs(tmp_path) == []


def test_limit_caps_embedded_designs(tmp_path):
    for i in range(4):
        _design(tmp_path, f"design_00000{i}")
    assert len(collect_designs(tmp_path, limit=2)) == 2


def test_selection_covers_every_family_before_repeating(tmp_path):
    """Slicing the first N by id looks balanced and is not.

    Design ids cycle through families, so a naive slice at limit=12 dropped
    reinforced_plate entirely -- and a family the viewer never shows reads as a
    family that does not exist.
    """
    families = ["plate", "flange", "spacer", "u_bracket"]
    for index in range(12):
        directory = _design(tmp_path, f"design_{index:06d}")
        requirement = json.loads((directory / "requirements.json").read_text())
        requirement["spec"]["kind"] = families[index % len(families)]
        (directory / "requirements.json").write_text(json.dumps(requirement))

    picked = collect_designs(tmp_path, limit=4)
    assert {d["family"] for d in picked} == set(families)


def test_selection_stays_sorted_by_id(tmp_path):
    """Round-robin picks them out of order; the list must still read in order."""
    for index in range(6):
        _design(tmp_path, f"design_{index:06d}")
    ids = [d["design_id"] for d in collect_designs(tmp_path, limit=6)]
    assert ids == sorted(ids)


class TestBucket:
    """`complete-k4-design_x` must bucket as "4", not as a build task."""

    def test_build_task(self):
        assert _bucket("build-design_000000") == "build"

    @pytest.mark.parametrize("k", [1, 2, 4, 8])
    def test_complete_task_strips_the_k_prefix(self, k):
        assert _bucket(f"complete-k{k}-design_000000") == str(k)

    def test_unknown_shape_falls_back_to_build(self):
        assert _bucket("weird") == "build"

    def test_complete_tasks_do_not_collapse_into_one_bucket(self):
        """The bug this catches flattens the entire success(k) curve."""
        buckets = {_bucket(f"complete-k{k}-d") for k in (1, 2, 4, 8)}
        assert len(buckets) == 4
        assert "build" not in buckets


def test_leaderboard_field_names_are_translated(tmp_path):
    rows = _normalize_scores({"scores": [{
        "policy": "bc", "progress_score": 0.445, "success_rate": 0.281,
        "validity_rate": 0.957, "episodes": 32, "milestone_rates": {"drew_geometry": 1.0},
    }]})
    assert rows[0]["progress_mean"] == 0.445
    assert rows[0]["validity"] == 0.957
    assert rows[0]["tasks"] == 32
    assert rows[0]["milestones"]["drew_geometry"] == 1.0


def test_bc_history_exposes_the_held_out_accuracy(tmp_path):
    """`operation_accuracy` is the held-out curve; nothing is named dev/val."""
    (tmp_path / "bc").mkdir()
    (tmp_path / "bc" / "report.json").write_text(json.dumps({
        "history": [
            {"epoch": 1, "train_accuracy": 0.66, "operation_accuracy": 0.87,
             "train_loss": 1.45, "val_loss": 0.44},
            {"epoch": 2, "train_accuracy": 0.98, "operation_accuracy": 0.981,
             "train_loss": 0.06, "val_loss": 0.05},
        ],
        "model": {"parameters": 1_140_000},
    }))
    training = collect_training(tmp_path)
    assert [r["held_out_accuracy"] for r in training["bc"]["history"]] == [0.87, 0.981]
    assert training["bc"]["best_held_out_accuracy"] == 0.981


def test_ablation_deltas_are_relative_to_the_intact_policy(tmp_path):
    (tmp_path / "leaderboard.json").write_text(json.dumps({"scores": [
        {"policy": "bc", "progress_score": 0.4},
        {"policy": "bc+shuffled-req", "progress_score": 0.3},
    ]}))
    rows = {r["name"]: r for r in collect_ablations(tmp_path)["rows"]}
    assert rows["bc"]["baseline"] is True
    assert rows["bc"]["delta"] == 0.0
    assert rows["bc+shuffled-req"]["delta"] == pytest.approx(-0.25)


def test_ablation_without_a_baseline_row_reports_no_delta(tmp_path):
    """Better an empty column than a delta against an arbitrary policy."""
    (tmp_path / "leaderboard.json").write_text(json.dumps({
        "scores": [{"policy": "bc+no-mask", "progress_score": 0.3}]
    }))
    assert collect_ablations(tmp_path)["rows"][0]["delta"] is None


def test_missing_run_directories_degrade_to_empty(tmp_path):
    """A partial repo must still build a page rather than crash the script."""
    training = collect_training(tmp_path / "nope")
    assert training["bc"]["history"] == []
    assert training["ppo"]["history"] == []


# ------------------------------------------------------------------- rollouts


def _traces(tmp_path, policy, rows):
    path = tmp_path / f"{policy}_traces.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def _episode(task_id="build-design_000000", family="corner_bracket", **overrides):
    row = {
        "task_id": task_id,
        "family": family,
        "requirement": "Design a 90-degree corner bracket.",
        "operations": ["CREATE_SKETCH", "PAD"],
        "accepted": [True, False],
        "rejections": ["", "no closed profile"],
        "steps": 2,
        "expert_steps": 20,
        "invalid_actions": 1,
        "progress_score": 0.25,
        "opened_a_sketch": True,
    }
    row.update(overrides)
    return row


def test_rollouts_report_every_policy_on_the_same_task(tmp_path):
    """Otherwise each strip describes a different task and comparing them lies."""
    _traces(tmp_path, "aaa", [_episode(), _episode(task_id="build-design_000001",
                                                  family="flange")])
    _traces(tmp_path, "zzz", [_episode()])
    out = collect_rollouts(tmp_path, per_family=1)

    first = next(t for t in out["tasks"] if t["task_id"] == "build-design_000000")
    assert sorted(e["policy"] for e in first["episodes"]) == ["aaa", "zzz"]


def test_rollouts_take_build_tasks_only(tmp_path):
    """A COMPLETE episode is mostly replayed expert prefix, so its strip would
    credit the policy with actions it never chose."""
    _traces(tmp_path, "bc", [
        _episode(task_id="complete-2-design_000000"),
        _episode(task_id="build-design_000000"),
    ])
    out = collect_rollouts(tmp_path)
    assert [t["task_id"] for t in out["tasks"]] == ["build-design_000000"]


def test_rollouts_keep_one_task_per_family(tmp_path):
    _traces(tmp_path, "bc", [
        _episode(task_id="build-design_000000", family="corner_bracket"),
        _episode(task_id="build-design_000008", family="corner_bracket"),
        _episode(task_id="build-design_000001", family="flange"),
    ])
    out = collect_rollouts(tmp_path, per_family=1)
    assert sorted(t["family"] for t in out["tasks"]) == ["corner_bracket", "flange"]


def test_an_older_trace_reports_no_per_step_record_rather_than_all_accepted(tmp_path):
    """The page distinguishes the two. An empty list must not read as a clean
    run: that would show a policy jamming as a policy succeeding."""
    _traces(tmp_path, "bc", [_episode(accepted=None, rejections=None)])
    episode = collect_rollouts(tmp_path)["tasks"][0]["episodes"][0]
    assert episode["accepted"] == []
    assert episode["operations"]


def test_per_step_record_is_truncated_to_the_operations_it_describes(tmp_path):
    """A longer accepted list than operations would shift every cell's meaning."""
    _traces(tmp_path, "bc", [_episode(accepted=[True, False, True, True])])
    episode = collect_rollouts(tmp_path)["tasks"][0]["episodes"][0]
    assert episode["accepted"] == [True, False]


def test_milestones_are_reported_in_reach_order(tmp_path):
    _traces(tmp_path, "bc", [_episode(drew_geometry=True, made_a_solid=True)])
    episode = collect_rollouts(tmp_path)["tasks"][0]["episodes"][0]
    assert episode["milestones"] == ["opened_a_sketch", "drew_geometry", "made_a_solid"]


def test_rollouts_without_traces_degrade_to_empty(tmp_path):
    assert collect_rollouts(tmp_path) == {"tasks": []}


def test_an_episode_without_a_rebuilt_solid_reports_none_not_a_missing_key(tmp_path):
    """The page distinguishes 'built nothing' from 'not rebuilt yet'. A missing
    key would read as neither and silently render an empty pane."""
    _traces(tmp_path, "bc", [_episode()])
    episode = collect_rollouts(tmp_path)["tasks"][0]["episodes"][0]
    assert "mesh" in episode
    assert episode["mesh"] is None


def test_a_rebuilt_solid_is_attached_to_its_own_policy(tmp_path):
    """Keyed by task and policy; crossing them would show one policy's part
    under another's name, which no reader could catch."""
    from kairos.dashboard.mesh import mesh_from_stl

    _traces(tmp_path, "bc", [_episode()])
    _traces(tmp_path, "ppo", [_episode()])

    # A minimal binary STL: one triangle is enough to exercise the path.
    import struct

    tri = struct.pack("<12fH", 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0)
    stl = b"\0" * 80 + struct.pack("<I", 1) + tri
    target = tmp_path / "rollout_meshes" / "build-design_000000"
    target.mkdir(parents=True)
    (target / "ppo.stl").write_bytes(stl)
    assert mesh_from_stl(target / "ppo.stl")["triangle_count"] == 1

    episodes = {e["policy"]: e for e in collect_rollouts(tmp_path)["tasks"][0]["episodes"]}
    assert episodes["ppo"]["mesh"]["triangle_count"] == 1
    assert episodes["bc"]["mesh"] is None


# --------------------------------------------------------------------- matrix


def test_matrix_columns_line_up_across_policies(tmp_path):
    """Every row indexes the same task list. A policy missing a task must leave
    a hole in place, or every cell after it describes the wrong task."""
    _traces(tmp_path, "aaa", [
        _episode(task_id="build-design_000000"),
        _episode(task_id="build-design_000001"),
    ])
    _traces(tmp_path, "zzz", [_episode(task_id="build-design_000001")])

    matrix = collect_matrix(tmp_path)
    assert [t["id"] for t in matrix["tasks"]] == [
        "build-design_000000", "build-design_000001"]
    assert matrix["cells"]["zzz"][0] is None
    assert matrix["cells"]["zzz"][1] is not None
    assert len(matrix["cells"]["aaa"]) == len(matrix["tasks"])


def test_an_aborted_episode_is_a_hole_not_a_zero(tmp_path):
    """A task the harness could not run is not a task the policy failed."""
    _traces(tmp_path, "bc", [_episode(aborted=True)])
    assert collect_matrix(tmp_path)["cells"]["bc"] == [None]


def test_matrix_counts_milestones_not_success(tmp_path):
    """Success is 0.000 for three of six policies; a success matrix would be
    one colour and would say nothing about where they differ."""
    _traces(tmp_path, "bc", [_episode(drew_geometry=True, made_a_solid=True,
                                      finished_successfully=False)])
    assert collect_matrix(tmp_path)["cells"]["bc"] == [3]


def test_matrix_without_traces_degrades_to_empty(tmp_path):
    assert collect_matrix(tmp_path) == {"tasks": [], "policies": [], "cells": {}}


# ------------------------------------------------------------------ failures


def test_failure_kind_collapses_the_instance_that_failed():
    """Pad002 and Pad007 failing the same way are one failure, not two."""
    from kairos.dashboard.bundle import _failure_kind

    a = _failure_kind('Pad failed to build: ["Pad002: state=[\'Invalid\']"]')
    b = _failure_kind('Pad failed to build: ["Pad007: state=[\'Invalid\']"]')
    assert a == b == "Pad failed to build"


def test_failure_kind_collapses_constraint_indices():
    """The indices name which geometry, not what went wrong. Leaving them in
    split one failure kind across 80 rows of count 1."""
    from kairos.dashboard.bundle import _failure_kind

    assert (_failure_kind("sketch rejected Coincident constraint (5, 2, 1, 1)")
            == _failure_kind("sketch rejected Coincident constraint (8, 8)")
            == "sketch rejected Coincident constraint")


def test_failure_kind_keeps_different_failures_apart():
    from kairos.dashboard.bundle import _failure_kind

    assert _failure_kind("Pad failed to build: [...]") != _failure_kind(
        "Pocket failed to build: [...]")


def test_failures_count_only_refused_steps(tmp_path):
    _traces(tmp_path, "bc", [_episode(
        operations=["CREATE_SKETCH", "PAD", "PAD"],
        accepted=[True, False, False],
        rejections=["", 'Pad failed to build: ["Pad001: x"]',
                    'Pad failed to build: ["Pad002: x"]'],
    )])
    row = collect_failures(tmp_path)["policies"]["bc"]
    assert row["steps"] == 3
    assert row["rejected"] == 2
    assert row["kinds"] == [{"kind": "Pad failed to build", "count": 2}]
    assert row["distinct"] == 1


def test_failures_report_what_the_top_n_left_out(tmp_path):
    """Otherwise the listed counts read as the whole of it."""
    rejections = [f"failure {i} of many" for i in range(6)]
    _traces(tmp_path, "bc", [_episode(
        operations=["PAD"] * 6, accepted=[False] * 6, rejections=rejections)])
    row = collect_failures(tmp_path, top=2)["policies"]["bc"]
    assert len(row["kinds"]) == 2
    assert row["other"] == 4
    assert row["distinct"] == 6


# ---------------------------------------------------------------- task types


def _leaderboard(tmp_path, by_task_type):
    (tmp_path / "leaderboard.json").write_text(
        json.dumps({"scores": [], "by_task_type": by_task_type}))


def test_task_type_split_keeps_both_kinds_per_policy(tmp_path):
    _leaderboard(tmp_path, {
        "bc": {
            "build": {"progress_score": 0.069, "success_rate": 0.0, "episodes": 16},
            "complete": {"progress_score": 0.562, "success_rate": 0.467, "episodes": 60},
        },
    })
    rows = collect_task_types(tmp_path)
    assert rows["kinds"] == ["build", "complete"]
    assert rows["rows"][0]["build"]["progress"] == 0.069
    assert rows["rows"][0]["complete"]["success"] == 0.467


def test_a_policy_missing_a_kind_reports_none_not_zero(tmp_path):
    """A kind a policy never ran is not a kind it scored zero on."""
    _leaderboard(tmp_path, {
        "bc": {"build": {"progress_score": 0.069, "success_rate": 0.0, "episodes": 16}},
        "ppo": {"complete": {"progress_score": 0.5, "success_rate": 0.4, "episodes": 60}},
    })
    rows = {r["policy"]: r for r in collect_task_types(tmp_path)["rows"]}
    assert rows["bc"]["complete"]["progress"] is None
    assert rows["ppo"]["build"]["progress"] is None


def test_task_types_without_a_leaderboard_degrade_to_empty(tmp_path):
    assert collect_task_types(tmp_path) == {"kinds": [], "rows": []}


# ------------------------------------------------------------------- funnel


def test_funnel_drop_is_measured_against_the_rung_before(tmp_path):
    (tmp_path / "leaderboard.json").write_text(json.dumps({"scores": [{
        "policy": "bc",
        "milestone_rates": {"opened_a_sketch": 0.9, "drew_geometry": 0.9,
                            "made_a_solid": 0.4},
    }]}))
    steps = collect_funnel(tmp_path)["rows"][0]["steps"]
    assert [round(s["drop"], 3) for s in steps] == [0.1, 0.0, 0.5]


def test_funnel_names_the_rung_that_lost_the_most(tmp_path):
    (tmp_path / "leaderboard.json").write_text(json.dumps({"scores": [{
        "policy": "scripted-spec",
        "milestone_rates": {"opened_a_sketch": 1.0, "drew_geometry": 1.0,
                            "made_a_solid": 1.0, "has_any_hole": 0.5},
    }]}))
    row = collect_funnel(tmp_path)["rows"][0]
    assert row["wall"] == "has_any_hole"
    assert row["wall_drop"] == 0.5


def test_a_policy_that_loses_nothing_has_no_wall(tmp_path):
    """The oracle reaches every rung; calling one of them its wall would be a
    label with no failure behind it."""
    (tmp_path / "leaderboard.json").write_text(json.dumps({"scores": [{
        "policy": "oracle-replay",
        "milestone_rates": {name: 1.0 for name in
                            ["opened_a_sketch", "drew_geometry", "made_a_solid"]},
    }]}))
    row = collect_funnel(tmp_path)["rows"][0]
    assert row["wall"] is None
    assert row["wall_drop"] == 0.0


def test_funnel_without_a_leaderboard_degrades_to_empty(tmp_path):
    assert collect_funnel(tmp_path)["rows"] == []


# ---------------------------------------------------------------------- jam


def test_jam_measures_the_tail_not_the_total(tmp_path):
    """Four refusals scattered and four in a row waste the same number of
    steps and mean different things."""
    _traces(tmp_path, "scattered", [_episode(
        operations=["PAD"] * 8,
        accepted=[True, False, True, False, True, False, True, False])])
    _traces(tmp_path, "jammed", [_episode(
        operations=["PAD"] * 8,
        accepted=[True, True, True, True, False, False, False, False])])

    rows = {r["policy"]: r for r in collect_jam(tmp_path)["rows"]}
    assert rows["scattered"]["tail_share"] == 0.125
    assert rows["jammed"]["tail_share"] == 0.5


def test_jam_counts_an_episode_as_recovered_on_any_later_acceptance(tmp_path):
    _traces(tmp_path, "back", [_episode(
        operations=["PAD"] * 4, accepted=[True, False, True, True])])
    _traces(tmp_path, "stopped", [_episode(
        operations=["PAD"] * 4, accepted=[True, False, False, False])])
    rows = {r["policy"]: r for r in collect_jam(tmp_path)["rows"]}
    assert rows["back"]["recovered"] == 1
    assert rows["stopped"]["recovered"] == 0


def test_an_episode_with_no_refusal_is_not_a_jam(tmp_path):
    _traces(tmp_path, "clean", [_episode(
        operations=["PAD"] * 3, accepted=[True, True, True])])
    row = collect_jam(tmp_path)["rows"][0]
    assert row["episodes"] == 1
    assert row["jammed"] == 0


def test_jam_skips_traces_with_no_per_step_record(tmp_path):
    """An older trace has no acceptance list; counting it as clean would put a
    jamming policy at zero."""
    _traces(tmp_path, "old", [_episode(accepted=None)])
    assert collect_jam(tmp_path)["rows"][0]["episodes"] == 0


# ----------------------------------------------------------------- families


def test_family_scores_are_means_within_the_family(tmp_path):
    _traces(tmp_path, "bc", [
        _episode(task_id="build-a", family="plate", progress_score=0.6),
        _episode(task_id="build-b", family="plate", progress_score=0.4),
        _episode(task_id="build-c", family="flange", progress_score=0.2),
    ])
    scored = collect_families(tmp_path)
    families = [f["name"] for f in scored["families"]]
    assert families == ["flange", "plate"]
    assert scored["cells"]["bc"] == [0.2, 0.5]


def test_a_family_a_policy_never_ran_is_none_not_zero(tmp_path):
    _traces(tmp_path, "aaa", [_episode(task_id="build-a", family="plate")])
    _traces(tmp_path, "zzz", [_episode(task_id="build-b", family="flange")])
    cells = collect_families(tmp_path)["cells"]
    assert cells["aaa"][0] is None and cells["aaa"][1] is not None
    assert cells["zzz"][0] is not None and cells["zzz"][1] is None


def test_aborted_episodes_do_not_drag_a_family_down(tmp_path):
    """An episode the harness could not run is not a score of zero."""
    _traces(tmp_path, "bc", [
        _episode(task_id="build-a", family="plate", progress_score=0.8),
        _episode(task_id="build-b", family="plate", aborted=True, progress_score=0.0),
    ])
    assert collect_families(tmp_path)["cells"]["bc"] == [0.8]


# ------------------------------------------------------------------ dataset


def test_dataset_counts_every_design_not_the_embedded_cap(tmp_path):
    """The bundle carries 24. The dataset is 1,080, and the page says so only
    if this reads the files rather than the embedded list."""
    for i in range(30):
        _design(tmp_path, f"design_{i:06d}")
    assert collect_dataset(tmp_path)["designs"] == 30


def test_dataset_buckets_mass_across_the_observed_range(tmp_path):
    for i, mass in enumerate([10.0, 20.0, 30.0]):
        _design(tmp_path, f"design_{i:06d}", mass_g=mass)
    stats = collect_dataset(tmp_path, buckets=2)
    assert stats["mass_min"] == 10.0 and stats["mass_max"] == 30.0
    assert sum(b["count"] for b in stats["mass_histogram"]) == 3


def test_the_heaviest_design_lands_in_the_last_bucket(tmp_path):
    """A value equal to the maximum divides to exactly `buckets` and would
    index one past the end."""
    for i, mass in enumerate([1.0, 100.0]):
        _design(tmp_path, f"design_{i:06d}", mass_g=mass)
    histogram = collect_dataset(tmp_path, buckets=4)["mass_histogram"]
    assert len(histogram) == 4
    assert histogram[-1]["count"] == 1


def test_dataset_without_designs_degrades_to_empty(tmp_path):
    assert collect_dataset(tmp_path) == {}


# ------------------------------------------------------------------- effort


def test_effort_separates_finished_episodes_from_all(tmp_path):
    """A policy that burns the budget and never finishes is unfinished, not
    inefficient, and one average over both says neither."""
    _traces(tmp_path, "bc", [
        _episode(task_id="a", steps=40, expert_steps=20, finished_successfully=False),
        _episode(task_id="b", steps=24, expert_steps=20, finished_successfully=True),
    ])
    row = collect_effort(tmp_path)["rows"][0]
    assert row["ratio"] == 1.6
    assert row["finished"] == 1
    assert row["ratio_finished"] == 1.2


def test_a_policy_that_never_finished_has_no_finished_ratio(tmp_path):
    _traces(tmp_path, "bc", [_episode(steps=40, expert_steps=20,
                                      finished_successfully=False)])
    assert collect_effort(tmp_path)["rows"][0]["ratio_finished"] is None


def test_effort_skips_episodes_with_no_expert_count(tmp_path):
    """Dividing by zero would produce Infinity and poison the mean."""
    _traces(tmp_path, "bc", [
        _episode(task_id="a", steps=10, expert_steps=0),
        _episode(task_id="b", steps=10, expert_steps=5),
    ])
    row = collect_effort(tmp_path)["rows"][0]
    assert row["episodes"] == 1 and row["ratio"] == 2.0


# ------------------------------------------------------- ablation intervals


def test_ablation_intervals_orient_every_row_as_ablated_minus_intact(tmp_path):
    """Otherwise the sign depends on alphabetical order of the pair names and
    a reader cannot tell a cost from a gain."""
    _traces(tmp_path, "bc", [
        _episode(task_id=f"t{i}", progress_score=0.8) for i in range(6)])
    _traces(tmp_path, "bc+shuffled-req", [
        _episode(task_id=f"t{i}", progress_score=0.2) for i in range(6)])

    rows = collect_ablation_intervals(tmp_path)["rows"]
    assert len(rows) == 1
    assert rows[0]["condition"] == "bc+shuffled-req"
    # The ablation scored lower, so the difference must be negative.
    assert rows[0]["difference"] < 0
    assert rows[0]["low"] <= rows[0]["difference"] <= rows[0]["high"]


def test_ablation_intervals_without_the_baseline_degrade_to_empty(tmp_path):
    _traces(tmp_path, "bc+no-mask", [_episode()])
    assert collect_ablation_intervals(tmp_path)["rows"] == []


# ----------------------------------------------------------------- sources


def test_sources_report_a_missing_artifact_as_missing(tmp_path):
    """Every collector degrades to empty on an absent artifact, so without
    this the page shows a blank section and no reason for it."""
    rows = {r["label"]: r for r in collect_sources(
        tmp_path / "dataset", tmp_path / "bench", tmp_path / "abl", tmp_path / "runs")}
    assert all(not r["present"] for r in rows.values())
    assert "codec audit" in rows


def test_sources_count_what_they_found(tmp_path):
    for i in range(3):
        _design(tmp_path / "dataset", f"design_{i:06d}")
    bench = tmp_path / "bench"
    bench.mkdir()
    (bench / "bc_traces.jsonl").write_text("{}\n")
    (bench / "ppo_traces.jsonl").write_text("{}\n")

    rows = {r["label"]: r for r in collect_sources(
        tmp_path / "dataset", bench, tmp_path / "abl", tmp_path / "runs")}
    assert rows["dataset"]["count"] == 3
    assert rows["benchmark traces"]["count"] == 2
    assert rows["dataset"]["present"] is True
