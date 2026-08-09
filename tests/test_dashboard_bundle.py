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
    collect_ablations,
    collect_designs,
    collect_rollouts,
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
