"""Tests for the dashboard data bundle.

Every assertion here guards a *silent* failure mode. Nothing in this pipeline
raises when it reads a field that does not exist — it renders a dash, or drops
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
