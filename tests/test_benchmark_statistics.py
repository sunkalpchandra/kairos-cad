"""Paired comparison tests — pure python, no torch, no FreeCAD."""

import pytest

from kairos.benchmark.statistics import (
    compare_all,
    paired_bootstrap,
    scores_by_task,
)


def test_identical_policies_do_not_separate():
    scores = {f"t{i}": 0.5 for i in range(20)}
    result = paired_bootstrap(scores, dict(scores), "a", "b")
    assert result.mean_difference == pytest.approx(0.0)
    assert not result.separates
    assert result.ties == 20


def test_a_consistent_advantage_separates():
    a = {f"t{i}": 0.6 for i in range(20)}
    b = {f"t{i}": 0.4 for i in range(20)}
    result = paired_bootstrap(a, b, "a", "b")
    assert result.mean_difference == pytest.approx(0.2)
    assert result.separates and result.ci_low > 0
    assert (result.wins, result.losses) == (20, 0)


def test_a_noisy_tie_does_not_separate():
    """Alternating wins and losses of equal size must read as no difference."""
    a = {f"t{i}": 0.5 + (0.3 if i % 2 else -0.3) for i in range(20)}
    b = {f"t{i}": 0.5 for i in range(20)}
    result = paired_bootstrap(a, b, "a", "b", seed=0)
    assert not result.separates
    assert result.ci_low < 0 < result.ci_high


def test_pairing_is_by_task_not_by_position():
    """Scores must be matched by task id, or the pairing is meaningless."""
    a = {"t1": 1.0, "t2": 0.0}
    b = {"t2": 0.0, "t1": 1.0}  # same scores, different insertion order
    assert paired_bootstrap(a, b, "a", "b").mean_difference == pytest.approx(0.0)


def test_only_tasks_both_policies_attempted_are_compared():
    """A task one policy could not attempt must not become an implicit zero."""
    a = {"t1": 0.8, "t2": 0.6, "t3": 0.9}
    b = {"t1": 0.4, "t2": 0.2}
    result = paired_bootstrap(a, b, "a", "b")
    assert result.n_pairs == 2
    assert result.mean_difference == pytest.approx(0.4)


def test_no_shared_tasks_yields_an_empty_comparison():
    result = paired_bootstrap({"t1": 1.0}, {"t2": 1.0}, "a", "b")
    assert result.n_pairs == 0 and not result.separates


def test_summary_names_the_winner_or_says_there_is_none():
    strong = paired_bootstrap(
        {f"t{i}": 0.9 for i in range(20)}, {f"t{i}": 0.1 for i in range(20)}, "a", "b"
    )
    assert "a > b" in strong.summary()
    tied = paired_bootstrap({f"t{i}": 0.5 for i in range(20)},
                            {f"t{i}": 0.5 for i in range(20)}, "a", "b")
    assert "no separation" in tied.summary()


def test_scores_by_task_skips_aborted_rows():
    rows = [
        {"task_id": "t1", "progress_score": 0.5, "aborted": False},
        {"task_id": "t2", "progress_score": 0.9, "aborted": True},
    ]
    assert scores_by_task(rows) == {"t1": 0.5}


def test_scores_by_task_averages_repeats():
    rows = [
        {"task_id": "t1", "progress_score": 0.4, "aborted": False},
        {"task_id": "t1", "progress_score": 0.6, "aborted": False},
    ]
    assert scores_by_task(rows)["t1"] == pytest.approx(0.5)


def test_compare_all_covers_every_pair_once():
    traces = {
        name: [{"task_id": f"t{i}", "progress_score": value, "aborted": False}
               for i in range(10)]
        for name, value in (("a", 0.9), ("b", 0.5), ("c", 0.1))
    }
    comparisons = compare_all(traces)
    assert len(comparisons) == 3  # 3 choose 2
    pairs = {(c.policy_a, c.policy_b) for c in comparisons}
    assert pairs == {("a", "b"), ("a", "c"), ("b", "c")}
    # Sorted by effect size, largest first.
    assert abs(comparisons[0].mean_difference) >= abs(comparisons[-1].mean_difference)


def test_comparison_serializes_with_its_verdict():
    import json

    payload = paired_bootstrap(
        {f"t{i}": 0.9 for i in range(20)}, {f"t{i}": 0.1 for i in range(20)}, "a", "b"
    ).to_dict()
    assert payload["separates"] is True
    json.dumps(payload)
