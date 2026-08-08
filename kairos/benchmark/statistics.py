"""Paired comparison between two policies on the same tasks.

Every policy faces the **same enumerated tasks in the same order**, which makes
the comparison paired: for each task there is a matched pair of scores, and the
quantity of interest is the distribution of their per-task difference. Treating
the two as independent samples throws that pairing away and widens the interval
for no reason — with 32 tasks and a shared task-difficulty spread, most of the
variance is *between tasks*, and pairing removes exactly that.

Phase 5 already showed what happens without this discipline: a 6-episode
evaluation reported 0.500 for a policy that scored 0.286 over 14 episodes, and
the difference was read as a result. A paired interval that straddles zero says
"this benchmark cannot separate these two", which is usually the honest answer
at these sample sizes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class PairedComparison:
    """The per-task difference between two policies, with its interval."""

    policy_a: str
    policy_b: str
    metric: str
    n_pairs: int
    mean_difference: float
    ci_low: float
    ci_high: float
    wins: int
    losses: int
    ties: int

    @property
    def separates(self) -> bool:
        """True when the interval excludes zero — a difference the data supports."""
        return self.ci_low > 0.0 or self.ci_high < 0.0

    def summary(self) -> str:
        verdict = (
            f"{self.policy_a} > {self.policy_b}"
            if self.separates and self.mean_difference > 0
            else f"{self.policy_b} > {self.policy_a}"
            if self.separates
            else "no separation"
        )
        return (
            f"{self.policy_a} vs {self.policy_b} on {self.metric}: "
            f"{self.mean_difference:+.3f} [{self.ci_low:+.3f}, {self.ci_high:+.3f}] "
            f"over {self.n_pairs} paired tasks "
            f"({self.wins}W/{self.losses}L/{self.ties}T) — {verdict}"
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["separates"] = self.separates
        return data


def paired_bootstrap(
    a: dict[str, float],
    b: dict[str, float],
    policy_a: str = "a",
    policy_b: str = "b",
    metric: str = "progress_score",
    samples: int = 5000,
    seed: int = 0,
    alpha: float = 0.05,
) -> PairedComparison:
    """Bootstrap the mean per-task difference ``a - b``.

    Args are ``{task_id: score}``; only tasks present in both are compared, so
    a task one policy could not attempt never becomes an implicit zero for it.
    """
    shared = sorted(set(a) & set(b))
    if not shared:
        return PairedComparison(policy_a, policy_b, metric, 0, 0.0, 0.0, 0.0, 0, 0, 0)

    differences = np.array([a[t] - b[t] for t in shared], dtype=float)
    rng = np.random.default_rng(seed)
    # Resample the *differences*, not the two score sets independently: the
    # pairing is the information that makes this test sharp.
    resampled = rng.choice(differences, size=(samples, len(differences)), replace=True)
    means = resampled.mean(axis=1)

    return PairedComparison(
        policy_a=policy_a,
        policy_b=policy_b,
        metric=metric,
        n_pairs=len(shared),
        mean_difference=float(differences.mean()),
        ci_low=float(np.quantile(means, alpha / 2)),
        ci_high=float(np.quantile(means, 1 - alpha / 2)),
        wins=int((differences > 0).sum()),
        losses=int((differences < 0).sum()),
        ties=int((differences == 0).sum()),
    )


def scores_by_task(rows: list[dict[str, Any]], metric: str = "progress_score") -> dict[str, float]:
    """Collapse trace rows to ``{task_id: mean score}``, skipping aborted ones."""
    grouped: dict[str, list[float]] = {}
    for row in rows:
        if row.get("aborted"):
            continue
        grouped.setdefault(row["task_id"], []).append(float(row.get(metric, 0.0)))
    return {task: sum(values) / len(values) for task, values in grouped.items()}


def compare_all(
    traces_by_policy: dict[str, list[dict[str, Any]]],
    metric: str = "progress_score",
    seed: int = 0,
) -> list[PairedComparison]:
    """Every pairwise comparison, strongest separation first."""
    scores = {name: scores_by_task(rows, metric) for name, rows in traces_by_policy.items()}
    names = sorted(scores)
    comparisons = [
        paired_bootstrap(scores[a], scores[b], a, b, metric, seed=seed)
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    ]
    return sorted(comparisons, key=lambda c: -abs(c.mean_difference))
