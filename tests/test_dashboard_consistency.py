"""The station shows one run through eight collectors. They have to agree.

Every view here is a different reduction of the same traces. Nothing forced
them to stay consistent, so a collector that quietly filtered differently --
dropping aborted episodes in one place and counting them in another -- would
put two numbers on the page that describe the same thing and disagree, and
neither would look wrong on its own.

These run against the committed run artifacts and skip where those are absent,
which includes CI: they are run outputs, not source.
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest

from kairos.dashboard.bundle import (
    _traces_by_policy,
    collect_benchmark,
    collect_families,
    collect_funnel,
    collect_task_types,
)

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs/benchmark_core"

pytestmark = pytest.mark.skipif(
    not list(RUNS.glob("*_traces.jsonl")),
    reason="benchmark traces are run artifacts, not source",
)

#: The traces store progress to about six decimals, so two reductions of them
#: agree to roughly that and not to machine epsilon.
TOLERANCE = 1e-4


def _leaderboard() -> dict[str, dict]:
    return {row["policy"]: row
            for row in collect_benchmark(RUNS)["leaderboard"]["policies"]}


def test_family_scores_reweight_to_the_leaderboard():
    """Weighted by episodes, the per-family means must recover the headline.

    They are not equal unweighted -- families have different episode counts,
    and bc reads 0.478 as a mean of family means against 0.458 over episodes.
    That difference is arithmetic; a difference after weighting would be a bug.
    """
    scored = collect_families(RUNS)
    names = [family["name"] for family in scored["families"]]
    leaderboard = _leaderboard()

    for policy, cells in scored["cells"].items():
        rows = [row for row in _traces_by_policy(RUNS)[policy] if not row.get("aborted")]
        counts = collections.Counter(row.get("family") or "unknown" for row in rows)
        total = sum(counts[name]
                    for name, cell in zip(names, cells, strict=True)
                    if cell is not None)
        weighted = sum(
            cell * counts[name]
            for name, cell in zip(names, cells, strict=True) if cell is not None
        ) / total
        assert weighted == pytest.approx(
            leaderboard[policy]["progress_mean"], abs=TOLERANCE
        ), f"{policy}: family scores do not reweight to the leaderboard"


def test_the_funnel_only_ever_narrows():
    """A funnel that widens is not a funnel.

    The leaderboard's `milestone_rates` count each flag independently and the
    flags are not nested: 8 of bc's 76 episodes drew geometry without the
    harness recording a sketch, so the raw rates read 0.89 then 1.00. Drawn
    from those, the funnel went up at the second rung. Prefix-reached is what
    `progress_score` uses and is the only reading that can narrow.
    """
    for row in collect_funnel(RUNS)["rows"]:
        rates = [step["rate"] for step in row["steps"]]
        for earlier, later in zip(rates, rates[1:], strict=False):
            assert later <= earlier + 1e-9, (
                f"{row['policy']}: reach rises from {earlier} to {later}"
            )


def test_the_funnel_and_the_leaderboard_agree_on_who_finished():
    """The last rung is the success rate by another name."""
    leaderboard = _leaderboard()
    for row in collect_funnel(RUNS)["rows"]:
        finished = next((step["rate"] for step in row["steps"]
                         if step["milestone"] == "finished_successfully"), None)
        if finished is None or row["policy"] not in leaderboard:
            continue
        assert finished == pytest.approx(
            leaderboard[row["policy"]]["success_rate"], abs=TOLERANCE
        ), f"{row['policy']}: funnel tail and success rate disagree"


def test_the_task_type_split_covers_every_episode():
    """BUILD plus COMPLETE has to be the whole suite. A task kind that fell
    into neither would vanish from the split and from nothing else."""
    split = collect_task_types(RUNS)
    if not split["rows"]:
        pytest.skip("no by_task_type in this leaderboard")

    raw = json.loads((RUNS / "leaderboard.json").read_text())
    by_policy = raw.get("by_task_type") or {}
    for policy, kinds in by_policy.items():
        counted = sum((scores or {}).get("episodes", 0) for scores in kinds.values())
        total = len([row for row in _traces_by_policy(RUNS)[policy]])
        assert counted == total, (
            f"{policy}: task-type split covers {counted} of {total} episodes"
        )


def test_every_policy_in_the_leaderboard_has_traces():
    """A scored policy with no traces would render a leaderboard row and an
    empty row in every view built from traces."""
    traces = _traces_by_policy(RUNS)
    for policy in _leaderboard():
        assert policy in traces, f"{policy} is scored but has no trace file"
