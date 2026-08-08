#!/usr/bin/env python
"""Regenerate benchmark tables from traces alone.

    python3 scripts/benchmark_report.py --runs runs/benchmark_core

Reads only ``*_traces.jsonl``. Every published table must be reproducible from
the traces without re-running the environment, otherwise a number can outlive
the run that produced it, which is how a stale figure shipped in Phase 4.

The headline output is the **success(k) curve**: success against how many
actions the policy had to supply. It measures compounding error directly, and
unlike a single success rate it cannot be uniformly zero.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


#: Columns of the headline table, in the order they are published.
LEADERBOARD_FIELDS = (
    "progress_score",
    "finished_successfully",
    "validity_rate",
    "satisfaction_rate",
    "efficiency",
)

#: The milestone ladder, in ascending order. Kept in step with
#: `kairos.benchmark.metrics.MILESTONES`; asserted below rather than trusted.
MILESTONES = (
    "opened_a_sketch",
    "drew_geometry",
    "made_a_solid",
    "solid_is_valid",
    "has_any_hole",
    "all_constraints_met",
    "finished_successfully",
)


def _suffix_of(task_id: str) -> int | None:
    """Suffix length from a task id, or None for BUILD tasks."""
    if not task_id.startswith("complete-k"):
        return None
    try:
        return int(task_id.split("-")[1][1:])
    except (IndexError, ValueError):
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/benchmark_core"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--metric", default="progress_score")
    args = parser.parse_args()

    traces = sorted(args.runs.glob("*_traces.jsonl"))
    if not traces:
        print(f"error: no traces under {args.runs}", file=sys.stderr)
        return 1

    # A silently reordered or renamed ladder would publish a table whose columns
    # do not mean what their headers say, which is worse than not publishing it.
    from kairos.benchmark.metrics import MILESTONES as LADDER

    if tuple(name for name, _ in LADDER) != MILESTONES:
        print(
            "error: the milestone ladder in kairos/benchmark/metrics.py no longer "
            f"matches this report's columns:\n  metrics: {[n for n, _ in LADDER]}\n"
            f"  report : {list(MILESTONES)}",
            file=sys.stderr,
        )
        return 1

    curves: dict[str, dict[int, list[bool]]] = defaultdict(lambda: defaultdict(list))
    builds: dict[str, list[bool]] = defaultdict(list)
    families: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    # Column -> per-policy values, for the leaderboard and milestone tables.
    columns: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for path in traces:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("aborted"):
                continue
            policy = row["policy"]
            k = _suffix_of(row["task_id"])
            if k is None:
                builds[policy].append(bool(row["finished_successfully"]))
            else:
                curves[policy][k].append(bool(row["finished_successfully"]))
            families[policy][row.get("family", "unknown")].append(
                float(row.get("progress_score", 0.0))
            )
            for field in (*LEADERBOARD_FIELDS, *MILESTONES):
                value = row.get(field)
                if value is not None:
                    columns[policy][field].append(float(value))

    def mean(policy: str, field: str) -> str:
        values = columns[policy].get(field, [])
        return f"{sum(values) / len(values):.3f}" if values else "-"

    def rate(policy: str, field: str) -> str:
        values = columns[policy].get(field, [])
        return f"{sum(values) / len(values):.2f}" if values else "-"

    ranked = sorted(
        columns,
        key=lambda p: -(sum(columns[p].get("progress_score", [0]))
                        / max(1, len(columns[p].get("progress_score", [0])))),
    )

    lines: list[str] = ["## leaderboard", ""]
    lines += [
        "| policy | " + " | ".join(f.replace("_", " ") for f in LEADERBOARD_FIELDS) + " |",
        "| --- |" + " --- |" * len(LEADERBOARD_FIELDS),
    ]
    for policy in ranked:
        cells = " | ".join(mean(policy, field) for field in LEADERBOARD_FIELDS)
        lines.append(f"| `{policy}` | {cells} |")

    lines += ["", "## milestone ladder (fraction of episodes reaching each rung)", ""]
    lines += [
        "| policy | " + " | ".join(m.replace("_", " ") for m in MILESTONES) + " |",
        "| --- |" + " --- |" * len(MILESTONES),
    ]
    for policy in ranked:
        cells = " | ".join(rate(policy, milestone) for milestone in MILESTONES)
        lines.append(f"| `{policy}` | {cells} |")

    lines += ["", "## success(k): finish the last k actions", ""]
    ks = sorted({k for policy in curves.values() for k in policy})
    header = "| policy | BUILD | " + " | ".join(f"k={k}" for k in ks) + " |"
    lines += [header, "| --- |" + " --- |" * (len(ks) + 1)]
    for policy in sorted(set(curves) | set(builds)):
        cells = []
        for k in ks:
            values = curves[policy].get(k, [])
            cells.append(f"{sum(values) / len(values):.2f}" if values else "-")
        build = builds.get(policy, [])
        build_cell = f"{sum(build) / len(build):.2f}" if build else "-"
        lines.append(f"| `{policy}` | {build_cell} | " + " | ".join(cells) + " |")

    lines += ["", "## progress by family", ""]
    all_families = sorted({f for p in families.values() for f in p})
    lines += [
        "| policy | " + " | ".join(all_families) + " |",
        "| --- |" + " --- |" * len(all_families),
    ]
    for policy in sorted(families):
        cells = []
        for family in all_families:
            values = families[policy].get(family, [])
            cells.append(f"{sum(values) / len(values):.2f}" if values else "-")
        lines.append(f"| `{policy}` | " + " | ".join(cells) + " |")

    # Paired comparisons: every policy faced identical tasks, so the per-task
    # difference is the right statistic and an interval spanning zero is the
    # honest answer rather than a ranking.
    from kairos.benchmark.statistics import compare_all

    by_policy: dict[str, list[dict]] = defaultdict(list)
    for path in traces:
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                by_policy[row["policy"]].append(row)

    lines += ["", "## paired comparisons (95% bootstrap CI on the per-task difference)", ""]
    lines += ["| comparison | difference | 95% CI | W/L/T | separates? |",
              "| --- | --- | --- | --- | --- |"]
    for c in compare_all(by_policy, metric=args.metric):
        if not c.n_pairs:
            continue
        lines.append(
            f"| `{c.policy_a}` vs `{c.policy_b}` | {c.mean_difference:+.3f} | "
            f"[{c.ci_low:+.3f}, {c.ci_high:+.3f}] | "
            f"{c.wins}/{c.losses}/{c.ties} | {'yes' if c.separates else '**no**'} |"
        )

    report = "\n".join(lines)
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n")
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
