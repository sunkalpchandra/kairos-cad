#!/usr/bin/env python3
"""Measure what fraction of expert steps the action codec can express.

    python3 scripts/audit_codec.py --root dataset

This is the number that caps every learned policy. An expert step the codec
cannot encode is a step behavioural cloning silently drops from its training
set and a step no policy can ever emit — so an oracle replaying the expert
through the codec cannot rebuild the part, and the benchmark's ceiling sits
below 1.0 for reasons that have nothing to do with learning.

Before the profile expansion this read 7.81% unrepresentable, all of it
`ADD_POLYGON` carrying an irregular vertex list.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kairos.actions.schema import Action  # noqa: E402
from kairos.rl.action_space import UnrepresentableAction, encode  # noqa: E402


def audit(root: Path) -> dict:
    """Encode every recorded expert action; count what fails and why."""
    total = 0
    unrepresentable = 0
    by_operation: collections.Counter[str] = collections.Counter()
    failures: collections.Counter[str] = collections.Counter()
    affected_designs: set[str] = set()

    for path in sorted(root.glob("designs/design_*/trajectory.json")):
        try:
            actions = json.loads(path.read_text()).get("actions", [])
        except (OSError, json.JSONDecodeError):
            continue
        for data in actions:
            total += 1
            operation = data.get("operation", "?")
            by_operation[operation] += 1
            try:
                encode(Action.from_dict(data))
            except UnrepresentableAction:
                unrepresentable += 1
                failures[operation] += 1
                affected_designs.add(path.parent.name)
            except Exception as err:  # a codec bug, not an expressiveness limit
                unrepresentable += 1
                failures[f"{operation}:{type(err).__name__}"] += 1
                affected_designs.add(path.parent.name)

    return {
        "root": str(root),
        "steps": total,
        "unrepresentable": unrepresentable,
        "unrepresentable_rate": unrepresentable / total if total else 0.0,
        "affected_designs": len(affected_designs),
        "operations_used": len(by_operation),
        "by_operation": dict(by_operation.most_common()),
        "failures": dict(failures.most_common()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("dataset"))
    parser.add_argument("--json", type=Path, default=None, help="write the report here")
    parser.add_argument(
        "--max-rate",
        type=float,
        default=None,
        help="exit non-zero if the unrepresentable rate exceeds this",
    )
    args = parser.parse_args()

    if not args.root.exists():
        print(f"no such dataset root: {args.root}", file=sys.stderr)
        return 2

    report = audit(args.root)
    if not report["steps"]:
        print(f"no trajectories under {args.root}", file=sys.stderr)
        return 2

    print(f"{report['root']}: {report['steps']} expert steps, "
          f"{report['operations_used']} distinct operations")
    print(f"  unrepresentable : {report['unrepresentable']} "
          f"({report['unrepresentable_rate'] * 100:.2f}%)")
    print(f"  designs affected: {report['affected_designs']}")
    for operation, count in report["failures"].items():
        print(f"    {operation}: {count}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2))
        print(f"  wrote {args.json}")

    if args.max_rate is not None and report["unrepresentable_rate"] > args.max_rate:
        print(
            f"FAIL: {report['unrepresentable_rate'] * 100:.2f}% exceeds the "
            f"{args.max_rate * 100:.2f}% ceiling",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
