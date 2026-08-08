#!/usr/bin/env python3
"""Measure what fraction of expert steps the action codec can express.

    python3 scripts/audit_codec.py --root dataset

This is the number that caps every learned policy. An expert step the codec
cannot encode is a step behavioural cloning silently drops from its training
set and a step no policy can ever emit, so an oracle replaying the expert
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
from kairos.rl.action_space import (  # noqa: E402
    TARGET_KIND,
    UnrepresentableAction,
    decode,
    encode,
)

#: A round trip should only lose the 3-decimal rounding decode applies, so any
#: error above this is a range problem, not quantization. The two failure modes
#: separate cleanly by two orders of magnitude in practice (0.0005 vs 0.09+).
DRIFT_TOLERANCE_MM = 0.01


def _synthetic_pool(action: Action) -> dict[str, list[str]]:
    """A target pool containing the action's own target, plus decoys.

    The audit has no live engine, so it builds a pool the recorded name is
    genuinely in. Decoys come first, so an unresolved index of 0 lands on the
    wrong name and the loss is detected rather than passing by luck.
    """
    kind = TARGET_KIND.get(action.operation)
    if kind is None or not action.target:
        return {}
    names = [n.strip() for n in str(action.target).split(",") if n.strip()]
    decoys = [f"__decoy{i}__" for i in range(3)]
    return {kind: decoys + names}


def _round_trip_error(action: Action) -> tuple[float, str]:
    """Largest absolute drift between an action and its decoded round trip.

    Encoding alone is not enough to trust: before slot ranges were unified,
    encode and decode disagreed about PAD length, so an in-range value came
    back 61.9 mm away having raised nothing.

    Targets and list-valued parameters are compared too. Walking only the
    numeric parameters left 286 target-bearing expert steps reported as
    losslessly representable while a fillet moved from Edge30 to Edge1, and
    that 0.00% is what the "codec is not the bottleneck" claim rests on.
    """
    pool = _synthetic_pool(action)
    decoded = decode(*encode(action, targets=pool), targets=pool)
    worst, where = 0.0, ""

    for key, value in (action.parameters or {}).items():
        got = (decoded.parameters or {}).get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            # Vertex lists (ADD_POLYGON.points) are geometry too; a mismatch
            # here is a different shape, so report it as a full failure rather
            # than skipping it for not being a float.
            if isinstance(value, (list, tuple)) and got != value:
                return float("inf"), key
            continue
        if isinstance(got, (int, float)) and not isinstance(got, bool):
            error = abs(float(got) - float(value))
            if error > worst:
                worst, where = error, key

    if action.target and pool:
        # decode resolves one name; the expert may have named several.
        wanted = str(action.target).split(",")[0].strip()
        if (decoded.target or "") != wanted:
            return float("inf"), "target"
    return worst, where


def audit(root: Path) -> dict:
    """Encode every recorded expert action; count what fails and why."""
    total = 0
    unrepresentable = 0
    drifted = 0
    by_operation: collections.Counter[str] = collections.Counter()
    failures: collections.Counter[str] = collections.Counter()
    drifts: collections.Counter[str] = collections.Counter()
    affected_designs: set[str] = set()
    worst_drift = 0.0

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
                action = Action.from_dict(data)
                error, where = _round_trip_error(action)
            except UnrepresentableAction:
                unrepresentable += 1
                failures[operation] += 1
                affected_designs.add(path.parent.name)
                continue
            except Exception as err:  # a codec bug, not an expressiveness limit
                unrepresentable += 1
                failures[f"{operation}:{type(err).__name__}"] += 1
                affected_designs.add(path.parent.name)
                continue

            worst_drift = max(worst_drift, error)
            if error > DRIFT_TOLERANCE_MM:
                drifted += 1
                drifts[f"{operation}.{where}"] += 1
                affected_designs.add(path.parent.name)

    broken = unrepresentable + drifted
    return {
        "root": str(root),
        "steps": total,
        "unrepresentable": unrepresentable,
        "drifted": drifted,
        "unusable": broken,
        "unrepresentable_rate": broken / total if total else 0.0,
        "worst_round_trip_mm": worst_drift,
        "affected_designs": len(affected_designs),
        "operations_used": len(by_operation),
        "by_operation": dict(by_operation.most_common()),
        "failures": dict(failures.most_common()),
        "drifts": dict(drifts.most_common()),
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
    print(f"  unrepresentable : {report['unrepresentable']} (encode refused)")
    print(f"  silently drifted: {report['drifted']} (round trip moved the value)")
    print(f"  unusable total  : {report['unusable']} "
          f"({report['unrepresentable_rate'] * 100:.2f}%)")
    print(f"  worst round trip: {report['worst_round_trip_mm']:.6f} mm")
    print(f"  designs affected: {report['affected_designs']}")
    for operation, count in report["failures"].items():
        print(f"    refused  {operation}: {count}")
    for parameter, count in report["drifts"].items():
        print(f"    drifted  {parameter}: {count}")

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
