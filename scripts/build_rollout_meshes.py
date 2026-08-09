#!/usr/bin/env python3
"""Rebuild the solid each policy produced, from its recorded trace.

    PYTHONPATH=. /path/to/freecad/python scripts/build_rollout_meshes.py \\
        --runs runs/benchmark_core

Runs under FreeCAD's interpreter. For every BUILD episode it replays the
actions the environment accepted and exports the resulting solid to
``<runs>/rollout_meshes/<task>/<policy>.stl``.

The leaderboard scores a policy. The Rollouts strip shows what it did. Neither
shows what it *made*, and for a project about building CAD that is the thing.
Replaying is possible at all because the trace now records the resolved action
the environment executed, not just its name.

A rejected action changed nothing, so it is skipped: replaying it would fail
here exactly as it failed there, and stop the rebuild at the first rejection
instead of at the end of what the policy actually built.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _episodes(runs: Path, per_family: int) -> list[dict]:
    """The same tasks the dashboard shows: one BUILD task per family."""
    rows: dict[str, list[dict]] = {}
    for path in sorted(runs.glob("*_traces.jsonl")):
        policy = path.name.replace("_traces.jsonl", "")
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not str(row.get("task_id", "")).startswith("build-"):
                continue
            row["policy"] = policy
            rows.setdefault(row["task_id"], []).append(row)

    by_family: dict[str, list[str]] = {}
    for task_id, group in sorted(rows.items()):
        family = group[0].get("family") or "unknown"
        by_family.setdefault(family, [])
        if len(by_family[family]) < per_family:
            by_family[family].append(task_id)

    keep = {task for tasks in by_family.values() for task in tasks}
    return [row for task_id in sorted(keep) for row in rows[task_id]]


def export_episode(
    row: dict, out_dir: Path, tolerance: float = 0.01
) -> tuple[bool, str, float]:
    """Replay one episode's accepted actions and export the solid it left.

    Returns ``(wrote, complaint, drift)``. The rebuild happens outside the environment
    that scored the episode, so a discrepancy between the two would show up as
    a wrong part rather than an error. The trace recorded the mass the runner
    measured; this checks the rebuilt solid against it, which is the one
    number that catches a replay that silently diverged.
    """
    from kairos.actions.executor import ActionExecutor
    from kairos.actions.schema import Action
    from kairos.cad.engine import CADEngine

    actions = row.get("actions") or []
    accepted = row.get("accepted") or []
    if not actions:
        return False, "", 0.0

    engine = CADEngine(f"rollout_{row['policy']}")
    try:
        executor = ActionExecutor(engine)
        for index, data in enumerate(actions):
            # An action the environment refused changed nothing there; replaying
            # it would only stop the rebuild early here.
            if index < len(accepted) and not accepted[index]:
                continue
            if not data:
                continue
            try:
                executor.execute(Action.from_dict(data))
            except Exception:
                break
        if not engine.has_solid():
            return False, "", 0.0
        complaint = ""
        drift = 0.0
        recorded = float(row.get("mass_g") or 0.0)
        if recorded > 0:
            rebuilt = float(engine.measure_mass())
            drift = abs(rebuilt - recorded) / recorded
            if drift > tolerance:
                complaint = (f"{row['task_id']} {row['policy']}: rebuilt mass "
                             f"{rebuilt:.3f} g vs {recorded:.3f} g recorded "
                             f"({drift * 100:.1f}% off)")
        out_dir.mkdir(parents=True, exist_ok=True)
        engine.export_stl(out_dir / f"{row['policy']}.stl")
        return True, complaint, drift
    except Exception:
        return False, "", 0.0
    finally:
        engine.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs/benchmark_core"))
    parser.add_argument("--per-family", type=int, default=1)
    parser.add_argument("--tolerance", type=float, default=0.01,
                        help="allowed relative drift from the recorded mass")
    args = parser.parse_args()

    from kairos.cad.backend import freecad_available

    if not freecad_available():
        print("error: run this under FreeCAD's interpreter; see docs/freecad-setup.md",
              file=sys.stderr)
        return 1

    episodes = _episodes(args.runs, args.per_family)
    if not episodes:
        print(f"error: no BUILD traces under {args.runs}", file=sys.stderr)
        return 1
    if not any(row.get("actions") for row in episodes):
        print("error: these traces predate action recording; re-run the benchmark",
              file=sys.stderr)
        return 1

    root = args.runs / "rollout_meshes"
    built = 0
    complaints = []
    worst = 0.0
    for row in episodes:
        wrote, complaint, drift = export_episode(
            row, root / row["task_id"], args.tolerance
        )
        built += int(wrote)
        worst = max(worst, drift)
        if complaint:
            complaints.append(complaint)

    # Say what did not build as well as what did: a policy that never made a
    # solid is a result, not a gap in the export.
    print(f"{built} of {len(episodes)} episodes left a solid; wrote {root}")
    if complaints:
        print(f"\n{len(complaints)} rebuilt solids disagree with the recorded mass:",
              file=sys.stderr)
        for line in complaints:
            print(f"  {line}", file=sys.stderr)
        print("\nThe replay diverged from the episode it is meant to reproduce.",
              file=sys.stderr)
        return 1
    # The worst drift, not just that it passed: a threshold with no number
    # behind it hides a check that is drifting toward its own limit.
    print(f"every rebuilt solid matches its recorded mass; worst drift {worst:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
