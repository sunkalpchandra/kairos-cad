#!/usr/bin/env python
"""Audit a generated dataset for completeness and internal consistency.

Checks every ``design_*`` directory for the full artifact set, parseable
JSON, and recorded validity; optionally deletes incomplete directories
(crash leftovers) and stray FreeCAD backup files.

Pure python — runs under any interpreter:

    python3 scripts/audit_dataset.py --root dataset [--fix]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REQUIRED_FILES = (
    "model.FCStd",
    "model.step",
    "model.stl",
    "iso.png",
    "front.png",
    "top.png",
    "right.png",
    "state.json",
    "requirements.json",
    "trajectory.json",
)


def audit_design(design_dir: Path) -> list[str]:
    """Return a list of problems (empty = complete and consistent)."""
    problems = [f"missing {name}" for name in REQUIRED_FILES if not (design_dir / name).exists()]
    if problems:
        return problems
    try:
        state = json.loads((design_dir / "state.json").read_text())
        if state.get("valid") is not True:
            problems.append("state.json records invalid geometry")
        trajectory = json.loads((design_dir / "trajectory.json").read_text())
        n = len(trajectory.get("actions", []))
        if n == 0 or n != len(trajectory.get("rewards", [])):
            problems.append("trajectory actions/rewards inconsistent")
    except (json.JSONDecodeError, OSError) as err:
        problems.append(f"unreadable json: {err}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("dataset"))
    parser.add_argument("--fix", action="store_true", help="delete incomplete design dirs and backups")
    args = parser.parse_args()

    designs_dir = args.root / "designs"
    trajectories_dir = args.root / "trajectories"
    complete, broken = [], {}
    by_family: dict[str, int] = {}

    for design_dir in sorted(designs_dir.glob("design_*")):
        problems = audit_design(design_dir)
        if problems:
            broken[design_dir.name] = problems
        else:
            complete.append(design_dir.name)
            try:
                kind = json.loads((design_dir / "state.json").read_text())["parameters"]["kind"]
                by_family[kind] = by_family.get(kind, 0) + 1
            except Exception:
                pass

    backups = list(designs_dir.glob("design_*/*.FCBak")) + list(
        designs_dir.glob("design_*/*.FCStd1")
    )
    orphan_trajectories = [
        p
        for p in (trajectories_dir.glob("trajectory_*.json") if trajectories_dir.is_dir() else [])
        if not (designs_dir / p.stem.replace("trajectory_", "design_")).is_dir()
    ]

    print(f"complete designs:     {len(complete)}")
    for family, count in sorted(by_family.items()):
        print(f"  {family:>18}: {count}")
    print(f"incomplete designs:   {len(broken)}")
    for name, problems in sorted(broken.items())[:15]:
        print(f"  {name}: {problems[0]}" + (f" (+{len(problems) - 1} more)" if len(problems) > 1 else ""))
    print(f"backup files:         {len(backups)}")
    print(f"orphan trajectories:  {len(orphan_trajectories)}")

    if args.fix:
        for name in broken:
            shutil.rmtree(designs_dir / name)
        for path in backups + orphan_trajectories:
            path.unlink()
        print(f"fixed: removed {len(broken)} dirs, {len(backups)} backups, "
              f"{len(orphan_trajectories)} orphan trajectories")
    return 0 if not broken or args.fix else 2


if __name__ == "__main__":
    raise SystemExit(main())
