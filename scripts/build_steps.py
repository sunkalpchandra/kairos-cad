#!/usr/bin/env python3
"""Export a mesh after every expert action, so the timeline can be scrubbed.

    PYTHONPATH=. /path/to/freecad/python scripts/build_steps.py --limit 8

Runs under FreeCAD's interpreter. For each design it replays the recorded
trajectory one action at a time and writes an STL whenever the solid changes,
into ``designs/<id>/steps/``.

A parametric CAD timeline lets you click a feature and see the part as it was
at that point. The station's timeline is the recorded trajectory, which is the
same object, but it could only highlight a node because no geometry existed for
intermediate states. This produces it.

Only designs that actually get scrubbed are worth the bytes: each step mesh is
a few KB in the bundle, and a full dataset would add megabytes to a page that
has to stay openable. ``--limit`` takes one design per family by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _families(root: Path, limit: int) -> list[Path]:
    """One design per family, lowest id first, up to `limit` families."""
    by_family: dict[str, Path] = {}
    for path in sorted(root.glob("designs/design_*/trajectory.json")):
        try:
            family = json.loads(path.read_text()).get("family", "unknown")
        except (OSError, json.JSONDecodeError):
            continue
        by_family.setdefault(family, path.parent)
    return [by_family[name] for name in sorted(by_family)][:limit]


def export_steps(design_dir: Path, tolerance: float) -> int:
    """Replay the trajectory, writing an STL after each solid-changing action."""
    from kairos.actions.executor import ActionExecutor
    from kairos.actions.schema import Action
    from kairos.cad.engine import CADEngine

    trajectory = json.loads((design_dir / "trajectory.json").read_text())
    actions = trajectory.get("actions") or []
    steps_dir = design_dir / "steps"
    steps_dir.mkdir(exist_ok=True)
    for stale in steps_dir.glob("*.stl"):
        stale.unlink()

    engine = CADEngine(f"steps_{design_dir.name}")
    written = 0
    try:
        executor = ActionExecutor(engine)
        for index, data in enumerate(actions):
            try:
                result = executor.execute(Action.from_dict(data))
            except Exception:
                # A step that will not replay ends the sequence rather than
                # failing the design: the earlier steps are still valid.
                break
            if not result.ok:
                break
            # Only a solid can be meshed, and only a change is worth a file.
            if not engine.has_solid():
                continue
            try:
                engine.export_stl(steps_dir / f"{index:03d}.stl")
                written += 1
            except Exception:
                continue
    finally:
        engine.close()
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("dataset"))
    parser.add_argument("--limit", type=int, default=8,
                        help="how many families to cover, one design each")
    parser.add_argument("--tolerance", type=float, default=0.35)
    args = parser.parse_args()

    from kairos.cad.backend import freecad_available

    if not freecad_available():
        print("error: run this under FreeCAD's interpreter; see docs/freecad-setup.md",
              file=sys.stderr)
        return 1

    designs = _families(args.root, args.limit)
    if not designs:
        print(f"error: no trajectories under {args.root}", file=sys.stderr)
        return 1

    total = 0
    for design_dir in designs:
        written = export_steps(design_dir, args.tolerance)
        total += written
        print(f"  {design_dir.name}: {written} step meshes")

    print(f"wrote {total} step meshes across {len(designs)} designs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
