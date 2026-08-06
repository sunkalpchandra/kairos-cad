#!/usr/bin/env python
"""KAIROS first end-to-end demo (project spec §50).

Natural-language requirement → engineering spec → CAD build via structured
actions → constraint checking → shaped rewards → exports + renders + metrics.

The design actions come from the procedural expert (the same recipes that
generate BC data); the learned policy replaces it in Phase 4/5. Everything
else — parsing, execution, inspection, reward, export — is the real
pipeline.

Run under FreeCAD's interpreter:

    PYTHONPATH=. /Applications/FreeCAD.app/Contents/Resources/bin/python \
        scripts/demo.py --out outputs/demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_REQUIREMENT = (
    "Create an L-bracket with 4 M5 mounting holes, 3 mm minimum wall "
    "thickness, 90 degree angle, symmetric hole placement, and minimum "
    "possible mass."
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirement", default=DEFAULT_REQUIREMENT)
    parser.add_argument("--out", type=Path, default=Path("outputs/demo"))
    args = parser.parse_args()

    from kairos.cad.backend import freecad_available, freecad_version

    if not freecad_available():
        print("error: FreeCAD unavailable; see docs/freecad-setup.md")
        return 1

    from kairos.actions.executor import ActionExecutor
    from kairos.cad.engine import CADEngine
    from kairos.data.families import get_family
    from kairos.data.trajectories import TrajectoryRecorder
    from kairos.language import parse_requirement

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    # 1-2. Parse requirement → engineering specification.
    print(f"KAIROS demo (FreeCAD {freecad_version()})")
    print(f"\nREQUIREMENT\n  {args.requirement}\n")
    spec = parse_requirement(args.requirement)
    print("ENGINEERING SPECIFICATION")
    for constraint in spec.constraints:
        print(f"  {constraint.kind:>24}: {constraint.value}")
    print(f"  {'objectives':>24}: {', '.join(spec.objectives) or '-'}\n")

    # 3. Open the CAD environment.
    engine = CADEngine("demo")
    executor = ActionExecutor(engine)
    recorder = TrajectoryRecorder(executor, args.requirement)

    # 4-14. Build with the procedural expert, matched to the parsed spec.
    family = get_family("l_bracket")
    holes_per_leg = max(1, (spec.hole_count or 4) // 2)
    params = family.params_cls(
        hole_diameter=spec.hole_diameter or 5.0,
        holes_per_leg=holes_per_leg,
        thickness=max(spec.min_wall_thickness or 3.0, 3.0),
        fillet_radius=2.0,
    )
    print("EXECUTING EXPERT ACTION SEQUENCE")
    family.build(executor, params)
    for step in recorder.steps:
        action = step["action"]
        print(
            f"  t={recorder.steps.index(step):>2} "
            f"{action['operation']:<16} reward {step['reward']['total']:+7.3f}"
        )

    # 15. Export.
    step_path = engine.export_step(out / "bracket")
    stl_path = engine.export_stl(out / "bracket")
    renders = engine.render(out)

    # 16-18. Trajectory, reward curve data, final metrics.
    trajectory_path = recorder.write(out / "trajectory.json")
    data = recorder.to_dict()
    final = data["final_metrics"]
    summary = final["summary"]
    constraints = final["constraints"]
    print("\nFINAL ENGINEERING METRICS")
    print(f"  mass:                  {summary['mass_g']:.1f} g")
    print(f"  volume:                {summary['volume_mm3'] / 1000:.1f} cm^3")
    print(f"  holes:                 {summary['hole_count']}")
    print(f"  geometry valid:        {'YES' if summary['valid'] else 'NO'}")
    print(f"  constraints satisfied: {constraints['satisfaction_rate'] * 100:.0f}% "
          f"(measured; {constraints['counts']['unmeasured']} unmeasured)")
    print(f"  actions:               {final['steps']} "
          f"({final['invalid_actions']} invalid)")
    print(f"  total reward:          {final['total_reward']:+.3f}")
    print("\nARTIFACTS")
    for label, path in [
        ("STEP", step_path), ("STL", stl_path), ("trajectory", trajectory_path),
        *[(f"render:{k}", v) for k, v in renders.items()],
    ]:
        print(f"  {label:<12} {path}")
    (out / "spec.json").write_text(json.dumps(spec.to_dict(), indent=2))
    engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
