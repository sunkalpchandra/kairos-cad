"""Validated dataset generation.

For each sampled design: execute the recipe, validate the result, and write
the dataset layout from the project spec —

    designs/design_NNNNNN/
        model.FCStd  model.step  model.stl
        iso.png  front.png  top.png  right.png
        state.json  requirements.json  trajectory.json

Invalid or infeasible designs are rejected (with a recorded reason), never
silently written.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kairos.actions.executor import ActionExecutor
from kairos.cad.engine import CADEngine
from kairos.data import procedural


@dataclass
class GenerationStats:
    attempted: int = 0
    written: int = 0
    infeasible: int = 0
    failed: int = 0
    invalid: int = 0
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "written": self.written,
            "infeasible": self.infeasible,
            "failed": self.failed,
            "invalid": self.invalid,
            "reasons": self.reasons,
        }


def _requirements(kind: str, params, hole_count: int) -> dict[str, Any]:
    """Structured requirement plus a template natural-language rendering."""
    d = params.hole_diameter
    if kind == "l_bracket":
        text = (
            f"Design a 90-degree L-bracket with {hole_count} mounting holes of "
            f"{d:.0f} mm diameter ({params.holes_per_leg} per leg), wall thickness "
            f"{params.thickness:.1f} mm, legs {params.leg1:.0f} mm and "
            f"{params.leg2:.0f} mm, width {params.width:.0f} mm. Minimize mass."
        )
    else:
        text = (
            f"Design a rectangular mounting plate {params.length:.0f} x "
            f"{params.width:.0f} x {params.thickness:.1f} mm with {hole_count} "
            f"through-holes of {d:.0f} mm diameter in a "
            f"{params.holes_x}x{params.holes_y} grid. Minimize mass."
        )
    return {
        "text": text,
        "spec": {
            "kind": kind,
            "hole_count": hole_count,
            "hole_diameter": d,
            "min_wall_thickness": params.thickness,
            "objective": "minimize_mass",
        },
    }


def generate_design(
    kind: str, rng: random.Random, out_dir: Path, design_id: int, stats: GenerationStats
) -> bool:
    """Generate one validated design; returns True if written."""
    stats.attempted += 1
    if kind == "l_bracket":
        params = procedural.LBracketParams.sample(rng)
        builder = procedural.build_l_bracket
    elif kind == "plate":
        params = procedural.PlateParams.sample(rng)
        builder = procedural.build_plate
    else:
        raise ValueError(f"unknown design kind {kind!r}")

    if not params.is_feasible():
        stats.infeasible += 1
        return False

    expected_holes = procedural.expected_hole_count(params)
    engine = CADEngine(f"design_{design_id:06d}")
    try:
        executor = ActionExecutor(engine)
        try:
            actions = builder(executor, params)
        except RuntimeError as err:
            stats.failed += 1
            stats.reasons.append(f"design_{design_id:06d}: {err}")
            return False

        report = engine.check_validity()
        holes = engine.find_holes(diameter=params.hole_diameter)
        if not report.is_valid:
            stats.invalid += 1
            stats.reasons.append(f"design_{design_id:06d}: invalid geometry {report.issues}")
            return False
        if len(holes) != expected_holes:
            stats.invalid += 1
            stats.reasons.append(
                f"design_{design_id:06d}: hole count {len(holes)} != expected {expected_holes}"
            )
            return False

        design_dir = out_dir / f"design_{design_id:06d}"
        design_dir.mkdir(parents=True, exist_ok=True)
        engine.render(design_dir)
        engine.export_step(design_dir / "model.step")
        engine.export_stl(design_dir / "model.stl")
        engine.save(design_dir / "model.FCStd")

        state = engine.summary()
        state["parameters"] = procedural.params_to_dict(params)
        state["validation"] = report.to_dict()
        (design_dir / "state.json").write_text(json.dumps(state, indent=2))
        (design_dir / "requirements.json").write_text(
            json.dumps(_requirements(kind, params, expected_holes), indent=2)
        )
        (design_dir / "trajectory.json").write_text(
            json.dumps(
                {
                    "actions": [a.to_dict() for a in actions],
                    "history": executor.trajectory(),
                },
                indent=2,
            )
        )
        stats.written += 1
        return True
    finally:
        engine.close()


def generate_dataset(
    out_dir: str | Path,
    count: int,
    seed: int = 0,
    kinds: tuple[str, ...] = ("l_bracket", "plate"),
    max_attempts_factor: int = 8,
) -> GenerationStats:
    """Generate ``count`` validated designs, alternating kinds, seeded."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    stats = GenerationStats()
    design_id = 0
    while stats.written < count and stats.attempted < count * max_attempts_factor:
        kind = kinds[design_id % len(kinds)]
        if generate_design(kind, rng, out_dir, design_id, stats):
            pass
        design_id += 1
    (out_dir / "generation_stats.json").write_text(json.dumps(stats.to_dict(), indent=2))
    return stats
