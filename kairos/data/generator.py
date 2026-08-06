"""Validated dataset generation over the design-family registry.

For each sampled design: execute the family recipe, validate the result, and
write the dataset layout from the project spec —

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
from kairos.data.families import family_names, get_family, params_to_dict
from kairos.data.trajectories import TrajectoryRecorder


@dataclass
class GenerationStats:
    attempted: int = 0
    written: int = 0
    infeasible: int = 0
    failed: int = 0
    invalid: int = 0
    by_family: dict[str, int] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "written": self.written,
            "infeasible": self.infeasible,
            "failed": self.failed,
            "invalid": self.invalid,
            "by_family": dict(self.by_family),
            "reasons": self.reasons,
        }


def _check_holes(engine: CADEngine, expected: list[tuple[float, int]]) -> str | None:
    """Verify each (diameter, count) hole requirement; returns issue or None."""
    for diameter, count in expected:
        found = len(engine.find_holes(diameter=diameter))
        if found != count:
            return f"hole count for d={diameter}: {found} != expected {count}"
    return None


def generate_design(
    kind: str,
    rng: random.Random,
    out_dir: Path,
    design_id: int,
    stats: GenerationStats,
    trajectories_dir: Path | None = None,
) -> bool:
    """Generate one validated design of the given family; True if written."""
    family = get_family(kind)
    stats.attempted += 1
    params = family.params_cls.sample(rng)
    if not params.is_feasible():
        stats.infeasible += 1
        return False

    requirement = family.requirements(params)["text"]
    engine = CADEngine(f"design_{design_id:06d}")
    try:
        executor = ActionExecutor(engine)
        recorder = TrajectoryRecorder(executor, requirement)
        try:
            actions = family.build(executor, params)
        except RuntimeError as err:
            stats.failed += 1
            stats.reasons.append(f"design_{design_id:06d} [{kind}]: {err}")
            return False

        report = engine.check_validity()
        if not report.is_valid:
            stats.invalid += 1
            stats.reasons.append(
                f"design_{design_id:06d} [{kind}]: invalid geometry {report.issues}"
            )
            return False
        hole_issue = _check_holes(engine, family.expected_holes(params))
        if hole_issue:
            stats.invalid += 1
            stats.reasons.append(f"design_{design_id:06d} [{kind}]: {hole_issue}")
            return False

        design_dir = out_dir / f"design_{design_id:06d}"
        design_dir.mkdir(parents=True, exist_ok=True)
        engine.render(design_dir)
        engine.export_step(design_dir / "model.step")
        engine.export_stl(design_dir / "model.stl")
        engine.save(design_dir / "model.FCStd")

        state = engine.summary()
        state["parameters"] = params_to_dict(kind, params)
        state["validation"] = report.to_dict()
        (design_dir / "state.json").write_text(json.dumps(state, indent=2))
        (design_dir / "requirements.json").write_text(
            json.dumps(family.requirements(params), indent=2)
        )
        trajectory = recorder.to_dict()
        trajectory["design_id"] = f"design_{design_id:06d}"
        trajectory["family"] = kind
        trajectory["recipe_actions"] = [a.to_dict() for a in actions]
        (design_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2))
        if trajectories_dir is not None:
            trajectories_dir.mkdir(parents=True, exist_ok=True)
            (trajectories_dir / f"trajectory_{design_id:06d}.json").write_text(
                json.dumps(trajectory, indent=2)
            )
        stats.written += 1
        stats.by_family[kind] = stats.by_family.get(kind, 0) + 1
        return True
    finally:
        engine.close()


def generate_dataset(
    out_dir: str | Path,
    count: int,
    seed: int = 0,
    kinds: tuple[str, ...] | None = None,
    max_attempts_factor: int = 8,
    start_id: int = 0,
) -> GenerationStats:
    """Generate ``count`` validated designs, cycling families, seeded.

    ``start_id`` offsets design directory numbering so shards can generate
    disjoint id ranges in parallel.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir = out_dir.parent / "trajectories"
    kinds = tuple(kinds) if kinds else tuple(family_names())
    rng = random.Random(seed)
    stats = GenerationStats()
    design_id = start_id
    while stats.written < count and stats.attempted < count * max_attempts_factor:
        kind = kinds[design_id % len(kinds)]
        generate_design(kind, rng, out_dir, design_id, stats, trajectories_dir)
        design_id += 1
    stats_path = out_dir / (
        "generation_stats.json" if start_id == 0 else f"generation_stats_{start_id:06d}.json"
    )
    stats_path.write_text(json.dumps(stats.to_dict(), indent=2))
    return stats
