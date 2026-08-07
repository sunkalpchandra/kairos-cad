#!/usr/bin/env python
"""Phase 6: minimize a design's mass while keeping it manufacturable.

    PYTHONPATH=. /Applications/FreeCAD.app/Contents/Resources/bin/python \\
        scripts/optimize_design.py --family plate --min-thickness 4.0

Samples the family in FreeCAD to fit a surrogate, searches parameter space with
it, then **builds the winner for real** and reports the verified numbers. Runs
under FreeCAD's interpreter; the surrogate is pure numpy so no torch is needed.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _build_and_measure(family, params_cls, values, wall_thickness=True):
    """Build one parameter set and return (mass_g, thickness_mm, valid)."""
    from kairos.actions.executor import ActionExecutor
    from kairos.cad.engine import CADEngine
    from kairos.representation import observe

    params = params_cls(**values)
    if not params.is_feasible():
        return None, None, False
    engine = CADEngine("optimize")
    try:
        family.build(ActionExecutor(engine), params)
        summary = observe(engine, wall_thickness=wall_thickness)["summary"]
        if not summary.get("valid"):
            return None, None, False
        return (
            float(summary.get("mass_g") or 0.0),
            summary.get("min_wall_thickness_mm"),
            True,
        )
    except Exception:
        return None, None, False
    finally:
        engine.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--family", default=None)
    parser.add_argument("--min-thickness", type=float, default=None)
    parser.add_argument("--samples", type=int, default=None, help="FreeCAD builds for fitting")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--population", type=int, default=None)
    parser.add_argument("--degree", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    from kairos.config import load_config

    # The config is the default; explicit flags win. Without this the
    # optimization: section would be decorative.
    section = dict(load_config(args.config).get("optimization", {}) or {})
    known = {"family", "min_thickness", "samples", "iterations", "population",
             "surrogate_degree", "out_dir"}
    unknown = set(section) - known
    if unknown:
        print(f"error: unknown optimization keys: {sorted(unknown)}", file=sys.stderr)
        return 2
    # `or` would silently replace an explicit 0 with the config default.
    args.family = args.family or section.get("family", "plate")
    args.min_thickness = (
        args.min_thickness if args.min_thickness is not None
        else float(section.get("min_thickness", 4.0))
    )
    args.samples = (
        args.samples if args.samples is not None
        else int(section.get("samples", 60))
    )
    args.iterations = (
        args.iterations if args.iterations is not None
        else int(section.get("iterations", 20))
    )
    args.population = (
        args.population if args.population is not None
        else int(section.get("population", 128))
    )
    args.degree = (
        args.degree if args.degree is not None
        else int(section.get("surrogate_degree", 3))
    )
    args.seed = args.seed if args.seed is not None else 0
    args.out = args.out or Path(section.get("out_dir", "runs/optimize"))

    from kairos.data.families import family_names, get_family
    from kairos.optimization import (
        Bounds,
        Sample,
        SurrogateData,
        optimize_design,
        train_surrogate,
        verify_result,
    )

    if args.family not in family_names():
        print(f"error: unknown family {args.family!r}; known: {family_names()}", file=sys.stderr)
        return 1
    family = get_family(args.family)
    params_cls = family.params_cls
    rng = random.Random(args.seed)

    # --- sample the family in FreeCAD ---------------------------------
    print(f"sampling {args.samples} {args.family} designs for the surrogate ...")
    started = time.perf_counter()
    numeric_fields = [
        name
        for name, f in params_cls.__dataclass_fields__.items()
        if f.type in ("float", float) or isinstance(getattr(params_cls(), name), float)
    ]
    data = SurrogateData(args.family, numeric_fields)
    for _ in range(args.samples):
        params = params_cls.sample(rng)
        values = {n: float(getattr(params, n)) for n in numeric_fields}
        mass, thickness, valid = _build_and_measure(family, params_cls, values)
        if mass is None:
            continue
        data.samples.append(Sample(values, mass, thickness, valid))
    usable = len(data.matrix()[0])
    print(f"  {usable} usable samples in {time.perf_counter() - started:.0f}s")
    if usable < 8:
        print("error: too few usable samples to fit a surrogate", file=sys.stderr)
        return 1

    # --- fit and search -----------------------------------------------
    model, metrics = train_surrogate(data, degree=args.degree, seed=args.seed)
    print(
        f"surrogate: mass R2 {metrics.mass_r2:.4f} (MAE {metrics.mass_mae:.2f} g), "
        f"thickness R2 {metrics.thickness_r2:.4f} (MAE {metrics.thickness_mae:.3f} mm), "
        f"{metrics.train_rows} train / {metrics.test_rows} test"
    )

    observed = data.matrix()[0]
    bounds = Bounds(
        {n: float(observed[:, i].min()) for i, n in enumerate(data.parameter_names)},
        {n: float(observed[:, i].max()) for i, n in enumerate(data.parameter_names)},
    )
    result = optimize_design(
        model,
        bounds,
        min_thickness_mm=args.min_thickness,
        is_feasible=lambda v: params_cls(**v).is_feasible(),
        iterations=args.iterations,
        population=args.population,
        seed=args.seed,
    )
    print(
        f"search: {result.evaluations} surrogate evaluations over "
        f"{result.iterations} iterations -> predicted mass "
        f"{result.predicted_mass_g:.2f} g, thickness {result.predicted_thickness_mm:.2f} mm"
    )

    # --- verify the winner in FreeCAD ---------------------------------
    baseline_mass, _, _ = _build_and_measure(
        family, params_cls, {n: float(getattr(params_cls(), n)) for n in data.parameter_names}
    )
    result.baseline_mass_g = baseline_mass
    verify_result(
        result,
        lambda v: _build_and_measure(family, params_cls, v),
        min_thickness_mm=args.min_thickness,
    )

    print("\nverified by building it:")
    print(f"  mass       {result.verified_mass_g:.2f} g (default design {baseline_mass:.2f} g)")
    print(f"  thickness  {result.verified_thickness_mm} mm (floor {args.min_thickness})")
    print(f"  manufacturable: {result.verified_feasible}")
    if result.mass_saving_pct is not None:
        print(f"  mass saving: {result.mass_saving_pct:+.1f}%")
    if result.surrogate_error_pct is not None:
        print(f"  surrogate mass error on the winner: {result.surrogate_error_pct:.1f}%")

    args.out.mkdir(parents=True, exist_ok=True)
    model.save(args.out / f"{args.family}_surrogate.json")
    data.save(args.out / f"{args.family}_samples.json")
    (args.out / f"{args.family}_result.json").write_text(
        json.dumps({"metrics": metrics.to_dict(), "result": result.to_dict()}, indent=2) + "\n"
    )
    print(f"\nwrote {args.out}/{args.family}_*.json")
    return 0 if result.verified_feasible else 2


if __name__ == "__main__":
    raise SystemExit(main())
