"""Constrained design optimization: minimize mass, keep the part manufacturable.

The loop is **propose cheaply, verify exactly**. A cross-entropy-method search
samples parameter sets, scores them with the surrogate (microseconds), keeps
the best, and refits the sampling distribution to them. Only the final winner
is built in FreeCAD and measured for real.

That split is the point. Scoring every candidate in FreeCAD would make a
10,000-candidate search take hours; scoring none of them exactly would let a
surrogate error ship as a result. So the search is surrogate-driven and the
*answer* is always a verified build — and when verification disagrees with the
prediction, the report says so rather than quietly reporting the prediction.

Infeasible parameter draws are rejected by the family's own ``is_feasible``
before the surrogate ever sees them, so the search never spends its budget on
geometry that cannot be built.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class Bounds:
    """Per-parameter search range."""

    lower: dict[str, float]
    upper: dict[str, float]

    def names(self) -> list[str]:
        return sorted(self.lower)

    def clip(self, values: dict[str, float]) -> dict[str, float]:
        return {
            k: float(min(max(v, self.lower[k]), self.upper[k])) for k, v in values.items()
        }

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        names = self.names()
        low = np.array([self.lower[n] for n in names])
        high = np.array([self.upper[n] for n in names])
        return rng.uniform(low, high, size=(n, len(names)))


@dataclass
class OptimizationResult:
    """What the search found, and whether reality agreed."""

    parameters: dict[str, float]
    predicted_mass_g: float
    predicted_thickness_mm: float
    verified_mass_g: float | None = None
    verified_thickness_mm: float | None = None
    verified_feasible: bool | None = None
    baseline_mass_g: float | None = None
    evaluations: int = 0
    iterations: int = 0
    history: list[float] = field(default_factory=list)

    @property
    def mass_saving_pct(self) -> float | None:
        """Mass reduction against the baseline, using verified numbers."""
        if self.baseline_mass_g in (None, 0) or self.verified_mass_g is None:
            return None
        return 100.0 * (self.baseline_mass_g - self.verified_mass_g) / self.baseline_mass_g

    @property
    def surrogate_error_pct(self) -> float | None:
        """How far the surrogate's mass prediction was from the truth."""
        if self.verified_mass_g in (None, 0):
            return None
        return 100.0 * abs(self.predicted_mass_g - self.verified_mass_g) / self.verified_mass_g

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["mass_saving_pct"] = self.mass_saving_pct
        data["surrogate_error_pct"] = self.surrogate_error_pct
        return data


def penalized_objective(
    mass_g: float,
    thickness_mm: float,
    min_thickness_mm: float,
    penalty_per_mm: float = 10.0,
) -> float:
    """Mass, scaled up by how far the wall falls under spec.

    Multiplicative, not additive. An additive penalty has to be picked in mass
    units, and a value large enough to forbid a real violation (1000 g) also
    makes a 0.01 mm prediction error at the boundary cost more than every mass
    difference in the search — so the optimum, which sits exactly on the
    boundary, is precisely where the search refuses to go. Scaling keeps the
    penalty commensurate with the objective at any part size.

    Soft rather than a hard reject: discarding infeasible candidates outright
    leaves no gradient in the region just outside the boundary.
    """
    shortfall = max(0.0, min_thickness_mm - thickness_mm)
    return float(mass_g * (1.0 + penalty_per_mm * shortfall))


def optimize_design(
    surrogate,
    bounds: Bounds,
    min_thickness_mm: float,
    is_feasible: Callable[[dict[str, float]], bool] | None = None,
    iterations: int = 20,
    population: int = 256,
    elite_fraction: float = 0.15,
    seed: int = 0,
    smoothing: float = 0.7,
) -> OptimizationResult:
    """Cross-entropy search for the lightest design that stays manufacturable.

    Args:
        surrogate: anything with ``predict_one(dict) -> (mass, thickness)``.
        bounds: per-parameter search ranges.
        min_thickness_mm: the manufacturing floor to respect.
        is_feasible: the family's own guard; rejected draws cost nothing.
        smoothing: how much of the previous distribution to keep each round.
            Refitting purely to the elite collapses the search in a few rounds.
    """
    names = bounds.names()
    rng = np.random.default_rng(seed)
    lower = np.array([bounds.lower[n] for n in names])
    upper = np.array([bounds.upper[n] for n in names])

    mean = (lower + upper) / 2.0
    spread = (upper - lower) / 4.0
    n_elite = max(2, int(round(population * elite_fraction)))

    best_row: np.ndarray | None = None
    best_score = math.inf
    best_prediction = (math.nan, math.nan)
    evaluations = 0
    history: list[float] = []

    for _ in range(iterations):
        draws = rng.normal(mean, spread, size=(population, len(names)))
        draws = np.clip(draws, lower, upper)

        scored: list[tuple[float, np.ndarray, tuple[float, float]]] = []
        for row in draws:
            values = dict(zip(names, row.tolist(), strict=True))
            if is_feasible is not None and not is_feasible(values):
                continue
            mass, thickness = surrogate.predict_one(values)
            evaluations += 1
            score = penalized_objective(mass, thickness, min_thickness_mm)
            scored.append((score, row, (mass, thickness)))

        if not scored:
            # Nothing feasible this round: widen rather than stall.
            spread = np.minimum(spread * 1.5, (upper - lower) / 2.0)
            continue

        scored.sort(key=lambda item: item[0])
        if scored[0][0] < best_score:
            best_score, best_row, best_prediction = scored[0]
        history.append(float(scored[0][0]))

        elite = np.array([row for _, row, _ in scored[:n_elite]])
        mean = smoothing * mean + (1 - smoothing) * elite.mean(axis=0)
        spread = smoothing * spread + (1 - smoothing) * np.maximum(elite.std(axis=0), 1e-6)

    if best_row is None:
        raise RuntimeError("no feasible candidate found in the search space")

    return OptimizationResult(
        parameters=dict(zip(names, best_row.tolist(), strict=True)),
        predicted_mass_g=best_prediction[0],
        predicted_thickness_mm=best_prediction[1],
        evaluations=evaluations,
        iterations=iterations,
        history=history,
    )


def verify_result(
    result: OptimizationResult,
    build_and_measure: Callable[[dict[str, float]], tuple[float, float | None, bool]],
    min_thickness_mm: float,
) -> OptimizationResult:
    """Build the winner for real and record what the geometry actually says.

    ``build_and_measure`` returns ``(mass_g, thickness_mm, valid)``. The
    verified numbers are what any report must quote; the predictions are kept
    only so the surrogate's error is visible.
    """
    mass, thickness, valid = build_and_measure(result.parameters)
    result.verified_mass_g = mass
    result.verified_thickness_mm = thickness
    result.verified_feasible = bool(
        valid and thickness is not None and thickness >= min_thickness_mm - 0.05
    )
    return result
