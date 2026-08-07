"""Multi-objective shaped reward for CAD episodes.

Implements the project reward decomposition

    R = R_validity + R_constraint + R_engineering + R_progress
        - R_invalid - R_complexity - R_action_cost

as an episode-scoped ``RewardTracker``: shaping bonuses (valid sketch, first
solid, each newly satisfied constraint, ...) are awarded exactly once per
episode; the mass-progress term only activates while every *measured*
constraint is satisfied, so the agent cannot farm "mass reduction" by never
building the required geometry.

The tracker consumes plain observation dicts (``kairos.representation.observe``)
and ``ConstraintReport``s — pure python, unit-testable without FreeCAD.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kairos.actions.schema import FEATURE_OPS, SKETCH_OPS, ActionResult, Operation
from kairos.evaluation.constraints import ConstraintReport, check_constraints
from kairos.language.spec import EngineeringSpec


@dataclass(frozen=True)
class RewardWeights:
    """Weights for each reward component (defaults follow the project spec)."""

    valid_sketch: float = 0.2
    fully_constrained_sketch: float = 0.5
    first_solid: float = 0.5
    constraint_satisfied: float = 0.5  # per newly satisfied constraint
    all_constraints: float = 2.0
    mass_progress: float = 1.0  # x normalized mass improvement
    finish_success: float = 5.0
    finish_failure: float = -1.0
    validity_broken: float = -1.0
    invalid_action: float = -0.5
    complexity: float = -0.02  # per successful feature operation
    action_cost: float = -0.01  # per step


@dataclass
class RewardBreakdown:
    """One step's reward, decomposed for logging and visualization."""

    total: float = 0.0
    components: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, value: float) -> None:
        if value == 0.0:
            return
        self.components[name] = self.components.get(name, 0.0) + value
        self.total += value

    def to_dict(self) -> dict[str, Any]:
        return {"total": round(self.total, 6), "components": {
            k: round(v, 6) for k, v in self.components.items()
        }}


class RewardTracker:
    """Stateful per-episode reward computation."""

    def __init__(
        self,
        spec: EngineeringSpec,
        weights: RewardWeights | None = None,
        context: dict | None = None,
    ) -> None:
        self.spec = spec
        self.weights = weights or RewardWeights()
        self.context = context or {}
        self._awarded: set[str] = set()
        self._satisfied_kinds: set[str] = set()
        self._had_solid = False
        self._was_valid = False
        self._mass_baseline: float | None = None
        self._best_mass: float | None = None
        self.last_report: ConstraintReport | None = None

    # ------------------------------------------------------------------ api

    def step(
        self,
        result: ActionResult,
        observation: dict[str, Any],
    ) -> RewardBreakdown:
        """Score one executed action given the post-action observation."""
        w = self.weights
        breakdown = RewardBreakdown()
        breakdown.add("action_cost", w.action_cost)

        if not result.ok:
            breakdown.add("invalid_action", w.invalid_action)
            return breakdown

        summary = observation.get("summary", {})
        report = check_constraints(observation, self.spec, self.context)
        self.last_report = report

        # -------------------------------------------------- shaping events
        sketch = observation.get("sketch") or {}
        if (
            result.operation in SKETCH_OPS
            and sketch.get("geometry_count", 0) > 0
            and self._award("valid_sketch")
        ):
            breakdown.add("valid_sketch", w.valid_sketch)

        if sketch.get("fully_constrained") and self._award("fully_constrained_sketch"):
            breakdown.add("fully_constrained_sketch", w.fully_constrained_sketch)

        has_solid = bool(summary.get("has_solid"))
        if has_solid and not self._had_solid and self._award("first_solid"):
            breakdown.add("first_solid", w.first_solid)
        self._had_solid = self._had_solid or has_solid

        # Validity regression (e.g. a legal action that fragments the part).
        is_valid = bool(summary.get("valid"))
        if self._was_valid and has_solid and not is_valid:
            breakdown.add("validity_broken", w.validity_broken)
        self._was_valid = is_valid

        # ---------------------------------------------- constraint progress
        for res in report.satisfied:
            key = f"constraint:{res.constraint.kind}"
            if res.constraint.kind not in self._satisfied_kinds:
                self._satisfied_kinds.add(res.constraint.kind)
                if self._award(key):
                    breakdown.add(key, w.constraint_satisfied)

        if report.all_measured_satisfied and report.satisfied and self._award("all_constraints"):
            breakdown.add("all_constraints", w.all_constraints)

        # ------------------------------------------------- objective progress
        if self.spec.has_objective("minimize_mass") and has_solid:
            mass = summary.get("mass_g")
            if mass is not None and report.all_measured_satisfied:
                if self._mass_baseline is None:
                    # First constraint-satisfying design sets the scale.
                    self._mass_baseline = mass
                    self._best_mass = mass
                elif self._best_mass is not None and mass < self._best_mass:
                    # Paid against the best mass so far, never the previous
                    # step's: the episode total then telescopes to
                    # (baseline - lightest)/baseline, so padding material back
                    # on and removing it again earns nothing the second time.
                    improvement = (self._best_mass - mass) / max(self._mass_baseline, 1e-9)
                    breakdown.add("mass_progress", w.mass_progress * improvement)
                    self._best_mass = mass

        # ------------------------------------------------------- complexity
        if result.operation in FEATURE_OPS:
            breakdown.add("complexity", w.complexity)

        # ------------------------------------------------------ termination
        if result.operation is Operation.FINISH_DESIGN:
            # ``all_measured_satisfied`` already encodes the right rule for an
            # empty spec (trivially satisfied) and for one whose constraints are
            # all unmeasured (no credit). Also demanding a non-empty
            # ``satisfied`` list would make every zero-constraint requirement —
            # e.g. the u_bracket and spacer families — unwinnable.
            success = has_solid and is_valid and report.all_measured_satisfied
            breakdown.add(
                "finish", w.finish_success if success else w.finish_failure
            )

        return breakdown

    # ------------------------------------------------------------- helpers

    def _award(self, key: str) -> bool:
        """True the first time ``key`` is awarded this episode."""
        if key in self._awarded:
            return False
        self._awarded.add(key)
        return True
