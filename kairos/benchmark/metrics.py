"""Benchmark metrics that discriminate when no policy succeeds.

Success rate alone ranks a policy that builds a valid solid with the right hole
count identically with one that emits only invalid actions. The headline metric
here is instead a progress score over milestones:

    sketch -> geometry -> solid -> valid solid -> holes -> constraints -> finished

Weights increase monotonically, so reaching a later milestone always scores more
than any combination of earlier ones. Credit stops at the first milestone missed:
a constraint check passes vacuously on geometry that was never built.

Efficiency, validity and constraint satisfaction are reported alongside, never
folded in. Validity is where PPO beat BC in Phase 5 (0.000 invalid against
0.018) and a single success number hid it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

#: Ordered milestones, each with the score awarded for reaching it. Later
#: milestones are worth more than every earlier one combined, so the ordering
#: of two policies never depends on the weights' exact values.
MILESTONES: tuple[tuple[str, float], ...] = (
    ("opened_a_sketch", 1.0),
    ("drew_geometry", 2.0),
    ("made_a_solid", 4.0),
    ("solid_is_valid", 8.0),
    ("has_any_hole", 16.0),
    ("all_constraints_met", 32.0),
    ("finished_successfully", 64.0),
)

MAX_PROGRESS = sum(weight for _, weight in MILESTONES)


@dataclass
class EpisodeOutcome:
    """What one episode achieved, in terms the benchmark can score.

    Every field is read from the environment's own step reports rather than
    re-derived, so the benchmark cannot disagree with the simulator about what
    happened.
    """

    requirement: str = ""
    family: str = "unknown"
    steps: int = 0
    expert_steps: int | None = None
    invalid_actions: int = 0
    opened_a_sketch: bool = False
    drew_geometry: bool = False
    made_a_solid: bool = False
    solid_is_valid: bool = False
    has_any_hole: bool = False
    all_constraints_met: bool = False
    finished_successfully: bool = False
    satisfaction_rate: float = 0.0
    mass_g: float = 0.0
    crashed: bool = False

    # ------------------------------------------------------------ scoring

    def milestones_reached(self) -> list[str]:
        """Milestones reached, stopping at the first one missed.

        Prefix semantics matter: a constraint check can pass vacuously on
        geometry that was never built, and awarding it would rank an empty
        document above a real but imperfect part.
        """
        reached: list[str] = []
        for name, _ in MILESTONES:
            if not getattr(self, name):
                break
            reached.append(name)
        return reached

    def progress_score(self) -> float:
        """Weighted milestone score in [0, 1]."""
        weights = dict(MILESTONES)
        return sum(weights[name] for name in self.milestones_reached()) / MAX_PROGRESS

    def efficiency(self) -> float | None:
        """Expert steps / steps taken, capped at 1.0. None without a reference.

        Only meaningful among episodes that reached the same milestone, so the
        aggregate reports it separately rather than folding it into the score.

        None unless a valid solid exists, because a ratio of steps is not an
        efficiency when nothing was built. `immediate-finish` quits in one step
        and scored a perfect 1.000 here, on a metric its own contract says it
        must lose: a policy that builds nothing is not efficient at building.
        """
        if not self.expert_steps or self.steps <= 0:
            return None
        if not self.solid_is_valid:
            return None
        return min(1.0, self.expert_steps / self.steps)

    def validity_rate(self) -> float:
        """Fraction of actions the engine accepted."""
        if self.steps <= 0:
            return 0.0
        return 1.0 - (self.invalid_actions / self.steps)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["progress_score"] = round(self.progress_score(), 4)
        data["milestones_reached"] = self.milestones_reached()
        data["validity_rate"] = round(self.validity_rate(), 4)
        data["efficiency"] = self.efficiency()
        return data


def outcome_from_episode(episode, expert_steps: int | None = None) -> EpisodeOutcome:
    """Build an outcome from a :class:`kairos.rl.collect.EpisodeSummary`.

    The RL collector records less than the benchmark wants, so the fields it
    cannot supply stay False rather than being guessed at.
    """
    return EpisodeOutcome(
        requirement=getattr(episode, "requirement", ""),
        steps=getattr(episode, "steps", 0),
        expert_steps=expert_steps,
        invalid_actions=getattr(episode, "invalid_actions", 0),
        made_a_solid=bool(getattr(episode, "has_solid", False)),
        solid_is_valid=bool(getattr(episode, "has_solid", False)),
        all_constraints_met=bool(getattr(episode, "satisfaction_rate", 0.0) >= 1.0)
        and bool(getattr(episode, "has_solid", False)),
        finished_successfully=bool(getattr(episode, "finished_successfully", False)),
        satisfaction_rate=float(getattr(episode, "satisfaction_rate", 0.0)),
        mass_g=float(getattr(episode, "mass_g", 0.0)),
        crashed=bool(getattr(episode, "crashed", False)),
    )


@dataclass
class BenchmarkScore:
    """Aggregate over a policy's episodes."""

    policy: str
    episodes: int
    progress_score: float
    success_rate: float
    validity_rate: float
    satisfaction_rate: float
    efficiency: float | None
    crash_rate: float
    milestone_rates: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_policy(policy: str, outcomes: list[EpisodeOutcome]) -> BenchmarkScore:
    """Aggregate one policy's episodes into a comparable score."""
    if not outcomes:
        return BenchmarkScore(policy, 0, 0.0, 0.0, 0.0, 0.0, None, 0.0, {})

    n = len(outcomes)
    efficiencies = [e for e in (o.efficiency() for o in outcomes) if e is not None]
    return BenchmarkScore(
        policy=policy,
        episodes=n,
        progress_score=sum(o.progress_score() for o in outcomes) / n,
        success_rate=sum(o.finished_successfully for o in outcomes) / n,
        validity_rate=sum(o.validity_rate() for o in outcomes) / n,
        satisfaction_rate=sum(o.satisfaction_rate for o in outcomes) / n,
        efficiency=(sum(efficiencies) / len(efficiencies)) if efficiencies else None,
        crash_rate=sum(o.crashed for o in outcomes) / n,
        milestone_rates={
            name: sum(getattr(o, name) for o in outcomes) / n for name, _ in MILESTONES
        },
    )


def format_scores(scores: list[BenchmarkScore]) -> str:
    """Render a leaderboard, progress first because success is often all zero."""
    header = (
        f"{'policy':>14}  {'episodes':>8}  {'progress':>8}  {'success':>8}  "
        f"{'valid':>7}  {'constr':>7}  {'effic':>7}  {'crash':>6}"
    )
    lines = [header, "-" * len(header)]
    for s in sorted(scores, key=lambda s: -s.progress_score):
        efficiency = "-" if s.efficiency is None else f"{s.efficiency:.3f}"
        lines.append(
            f"{s.policy:>14}  {s.episodes:>8}  {s.progress_score:>8.3f}  "
            f"{s.success_rate:>8.3f}  {s.validity_rate:>7.3f}  "
            f"{s.satisfaction_rate:>7.3f}  {efficiency:>7}  {s.crash_rate:>6.3f}"
        )
    if scores:
        lines += ["", "milestone reach rates:"]
        names = [name for name, _ in MILESTONES]
        lines.append("  " + "  ".join(f"{n[:9]:>9}" for n in names))
        for s in sorted(scores, key=lambda s: -s.progress_score):
            row = "  ".join(f"{s.milestone_rates.get(n, 0.0):>9.2f}" for n in names)
            lines.append(f"  {row}   <- {s.policy}")
    return "\n".join(lines)
