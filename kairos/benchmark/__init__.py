"""Phase 7: the KAIROS-CAD benchmark.

Pure python — no torch — so the benchmark can score results anywhere, including
under FreeCAD's interpreter where the environment runs.
"""

from kairos.benchmark.metrics import (
    MAX_PROGRESS,
    MILESTONES,
    BenchmarkScore,
    EpisodeOutcome,
    format_scores,
    outcome_from_episode,
    score_policy,
)

__all__ = [
    "MAX_PROGRESS",
    "MILESTONES",
    "BenchmarkScore",
    "EpisodeOutcome",
    "format_scores",
    "outcome_from_episode",
    "score_policy",
]
