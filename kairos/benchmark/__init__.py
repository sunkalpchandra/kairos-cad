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
from kairos.benchmark.splits import (
    SPLIT_NAMES,
    ContaminationError,
    Split,
    SplitSet,
    assert_disjoint,
    build_splits,
    load_requirements_by_design,
    requirements_for,
    text_hash,
)

#: Frozen suite identity; every artifact records it.
SUITE_VERSION = "kairos-cad-v1"

__all__ = [
    "MAX_PROGRESS",
    "MILESTONES",
    "SPLIT_NAMES",
    "SUITE_VERSION",
    "ContaminationError",
    "Split",
    "SplitSet",
    "assert_disjoint",
    "build_splits",
    "load_requirements_by_design",
    "requirements_for",
    "text_hash",
    "BenchmarkScore",
    "EpisodeOutcome",
    "format_scores",
    "outcome_from_episode",
    "score_policy",
]
