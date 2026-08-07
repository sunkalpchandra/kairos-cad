"""Phase 6: engineering optimization over a learned surrogate.

Pure numpy — no torch — so the search runs under FreeCAD's interpreter, where
the verification build happens.
"""

from kairos.optimization.optimizer import (
    Bounds,
    OptimizationResult,
    optimize_design,
    penalized_objective,
    verify_result,
)
from kairos.optimization.surrogate import (
    RidgeSurrogate,
    Sample,
    SurrogateData,
    train_surrogate,
)

__all__ = [
    "Bounds",
    "OptimizationResult",
    "RidgeSurrogate",
    "Sample",
    "SurrogateData",
    "optimize_design",
    "penalized_objective",
    "train_surrogate",
    "verify_result",
]
