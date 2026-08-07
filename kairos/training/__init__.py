"""Phase 4 training: behavioral cloning over the recorded expert trajectories.

Requires the optional torch extra (``pip install -e ".[learn]"``), like
:mod:`kairos.models`.
"""

from __future__ import annotations

__all__ = ["BCBatch", "TrajectoryDataset", "build_examples"]


def __getattr__(name: str):
    if name in __all__:
        from kairos.training import bc_dataset

        return getattr(bc_dataset, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
