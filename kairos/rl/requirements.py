"""Requirement pools for RL training and evaluation.

Episodes draw their requirement from a pool so the language encoder receives
signal — a single fixed requirement teaches one build, not requirement reading.

Pools are drawn from the generated dataset when it is available, because those
requirements are exactly the distribution BC was fitted to, and PPO starting
from a BC policy should stay in that distribution. A small hardcoded fallback
keeps training runnable without a dataset on disk.

``train`` and ``held_out`` are split **by design index** the same way BC splits
were, so a policy cannot be evaluated on a requirement it trained against.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

#: Used when no dataset is present. One per family, so every build shape the
#: policy might see is represented.
FALLBACK_REQUIREMENTS: tuple[str, ...] = (
    "Design a rectangular mounting plate 80 x 50 x 6 mm with 4 through-holes "
    "of 5 mm diameter in a 2x2 grid. Minimize mass.",
    "Design a 90-degree L-bracket with 4 mounting holes of 5 mm diameter "
    "(2 per leg), wall thickness 5.0 mm, legs 70 mm and 55 mm, width 40 mm. "
    "Minimize mass.",
    "Design a cylindrical spacer of outer diameter 30 mm and height 20 mm "
    "with a 12 mm diameter through-bore. Minimize mass.",
    "Design a circular flange of outer diameter 90 mm and thickness 8.0 mm "
    "with a 32 mm diameter hub raised 10 mm, a 14 mm central bore, and 6 bolt "
    "holes of 5 mm diameter on a 64 mm bolt circle. Minimize mass.",
    "Design a U-channel bracket 70 mm wide, 45 mm tall and 50 mm deep, with "
    "4.0 mm side walls and a 5.0 mm base. Provide 2 base mounting holes and "
    "2 cross-wall holes, all 5 mm diameter. Minimize mass.",
)


def load_requirements(
    root: str | Path = "dataset",
    limit: int | None = None,
    families: list[str] | None = None,
) -> list[str]:
    """Read requirement texts from a generated dataset, newest ids last."""
    paths = sorted(Path(root).glob("designs/design_*/requirements.json"))
    if limit is not None:
        paths = paths[:limit]

    requirements: list[str] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        text = payload.get("text")
        kind = (payload.get("spec") or {}).get("kind")
        if not text:
            continue
        if families and kind not in families:
            continue
        requirements.append(text)
    return requirements


def three_way_pools(
    root: str | Path = "dataset",
    splits_path: str | Path | None = None,
    seed: int = 0,
) -> tuple[list[str], list[str], list[str]]:
    """Return ``(train, dev, test)`` requirement pools from a frozen split.

    Prefer this over :func:`requirement_pools`. Checkpoint selection must read
    ``dev`` and the benchmark must read ``test``; selecting on the set you then
    report is model selection on the evaluation set, which inflates the number
    just as surely as training on it.
    """
    from kairos.benchmark.splits import (
        SplitSet,
        build_splits,
        load_requirements_by_design,
        requirements_for,
    )

    designs = load_requirements_by_design(root)
    if not designs:
        pool = list(FALLBACK_REQUIREMENTS)
        return pool, pool, pool

    if splits_path is not None and Path(splits_path).exists():
        splits = SplitSet.load(splits_path)
    else:
        splits = build_splits(designs, seed=seed)
    return tuple(requirements_for(splits, name, designs) for name in ("train", "dev", "test"))


def requirement_pools(
    root: str | Path = "dataset",
    limit: int | None = None,
    held_out_fraction: float = 0.15,
    seed: int = 0,
    families: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Return ``(train, held_out)`` requirement pools.

    Falls back to :data:`FALLBACK_REQUIREMENTS` when no dataset is present, in
    which case both pools are the same small set and evaluation numbers should
    be read as in-distribution.

    .. deprecated::
        Two-way only, and derived at call time — calling it twice with
        different ``limit`` values draws a different boundary, which is how a
        contaminated comparison shipped. Use :func:`three_way_pools`.
    """
    requirements = load_requirements(root, limit=limit, families=families)
    if not requirements:
        return list(FALLBACK_REQUIREMENTS), list(FALLBACK_REQUIREMENTS)

    unique = list(dict.fromkeys(requirements))  # stable dedupe
    if len(unique) < 2:
        return unique, unique

    order = np.random.default_rng(seed).permutation(len(unique))
    n_held_out = max(1, int(round(len(unique) * held_out_fraction)))
    held_out = [unique[i] for i in order[:n_held_out]]
    train = [unique[i] for i in order[n_held_out:]]
    return train, held_out
