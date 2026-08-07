"""Three-way splits, recorded as data rather than re-derived.

Two contamination bugs have already shipped in this project, and both had the
same root cause: **the split was a function, evaluated more than once, with
different arguments.** `requirement_pools(limit=64)` and
`requirement_pools(limit=40)` permute over different pool sizes and therefore
draw a different boundary, which is how three of six "held-out" requirements
turned out to be trained on.

So a split here is a frozen artifact — an explicit list of ids and text hashes,
written once and read back — not a computation to repeat.

**Why three ways and not two.** PPO picks its best checkpoint by success rate on
the pool it is then scored against. Nothing is *trained* on that pool, so the
existing leak check passes; but choosing which checkpoint to ship using the
evaluation set is model selection on the test set, and it inflates the reported
number just the same. `dev` exists to absorb every such choice — checkpoint
selection, hyperparameters, ablation decisions — leaving `test` for a number
quoted once.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Split names, in the order they are consumed.
SPLIT_NAMES = ("train", "dev", "test")


def text_hash(text: str) -> str:
    """Stable hash of a requirement, normalized for whitespace and case.

    Splitting by design id alone is not enough: two designs can carry
    near-identical requirement texts, and a policy that memorized one has
    effectively seen the other.
    """
    normalized = " ".join(text.split()).strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


class ContaminationError(RuntimeError):
    """Two splits share a design or a requirement, so results are meaningless."""


@dataclass
class Split:
    """One split: which designs, and the hash of every requirement in it."""

    name: str
    design_ids: list[str] = field(default_factory=list)
    text_hashes: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.design_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "design_ids": self.design_ids,
            "text_hashes": self.text_hashes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Split:
        return cls(
            name=payload["name"],
            design_ids=list(payload["design_ids"]),
            text_hashes=list(payload["text_hashes"]),
        )


@dataclass
class SplitSet:
    """The frozen train/dev/test partition of a dataset."""

    splits: dict[str, Split] = field(default_factory=dict)
    dataset_root: str = ""
    seed: int = 0

    def __getitem__(self, name: str) -> Split:
        return self.splits[name]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_root": self.dataset_root,
            "seed": self.seed,
            "splits": {name: s.to_dict() for name, s in self.splits.items()},
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SplitSet:
        return cls(
            splits={n: Split.from_dict(s) for n, s in payload["splits"].items()},
            dataset_root=payload.get("dataset_root", ""),
            seed=int(payload.get("seed", 0)),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> SplitSet:
        return cls.from_dict(json.loads(Path(path).read_text()))


def assert_disjoint(splits: SplitSet) -> None:
    """Raise unless every pair of splits is disjoint by id *and* by text."""
    names = [n for n in SPLIT_NAMES if n in splits.splits]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared_ids = set(splits[a].design_ids) & set(splits[b].design_ids)
            if shared_ids:
                raise ContaminationError(
                    f"{a} and {b} share {len(shared_ids)} designs, e.g. "
                    f"{sorted(shared_ids)[:3]}"
                )
            shared_texts = set(splits[a].text_hashes) & set(splits[b].text_hashes)
            if shared_texts:
                raise ContaminationError(
                    f"{a} and {b} share {len(shared_texts)} requirement texts; "
                    "splitting by design id alone let a duplicate through"
                )


def build_splits(
    designs: dict[str, str],
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> SplitSet:
    """Partition ``{design_id: requirement_text}`` into train/dev/test.

    Grouped by requirement hash before splitting, so two designs sharing a text
    always land in the same split rather than straddling the boundary.
    """
    if abs(sum(fractions) - 1.0) > 1e-6:
        raise ValueError(f"fractions must sum to 1.0, got {sum(fractions)}")

    grouped: dict[str, list[str]] = {}
    for design_id, text in designs.items():
        grouped.setdefault(text_hash(text), []).append(design_id)

    # Sort before shuffling so the result depends on the seed alone, not on
    # dict ordering or filesystem enumeration order.
    keys = sorted(grouped)
    rng = _Random(seed)
    rng.shuffle(keys)

    n = len(keys)
    n_train = int(round(n * fractions[0]))
    n_dev = int(round(n * fractions[1]))
    chunks = {
        "train": keys[:n_train],
        "dev": keys[n_train : n_train + n_dev],
        "test": keys[n_train + n_dev :],
    }

    splits = SplitSet(seed=seed)
    for name in SPLIT_NAMES:
        hashes = chunks[name]
        splits.splits[name] = Split(
            name=name,
            design_ids=sorted(d for h in hashes for d in grouped[h]),
            text_hashes=sorted(hashes),
        )
    assert_disjoint(splits)
    return splits


class _Random:
    """Deterministic shuffler that does not depend on numpy or global state."""

    def __init__(self, seed: int) -> None:
        import random

        self._rng = random.Random(seed)

    def shuffle(self, items: list) -> None:
        self._rng.shuffle(items)


def load_requirements_by_design(root: str | Path) -> dict[str, str]:
    """Read ``{design_id: requirement_text}`` from a generated dataset."""
    designs: dict[str, str] = {}
    for path in sorted(Path(root).glob("designs/design_*/requirements.json")):
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        text = payload.get("text")
        if text:
            designs[path.parent.name] = text
    return designs


def requirements_for(splits: SplitSet, name: str, designs: dict[str, str]) -> list[str]:
    """Requirement texts belonging to one split, de-duplicated and ordered."""
    wanted = set(splits[name].design_ids)
    seen: dict[str, None] = {}
    for design_id in sorted(wanted):
        text = designs.get(design_id)
        if text is not None:
            seen.setdefault(text, None)
    return list(seen)
