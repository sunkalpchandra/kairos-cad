"""Behavioral-cloning dataset over recorded expert trajectories.

Three details decide whether this data teaches anything real:

**States are recorded after their action.** ``TrajectoryRecorder`` observes in
the post-action callback, so ``states[i]`` already contains the effect of
``actions[i]``. Training on that pair would leak the answer — the model would
read "a pad exists" and predict PAD. Step ``i`` is therefore supervised from
``states[i - 1]``, and step 0 from the encoding of an empty document.

**Not every expert action is representable.** The codec expresses ADD_POLYGON
only as a regular n-gon, while the L, U, corner, spacer, flange, and support
recipes sketch irregular profiles. Those steps are dropped rather than
approximated, and the count is reported — silently fitting them to the nearest
hexagon would train the policy toward a shape the expert never drew.

**Targets are not supervised.** ``encode`` cannot recover a target *index*
without the live edge/face list, which trajectories do not record. Supervising
the recorded 0 would teach "always pick the first edge", so the target head is
left to RL in Phase 5.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from kairos.actions.masking import StateFlags, operation_mask
from kairos.actions.schema import Action, Operation
from kairos.language import parse_requirement
from kairos.language import tokenizer as tk
from kairos.representation.feature_encoder import encode_history
from kairos.representation.numerical_encoder import ENCODING_DIM, encode_numeric
from kairos.rl.action_space import OPERATIONS, PARAM_SLOTS, UnrepresentableAction, encode

#: Indices into the frozen numeric vector used to rebuild legality flags.
_HAS_SOLID, _FACES, _EDGES, _FEATURES = 0, 9, 10, 12
_SKETCH_OPEN, _SKETCH_GEOMETRY = 13, 14


@dataclass
class BuildStats:
    """What the builder kept and what it had to drop."""

    trajectories: int = 0
    steps_seen: int = 0
    steps_kept: int = 0
    dropped_unrepresentable: int = 0
    dropped_unknown_operation: int = 0
    expert_action_illegal: int = 0

    @property
    def coverage(self) -> float:
        return self.steps_kept / self.steps_seen if self.steps_seen else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectories": self.trajectories,
            "steps_seen": self.steps_seen,
            "steps_kept": self.steps_kept,
            "coverage": round(self.coverage, 4),
            "dropped_unrepresentable": self.dropped_unrepresentable,
            "dropped_unknown_operation": self.dropped_unknown_operation,
            "expert_action_illegal": self.expert_action_illegal,
        }


def _flags_from_numeric(numeric: np.ndarray) -> StateFlags:
    """Rebuild legality flags from a recorded numeric state.

    The masking layer normally reads a live engine; recorded trajectories keep
    the same information in the frozen numeric vector (counts are normalized,
    so "> 0" is the presence test).
    """
    return StateFlags(
        has_sketch=numeric[_SKETCH_OPEN] > 0.5,
        sketch_has_geometry=numeric[_SKETCH_GEOMETRY] > 0.0,
        has_solid=numeric[_HAS_SOLID] > 0.5,
        has_edges=numeric[_EDGES] > 0.0,
        has_faces=numeric[_FACES] > 0.0,
        has_features=numeric[_FEATURES] > 0.0,
    )


def build_examples(
    root: str | Path = "dataset",
    max_text_length: int = 64,
    history_length: int = 16,
    max_steps: int = 64,
    limit: int | None = None,
) -> tuple[dict[str, np.ndarray], BuildStats]:
    """Assemble every supervisable step under ``root`` into numpy arrays."""
    root = Path(root)
    paths = sorted(root.glob("designs/design_*/trajectory.json"))
    if limit is not None:
        paths = paths[:limit]

    stats = BuildStats()
    columns: dict[str, list] = {
        key: []
        for key in (
            "token_ids", "token_values", "token_mask", "numeric", "history",
            "operation_mask", "operation", "parameters", "design_index",
        )
    }
    operation_index = {op: i for i, op in enumerate(OPERATIONS)}

    for design_index, path in enumerate(paths):
        trajectory = json.loads(path.read_text())
        stats.trajectories += 1

        requirement = trajectory["requirement"]
        spec = parse_requirement(requirement)
        ids, values, mask = tk.encode(requirement, max_length=max_text_length)
        states = trajectory["states"]
        summaries = trajectory.get("step_summaries", [])
        # The document is empty before the first action; every .get() in
        # encode_numeric falls back to zero, so {} is exactly that state.
        initial = encode_numeric({}, spec, step=0, max_steps=max_steps)

        for i, raw_action in enumerate(trajectory["actions"]):
            stats.steps_seen += 1
            try:
                operation = Operation(raw_action["operation"])
            except ValueError:
                stats.dropped_unknown_operation += 1
                continue
            if operation not in operation_index:
                stats.dropped_unknown_operation += 1
                continue

            action = Action(
                operation,
                target=raw_action.get("target"),
                parameters=raw_action.get("parameters") or {},
            )
            try:
                _, parameters, _ = encode(action)
            except UnrepresentableAction:
                stats.dropped_unrepresentable += 1
                continue

            numeric = np.asarray(states[i - 1], dtype=np.float32) if i > 0 else initial
            history = summaries[i - 1].get("feature_history", []) if i > 0 else []
            history_ids, _ = encode_history(list(history), max_length=history_length)

            legality = np.asarray(
                operation_mask(_flags_from_numeric(numeric), list(OPERATIONS)), dtype=np.int64
            )
            if not legality[operation_index[operation]]:
                # The expert did it, so it was legal in the live engine; a
                # mismatch means the reconstructed flags are wrong. Count it
                # and keep the step, with the label forced legal.
                stats.expert_action_illegal += 1
                legality[operation_index[operation]] = 1

            columns["token_ids"].append(ids)
            columns["token_values"].append(values)
            columns["token_mask"].append(mask)
            columns["numeric"].append(numeric)
            columns["history"].append(history_ids)
            columns["operation_mask"].append(legality)
            columns["operation"].append(operation_index[operation])
            columns["parameters"].append(parameters)
            columns["design_index"].append(design_index)
            stats.steps_kept += 1

    dtypes = {
        "token_ids": np.int64, "token_values": np.float32, "token_mask": np.int64,
        "numeric": np.float32, "history": np.int64, "operation_mask": np.int64,
        "operation": np.int64, "parameters": np.float32, "design_index": np.int64,
    }
    arrays = {
        key: np.asarray(value, dtype=dtypes[key])
        if value
        else np.zeros((0,), dtype=dtypes[key])
        for key, value in columns.items()
    }
    return arrays, stats


@dataclass
class BCBatch:
    """One collated training batch."""

    token_ids: torch.Tensor
    token_values: torch.Tensor
    token_mask: torch.Tensor
    numeric: torch.Tensor
    history: torch.Tensor
    operation_mask: torch.Tensor
    operation: torch.Tensor
    parameters: torch.Tensor

    def to(self, device: torch.device | str) -> BCBatch:
        return BCBatch(**{k: v.to(device) for k, v in self.__dict__.items()})

    def model_inputs(self) -> dict[str, torch.Tensor]:
        """Keyword arguments for :meth:`KairosVLA.forward` (teacher forced)."""
        return {
            "token_ids": self.token_ids,
            "token_values": self.token_values,
            "token_mask": self.token_mask,
            "numeric": self.numeric,
            "history": self.history,
            "operation_mask": self.operation_mask,
            "operation": self.operation,
        }


class TrajectoryDataset(Dataset):
    """Torch dataset over pre-built arrays."""

    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.arrays = arrays
        self.length = len(arrays["operation"])
        if arrays["numeric"].size and arrays["numeric"].shape[1] != ENCODING_DIM:
            raise ValueError(
                f"numeric states are {arrays['numeric'].shape[1]}-dim, expected {ENCODING_DIM}"
            )
        if arrays["parameters"].size and arrays["parameters"].shape[1] != PARAM_SLOTS:
            raise ValueError(
                f"parameters are {arrays['parameters'].shape[1]}-wide, expected {PARAM_SLOTS}"
            )

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            key: torch.from_numpy(np.asarray(self.arrays[key][index]))
            for key in (
                "token_ids", "token_values", "token_mask", "numeric",
                "history", "operation_mask", "operation", "parameters",
            )
        }

    @property
    def design_index(self) -> np.ndarray:
        """Which design each row came from (for leak-free train/val splits)."""
        return self.arrays["design_index"]


def collate(rows: list[dict[str, torch.Tensor]]) -> BCBatch:
    """Stack per-row tensors into a :class:`BCBatch`."""
    stacked = {key: torch.stack([row[key] for row in rows]) for key in rows[0]}
    return BCBatch(**stacked)
