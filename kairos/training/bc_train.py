"""Behavioral cloning: fit :class:`KairosVLA` to the expert trajectories.

The loss is the sum of two supervised terms:

- **operation**: cross-entropy over legal operations. Illegal ones already
  carry ``-1e9`` logits, so they contribute no gradient.
- **parameters**: mean squared error in the codec's normalized [0, 1] space,
  applied only to the slots the predicted operation actually uses. Averaging
  over all six slots would teach every operation to emit 0.5 in the slots it
  ignores, and that noise competes with the slots that matter.

The target head is not supervised — trajectories do not record the live
edge/face list an index would refer to (see :mod:`kairos.training.bc_dataset`).

Splits are **by design, not by step**: steps from one design are highly
correlated, so a random step-level split would put a design's step 3 in train
and its step 4 in validation and report memorization as generalization.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from kairos.models.vla import KairosVLA, VLAConfig
from kairos.rl.action_space import OPERATIONS, PARAM_SLOTS, decode
from kairos.training.bc_dataset import TrajectoryDataset, collate

#: Non-empty pools so target-taking operations decode without tripping the
#: "no target available" path while their slots are probed.
_PROBE_TARGETS = {"edges": ["Edge1"], "faces": ["Face1"], "features": ["Pad"]}


def _slots_used_by_operation() -> torch.Tensor:
    """``[n_operations, PARAM_SLOTS]`` mask of slots each operation decodes.

    Probed from the codec rather than read off the action schema: the two do
    not agree (ADD_POLYGON takes one schema parameter, ``points``, but consumes
    five slots for centre, radius, sides, and rotation). Perturbing one slot at
    a time and watching the decoded parameters is self-maintaining — change a
    range in the codec and this mask follows.
    """
    mask = torch.zeros(len(OPERATIONS), PARAM_SLOTS)
    base = np.full(PARAM_SLOTS, 0.3)
    for index in range(len(OPERATIONS)):
        reference = decode(index, base, 0, _PROBE_TARGETS).parameters
        for slot in range(PARAM_SLOTS):
            probe = base.copy()
            probe[slot] = 0.8  # crosses every boolean threshold and bucket edge
            if decode(index, probe, 0, _PROBE_TARGETS).parameters != reference:
                mask[index, slot] = 1.0
    return mask


@dataclass
class TrainConfig:
    """Optimization hyperparameters for a BC run."""

    epochs: int = 12
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    parameter_loss_weight: float = 1.0
    val_fraction: float = 0.15
    seed: int = 0
    device: str = "auto"
    grad_clip: float = 1.0
    warmup_fraction: float = 0.05
    #: "none" or "inverse_sqrt". The expert action mix is heavily skewed (a
    #: quarter of steps are ADD_CIRCLE, while FILLET is under 1%), and an
    #: unweighted objective simply never predicts the rare operations.
    #: Inverse-square-root rather than inverse frequency: full inverse
    #: weighting swings too far and starts costing the common operations.
    class_weighting: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EpochMetrics:
    """One epoch's train and validation numbers."""

    epoch: int
    train_loss: float = 0.0
    val_loss: float = 0.0
    operation_accuracy: float = 0.0
    operation_top3: float = 0.0
    parameter_mae: float = 0.0
    seconds: float = 0.0
    extras: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(data.pop("extras"))
        return data


def _labels_of(subset) -> np.ndarray:
    """Operation labels of a split, read straight from the backing array.

    Indexing the Dataset row by row would build every tensor column for all
    ~10k rows just to read one integer each.
    """
    if isinstance(subset, Subset):
        return np.asarray(subset.dataset.arrays["operation"])[np.asarray(subset.indices)]
    return np.asarray(subset.arrays["operation"])


def configs_from(config: dict[str, Any]) -> tuple[VLAConfig, TrainConfig, str]:
    """Build ``(VLAConfig, TrainConfig, out_dir)`` from a loaded YAML config.

    Validation lives here rather than in :mod:`kairos.config` so the dataclass
    fields stay the single source of truth — and so ``kairos.config`` never has
    to import torch.
    """
    model_section = dict(config.get("model", {}) or {})
    bc_section = dict(config.get("behavioral_cloning", {}) or {})
    out_dir = str(bc_section.pop("out_dir", "runs/bc"))

    unknown_model = set(model_section) - set(VLAConfig.__dataclass_fields__)
    if unknown_model:
        raise ValueError(f"unknown model config keys: {sorted(unknown_model)}")
    unknown_train = set(bc_section) - set(TrainConfig.__dataclass_fields__)
    if unknown_train:
        raise ValueError(f"unknown behavioral_cloning keys: {sorted(unknown_train)}")

    if "seed" in config and "seed" not in bc_section:
        bc_section["seed"] = config["seed"]
    return VLAConfig.from_dict(model_section), TrainConfig(**bc_section), out_dir


def resolve_device(name: str = "auto") -> torch.device:
    """Pick a device, preferring Apple's Metal backend when available."""
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def split_by_design(
    dataset: TrajectoryDataset, val_fraction: float, seed: int
) -> tuple[Subset, Subset]:
    """Split steps into train/val so no design appears in both."""
    designs = np.unique(dataset.design_index)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(designs)
    n_val = max(1, int(round(len(shuffled) * val_fraction))) if len(shuffled) > 1 else 0
    val_designs = set(shuffled[:n_val].tolist())

    val_rows, train_rows = [], []
    for row, design in enumerate(dataset.design_index):
        (val_rows if design in val_designs else train_rows).append(row)
    return Subset(dataset, train_rows), Subset(dataset, val_rows)


class BCTrainer:
    """Trains a :class:`KairosVLA` by behavioral cloning."""

    def __init__(
        self,
        model: KairosVLA,
        config: TrainConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.config = config or TrainConfig()
        self.device = device or resolve_device(self.config.device)
        self.model = model.to(self.device)
        self.slot_mask = _slots_used_by_operation().to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.history: list[EpochMetrics] = []
        self.best_accuracy = -1.0
        self.best_epoch = -1
        self._best_state: dict | None = None
        self.class_weights: torch.Tensor | None = None

    def compute_class_weights(self, labels: np.ndarray) -> torch.Tensor | None:
        """Inverse-sqrt-frequency weights over observed operations.

        Operations absent from the training split keep weight 1.0 — they
        contribute no gradient anyway, and a zero-count denominator would send
        their weight to infinity.
        """
        if self.config.class_weighting == "none":
            return None
        if self.config.class_weighting != "inverse_sqrt":
            raise ValueError(f"unknown class_weighting {self.config.class_weighting!r}")

        counts = np.bincount(np.asarray(labels), minlength=len(OPERATIONS)).astype(np.float64)
        weights = np.ones_like(counts)
        seen = counts > 0
        weights[seen] = 1.0 / np.sqrt(counts[seen])
        weights[seen] *= seen.sum() / weights[seen].sum()  # mean weight 1 over seen classes
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    # ----------------------------------------------------------------- loss

    def compute_loss(self, batch, outputs) -> tuple[torch.Tensor, dict[str, float]]:
        """Cross-entropy on the operation plus masked MSE on its parameters."""
        operation_loss = nn.functional.cross_entropy(
            outputs["operation_logits"], batch.operation, weight=self.class_weights
        )

        slots = self.slot_mask[batch.operation]
        error = (outputs["parameters"] - batch.parameters) ** 2 * slots
        denominator = slots.sum().clamp(min=1.0)
        parameter_loss = error.sum() / denominator

        total = operation_loss + self.config.parameter_loss_weight * parameter_loss
        return total, {
            "operation_loss": float(operation_loss.detach()),
            "parameter_loss": float(parameter_loss.detach()),
        }

    # ------------------------------------------------------------- training

    def _run_epoch(self, loader: DataLoader, train: bool) -> dict[str, float]:
        self.model.train(train)
        totals = {"loss": 0.0, "operation_loss": 0.0, "parameter_loss": 0.0}
        correct = top3 = seen = 0
        absolute_error = 0.0
        slot_total = 0.0

        for rows in loader:
            batch = rows.to(self.device)
            with torch.set_grad_enabled(train):
                outputs = self.model(**batch.model_inputs())
                loss, parts = self.compute_loss(batch, outputs)

            if train:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.config.grad_clip:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()

            size = batch.operation.shape[0]
            seen += size
            totals["loss"] += float(loss.detach()) * size
            for key, value in parts.items():
                totals[key] += value * size

            with torch.no_grad():
                logits = outputs["operation_logits"]
                correct += int((logits.argmax(dim=-1) == batch.operation).sum())
                k = min(3, logits.shape[-1])
                top_k = logits.topk(k, dim=-1).indices
                top3 += int((top_k == batch.operation[:, None]).any(dim=-1).sum())

                slots = self.slot_mask[batch.operation]
                absolute_error += float(
                    ((outputs["parameters"] - batch.parameters).abs() * slots).sum()
                )
                slot_total += float(slots.sum())

        if seen == 0:
            return {"loss": math.nan, "accuracy": math.nan, "top3": math.nan, "mae": math.nan}
        return {
            "loss": totals["loss"] / seen,
            "operation_loss": totals["operation_loss"] / seen,
            "parameter_loss": totals["parameter_loss"] / seen,
            "accuracy": correct / seen,
            "top3": top3 / seen,
            "mae": absolute_error / max(slot_total, 1.0),
        }

    def fit(
        self,
        train_set,
        val_set,
        on_epoch=None,
    ) -> list[EpochMetrics]:
        """Train for ``config.epochs``, returning per-epoch metrics."""
        c = self.config
        torch.manual_seed(c.seed)
        # Weights come from the training split only — deriving them from the
        # whole dataset would leak validation label frequencies into training.
        if c.class_weighting != "none":
            self.class_weights = self.compute_class_weights(_labels_of(train_set))
        train_loader = DataLoader(
            train_set, batch_size=c.batch_size, shuffle=True, collate_fn=collate
        )
        val_loader = DataLoader(
            val_set, batch_size=c.batch_size, shuffle=False, collate_fn=collate
        )

        for epoch in range(1, c.epochs + 1):
            started = time.perf_counter()
            train_stats = self._run_epoch(train_loader, train=True)
            with torch.no_grad():
                val_stats = self._run_epoch(val_loader, train=False)

            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_stats["loss"],
                val_loss=val_stats["loss"],
                operation_accuracy=val_stats["accuracy"],
                operation_top3=val_stats["top3"],
                parameter_mae=val_stats["mae"],
                seconds=time.perf_counter() - started,
                extras={
                    "train_accuracy": train_stats["accuracy"],
                    "val_operation_loss": val_stats.get("operation_loss", math.nan),
                    "val_parameter_loss": val_stats.get("parameter_loss", math.nan),
                },
            )
            self.history.append(metrics)
            # Keep the best epoch's weights. Reporting the best accuracy
            # while saving the final epoch attributes a number to a
            # checkpoint that never produced it.
            if metrics.operation_accuracy > self.best_accuracy:
                self.best_accuracy = metrics.operation_accuracy
                self.best_epoch = epoch
                self._best_state = {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }
            if on_epoch is not None:
                on_epoch(metrics)

        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)
        return self.history

    # ---------------------------------------------------------- persistence

    def save(self, path: str | Path, extra: dict[str, Any] | None = None) -> Path:
        """Write weights, architecture, and training config to one file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "model_config": self.model.config.to_dict(),
                "train_config": self.config.to_dict(),
                "history": [m.to_dict() for m in self.history],
                **(extra or {}),
            },
            path,
        )
        return path


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> KairosVLA:
    """Rebuild the exact architecture a checkpoint was trained with."""
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    model = KairosVLA(VLAConfig.from_dict(payload["model_config"]))
    model.load_state_dict(payload["model_state"])
    return model.to(device)


def write_report(path: str | Path, payload: dict[str, Any]) -> Path:
    """Persist a run's metrics as JSON next to its checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
