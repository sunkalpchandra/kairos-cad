"""Offline evaluation of a trained policy against held-out expert steps.

A single accuracy number hides the thing that matters. The action distribution
is heavily skewed — a quarter of all steps are ADD_CIRCLE — so a policy that
learned only the majority operation can look competent. These breakdowns make
that visible: per-operation recall exposes collapse onto frequent classes, and
per-family accuracy exposes a policy that handles plates but not flanges.

``majority_baseline`` is reported alongside, because "82% accurate" means
something different when always guessing ADD_CIRCLE scores 26%.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from kairos.rl.action_space import OPERATIONS
from kairos.training.bc_dataset import TrajectoryDataset, collate
from kairos.training.bc_train import _slots_used_by_operation


def _rows_of(subset: Subset | TrajectoryDataset) -> np.ndarray:
    """Row indices a subset covers, in the parent dataset's numbering."""
    if isinstance(subset, Subset):
        return np.asarray(subset.indices, dtype=np.int64)
    return np.arange(len(subset), dtype=np.int64)


def _parent_of(subset: Subset | TrajectoryDataset) -> TrajectoryDataset:
    return subset.dataset if isinstance(subset, Subset) else subset


@torch.no_grad()
def evaluate(
    model,
    subset: Subset | TrajectoryDataset,
    device: torch.device | str = "cpu",
    batch_size: int = 256,
) -> dict[str, Any]:
    """Score ``model`` over ``subset``; returns a JSON-serializable report."""
    parent = _parent_of(subset)
    rows = _rows_of(subset)
    if len(rows) == 0:
        return {"steps": 0}

    model = model.to(device).eval()
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    # Same masking the trainer uses: slots an operation never decodes carry no
    # meaningful target, and averaging over them would report a large error for
    # predictions that are in fact irrelevant.
    slot_mask = _slots_used_by_operation().to(device)

    predictions: list[np.ndarray] = []
    truths: list[np.ndarray] = []
    top3_hits: list[np.ndarray] = []
    parameter_errors: list[np.ndarray] = []
    parameter_error_sum = parameter_slot_count = 0.0
    own_error_sum = own_slot_count = 0.0

    for rows_batch in loader:
        batch = rows_batch.to(device)
        outputs = model(**batch.model_inputs())
        logits = outputs["operation_logits"]

        predictions.append(logits.argmax(dim=-1).cpu().numpy())
        truths.append(batch.operation.cpu().numpy())
        k = min(3, logits.shape[-1])
        hits = (logits.topk(k, dim=-1).indices == batch.operation[:, None]).any(dim=-1)
        top3_hits.append(hits.cpu().numpy())
        slots = slot_mask[batch.operation]
        absolute = (outputs["parameters"] - batch.parameters).abs() * slots
        # Micro-average over slots, matching how the trainer computes it.
        # Averaging per row instead would enter every zero-slot operation
        # (FINISH_DESIGN) as an exact 0.0 sample and drag the mean down.
        parameter_error_sum += float(absolute.sum())
        parameter_slot_count += float(slots.sum())
        parameter_errors.append(
            (absolute.sum(dim=-1) / slots.sum(dim=-1).clamp(min=1.0)).cpu().numpy()
        )

        # The parameter head above was conditioned on the EXPERT's operation.
        # That is right for the training loss and wrong as a measure of the
        # deployed policy, which must condition on its own choice; reporting
        # only the teacher-forced number understates real error.
        own = model(**{**batch.model_inputs(), "operation": logits.argmax(dim=-1)})
        own_slots = slot_mask[logits.argmax(dim=-1)]
        own_error_sum += float(((own["parameters"] - batch.parameters).abs() * own_slots).sum())
        own_slot_count += float(own_slots.sum())

    predicted = np.concatenate(predictions)
    actual = np.concatenate(truths)
    top3 = np.concatenate(top3_hits)
    errors = np.concatenate(parameter_errors)

    counts = Counter(actual.tolist())
    majority = max(counts.values()) / len(actual) if counts else 0.0

    per_operation: dict[str, dict[str, Any]] = {}
    for op_id, support in sorted(counts.items(), key=lambda kv: -kv[1]):
        selected = actual == op_id
        predicted_as = predicted == op_id
        recall = float((predicted[selected] == op_id).mean())
        precision = float((actual[predicted_as] == op_id).mean()) if predicted_as.any() else 0.0
        per_operation[OPERATIONS[op_id].value] = {
            "support": int(support),
            "recall": round(recall, 4),
            "precision": round(precision, 4),
        }

    per_family: dict[str, dict[str, Any]] = {}
    families = parent.families
    if families:
        family_ids = parent.family_index[rows]
        grouped: dict[int, list[int]] = defaultdict(list)
        for position, family_id in enumerate(family_ids):
            grouped[int(family_id)].append(position)
        for family_id, positions in sorted(grouped.items()):
            index = np.asarray(positions)
            name = families[family_id] if family_id < len(families) else str(family_id)
            per_family[name] = {
                "steps": len(index),
                "accuracy": round(float((predicted[index] == actual[index]).mean()), 4),
                "parameter_mae": round(float(errors[index].mean()), 4),
            }

    return {
        "steps": int(len(actual)),
        "operation_accuracy": round(float((predicted == actual).mean()), 4),
        "operation_top3": round(float(top3.mean()), 4),
        "majority_baseline": round(float(majority), 4),
        # Micro-averaged, so this is directly comparable to the training loss.
        "parameter_mae": round(parameter_error_sum / max(parameter_slot_count, 1.0), 4),
        "parameter_mae_self_conditioned": round(
            own_error_sum / max(own_slot_count, 1.0), 4
        ),
        "per_operation": per_operation,
        "per_family": per_family,
    }


def format_report(report: dict[str, Any]) -> str:
    """Render :func:`evaluate` output as readable text."""
    if not report.get("steps"):
        return "no evaluation steps"
    lines = [
        f"steps: {report['steps']}",
        f"operation accuracy: {report['operation_accuracy']:.3f} "
        f"(top-3 {report['operation_top3']:.3f}, "
        f"majority baseline {report['majority_baseline']:.3f})",
        f"parameter MAE: {report['parameter_mae']:.4f} teacher-forced, "
        f"{report.get('parameter_mae_self_conditioned', float('nan')):.4f} "
        "conditioned on the policy's own operation",
        "",
        f"{'operation':>18}  {'support':>7}  {'recall':>7}  {'precision':>9}",
    ]
    for name, row in report["per_operation"].items():
        lines.append(
            f"{name:>18}  {row['support']:>7}  {row['recall']:>7.3f}  {row['precision']:>9.3f}"
        )
    if report["per_family"]:
        lines += ["", f"{'family':>18}  {'steps':>7}  {'accuracy':>8}  {'param MAE':>9}"]
        for name, row in report["per_family"].items():
            lines.append(
                f"{name:>18}  {row['steps']:>7}  {row['accuracy']:>8.3f}  "
                f"{row['parameter_mae']:>9.4f}"
            )
    return "\n".join(lines)
