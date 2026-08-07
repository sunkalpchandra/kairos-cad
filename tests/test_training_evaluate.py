"""Evaluation tests (skipped without the optional torch extra)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="requires the 'learn' extra")

from torch import nn  # noqa: E402

from kairos.actions.schema import Operation  # noqa: E402
from kairos.rl.action_space import MAX_TARGETS, OPERATIONS, PARAM_SLOTS  # noqa: E402
from kairos.training.bc_dataset import TrajectoryDataset  # noqa: E402
from kairos.training.evaluate import evaluate, format_report  # noqa: E402

PAD_ID = OPERATIONS.index(Operation.PAD)
FILLET_ID = OPERATIONS.index(Operation.FILLET)


class StubPolicy(nn.Module):
    """Predicts a fixed operation and echoes a fixed parameter vector."""

    def __init__(self, operation_id, parameters=None):
        super().__init__()
        self.operation_id = operation_id
        self.parameters_out = parameters

    def forward(self, **batch):
        rows = batch["numeric"].shape[0]
        logits = torch.full((rows, len(OPERATIONS)), -10.0)
        logits[:, self.operation_id] = 10.0
        params = (
            self.parameters_out.expand(rows, PARAM_SLOTS)
            if self.parameters_out is not None
            else torch.full((rows, PARAM_SLOTS), 0.5)
        )
        return {
            "operation_logits": logits,
            "parameters": params,
            "target_logits": torch.zeros(rows, MAX_TARGETS),
        }

    def to(self, *args, **kwargs):
        return self


def _arrays(operations, families=None, parameters=None):
    n = len(operations)
    families = families if families is not None else [0] * n
    return {
        "token_ids": np.ones((n, 8), dtype=np.int64),
        "token_values": np.zeros((n, 8), dtype=np.float32),
        "token_mask": np.ones((n, 8), dtype=np.int64),
        "numeric": np.zeros((n, 24), dtype=np.float32),
        "history": np.zeros((n, 4), dtype=np.int64),
        "operation_mask": np.ones((n, len(OPERATIONS)), dtype=np.int64),
        "operation": np.asarray(operations, dtype=np.int64),
        "parameters": (
            np.asarray(parameters, dtype=np.float32)
            if parameters is not None
            else np.zeros((n, PARAM_SLOTS), dtype=np.float32)
        ),
        "design_index": np.arange(n, dtype=np.int64),
        "family_index": np.asarray(families, dtype=np.int64),
        "families": np.asarray(["plate", "flange"], dtype=object),
    }


def test_perfect_predictions_score_one():
    dataset = TrajectoryDataset(_arrays([PAD_ID] * 6))
    report = evaluate(StubPolicy(PAD_ID), dataset)
    assert report["operation_accuracy"] == 1.0
    assert report["per_operation"]["PAD"]["recall"] == 1.0


def test_majority_baseline_exposes_class_collapse():
    """A policy that only ever says PAD must not look good by accident."""
    labels = [PAD_ID] * 7 + [FILLET_ID] * 3
    report = evaluate(StubPolicy(PAD_ID), TrajectoryDataset(_arrays(labels)))
    assert report["majority_baseline"] == pytest.approx(0.7)
    assert report["operation_accuracy"] == pytest.approx(0.7)
    # The collapse is visible in the minority class's recall.
    assert report["per_operation"]["FILLET"]["recall"] == 0.0


def test_per_family_steps_sum_to_total():
    labels = [PAD_ID] * 6
    report = evaluate(StubPolicy(PAD_ID), TrajectoryDataset(_arrays(labels, [0, 0, 0, 1, 1, 1])))
    assert sum(row["steps"] for row in report["per_family"].values()) == report["steps"]
    assert set(report["per_family"]) == {"plate", "flange"}


def test_parameter_mae_ignores_slots_the_operation_does_not_use():
    """FILLET reads slot 0 only; error in unused slots must not be counted."""
    truth = np.zeros((4, PARAM_SLOTS), dtype=np.float32)
    dataset = TrajectoryDataset(_arrays([FILLET_ID] * 4, parameters=truth))
    prediction = torch.zeros(1, PARAM_SLOTS)
    prediction[0, PARAM_SLOTS - 1] = 1.0  # a full-scale error in an unused slot
    report = evaluate(StubPolicy(FILLET_ID, prediction), dataset)
    assert report["parameter_mae"] == pytest.approx(0.0, abs=1e-6)


def test_parameter_mae_counts_slots_the_operation_uses():
    truth = np.zeros((4, PARAM_SLOTS), dtype=np.float32)
    dataset = TrajectoryDataset(_arrays([FILLET_ID] * 4, parameters=truth))
    prediction = torch.zeros(1, PARAM_SLOTS)
    prediction[0, 0] = 1.0  # FILLET's radius slot
    report = evaluate(StubPolicy(FILLET_ID, prediction), dataset)
    assert report["parameter_mae"] == pytest.approx(1.0, abs=1e-6)


def test_empty_subset_reports_no_steps():
    from torch.utils.data import Subset

    dataset = TrajectoryDataset(_arrays([PAD_ID] * 3))
    assert evaluate(StubPolicy(PAD_ID), Subset(dataset, []))["steps"] == 0
    assert format_report({"steps": 0}) == "no evaluation steps"


def test_report_formats_without_raising():
    dataset = TrajectoryDataset(_arrays([PAD_ID, FILLET_ID], [0, 1]))
    text = format_report(evaluate(StubPolicy(PAD_ID), dataset))
    assert "operation accuracy" in text and "majority baseline" in text
