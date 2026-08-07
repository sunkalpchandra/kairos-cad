"""Replay tests (skipped without the optional torch extra)."""

import json

import pytest

torch = pytest.importorskip("torch", reason="requires the 'learn' extra")

from torch import nn  # noqa: E402

from kairos.actions.parameters import validate_action  # noqa: E402
from kairos.actions.schema import Operation  # noqa: E402
from kairos.rl.action_space import MAX_TARGETS, OPERATIONS, PARAM_SLOTS  # noqa: E402
from kairos.training.rollout import (  # noqa: E402
    format_replay,
    predict_action,
    replay_trajectory,
)

REQUIREMENT = "Design a rectangular mounting plate 100 x 60 x 6 mm with 4 holes of 5 mm diameter"


class StubPolicy(nn.Module):
    """Always prefers one operation; carries a parameter so device lookup works."""

    def __init__(self, operation):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.operation_id = OPERATIONS.index(operation)

    def forward(self, **batch):
        rows = batch["numeric"].shape[0]
        logits = torch.zeros(rows, len(OPERATIONS))
        logits[:, self.operation_id] = 20.0
        return {
            "operation_logits": logits,
            "parameters": torch.full((rows, PARAM_SLOTS), 0.5),
            "target_logits": torch.zeros(rows, MAX_TARGETS),
        }


def test_predicts_a_valid_action_from_an_empty_document():
    action, probabilities = predict_action(
        StubPolicy(Operation.CREATE_SKETCH), REQUIREMENT
    )
    assert action.operation is Operation.CREATE_SKETCH
    validate_action(action)  # must not raise
    assert probabilities.shape == (len(OPERATIONS),)
    assert probabilities.sum() == pytest.approx(1.0, abs=1e-5)


def test_masking_blocks_an_operation_the_empty_document_forbids():
    """PAD needs a sketch with geometry; from empty it must not be selectable."""
    model = StubPolicy(Operation.PAD)
    masked, _ = predict_action(model, REQUIREMENT, apply_mask=True)
    assert masked.operation is not Operation.PAD
    unmasked, _ = predict_action(model, REQUIREMENT, apply_mask=False)
    assert unmasked.operation is Operation.PAD


def test_replay_reports_per_step_agreement(tmp_path):
    trajectory = {
        "requirement": REQUIREMENT,
        "design_id": 7,
        "family": "plate",
        "states": [[0.0] * 24, [0.0] * 24],
        "actions": [
            {"operation": "CREATE_SKETCH", "target": None, "parameters": {"plane": "XY"}},
            {"operation": "PAD", "target": None, "parameters": {"length": 5.0}},
        ],
        "step_summaries": [{"feature_history": []}, {"feature_history": ["Pad"]}],
    }
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(trajectory))

    report = replay_trajectory(StubPolicy(Operation.CREATE_SKETCH), path)
    assert report["teacher_forced"] is True
    assert [s["expert"] for s in report["steps"]] == ["CREATE_SKETCH", "PAD"]
    assert report["steps"][0]["agrees"] is True
    assert report["steps"][1]["agrees"] is False  # stub never says PAD
    assert report["agreement"] == pytest.approx(0.5)
    assert "agreement: 0.500" in format_replay(report)


def test_replay_uses_the_previous_state_not_the_current_one(tmp_path):
    """Replay must apply the same anti-leak shift the training data does."""
    marked = [0.0] * 24
    marked[2] = 0.75
    trajectory = {
        "requirement": REQUIREMENT,
        "states": [marked, [0.0] * 24],
        "actions": [
            {"operation": "CREATE_SKETCH", "target": None, "parameters": {"plane": "XY"}},
            {"operation": "CREATE_SKETCH", "target": None, "parameters": {"plane": "XY"}},
        ],
        "step_summaries": [{"feature_history": []}, {"feature_history": []}],
    }
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(trajectory))

    seen = []

    class Recorder(StubPolicy):
        def forward(self, **batch):
            seen.append(float(batch["numeric"][0, 2]))
            return super().forward(**batch)

    replay_trajectory(Recorder(Operation.CREATE_SKETCH), path)
    assert seen[0] == pytest.approx(0.0)  # step 0 reads the empty document
    assert seen[1] == pytest.approx(0.75)  # step 1 reads states[0], not states[1]
