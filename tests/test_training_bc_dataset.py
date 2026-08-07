"""BC dataset tests (skipped without the optional torch extra)."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="requires the 'learn' extra")

from kairos.rl.action_space import OPERATIONS, PARAM_SLOTS  # noqa: E402
from kairos.training.bc_dataset import (  # noqa: E402
    TrajectoryDataset,
    build_examples,
    collate,
)

REQUIREMENT = "Design a rectangular plate 100 x 60 x 6 mm with 4 holes of 5 mm diameter"


def _write_trajectory(root, design_id, actions, states, histories):
    design_dir = root / "designs" / f"design_{design_id:06d}"
    design_dir.mkdir(parents=True)
    (design_dir / "trajectory.json").write_text(
        json.dumps(
            {
                "requirement": REQUIREMENT,
                "states": states,
                "actions": actions,
                "step_summaries": [{"feature_history": h} for h in histories],
                "family": "plate",
                "design_id": design_id,
            }
        )
    )
    return design_dir


def _state(marker: float, sketch=False, geometry=False, solid=False):
    """A 24-dim state carrying a step marker plus the legality-relevant flags.

    Slot 2 (log_volume) is unused by the flag reconstruction, so it is free to
    tag which step produced the state.
    """
    vector = [0.0] * 24
    vector[2] = marker
    vector[0] = 1.0 if solid else 0.0  # has_solid
    vector[13] = 1.0 if sketch else 0.0  # sketch_open
    vector[14] = 1.0 / 12.0 if geometry else 0.0  # sketch_geometry
    return vector


@pytest.fixture
def dataset_root(tmp_path):
    actions = [
        {"operation": "CREATE_SKETCH", "target": None, "parameters": {"plane": "XY"}},
        {
            "operation": "ADD_RECTANGLE",
            "target": None,
            "parameters": {"x": 0.0, "y": 0.0, "width": 40.0, "height": 20.0},
        },
        {
            "operation": "PAD",
            "target": None,
            "parameters": {"length": 10.0, "midplane": False, "reversed": False},
        },
    ]
    _write_trajectory(
        tmp_path,
        1,
        actions,
        # As the engine really evolves: sketch opened, then geometry in it,
        # then a solid once the pad runs.
        states=[
            _state(0.1, sketch=True),
            _state(0.2, sketch=True, geometry=True),
            _state(0.3, solid=True),
        ],
        histories=[[], [], ["Pad"]],
    )
    return tmp_path


def test_states_are_shifted_so_the_label_cannot_leak(dataset_root):
    """states[i] is recorded AFTER action i, so step i must read states[i-1]."""
    arrays, stats = build_examples(dataset_root)
    assert stats.steps_kept == 3
    # Step 0 sees the empty document, not the post-CREATE_SKETCH state.
    assert arrays["numeric"][0][2] == pytest.approx(0.0)
    # Steps 1 and 2 see the *previous* recorded state (markers 0.1 and 0.2).
    assert arrays["numeric"][1][2] == pytest.approx(0.1)
    assert arrays["numeric"][2][2] == pytest.approx(0.2)


def test_history_is_shifted_the_same_way(dataset_root):
    """The Pad in step 2's summary must not appear in step 2's input."""
    arrays, _ = build_examples(dataset_root, history_length=8)
    assert arrays["history"][2].sum() == 0  # history before PAD is empty


def test_labels_match_the_expert_actions(dataset_root):
    arrays, _ = build_examples(dataset_root)
    names = [OPERATIONS[i].value for i in arrays["operation"]]
    assert names == ["CREATE_SKETCH", "ADD_RECTANGLE", "PAD"]
    assert arrays["parameters"].shape == (3, PARAM_SLOTS)
    assert ((arrays["parameters"] >= 0.0) & (arrays["parameters"] <= 1.0)).all()


def test_expert_action_is_always_legal_under_the_reconstructed_mask(dataset_root):
    arrays, stats = build_examples(dataset_root)
    for row, label in zip(arrays["operation_mask"], arrays["operation"], strict=True):
        assert row[label] == 1
    assert stats.expert_action_illegal == 0


def test_irregular_polygons_are_dropped_and_counted(tmp_path):
    """Fitting an L profile to the nearest hexagon would be fabricated data."""
    actions = [
        {"operation": "CREATE_SKETCH", "target": None, "parameters": {"plane": "XZ"}},
        {
            "operation": "ADD_POLYGON",
            "target": None,
            "parameters": {"points": [[0, 0], [80, 0], [80, 6], [6, 6], [6, 60], [0, 60]]},
        },
    ]
    _write_trajectory(tmp_path, 2, actions, [_state(0.1), _state(0.2)], [[], []])
    _, stats = build_examples(tmp_path)
    assert stats.steps_seen == 2
    assert stats.steps_kept == 1
    assert stats.dropped_unrepresentable == 1
    assert stats.coverage == pytest.approx(0.5)


def test_regular_polygons_are_kept(tmp_path):
    import math

    hexagon = [
        [round(10 * math.cos(i * math.pi / 3), 6), round(10 * math.sin(i * math.pi / 3), 6)]
        for i in range(6)
    ]
    actions = [{"operation": "ADD_POLYGON", "target": None, "parameters": {"points": hexagon}}]
    _write_trajectory(tmp_path, 3, actions, [_state(0.1)], [[]])
    _, stats = build_examples(tmp_path)
    assert stats.steps_kept == 1
    assert stats.dropped_unrepresentable == 0


def test_dataset_and_collate_round_trip(dataset_root):
    arrays, _ = build_examples(dataset_root)
    dataset = TrajectoryDataset(arrays)
    assert len(dataset) == 3
    batch = collate([dataset[i] for i in range(3)])
    assert batch.operation.shape == (3,)
    assert batch.parameters.shape == (3, PARAM_SLOTS)
    assert set(batch.model_inputs()) == {
        "token_ids", "token_values", "token_mask", "numeric",
        "history", "operation_mask", "operation",
    }


def test_dataset_rejects_a_drifted_numeric_width(dataset_root):
    arrays, _ = build_examples(dataset_root)
    arrays["numeric"] = np.zeros((3, 12), dtype=np.float32)
    with pytest.raises(ValueError, match="expected 24"):
        TrajectoryDataset(arrays)


def test_design_index_supports_leak_free_splits(dataset_root):
    arrays, _ = build_examples(dataset_root)
    assert set(TrajectoryDataset(arrays).design_index.tolist()) == {0}


def test_empty_root_yields_an_empty_dataset(tmp_path):
    arrays, stats = build_examples(tmp_path)
    assert stats.steps_kept == 0
    assert len(TrajectoryDataset(arrays)) == 0
