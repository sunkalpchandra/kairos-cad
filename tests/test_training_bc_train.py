"""BC trainer tests (skipped without the optional torch extra)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="requires the 'learn' extra")

from kairos.actions.schema import Operation  # noqa: E402
from kairos.models.vla import KairosVLA, VLAConfig  # noqa: E402
from kairos.rl.action_space import OPERATIONS, PARAM_SLOTS  # noqa: E402
from kairos.training.bc_dataset import TrajectoryDataset, collate  # noqa: E402
from kairos.training.bc_train import (  # noqa: E402
    BCTrainer,
    TrainConfig,
    _slots_used_by_operation,
    load_checkpoint,
    split_by_design,
)

TINY = VLAConfig(
    embed_dim=16, language_depth=1, language_heads=2, max_text_length=16,
    vision_widths=(8,), fusion_depth=1, fusion_heads=2, hidden_dim=16, dropout=0.0,
)


def _arrays(n_rows=40, n_designs=8, seed=0):
    rng = np.random.default_rng(seed)
    # Two operations, perfectly predictable from one numeric feature: a model
    # that learns anything at all must reach high accuracy here.
    labels = rng.integers(0, 2, size=n_rows)
    numeric = np.zeros((n_rows, 24), dtype=np.float32)
    numeric[:, 0] = labels.astype(np.float32)
    op_ids = np.array(
        [OPERATIONS.index(Operation.PAD), OPERATIONS.index(Operation.FILLET)]
    )[labels]
    return {
        "token_ids": np.ones((n_rows, 16), dtype=np.int64),
        "token_values": np.zeros((n_rows, 16), dtype=np.float32),
        "token_mask": np.ones((n_rows, 16), dtype=np.int64),
        "numeric": numeric,
        "history": np.zeros((n_rows, 8), dtype=np.int64),
        "operation_mask": np.ones((n_rows, len(OPERATIONS)), dtype=np.int64),
        "operation": op_ids.astype(np.int64),
        "parameters": np.full((n_rows, PARAM_SLOTS), 0.25, dtype=np.float32),
        "design_index": rng.integers(0, n_designs, size=n_rows).astype(np.int64),
    }


def test_slot_mask_matches_the_codec():
    mask = _slots_used_by_operation()
    used = {op: int(mask[i].sum()) for i, op in enumerate(OPERATIONS)}
    assert used[Operation.PAD] == 3  # length, reversed, midplane
    assert used[Operation.FILLET] == 1  # radius
    assert used[Operation.ADD_RECTANGLE] == 4  # x, y, width, height
    assert used[Operation.ADD_POLYGON] == 5  # centre, radius, sides, rotation
    assert used[Operation.FINISH_DESIGN] == 0  # takes no parameters


def test_split_never_shares_a_design():
    dataset = TrajectoryDataset(_arrays())
    train, val = split_by_design(dataset, val_fraction=0.25, seed=0)
    train_designs = {dataset.design_index[i] for i in train.indices}
    val_designs = {dataset.design_index[i] for i in val.indices}
    assert train_designs and val_designs
    assert train_designs.isdisjoint(val_designs)


def test_split_is_deterministic_for_a_seed():
    dataset = TrajectoryDataset(_arrays())
    first = split_by_design(dataset, 0.25, seed=7)[1].indices
    second = split_by_design(dataset, 0.25, seed=7)[1].indices
    assert first == second


def test_parameter_loss_ignores_slots_the_operation_does_not_use():
    """FILLET reads slot 0 only; noise in slot 5 must not be penalized."""
    trainer = BCTrainer(KairosVLA(TINY), TrainConfig(device="cpu"))
    dataset = TrajectoryDataset(_arrays(n_rows=4))
    batch = collate([dataset[i] for i in range(4)]).to(trainer.device)

    outputs = {
        "operation_logits": torch.zeros(4, len(OPERATIONS)),
        "parameters": batch.parameters.clone(),
        "target_logits": torch.zeros(4, 64),
    }
    baseline, _ = trainer.compute_loss(batch, outputs)

    perturbed = dict(outputs)
    perturbed["parameters"] = batch.parameters.clone()
    perturbed["parameters"][:, PARAM_SLOTS - 1] += 0.5  # last slot: unused by both ops
    after, _ = trainer.compute_loss(batch, perturbed)
    assert float(after) == pytest.approx(float(baseline), abs=1e-6)


def test_parameter_loss_reacts_to_slots_the_operation_uses():
    trainer = BCTrainer(KairosVLA(TINY), TrainConfig(device="cpu"))
    dataset = TrajectoryDataset(_arrays(n_rows=4))
    batch = collate([dataset[i] for i in range(4)]).to(trainer.device)
    outputs = {
        "operation_logits": torch.zeros(4, len(OPERATIONS)),
        "parameters": batch.parameters.clone(),
        "target_logits": torch.zeros(4, 64),
    }
    baseline, _ = trainer.compute_loss(batch, outputs)
    outputs["parameters"] = batch.parameters.clone()
    outputs["parameters"][:, 0] += 0.5  # slot 0 is read by every operation here
    after, _ = trainer.compute_loss(batch, outputs)
    assert float(after) > float(baseline)


def test_training_reduces_loss_on_a_learnable_task():
    dataset = TrajectoryDataset(_arrays(n_rows=64, n_designs=8, seed=1))
    train, val = split_by_design(dataset, val_fraction=0.25, seed=1)
    trainer = BCTrainer(
        KairosVLA(TINY), TrainConfig(epochs=15, batch_size=16, learning_rate=3e-3, device="cpu")
    )
    history = trainer.fit(train, val)
    assert history[-1].train_loss < history[0].train_loss
    assert history[-1].operation_accuracy >= 0.9


def test_checkpoint_round_trips(tmp_path):
    trainer = BCTrainer(KairosVLA(TINY), TrainConfig(epochs=1, device="cpu"))
    dataset = TrajectoryDataset(_arrays(n_rows=8))
    batch = collate([dataset[i] for i in range(8)]).to(trainer.device)

    trainer.model.eval()
    with torch.no_grad():
        before = trainer.model(**batch.model_inputs())["operation_logits"]

    path = trainer.save(tmp_path / "bc.pt", extra={"note": "test"})
    restored = load_checkpoint(path).eval()
    with torch.no_grad():
        after = restored(**batch.model_inputs())["operation_logits"]
    assert torch.allclose(before, after, atol=1e-6)


def test_empty_validation_set_does_not_crash():
    """A one-design dataset has nothing to hold out; metrics go NaN, not boom."""
    arrays = _arrays(n_rows=8, n_designs=1)
    dataset = TrajectoryDataset(arrays)
    train, val = split_by_design(dataset, val_fraction=0.2, seed=0)
    assert len(val) == 0
    trainer = BCTrainer(KairosVLA(TINY), TrainConfig(epochs=1, batch_size=4, device="cpu"))
    history = trainer.fit(train, val)
    assert np.isnan(history[-1].val_loss)


def test_class_weighting_is_off_by_default():
    trainer = BCTrainer(KairosVLA(TINY), TrainConfig(device="cpu"))
    assert trainer.compute_class_weights(np.array([0, 0, 0, 1])) is None


def test_inverse_sqrt_weights_favour_rare_operations():
    trainer = BCTrainer(
        KairosVLA(TINY), TrainConfig(device="cpu", class_weighting="inverse_sqrt")
    )
    labels = np.array([0] * 100 + [1] * 4)  # one common, one rare
    weights = trainer.compute_class_weights(labels)
    assert weights[1] > weights[0]
    assert float(weights[0]) == pytest.approx(float(weights[1]) * np.sqrt(4 / 100), rel=1e-5)
    # Unobserved operations stay neutral rather than going infinite.
    assert float(weights[5]) == pytest.approx(1.0)


def test_unknown_class_weighting_is_rejected():
    trainer = BCTrainer(KairosVLA(TINY), TrainConfig(device="cpu", class_weighting="bogus"))
    with pytest.raises(ValueError, match="unknown class_weighting"):
        trainer.compute_class_weights(np.array([0, 1]))


def test_class_weights_come_from_the_training_split_only():
    """Deriving them from all rows would leak validation label frequencies."""
    dataset = TrajectoryDataset(_arrays(n_rows=64, n_designs=8, seed=2))
    train, val = split_by_design(dataset, val_fraction=0.25, seed=2)
    trainer = BCTrainer(
        KairosVLA(TINY),
        TrainConfig(epochs=1, batch_size=16, device="cpu", class_weighting="inverse_sqrt"),
    )
    trainer.fit(train, val)
    train_labels = np.array([int(train[i]["operation"]) for i in range(len(train))])
    assert torch.allclose(trainer.class_weights, trainer.compute_class_weights(train_labels))
