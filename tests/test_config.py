"""Pure-python tests of YAML config loading."""

import pytest

from kairos.config import (
    environment_kwargs_from,
    load_config,
    reward_weights_from,
)


def test_default_config_loads_and_builds():
    config = load_config()
    weights = reward_weights_from(config)
    assert weights.finish_success == 5.0
    assert weights.invalid_action == -0.5
    kwargs = environment_kwargs_from(config)
    assert kwargs["max_steps"] == 40
    assert "L-bracket" in kwargs["requirement"]
    assert kwargs["reward_weights"] is weights or kwargs["reward_weights"] == weights


def test_partial_reward_section_uses_defaults(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("reward:\n  invalid_action: -2.0\n")
    weights = reward_weights_from(load_config(path))
    assert weights.invalid_action == -2.0
    assert weights.finish_success == 5.0  # default preserved


def test_unknown_reward_key_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("reward:\n  bonus_for_vibes: 3.0\n")
    with pytest.raises(ValueError, match="unknown reward weight"):
        reward_weights_from(load_config(path))


def test_non_mapping_config_rejected(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(path)


def test_model_and_bc_sections_reach_the_training_dataclasses():
    """The YAML must actually drive training, not sit there decoratively."""
    pytest.importorskip("torch", reason="requires the 'learn' extra")
    from kairos.training.bc_train import configs_from

    model, train, out_dir = configs_from(load_config())
    assert model.embed_dim == 128
    assert isinstance(model.vision_widths, tuple)  # YAML lists must become tuples
    assert train.epochs > 0 and 0.0 < train.val_fraction < 1.0
    assert out_dir


def test_unknown_learning_keys_are_rejected():
    pytest.importorskip("torch", reason="requires the 'learn' extra")
    from kairos.training.bc_train import configs_from

    with pytest.raises(ValueError, match="unknown model config keys"):
        configs_from({"model": {"embed_dim": 8, "typo_here": 1}})
    with pytest.raises(ValueError, match="unknown behavioral_cloning keys"):
        configs_from({"behavioral_cloning": {"epochs": 1, "lr": 0.1}})


def test_top_level_seed_flows_into_training():
    pytest.importorskip("torch", reason="requires the 'learn' extra")
    from kairos.training.bc_train import configs_from

    _, train, _ = configs_from({"seed": 42})
    assert train.seed == 42


def test_unknown_top_level_sections_are_rejected():
    """A typo like 'rewrad:' silently left a run on default weights."""
    from kairos.config import validate_sections

    validate_sections(load_config())  # the real config must pass
    with pytest.raises(ValueError, match="unknown config sections"):
        validate_sections({"rewrad": {"valid_sketch": 9.0}})


def test_unknown_environment_keys_are_rejected():
    from kairos.config import environment_kwargs_from

    with pytest.raises(ValueError, match="unknown environment keys"):
        environment_kwargs_from({"environment": {"max_stpes": 5}})


def test_optimization_section_is_known_and_complete():
    config = load_config()
    section = config["optimization"]
    assert section["surrogate_degree"] == 3  # a quadratic gets thickness wrong
    assert section["min_thickness"] > 0
    assert section["samples"] >= 8  # the surrogate needs at least this many


def test_dataset_section_is_rejected_now_that_nothing_reads_it():
    """A whitelisted section read by nothing is worse than an unknown one.

    `dataset:` validated fine while scripts/generate_dataset.sh took its counts
    from argv, so a config restricting families would pass and then generate the
    default dataset anyway.
    """
    import pytest

    from kairos.config import validate_sections

    with pytest.raises(ValueError, match="unknown config sections"):
        validate_sections({"dataset": {"designs": 10}})


def test_shipped_config_has_no_section_nothing_reads():
    from pathlib import Path

    from kairos.config import load_config, validate_sections

    root = Path(__file__).resolve().parent.parent
    validate_sections(load_config(root / "configs" / "default.yaml"))
