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
