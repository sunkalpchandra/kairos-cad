"""YAML experiment configuration loading.

Configs are plain dicts with a few typed accessors; experiments pass a path
(e.g. ``configs/default.yaml``) and everything downstream, environment,
reward weights, dataset generation, is constructed from it. No experiment
parameter may be hardcoded at a call site.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from kairos.rl.rewards import RewardWeights

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load a YAML config; ``None`` loads ``configs/default.yaml``."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(config_path) as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"config {config_path} did not parse to a mapping")
    validate_sections(data)
    return data


def reward_weights_from(config: dict[str, Any]) -> RewardWeights:
    """Build RewardWeights from the ``reward`` section (defaults fill gaps)."""
    section = config.get("reward", {}) or {}
    known = {f for f in RewardWeights.__dataclass_fields__}
    unknown = set(section) - known
    if unknown:
        raise ValueError(f"unknown reward weight keys: {sorted(unknown)}")
    return RewardWeights(**{k: float(v) for k, v in section.items()})


#: Sections a config may declare. A section listed here but read by nothing
#: is worse than an unknown one: `dataset:` sat here while
#: scripts/generate_dataset.sh took its counts from argv, so a config
#: restricting families passed validation and generated the default dataset
#: anyway.
KNOWN_SECTIONS = frozenset(
    {"seed", "environment", "reward", "model", "behavioral_cloning", "ppo", "optimization"}
)
_ENVIRONMENT_KEYS = frozenset({"requirement", "max_steps", "material"})


def validate_sections(config: dict[str, Any]) -> None:
    """Reject unknown top-level sections.

    A typo like ``rewrad:`` otherwise leaves the run silently on default
    weights, an ablation that changes nothing and reports success.
    """
    unknown = set(config) - KNOWN_SECTIONS
    if unknown:
        raise ValueError(
            f"unknown config sections: {sorted(unknown)}; "
            f"known sections are {sorted(KNOWN_SECTIONS)}"
        )


def environment_kwargs_from(config: dict[str, Any]) -> dict[str, Any]:
    """Build KairosCADEnv constructor kwargs from the ``environment`` section."""
    section = config.get("environment", {}) or {}
    unknown = set(section) - _ENVIRONMENT_KEYS
    if unknown:
        raise ValueError(f"unknown environment keys: {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    if "requirement" in section:
        kwargs["requirement"] = str(section["requirement"]).strip()
    if "max_steps" in section:
        kwargs["max_steps"] = int(section["max_steps"])
    if "material" in section:
        kwargs["material"] = str(section["material"])
    kwargs["reward_weights"] = reward_weights_from(config)
    return kwargs
