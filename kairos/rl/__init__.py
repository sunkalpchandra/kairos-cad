"""Reinforcement learning: rewards, action codec, environment, and PPO.

The modules here split along a hard line: everything up to and including
``environment`` runs under FreeCAD's interpreter, while the PPO half needs
torch and runs under the system interpreter. They meet over the JSON bridge in
``protocol`` / ``env_server`` / ``env_client``.

Torch-dependent names are exported lazily so importing this package under
FreeCAD's python, which has no torch, keeps working.
"""

from __future__ import annotations

#: Importable anywhere: no torch, no FreeCAD.
_PORTABLE = {
    "protocol": ("PROTOCOL_VERSION", "ProtocolError"),
    "requirements": ("FALLBACK_REQUIREMENTS", "load_requirements", "requirement_pools"),
}

#: Needs torch (the learning half).
_TORCH_ONLY = {
    "buffer": ("RolloutBuffer", "Transition"),
    "collect": ("RolloutCollector", "build_inputs", "summarize_episodes"),
    "env_client": ("RemoteCADEnv", "probe_environment"),
    "evaluate": ("RandomPolicy", "compare_policies", "evaluate_policy"),
    "ppo": ("PPOConfig", "PPOTrainer"),
    "train_loop": ("LoopConfig", "PPOTrainingLoop"),
}

_LOOKUP = {
    name: module
    for group in (_PORTABLE, _TORCH_ONLY)
    for module, names in group.items()
    for name in names
}

__all__ = sorted(_LOOKUP)


def __getattr__(name: str):
    """Resolve exports on demand so torch stays optional."""
    module_name = _LOOKUP.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    try:
        module = importlib.import_module(f"kairos.rl.{module_name}")
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise AttributeError(
            f"{name} lives in kairos.rl.{module_name}, which needs the optional "
            f'torch extra (pip install -e ".[learn]"): {err}'
        ) from err
    return getattr(module, name)
