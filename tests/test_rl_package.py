"""kairos.rl export surface, must import cleanly in both interpreters."""

import pytest

import kairos.rl as rl

try:
    import torch  # noqa: F401

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on the interpreter
    TORCH_AVAILABLE = False


def test_package_imports_without_torch():
    """kairos.rl is imported under FreeCAD's python, which has no torch."""
    assert rl.__all__


def test_portable_names_resolve_anywhere():
    assert rl.PROTOCOL_VERSION >= 1
    assert isinstance(rl.FALLBACK_REQUIREMENTS, tuple)
    assert callable(rl.requirement_pools)


def test_unknown_attribute_raises_attribute_error():
    with pytest.raises(AttributeError, match="no attribute"):
        _ = rl.definitely_not_exported


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="requires the 'learn' extra")
def test_torch_names_resolve_when_available():
    assert rl.RolloutBuffer is not None
    assert rl.PPOTrainer is not None
    assert rl.RemoteCADEnv is not None


@pytest.mark.skipif(TORCH_AVAILABLE, reason="only meaningful without torch")
def test_torch_names_explain_the_missing_extra():  # pragma: no cover - env dependent
    with pytest.raises(AttributeError, match="learn"):
        _ = rl.PPOTrainer
