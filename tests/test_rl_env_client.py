"""Env client tests.

Most run the real server module under the *system* interpreter: `env_server`
imports FreeCAD lazily, so a handshake succeeds anywhere and only `reset` needs
the CAD stack. The one test that needs real geometry is marked `cad`.
"""

import sys
from pathlib import Path

import pytest

from kairos.rl.env_client import (
    RemoteCADEnv,
    RemoteEnvError,
    probe_environment,
    resolve_freecad_python,
)
from kairos.rl.protocol import ProtocolError


def _running_under_freecad() -> bool:
    """True when the test runner *is* the FreeCAD interpreter.

    Two tests below assert the behavior when the spawned interpreter cannot
    import FreeCAD. They spawn ``sys.executable``, so under `make test-cad`
    that assumption inverts and the assertions are meaningless rather than
    failing for a real reason.
    """
    try:
        return Path(sys.executable).resolve() == Path(resolve_freecad_python()).resolve()
    except FileNotFoundError:
        return False


needs_a_freecad_free_interpreter = pytest.mark.skipif(
    _running_under_freecad(),
    reason="this interpreter can import FreeCAD, so the no-FreeCAD path cannot be tested",
)


def _client(**kwargs):
    """A client backed by the system interpreter (no FreeCAD required)."""
    return RemoteCADEnv(python=sys.executable, timeout=30.0, **kwargs)


def test_resolve_prefers_the_explicit_interpreter():
    assert resolve_freecad_python(sys.executable) == sys.executable


def test_resolve_falls_back_to_the_environment_variable(monkeypatch):
    monkeypatch.setenv("KAIROS_FREECAD_PYTHON", sys.executable)
    assert resolve_freecad_python(None) == sys.executable


def test_resolve_reports_a_missing_interpreter(monkeypatch):
    monkeypatch.delenv("KAIROS_FREECAD_PYTHON", raising=False)
    monkeypatch.setattr(
        "kairos.rl.env_client.DEFAULT_FREECAD_PYTHON", "/nonexistent/python"
    )
    with pytest.raises(FileNotFoundError, match="KAIROS_FREECAD_PYTHON"):
        resolve_freecad_python(None)


def test_client_starts_and_handshakes():
    env = _client()
    try:
        assert env.alive
    finally:
        env.close()
    assert not env.alive


def test_bad_interpreter_fails_loudly_at_construction():
    with pytest.raises((RemoteEnvError, FileNotFoundError)):
        RemoteCADEnv(python="/nonexistent/python")


@needs_a_freecad_free_interpreter
def test_server_error_surfaces_on_reset():
    """Without FreeCAD the server reports a failed reset rather than hanging."""
    env = _client(auto_restart=False)
    try:
        with pytest.raises(ProtocolError, match="reset failed"):
            env.reset()
    finally:
        env.close()


def test_a_dead_server_truncates_the_episode_and_restarts():
    """FreeCAD can die on a pathological recompute; training must continue."""
    env = _client()
    try:
        env._process.kill()
        env._process.wait(timeout=5)
        obs, reward, terminated, truncated, info = env.step(0, [0.5] * 6)
        assert obs is None
        assert (terminated, truncated) == (False, True)
        assert info["crashed"] is True
        assert env.restarts == 1
        assert env.alive  # a fresh server is already up for the next episode
    finally:
        env.close()


def test_a_hung_server_times_out_and_restarts(monkeypatch):
    env = _client()
    try:
        env.timeout = 0.05
        monkeypatch.setattr(env, "_send", lambda payload: None)  # request never arrives
        obs, _, _, truncated, info = env.step(0, [0.5] * 6)
        assert obs is None and truncated is True
        assert info["crashed"] is True
        # Counters are diagnostics, not exact contracts: the patched _send also
        # stalls the restart handshake, so more than one timeout is expected.
        assert env.timeouts >= 1 and env.restarts >= 1
    finally:
        env.close()


@needs_a_freecad_free_interpreter
def test_probe_reports_unavailable_without_raising():
    available, message = probe_environment(python=sys.executable)
    assert available is False  # no FreeCAD in this interpreter
    assert isinstance(message, str) and message


@pytest.mark.cad
def test_bridged_environment_builds_real_geometry():
    """The whole point: a torch-side caller driving a real FreeCAD build."""
    from kairos.actions.schema import Operation
    from kairos.rl.action_space import OPERATIONS

    env = RemoteCADEnv(
        requirement="Design a rectangular mounting plate 60 x 40 x 5 mm with 4 holes."
    )
    try:
        observation = env.reset()
        assert observation["numeric"].shape == (24,)
        assert observation["action_mask"].sum() >= 1
        assert observation["has_solid"] is False

        for operation, params in (
            (Operation.CREATE_SKETCH, [0.0, 0.5, 0, 0, 0, 0]),
            (Operation.ADD_RECTANGLE, [0.5, 0.5, 0.3, 0.2, 0, 0]),
            (Operation.PAD, [0.1, 0.0, 0.0, 0, 0, 0]),
        ):
            observation, reward, terminated, truncated, info = env.step(
                OPERATIONS.index(operation), params
            )
            assert info["ok"] is True, info
            assert not truncated

        assert observation["has_solid"] is True
        assert observation["mass_g"] > 0.0
        assert "Pad" in " ".join(observation["feature_history"])
        assert env.restarts == 0
    finally:
        env.close()


def test_a_desynced_stream_restarts_instead_of_escaping():
    """A malformed line is a ProtocolError, which is not a RemoteEnvError --
    it used to propagate straight out of the rollout loop."""
    env = _client()
    try:
        original = env._read_line
        env._read_line = lambda timeout=None: "this is not json\n"
        obs, _, _, truncated, info = env.step(0, [0.5] * 6)
        env._read_line = original
        assert obs is None and truncated is True
        assert info["crashed"] is True
        assert env.restarts >= 1
    finally:
        env.close()


def test_max_steps_reaches_the_server():
    """The client's cap was stored and never sent; the server used its own."""
    env = RemoteCADEnv(python=sys.executable, max_steps=7, timeout=30.0)
    try:
        command = env._process.args
        assert "--max-steps" in command
        assert command[command.index("--max-steps") + 1] == "7"
    finally:
        env.close()
