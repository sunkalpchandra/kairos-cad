"""Env server tests using a fake environment (no FreeCAD, no torch)."""

import io

import numpy as np
import pytest

from kairos.rl import protocol as p
from kairos.rl.env_server import EnvironmentServer


class FakeEnv:
    """Stands in for KairosCADEnv with a scripted response."""

    def __init__(self, step_error=None, n_ops=38, n_numeric=24):
        self.step_error = step_error
        self.n_ops = n_ops
        self.n_numeric = n_numeric
        self.closed = False
        self.steps = []
        self.reset_calls = []

    def _obs(self):
        return {
            "numeric": np.arange(self.n_numeric, dtype=np.float32) / 100.0,
            "action_mask": np.ones(self.n_ops, dtype=np.int8),
        }

    def reset(self, seed=None, options=None):
        self.reset_calls.append((seed, options))
        return self._obs(), {"requirement": (options or {}).get("requirement", "default")}

    def step(self, action):
        self.steps.append(action)
        if self.step_error is not None:
            raise self.step_error
        info = {
            "action": {"operation": "PAD"},
            "result": {"ok": True, "message": "padded"},
            "reward_breakdown": {"components": {"action_cost": -0.01}},
            "constraints": {"satisfaction_rate": 0.5, "all_measured_satisfied": False},
            "observation": {"summary": {"feature_history": ["Pad"], "has_solid": True,
                                        "valid": True, "mass_g": 12.5}},
        }
        return self._obs(), 1.25, False, False, info

    def close(self):
        self.closed = True


def _server(env):
    return EnvironmentServer(stdin=io.StringIO(), stdout=io.StringIO(), env=env)


def test_handshake_reports_the_protocol_version():
    response = _server(FakeEnv()).handle(p.handshake_request())
    p.check_version(response)  # must not raise


def test_reset_returns_a_json_safe_observation():
    env = FakeEnv()
    response = _server(env).handle(p.reset_request("Design a plate", seed=7))
    p.raise_for_error(response)
    observation = response["observation"]
    assert isinstance(observation["numeric"], list)
    assert all(isinstance(v, float) for v in observation["numeric"])
    assert all(isinstance(v, int) for v in observation["action_mask"])
    assert env.reset_calls == [(7, {"requirement": "Design a plate"})]
    p.encode_message(response)  # numpy types would break this


def test_step_carries_reward_history_and_constraint_state():
    response = _server(FakeEnv()).handle(p.step_request(3, [0.5] * 6, target=1))
    p.raise_for_error(response)
    assert response["reward"] == pytest.approx(1.25)
    assert response["observation"]["feature_history"] == ["Pad"]
    assert response["observation"]["mass_g"] == pytest.approx(12.5)
    assert response["info"]["satisfaction_rate"] == pytest.approx(0.5)
    assert response["info"]["ok"] is True


def test_a_raising_step_ends_the_episode_instead_of_the_server():
    """A broken recompute is an ordinary CAD event; the rollout must survive."""
    response = _server(FakeEnv(step_error=RuntimeError("recompute exploded"))).handle(
        p.step_request(3, [0.5] * 6)
    )
    p.raise_for_error(response)  # still a successful *response*
    assert response["truncated"] is True
    assert response["observation"] is None
    assert response["info"]["crashed"] is True
    assert "recompute exploded" in response["info"]["message"]


def test_close_closes_the_environment():
    env = FakeEnv()
    server = _server(env)
    p.raise_for_error(server.handle(p.close_request()))
    assert env.closed is True


def test_unknown_commands_are_refused_not_fatal():
    response = _server(FakeEnv()).handle({"cmd": "sudo_make_me_a_sandwich"})
    assert response["ok"] is False
    assert response["kind"] == "unknown_command"


def test_serve_loop_answers_line_by_line_and_stops_on_close():
    env = FakeEnv()
    requests = "\n".join(
        p.encode_message(m)
        for m in (p.handshake_request(), p.reset_request(), p.close_request(), p.reset_request())
    )
    stdout = io.StringIO()
    server = EnvironmentServer(stdin=io.StringIO(requests), stdout=stdout, env=env)
    server.serve()

    lines = [line for line in stdout.getvalue().splitlines() if line]
    assert len(lines) == 3  # the request after close is never served
    assert p.decode_message(lines[0])["version"] == p.PROTOCOL_VERSION
    assert env.closed is True


def test_malformed_line_is_reported_without_killing_the_loop():
    stdout = io.StringIO()
    stdin = io.StringIO("{not json\n" + p.encode_message(p.handshake_request()) + "\n")
    EnvironmentServer(stdin=stdin, stdout=stdout, env=FakeEnv()).serve()

    lines = [p.decode_message(line) for line in stdout.getvalue().splitlines() if line]
    assert lines[0]["ok"] is False and lines[0]["kind"] == "protocol"
    assert lines[1]["ok"] is True  # the loop kept going
