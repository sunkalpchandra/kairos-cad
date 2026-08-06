"""CAD integration tests: the Gymnasium environment end-to-end."""

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from kairos.actions.schema import Operation  # noqa: E402
from kairos.rl.action_space import OPERATIONS, PARAM_SLOTS  # noqa: E402
from kairos.rl.environment import KairosCADEnv  # noqa: E402

pytestmark = pytest.mark.cad


@pytest.fixture
def env():
    environment = KairosCADEnv(max_steps=25)
    yield environment
    environment.close()


def _act(op, params=None, target=0):
    vector = np.full(PARAM_SLOTS, 0.5, dtype=np.float32)
    for slot, value in (params or {}).items():
        vector[slot] = value
    return {"operation": OPERATIONS.index(op), "params": vector, "target": target}


def test_reset_returns_valid_observation(env):
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert "L-bracket" in info["requirement"]
    mask = obs["action_mask"]
    assert mask[OPERATIONS.index(Operation.CREATE_SKETCH)] == 1
    assert mask[OPERATIONS.index(Operation.PAD)] == 0


def test_step_rewards_and_mask_evolution(env):
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(_act(Operation.CREATE_SKETCH))
    assert info["result"]["ok"] and not terminated
    obs, reward, *_ , info = env.step(
        _act(Operation.ADD_CIRCLE, {0: 0.6, 1: 0.55, 2: 0.3})
    )
    assert info["result"]["ok"]
    assert reward > 0  # valid_sketch bonus outweighs action cost
    assert obs["action_mask"][OPERATIONS.index(Operation.PAD)] == 1
    obs, reward, *_ , info = env.step(_act(Operation.PAD, {0: 0.1, 1: 0.0, 2: 0.0}))
    assert info["result"]["ok"], info["result"]["message"]
    assert info["observation"]["summary"]["has_solid"]
    assert obs["action_mask"][OPERATIONS.index(Operation.FILLET)] == 1


def test_illegal_action_penalized_not_fatal(env):
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(_act(Operation.PAD))
    assert not info["result"]["ok"]
    assert reward < 0
    assert not terminated and not truncated
    # Environment still functional afterwards.
    _, _, _, _, info = env.step(_act(Operation.CREATE_SKETCH))
    assert info["result"]["ok"]


def test_finish_terminates_episode(env):
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(_act(Operation.FINISH_DESIGN))
    assert terminated and not truncated
    assert info["reward_breakdown"]["components"]["finish"] == pytest.approx(-1.0)


def test_truncation_at_max_steps():
    env = KairosCADEnv(max_steps=3)
    try:
        env.reset(seed=0)
        for _ in range(2):
            _, _, terminated, truncated, _ = env.step(_act(Operation.CHECK_VALIDITY))
            assert not truncated
        _, _, terminated, truncated, _ = env.step(_act(Operation.CHECK_VALIDITY))
        assert truncated and not terminated
    finally:
        env.close()


def test_custom_requirement_via_reset_options(env):
    _, info = env.reset(options={"requirement": "Plate with 2 M5 holes. Minimize mass."})
    assert env.spec.hole_count == 2
    assert info["requirement"].startswith("Plate")


def test_constraint_info_reported(env):
    env.reset(seed=0)
    *_, info = env.step(_act(Operation.CHECK_VALIDITY))
    counts = info["constraints"]["counts"]
    assert counts["violated"] >= 1  # no holes yet
    assert info["constraints"]["satisfaction_rate"] < 1.0
