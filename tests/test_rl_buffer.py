"""Rollout buffer and GAE tests (skipped without the optional torch extra)."""

import pytest

torch = pytest.importorskip("torch", reason="requires the 'learn' extra")

from kairos.rl.buffer import RolloutBuffer  # noqa: E402


def _add(buffer, reward, value=0.0, terminated=False, truncated=False, bootstrap=None, ok=True):
    buffer.add(
        inputs={"numeric": torch.zeros(24)},
        action={"operation": torch.tensor(0), "parameters": torch.zeros(6)},
        log_prob=torch.tensor(-1.0),
        value=torch.tensor(float(value)),
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        info={"ok": ok},
        bootstrap_value=bootstrap,
    )


def test_empty_buffer_yields_empty_advantages():
    advantages, returns = RolloutBuffer().compute_advantages()
    assert advantages.numel() == 0 and returns.numel() == 0


def test_single_terminated_step_advantage_is_reward_minus_value():
    buffer = RolloutBuffer(gamma=0.99)
    _add(buffer, reward=5.0, value=1.0, terminated=True)
    advantages, returns = buffer.compute_advantages()
    assert float(advantages[0]) == pytest.approx(4.0)
    assert float(returns[0]) == pytest.approx(5.0)


def test_termination_assumes_no_future_value():
    """FINISH_DESIGN really is the end; nothing follows it."""
    buffer = RolloutBuffer(gamma=0.9)
    _add(buffer, reward=1.0, value=0.0, terminated=True)
    advantages, _ = buffer.compute_advantages()
    assert float(advantages[0]) == pytest.approx(1.0)  # no gamma * V(next) term


def test_truncation_bootstraps_instead_of_assuming_zero():
    """The step budget ending an episode must not read as catastrophe."""
    buffer = RolloutBuffer(gamma=0.9)
    _add(buffer, reward=1.0, value=0.0, truncated=True, bootstrap=10.0)
    advantages, _ = buffer.compute_advantages()
    assert float(advantages[0]) == pytest.approx(1.0 + 0.9 * 10.0)


def test_truncated_and_terminated_differ():
    ended = RolloutBuffer(gamma=0.9)
    _add(ended, reward=1.0, value=0.0, terminated=True)
    cut = RolloutBuffer(gamma=0.9)
    _add(cut, reward=1.0, value=0.0, truncated=True, bootstrap=5.0)
    assert float(ended.compute_advantages()[0][0]) != pytest.approx(
        float(cut.compute_advantages()[0][0])
    )


def test_advantage_does_not_leak_across_an_episode_boundary():
    """Step 0 ends an episode; its advantage must ignore the next episode."""
    buffer = RolloutBuffer(gamma=0.99, gae_lambda=0.95)
    _add(buffer, reward=1.0, value=0.0, terminated=True)
    _add(buffer, reward=100.0, value=0.0, terminated=True)
    advantages, _ = buffer.compute_advantages()
    assert float(advantages[0]) == pytest.approx(1.0)
    assert float(advantages[1]) == pytest.approx(100.0)


def test_gae_accumulates_within_an_episode():
    buffer = RolloutBuffer(gamma=1.0, gae_lambda=1.0)
    for reward in (1.0, 1.0, 1.0):
        _add(buffer, reward=reward, value=0.0)
    _add(buffer, reward=1.0, value=0.0, terminated=True)
    advantages, returns = buffer.compute_advantages()
    # With gamma=lambda=1 and V=0, the advantage is the remaining reward sum.
    assert [float(a) for a in advantages] == pytest.approx([4.0, 3.0, 2.0, 1.0])
    assert [float(r) for r in returns] == pytest.approx([4.0, 3.0, 2.0, 1.0])


def test_returns_equal_advantages_plus_values():
    buffer = RolloutBuffer()
    for i in range(5):
        _add(buffer, reward=float(i), value=float(i) / 2)
    _add(buffer, reward=1.0, value=0.5, terminated=True)
    advantages, returns = buffer.compute_advantages()
    values = torch.tensor([float(t.value) for t in buffer.transitions])
    assert torch.allclose(returns, advantages + values, atol=1e-5)


def test_batches_cover_every_transition_exactly_once():
    buffer = RolloutBuffer()
    for i in range(10):
        _add(buffer, reward=float(i), value=0.0)
    _add(buffer, reward=0.0, terminated=True)

    seen = 0
    for batch in buffer.batches(batch_size=4, seed=0):
        seen += batch["advantage"].shape[0]
        assert batch["inputs"]["numeric"].shape[1:] == (24,)
        assert batch["action"]["operation"].shape[0] == batch["advantage"].shape[0]
        assert batch["log_prob"].dim() == 1
    assert seen == len(buffer)


def test_statistics_count_endings_and_failures():
    buffer = RolloutBuffer()
    _add(buffer, reward=1.0, ok=False)
    _add(buffer, reward=2.0, terminated=True)
    _add(buffer, reward=3.0, truncated=True, bootstrap=0.0)
    stats = buffer.statistics()
    assert stats["transitions"] == 3
    assert stats["episodes"] == 2
    assert stats["terminated"] == 1 and stats["truncated"] == 1
    assert stats["invalid_actions"] == 1
    assert stats["reward_sum"] == pytest.approx(6.0)


def test_clear_resets_bootstraps_too():
    buffer = RolloutBuffer()
    _add(buffer, reward=1.0, truncated=True, bootstrap=9.0)
    buffer.clear()
    _add(buffer, reward=1.0, value=0.0, terminated=True)
    assert float(buffer.compute_advantages()[0][0]) == pytest.approx(1.0)
