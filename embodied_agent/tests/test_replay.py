import numpy as np
import pytest

from embodied_agent.agent.replay import ReplayBuffer


def _fake_obs(i):
    return {"vision": np.full(4, i, np.float32),
            "intero": np.full(3, i, np.float32)}


def test_add_and_sample_shapes():
    buf = ReplayBuffer(capacity=10000)
    rng = np.random.default_rng(0)
    for ep in range(3):
        buf.start_episode()
        for t in range(40):
            buf.add(_fake_obs(t), np.zeros(2), float(t), 1.0)
        buf.end_episode()
    assert buf.can_sample(16)
    batch = buf.sample(batch_size=8, seq_len=16, rng=rng)
    assert batch["prev_action"].shape == (8, 16, 2)
    assert batch["reward"].shape == (8, 16)
    assert batch["cont"].shape == (8, 16)
    assert batch["obs"]["vision"].shape == (8, 16, 4)


def test_sequences_are_contiguous():
    buf = ReplayBuffer(capacity=10000)
    rng = np.random.default_rng(1)
    buf.start_episode()
    for t in range(60):
        buf.add(_fake_obs(t), np.zeros(2), float(t), 1.0)
    buf.end_episode()
    batch = buf.sample(4, 10, rng)
    # rewards were the timestep index, so a contiguous slice increments by 1
    diffs = np.diff(batch["reward"].numpy(), axis=1)
    assert np.allclose(diffs, 1.0)


def test_capacity_evicts_old_episodes():
    buf = ReplayBuffer(capacity=100)
    for ep in range(10):
        buf.start_episode()
        for t in range(40):
            buf.add(_fake_obs(t), np.zeros(2), 0.0, 1.0)
        buf.end_episode()
    assert buf.num_steps <= 100 + 40  # bounded near capacity


def test_short_episode_not_sampled():
    buf = ReplayBuffer(capacity=1000)
    rng = np.random.default_rng(2)
    buf.start_episode()
    for t in range(5):
        buf.add(_fake_obs(t), np.zeros(2), 0.0, 1.0)
    buf.end_episode()
    assert not buf.can_sample(16)
    with pytest.raises(ValueError):
        buf.sample(2, 16, rng)
