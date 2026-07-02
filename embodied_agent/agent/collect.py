"""Environment interaction: fill the replay buffer with the prev-action
alignment the world model expects (see agent/replay.py)."""
from __future__ import annotations

import numpy as np

from .replay import ReplayBuffer


def smooth_random_policy(rng: np.random.Generator, smooth: float = 0.8):
    """A temporally-correlated random policy so the agent actually explores."""
    state = {"a": np.zeros(2)}

    def policy(obs):
        state["a"] = smooth * state["a"] + (1 - smooth) * rng.uniform(-1, 1, 2)
        return state["a"].copy()

    return policy


def collect_random(env, buffer: ReplayBuffer, steps: int,
                   rng: np.random.Generator) -> dict:
    """Roll a smoothed-random policy for `steps` env steps into the buffer."""
    policy = smooth_random_policy(rng)
    obs, _ = env.reset()
    buffer.start_episode()
    buffer.add(obs, np.zeros(2), 0.0, 1.0)
    eaten = deaths = 0
    for _ in range(steps):
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        buffer.add(obs, action, reward, 0.0 if terminated else 1.0)
        eaten += info["food_eaten"]
        if terminated or truncated:
            deaths += int(terminated)
            buffer.end_episode()
            policy = smooth_random_policy(rng)
            obs, _ = env.reset()
            buffer.start_episode()
            buffer.add(obs, np.zeros(2), 0.0, 1.0)
    buffer.end_episode()
    return {"food_eaten": eaten, "deaths": deaths}
