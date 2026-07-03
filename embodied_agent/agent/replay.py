"""Replay buffer of real trajectories for training the world model.

Stores whole episodes as contiguous arrays and samples fixed-length
sequences that never cross an episode boundary. Each timestep records the
observation, the *previous* action (the one that led into this observation),
the reward received on arriving here, and a continue flag. This
prev-action alignment lets the world model predict reward_t and the
reconstruction of obs_t from the same latent state s_t (see model/rssm.py).
"""
from __future__ import annotations

import numpy as np
import torch


class Episode:
    def __init__(self):
        self.obs: dict[str, list] = {}
        self.prev_action: list = []
        self.reward: list = []
        self.cont: list = []

    def add(self, obs, prev_action, reward, cont):
        for k, v in obs.items():
            self.obs.setdefault(k, []).append(np.asarray(v, dtype=np.float32))
        self.prev_action.append(np.asarray(prev_action, dtype=np.float32))
        self.reward.append(np.float32(reward))
        self.cont.append(np.float32(cont))

    def finalize(self) -> dict:
        reward = np.asarray(self.reward, dtype=np.float32)
        return {
            "obs": {k: np.stack(v) for k, v in self.obs.items()},
            "prev_action": np.stack(self.prev_action),
            "reward": reward,
            "cont": np.asarray(self.cont, dtype=np.float32),
            # salience for prioritized ("emotional") replay: the peak reward
            # magnitude (feeding / collision / death spikes) and how eventful
            # the episode was (reward variability). Cheap and available at
            # store time, so no forward pass is needed to weight memory.
            "salience": {
                "peak": float(np.abs(reward).max()) if len(reward) else 0.0,
                "var": float(reward.std()) if len(reward) else 0.0,
            },
        }

    def __len__(self):
        return len(self.reward)


class ReplayBuffer:
    def __init__(self, capacity: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.episodes: list[dict] = []
        self._size = 0
        self._current = Episode()

    def start_episode(self):
        self._current = Episode()

    def add(self, obs, prev_action, reward, cont):
        self._current.add(obs, prev_action, reward, cont)

    def end_episode(self):
        if len(self._current) >= 2:
            ep = self._current.finalize()
            self.episodes.append(ep)
            self._size += len(self._current)
            while self._size > self.capacity and len(self.episodes) > 1:
                dropped = self.episodes.pop(0)
                self._size -= len(dropped["reward"])
        self._current = Episode()

    def ingest(self, episode: "Episode"):
        """Finalise and store an externally-built Episode. Lets several streams
        (e.g. the many creatures of a colony) feed one shared buffer."""
        if len(episode) >= 2:
            ep = episode.finalize()
            self.episodes.append(ep)
            self._size += len(episode)
            while self._size > self.capacity and len(self.episodes) > 1:
                dropped = self.episodes.pop(0)
                self._size -= len(dropped["reward"])

    @property
    def num_steps(self) -> int:
        return self._size

    def can_sample(self, seq_len: int) -> bool:
        return any(len(ep["reward"]) >= seq_len for ep in self.episodes)

    def _priorities(self, eligible: list[dict], sleep_cfg) -> np.ndarray:
        """Per-episode sampling weights from stored salience. Returns a uniform
        vector unless prioritized replay is requested (B2)."""
        n = len(eligible)
        if sleep_cfg is None or not sleep_cfg.prioritized:
            return np.full(n, 1.0 / n)
        sal = np.array([
            sleep_cfg.reward_salience * ep["salience"]["peak"]
            + sleep_cfg.surprise_salience * ep["salience"]["var"]
            for ep in eligible], dtype=np.float64)
        p = (sal + sleep_cfg.priority_eps) ** sleep_cfg.priority_exponent
        return p / p.sum()

    def episode_priorities(self, seq_len: int, sleep_cfg) -> tuple:
        """(salience, probability) per eligible episode -- for the B2 demo."""
        eligible = [ep for ep in self.episodes
                    if len(ep["reward"]) >= seq_len]
        sal = np.array([ep["salience"]["peak"] for ep in eligible])
        return sal, self._priorities(eligible, sleep_cfg)

    def sample(self, batch_size: int, seq_len: int, rng: np.random.Generator,
               sleep_cfg=None):
        """Return a dict of batched tensors shaped (B, T, ...).

        With a SleepConfig whose ``prioritized`` is set, episodes are drawn in
        proportion to their salience (emotional / near-death memories replayed
        preferentially); otherwise sampling is uniform (the baseline)."""
        eligible = [ep for ep in self.episodes
                    if len(ep["reward"]) >= seq_len]
        if not eligible:
            raise ValueError("no episode long enough to sample")
        probs = self._priorities(eligible, sleep_cfg)
        obs_keys = list(eligible[0]["obs"].keys())
        obs_batch = {k: [] for k in obs_keys}
        act_batch, rew_batch, cont_batch = [], [], []
        for _ in range(batch_size):
            ep = eligible[rng.choice(len(eligible), p=probs)]
            n = len(ep["reward"])
            start = int(rng.integers(0, n - seq_len + 1))
            sl = slice(start, start + seq_len)
            for k in obs_keys:
                obs_batch[k].append(ep["obs"][k][sl])
            act_batch.append(ep["prev_action"][sl])
            rew_batch.append(ep["reward"][sl])
            cont_batch.append(ep["cont"][sl])

        def to_t(x):
            return torch.as_tensor(np.stack(x), device=self.device)

        return {
            "obs": {k: to_t(v) for k, v in obs_batch.items()},
            "prev_action": to_t(act_batch),
            "reward": to_t(rew_batch),
            "cont": to_t(cont_batch),
        }
