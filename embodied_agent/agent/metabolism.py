"""Metabolic cost of cognition & sensorimotor realism (B4).

Biology: the brain is ~20% of the metabolic budget -- thinking burns energy,
so cognition trades off against the very drive it serves. A starving animal
cannot afford to deliberate far ahead (bounded planning). Sensing and acting
are neither instantaneous nor noiseless (conduction latency, motor noise).

Two small, config-switchable pieces, both no-ops unless ``metab.enabled`` so
the baseline (free imagination, instantaneous noiseless control) is preserved:

  * ``Metabolism`` -- an energy price on imagination and an energy-gated
    planning horizon.
  * ``Sensorimotor`` -- fixed sensory/motor latency and motor noise on the
    closed loop, applied identically in training collection and evaluation.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..config import MetabolismConfig


class Metabolism:
    def __init__(self, cfg: MetabolismConfig):
        self.cfg = cfg

    def planning_horizon(self, energy: float, base: int) -> int:
        """Imagination horizon as a function of current body energy: full at
        satiation, shrinking toward ``min_horizon`` as energy -> 0."""
        c = self.cfg
        if not (c.enabled and c.bounded_planning):
            return base
        e = float(np.clip(energy, 0.0, 1.0))
        h = c.min_horizon + (base - c.min_horizon) * e
        return max(c.min_horizon, min(base, int(round(h))))

    def cognition_cost(self, imagined_steps: int) -> float:
        """Energy debited for a bout of imagination of `imagined_steps` total
        rolled-out steps (horizon x batch x passes)."""
        c = self.cfg
        if not (c.enabled and c.cognition_cost):
            return 0.0
        return c.imagination_energy_cost * imagined_steps


class Sensorimotor:
    """Sensory + motor latency and motor noise on the closed loop.

    ``perceive`` returns the observation the agent actually sees (delayed by
    ``obs_delay`` steps); ``execute`` returns the action the world actually
    receives (delayed by ``action_delay`` steps, plus Gaussian motor noise).
    No-op when both delays and the noise are zero.
    """

    def __init__(self, cfg: MetabolismConfig, action_dim: int = 2, seed: int = 0):
        self.cfg = cfg
        self.action_dim = action_dim
        self._rng = np.random.default_rng(seed)
        self._obs: deque = deque()
        self._act: deque = deque()

    @property
    def active(self) -> bool:
        c = self.cfg
        return c.enabled and (c.obs_delay or c.action_delay or c.motor_noise)

    def reset(self, obs: dict):
        c = self.cfg
        self._obs = deque([obs] * (c.obs_delay + 1), maxlen=c.obs_delay + 1)
        self._act = deque([np.zeros(self.action_dim)] * (c.action_delay + 1),
                          maxlen=c.action_delay + 1)

    def perceive(self, obs: dict) -> dict:
        if not self.active:
            return obs
        self._obs.append(obs)
        return self._obs[0]

    def execute(self, action: np.ndarray) -> np.ndarray:
        if not self.active:
            return action
        a = np.asarray(action, dtype=np.float64)
        if self.cfg.motor_noise:
            a = a + self.cfg.motor_noise * self._rng.standard_normal(self.action_dim)
        self._act.append(a)
        return np.clip(self._act[0], -1.0, 1.0)
