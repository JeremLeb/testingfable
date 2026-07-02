"""Environment package: arena physics, sensors, homeostasis."""
from __future__ import annotations

import numpy as np

from ..config import Config
from .arena import PetriEnv
from .homeostasis import Homeostasis
from .sensors import SensorSuite


def make_env(cfg: Config, seed: int = 0) -> PetriEnv:
    """Build the full embodied environment: arena + sensors + interoception."""
    env = PetriEnv(cfg.env, cfg.sensor, seed=seed)
    env.sensors = SensorSuite(cfg.sensor, cfg.env,
                              np.random.default_rng(seed + 1))
    env.homeostasis = Homeostasis(cfg.env, reward_cfg=cfg.reward)
    env.reset()
    return env
