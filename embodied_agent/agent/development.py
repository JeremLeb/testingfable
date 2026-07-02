"""Development & critical periods (B3).

Biology: an organism lives one irreversible life. Plasticity is high in an
early *critical period* and anneals as the brain matures, so early experience
imprints disproportionately and is hard to overwrite later. Learning is online
from a single non-stationary stream, not resampled i.i.d. episodes.

This object owns the age-dependent plasticity schedule. The continual-life
reset semantics live in ``train.py``. It is a no-op (gain 1.0) unless
``dev.enabled`` -- with it off the training loop keeps constant plasticity and
episodic resets (the deep-RL baseline).
"""
from __future__ import annotations

import math

from ..config import DevelopmentConfig


class Development:
    def __init__(self, cfg: DevelopmentConfig):
        self.cfg = cfg

    def plasticity_gain(self, age: int) -> float:
        """LR multiplier as a function of age (steps since birth): high at
        birth (``young_gain``), decaying exponentially toward ``floor_gain``."""
        c = self.cfg
        if not (c.enabled and c.critical_period):
            return 1.0
        tau = max(1, c.critical_period_steps)
        return c.floor_gain + (c.young_gain - c.floor_gain) * math.exp(-age / tau)

    def metrics(self, age: int) -> dict:
        return {"age": float(age), "plasticity_gain": self.plasticity_gain(age)}
