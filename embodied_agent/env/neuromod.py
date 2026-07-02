"""Neuromodulation & allostasis (B1).

Two biologically-motivated pieces, both off unless `neuromod.enabled`:

1. Allostatic drive weighting (pure function, used inside the reward). Priorities
   are not fixed: an interoceptive deficit amplifies its own drive's weight
   super-linearly, so a starving body fixates on food and a near-death body
   fixates on integrity. Arbitration becomes state-dependent and emergent
   instead of a hand-set constant (the project's known weak point). This is
   allostasis (Sterling): the setpoints' *influence* shifts with context.

2. Neuromodulators (a small stateful object, used in the training loop):
   - norepinephrine (NE): a running estimate of surprise (world-model
     prediction error). Unexpected uncertainty raises effective plasticity, so
     the learning rate is scaled up when the world is surprising.
   - dopamine (DA): a running estimate of |reward-prediction-error| (the
     critic's TD error). Logged as motivational tone and optionally used to
     gate the actor's learning rate.

These map named neuromodulators onto concrete integration points; they are not
a full neurochemical model.
"""
from __future__ import annotations

from ..config import NeuromodConfig, RewardConfig


def allostatic_weights(drives: dict[str, float], reward_cfg: RewardConfig,
                       nm: NeuromodConfig) -> dict[str, float]:
    """Return per-drive weights. With allostasis on, each base weight is
    multiplied by an urgency factor that grows super-linearly with that
    drive's current deficit (distance from setpoint, already in [0, 1])."""
    base = {
        "energy": reward_cfg.w_energy,
        "thermal": reward_cfg.w_thermal,
        "integrity": reward_cfg.w_integrity,
    }
    if not (nm.enabled and nm.allostatic):
        return base
    gains = {"energy": nm.energy_gain, "thermal": nm.thermal_gain,
             "integrity": nm.integrity_gain}
    p = nm.urgency_power
    return {k: base[k] * (1.0 + gains[k] * (max(0.0, drives[k]) ** p))
            for k in base}


class Neuromodulators:
    """Training-side modulators. Tracks EMAs of surprise and RPE and turns them
    into multiplicative gains on learning rates."""

    def __init__(self, cfg: NeuromodConfig):
        self.cfg = cfg
        self._ne_ema: float | None = None    # surprise baseline
        self._da_ema: float | None = None    # |RPE| baseline
        self.ne_gain = 1.0
        self.da_tone = 0.0

    @staticmethod
    def _ema(prev, x, rate):
        return x if prev is None else rate * prev + (1 - rate) * x

    def update_surprise(self, surprise: float) -> float:
        """Feed the world-model prediction error; return the plasticity gain
        in [1, ne_gain]. Above-baseline surprise raises the effective LR."""
        if not (self.cfg.enabled and self.cfg.ne_enabled):
            self.ne_gain = 1.0
            return 1.0
        self._ne_ema = self._ema(self._ne_ema, surprise, self.cfg.ne_ema)
        base = max(self._ne_ema, 1e-6)
        rel = surprise / base                       # >1 == more surprising
        # smoothly map relative surprise into [1, ne_gain]
        excess = max(0.0, rel - 1.0)
        self.ne_gain = 1.0 + (self.cfg.ne_gain - 1.0) * (excess / (1.0 + excess))
        return self.ne_gain

    def update_rpe(self, rpe_abs: float) -> float:
        """Feed |reward-prediction-error|; return a dopamine tone in [0, 1]."""
        if not (self.cfg.enabled and self.cfg.da_enabled):
            self.da_tone = 0.0
            return 0.0
        self._da_ema = self._ema(self._da_ema, rpe_abs, self.cfg.da_ema)
        base = max(self._da_ema, 1e-6)
        self.da_tone = min(1.0, rpe_abs / base * 0.5)
        return self.da_tone

    def actor_lr_gain(self) -> float:
        """Dopamine gates actor plasticity: learn more from surprising RPE."""
        if not (self.cfg.enabled and self.cfg.da_enabled):
            return 1.0
        return 1.0 + self.cfg.da_gain * self.da_tone

    def metrics(self) -> dict:
        return {"ne_gain": self.ne_gain, "da_tone": self.da_tone}
