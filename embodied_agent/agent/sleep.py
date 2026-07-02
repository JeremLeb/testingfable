"""Circadian wake/sleep scheduling for consolidation (B2).

Biology (complementary learning systems, McClelland): fast, light adaptation
happens online during *wake*; the bulk of cortical consolidation happens
offline during *sleep*, replaying emotionally salient experience and
recombining it (dreaming). A circadian clock gates the cycle.

This controller owns only the *timing*. The actual consolidation (prioritized
replay, dreaming) lives in ``train.py``; the buffer owns salience weighting.
It is a no-op unless ``sleep.enabled`` -- with it off the training loop keeps
its original interleaved-uniform-update behaviour (the deep-RL baseline).
"""
from __future__ import annotations

from ..config import SleepConfig


class SleepController:
    def __init__(self, cfg: SleepConfig):
        self.cfg = cfg
        self.day = max(1, cfg.day_steps)
        self.wake_steps = max(1, int(cfg.wake_frac * self.day))
        self.sleep_steps = max(1, self.day - self.wake_steps)
        # spread the night's consolidation budget across the sleep window so
        # wall-clock stays proportional and sleep genuinely "takes time".
        self.updates_per_sleep_step = max(
            1, round(cfg.sleep_updates / self.sleep_steps))

    # ------------------------------------------------------------ phase

    def time_of_day(self, step: int) -> int:
        return step % self.day

    def phase(self, step: int) -> float:
        """Circadian phase in [0, 1); 0 = dawn, wake_frac = dusk."""
        return self.time_of_day(step) / self.day

    def is_awake(self, step: int) -> bool:
        return self.time_of_day(step) < self.wake_steps

    def just_fell_asleep(self, step: int) -> bool:
        return self.time_of_day(step) == self.wake_steps

    def just_woke(self, step: int) -> bool:
        return self.time_of_day(step) == 0

    def metrics(self, step: int) -> dict:
        return {
            "circadian_phase": self.phase(step),
            "asleep": 0.0 if self.is_awake(step) else 1.0,
        }
