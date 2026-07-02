"""Interoceptive viability variables and their dynamics.

Three variables in [0, 1]: energy (drains with metabolism and movement,
restored by food), internal temperature (relaxes toward the ambient field;
extreme values burn/freeze integrity), and integrity (damaged by collisions
and thermal extremes, regenerates slowly). Death — energy or integrity
hitting zero — terminates the episode.

These are simultaneously a *sensory modality* (the intero channel the world
model must predict) and, from milestone 3, the source of reward via drive
reduction. Nothing else in the system invents pleasure or pain scalars.
"""
from __future__ import annotations

import numpy as np

from ..config import EnvConfig


class Homeostasis:
    def __init__(self, cfg: EnvConfig, reward_cfg=None):
        self.cfg = cfg
        self.reward_cfg = reward_cfg
        self.reset()

    def reset(self):
        self.energy = 1.0
        self.temp = self.cfg.temp_setpoint
        self.integrity = 1.0
        self.dead = False
        self.last_reward_terms: dict[str, float] = {}

    # ------------------------------------------------------------ drives

    def drives(self) -> dict[str, float]:
        """Distance of each viability variable from its setpoint."""
        return {
            "energy": 1.0 - self.energy,
            "thermal": abs(self.temp - self.cfg.temp_setpoint),
            "integrity": 1.0 - self.integrity,
        }

    # ------------------------------------------------------------ dynamics

    def update(self, env, food_eaten: int) -> tuple[float, bool]:
        cfg = self.cfg
        prev_drives = self.drives()

        self.energy -= cfg.base_metabolism
        self.energy -= cfg.move_cost * abs(env.last_action[0])
        self.energy += cfg.food_energy * food_eaten
        self.energy = float(np.clip(self.energy, 0.0, 1.0))

        ambient = float(env.temperature(env.pos))
        self.temp += cfg.temp_coupling * (ambient - self.temp)

        self.integrity -= cfg.collision_damage * env.impact_speed
        excess = abs(self.temp - cfg.temp_setpoint) - cfg.temp_danger
        if excess > 0:
            self.integrity -= cfg.temp_damage_rate * (excess / cfg.temp_danger)
        self.integrity += cfg.integrity_regen
        self.integrity = float(np.clip(self.integrity, 0.0, 1.0))

        self.dead = self.energy <= 0.0 or self.integrity <= 0.0
        reward = self._reward(prev_drives) if self.reward_cfg else 0.0
        return reward, self.dead

    def _reward(self, prev_drives: dict[str, float]) -> float:
        raise NotImplementedError("wired up in milestone 3")

    # ------------------------------------------------------------ logging

    def info(self) -> dict:
        d = self.drives()
        out = {
            "energy": self.energy, "temp": self.temp,
            "integrity": self.integrity, "dead": self.dead,
            "drive_energy": d["energy"], "drive_thermal": d["thermal"],
            "drive_integrity": d["integrity"],
        }
        out.update(self.last_reward_terms)
        return out
