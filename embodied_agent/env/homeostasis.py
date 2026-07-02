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
from .neuromod import allostatic_weights


class Homeostasis:
    def __init__(self, cfg: EnvConfig, reward_cfg=None, neuromod_cfg=None,
                 evo_cfg=None):
        self.cfg = cfg
        self.reward_cfg = reward_cfg
        self.neuromod_cfg = neuromod_cfg
        self.evo_cfg = evo_cfg
        self.reset()

    def reset(self):
        self.energy = 1.0
        self.temp = self.cfg.temp_setpoint
        self.integrity = 1.0
        self.dead = False
        self.repro_readiness = 0.0   # B5 reproductive drive accumulator
        self.offspring = 0
        self.last_reward_terms: dict[str, float] = {}

    def _update_reproduction(self):
        """B5: sustained energy surplus accrues reproductive readiness; crossing
        the threshold bears one offspring (paid for in energy). No-op unless the
        evolution reproduction drive is enabled."""
        evo = self.evo_cfg
        if not (evo is not None and evo.reproduction):
            return 0
        born = 0
        if self.energy >= evo.repro_energy_threshold:
            self.repro_readiness += evo.repro_rate
        if self.repro_readiness >= 1.0 and self.energy > evo.repro_cost:
            self.repro_readiness -= 1.0
            self.energy = float(np.clip(self.energy - evo.repro_cost, 0.0, 1.0))
            self.offspring += 1
            born = 1
        return born

    # ------------------------------------------------------------ drives

    def drives(self) -> dict[str, float]:
        """Distance of each viability variable from its setpoint."""
        return {
            "energy": 1.0 - self.energy,
            "thermal": abs(self.temp - self.cfg.temp_setpoint),
            "integrity": 1.0 - self.integrity,
        }

    def spend_energy(self, amount: float):
        """Debit energy for an external cost (e.g. the metabolic cost of
        cognition, B4). Kept in [0, 1]; can push the body toward death."""
        if amount:
            self.energy = float(np.clip(self.energy - amount, 0.0, 1.0))
            self.dead = self.dead or self.energy <= 0.0

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
        self.last_born = self._update_reproduction()
        reward = self._reward(prev_drives) if self.reward_cfg else 0.0
        return reward, self.dead

    def _reward(self, prev_drives: dict[str, float]) -> float:
        """Reward = weighted reduction in distance-from-setpoint (drive
        reduction), plus a one-off death penalty. Each drive's instantaneous
        contribution is stored so the logs answer 'which drive is winning'.

        Sign convention: a drive *falling* (moving toward its setpoint) is
        good, so contribution = w * (prev - curr). Eating raises energy ->
        energy drive falls -> positive reward. Taking a hit raises the
        integrity drive -> negative reward.
        """
        rc = self.reward_cfg
        curr = self.drives()
        # allostatic weighting: current bodily deficits set how much each drive
        # matters right now (falls back to the fixed weights if neuromod off).
        weights = allostatic_weights(curr, rc, self.neuromod_cfg) \
            if self.neuromod_cfg is not None else {
                "energy": rc.w_energy, "thermal": rc.w_thermal,
                "integrity": rc.w_integrity}
        terms = {}
        total = 0.0
        for name, w in weights.items():
            delta = (prev_drives[name] - curr[name]) * w * rc.reward_scale
            terms[f"reward_{name}"] = delta
            terms[f"weight_{name}"] = w
            total += delta
        if self.dead:
            terms["reward_death"] = -rc.death_penalty
            total -= rc.death_penalty
        else:
            terms["reward_death"] = 0.0
        terms["reward_total"] = total
        self.last_reward_terms = terms
        return total

    # ------------------------------------------------------------ logging

    def info(self) -> dict:
        d = self.drives()
        out = {
            "energy": self.energy, "temp": self.temp,
            "integrity": self.integrity, "dead": self.dead,
            "drive_energy": d["energy"], "drive_thermal": d["thermal"],
            "drive_integrity": d["integrity"],
            "offspring": self.offspring,
            "repro_readiness": self.repro_readiness,
        }
        out.update(self.last_reward_terms)
        return out
