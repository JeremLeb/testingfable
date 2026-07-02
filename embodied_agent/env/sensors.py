"""Multi-rate egocentric sensor suite.

Every modality is a named float32 vector in the observation dict:
  vision   (n_rays * 4,)  ray distance + one-hot channel (food/wall/none)
  touch    (sectors,)     contact pressure per body sector, decaying
  proprio  (6,)           speed, sin/cos heading, efference copy, collision flag
  intero   (3,)           energy, internal temperature, integrity
  smell    (3,)           plume concentration + egocentric gradient (slow rate)

Smell deliberately refreshes only every `smell_period` steps (holding its last
value in between) so the world model has to fuse senses with different rates.
"""
from __future__ import annotations

import numpy as np

from ..config import EnvConfig, SensorConfig
from . import geometry as geo


class SensorSuite:
    def __init__(self, cfg: SensorConfig, env_cfg: EnvConfig,
                 rng: np.random.Generator):
        self.cfg = cfg
        self.env_cfg = env_cfg
        self.rng = rng
        self.reset()

    @property
    def spaces(self) -> dict[str, int]:
        return {
            "vision": self.cfg.n_rays * 4,
            "touch": self.cfg.touch_sectors,
            "proprio": 6,
            "intero": 3,
            "smell": 3,
        }

    def reset(self):
        self._touch = np.zeros(self.cfg.touch_sectors, dtype=np.float64)
        self._smell = np.zeros(3, dtype=np.float64)
        self._steps = 0

    # ------------------------------------------------------------ modalities

    def _vision(self, env) -> np.ndarray:
        cfg = self.cfg
        half = np.deg2rad(cfg.fov_deg) / 2
        angles = env.heading + np.linspace(-half, half, cfg.n_rays)
        out = np.zeros((cfg.n_rays, 4))
        for i, ang in enumerate(angles):
            d = np.array([np.cos(ang), np.sin(ang)])
            wall_d = geo.ray_bounds(env.pos, d, env.cfg.arena_size)
            for rect in env.obstacles:
                wall_d = min(wall_d, geo.ray_rect(env.pos, d, rect))
            food_d = np.inf
            for f in env.foods:
                if f.active:
                    food_d = min(food_d, geo.ray_circle(
                        env.pos, d, f.pos, env.cfg.food_radius))
            dist = min(wall_d, food_d)
            if dist > cfg.ray_max_dist:
                out[i] = [1.0, 0.0, 0.0, 1.0]           # nothing in range
            elif food_d < wall_d:
                out[i] = [dist / cfg.ray_max_dist, 1.0, 0.0, 0.0]  # food
            else:
                out[i] = [dist / cfg.ray_max_dist, 0.0, 1.0, 0.0]  # wall
        return out.reshape(-1)

    def _update_touch(self, env):
        self._touch *= self.cfg.touch_decay
        sectors = self.cfg.touch_sectors
        for contact_angle in env.contacts:
            rel = (contact_angle - env.heading) % (2 * np.pi)
            idx = int(rel / (2 * np.pi) * sectors) % sectors
            self._touch[idx] = 1.0

    def _proprio(self, env) -> np.ndarray:
        return np.array([
            env.v / env.cfg.v_max,
            np.sin(env.heading), np.cos(env.heading),
            env.last_action[0], env.last_action[1],
            1.0 if env.contacts else 0.0,
        ])

    def _intero(self, env) -> np.ndarray:
        h = env.homeostasis
        if h is None:
            return np.array([1.0, 0.5, 1.0])
        return np.array([h.energy, h.temp, h.integrity])

    def _update_smell(self, env):
        conc, grad = env.smell_field(env.pos)
        cos_h, sin_h = np.cos(env.heading), np.sin(env.heading)
        fwd = grad[0] * cos_h + grad[1] * sin_h
        lat = -grad[0] * sin_h + grad[1] * cos_h
        scale = self.env_cfg.arena_size
        self._smell = np.array([
            np.tanh(conc), np.tanh(fwd * scale), np.tanh(lat * scale)])

    # ------------------------------------------------------------ observe

    def observe(self, env) -> dict[str, np.ndarray]:
        self._update_touch(env)
        if self._steps % self.cfg.smell_period == 0:
            self._update_smell(env)
        self._steps += 1
        obs = {
            "vision": self._vision(env),
            "touch": self._touch.copy(),
            "proprio": self._proprio(env),
            "intero": self._intero(env),
            "smell": self._smell.copy(),
        }
        for name, arr in obs.items():
            std = self.cfg.noise.get(name, 0.0)
            if std > 0:
                arr = arr + self.rng.normal(0.0, std, arr.shape)
            obs[name] = arr.astype(np.float32)
        return obs
