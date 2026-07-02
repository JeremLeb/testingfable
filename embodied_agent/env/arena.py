"""The petri arena: kinematics, food, temperature field, walls, collisions.

Gymnasium-style API (reset/step) without a gymnasium dependency. Homeostatic
dynamics and the full sensor suite are layered on in env/homeostasis.py and
env/sensors.py; this module owns the physical world.
"""
from __future__ import annotations

import numpy as np

from ..config import EnvConfig, SensorConfig
from . import geometry as geo


class Food:
    __slots__ = ("pos", "active", "timer")

    def __init__(self, pos):
        self.pos = np.asarray(pos, dtype=np.float64)
        self.active = True
        self.timer = 0


class PetriEnv:
    """Continuous 2D arena with a circular unicycle agent.

    Action: np.array([thrust, turn]) in [-1, 1]^2.
    step() returns (obs, reward, terminated, truncated, info). Until the
    sensor/homeostasis layers are attached, obs is a minimal proprio dict
    and reward is 0.
    """

    def __init__(self, cfg: EnvConfig, sensor_cfg: SensorConfig | None = None,
                 seed: int = 0):
        self.cfg = cfg
        self.sensor_cfg = sensor_cfg
        self.rng = np.random.default_rng(seed)
        self.sensors = None       # attached by env/sensors.py
        self.homeostasis = None   # attached by env/homeostasis.py
        self._build_layout()
        self.reset()

    # ------------------------------------------------------------ layout

    def _build_layout(self):
        cfg = self.cfg
        S = cfg.arena_size
        if cfg.fixed_layout:
            self.obstacles = [
                (S * 0.30, S * 0.55, S * 0.15, S * 0.08),
                (S * 0.60, S * 0.20, S * 0.08, S * 0.20),
                (S * 0.15, S * 0.15, S * 0.12, S * 0.10),
            ][: cfg.n_obstacles]
            hot = [(S * 0.80, S * 0.80), (S * 0.20, S * 0.75)][: cfg.n_hot]
            cold = [(S * 0.75, S * 0.15), (S * 0.10, S * 0.40)][: cfg.n_cold]
        else:
            self.obstacles = []
            for _ in range(cfg.n_obstacles):
                w = self.rng.uniform(cfg.obstacle_min, cfg.obstacle_max)
                h = self.rng.uniform(cfg.obstacle_min, cfg.obstacle_max)
                x = self.rng.uniform(1.0, S - w - 1.0)
                y = self.rng.uniform(1.0, S - h - 1.0)
                self.obstacles.append((x, y, w, h))
            hot = [self.rng.uniform(0.1 * S, 0.9 * S, 2) for _ in range(cfg.n_hot)]
            cold = [self.rng.uniform(0.1 * S, 0.9 * S, 2) for _ in range(cfg.n_cold)]
        sigma = cfg.temp_sigma_frac * S
        self.temp_sources = (
            [(np.asarray(c, float), cfg.temp_amp, sigma) for c in hot]
            + [(np.asarray(c, float), -cfg.temp_amp, sigma) for c in cold]
        )

    def _free_position(self, clearance: float) -> np.ndarray:
        S = self.cfg.arena_size
        for _ in range(200):
            p = self.rng.uniform(clearance, S - clearance, 2)
            if all(geo.resolve_circle_rect(p, clearance, r)[1] is None
                   for r in self.obstacles):
                return p
        return np.array([S / 2, S / 2])

    # ------------------------------------------------------------ fields

    def temperature(self, xy: np.ndarray) -> np.ndarray:
        """Ambient temperature at points xy of shape (..., 2), in [0, 1]."""
        xy = np.asarray(xy, dtype=np.float64)
        t = np.full(xy.shape[:-1], 0.5)
        for center, amp, sigma in self.temp_sources:
            d2 = np.sum((xy - center) ** 2, axis=-1)
            t = t + amp * np.exp(-d2 / (2 * sigma * sigma))
        return np.clip(t, 0.0, 1.0)

    def smell_field(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Chemical concentration and its gradient from active food plumes."""
        xy = np.asarray(xy, dtype=np.float64)
        sigma = self.sensor_cfg.smell_sigma_frac * self.cfg.arena_size \
            if self.sensor_cfg else 0.25 * self.cfg.arena_size
        conc = np.zeros(xy.shape[:-1])
        grad = np.zeros(xy.shape[:-1] + (2,))
        for f in self.foods:
            if not f.active:
                continue
            delta = f.pos - xy
            d2 = np.sum(delta ** 2, axis=-1)
            c = np.exp(-d2 / (2 * sigma * sigma))
            conc = conc + c
            grad = grad + (c / (sigma * sigma))[..., None] * delta
        return conc, grad

    # ------------------------------------------------------------ api

    def reset(self, seed: int | None = None):
        cfg = self.cfg
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            if not cfg.fixed_layout:
                self._build_layout()
        self.foods = [Food(self._free_position(cfg.agent_radius + cfg.food_radius))
                      for _ in range(cfg.n_food)]
        if cfg.fixed_layout:
            self.pos = np.array([cfg.arena_size / 2, cfg.arena_size * 0.4])
        else:
            self.pos = self._free_position(cfg.agent_radius * 2)
        self.heading = self.rng.uniform(-np.pi, np.pi)
        self.v = 0.0
        self.last_action = np.zeros(2)
        self.step_count = 0
        self.contacts = []        # world-frame contact normal angles this step
        self.impact_speed = 0.0
        if self.homeostasis is not None:
            self.homeostasis.reset()
        if self.sensors is not None:
            self.sensors.reset()
        obs = self._observe()
        return obs, self._info(food_eaten=0)

    def step(self, action):
        cfg = self.cfg
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        thrust, turn = action

        # unicycle integration
        self.heading = (self.heading + turn * cfg.turn_rate * cfg.dt
                        + np.pi) % (2 * np.pi) - np.pi
        self.v += thrust * cfg.accel * cfg.dt
        self.v *= (1.0 - cfg.drag)
        self.v = np.clip(self.v, -cfg.reverse_frac * cfg.v_max, cfg.v_max)
        direction = np.array([np.cos(self.heading), np.sin(self.heading)])
        new_pos = self.pos + direction * self.v * cfg.dt

        # collisions: arena bounds then obstacles
        self.contacts = []
        self.impact_speed = 0.0
        lo, hi = cfg.agent_radius, cfg.arena_size - cfg.agent_radius
        if new_pos[0] < lo:
            new_pos[0] = lo
            self._register_contact(np.array([1.0, 0.0]), direction)
        elif new_pos[0] > hi:
            new_pos[0] = hi
            self._register_contact(np.array([-1.0, 0.0]), direction)
        if new_pos[1] < lo:
            new_pos[1] = lo
            self._register_contact(np.array([0.0, 1.0]), direction)
        elif new_pos[1] > hi:
            new_pos[1] = hi
            self._register_contact(np.array([0.0, -1.0]), direction)
        for rect in self.obstacles:
            new_pos, normal = geo.resolve_circle_rect(
                new_pos, cfg.agent_radius, rect)
            if normal is not None:
                self._register_contact(normal, direction)
        if self.contacts:
            self.v *= 0.1  # nearly dead stop on impact
        self.pos = new_pos

        # food
        food_eaten = 0
        for f in self.foods:
            if f.active and np.linalg.norm(self.pos - f.pos) < \
                    cfg.agent_radius + cfg.food_radius:
                f.active = False
                f.timer = cfg.food_respawn_steps
                food_eaten += 1
            elif not f.active:
                f.timer -= 1
                if f.timer <= 0:
                    f.active = True
                    if not cfg.fixed_layout:
                        f.pos = self._free_position(
                            cfg.agent_radius + cfg.food_radius)

        self.last_action = action
        self.step_count += 1

        terminated = False
        reward = 0.0
        if self.homeostasis is not None:
            reward, terminated = self.homeostasis.update(
                self, food_eaten=food_eaten)
        truncated = self.step_count >= cfg.max_episode_steps
        return (self._observe(), reward, terminated, truncated,
                self._info(food_eaten=food_eaten))

    # ------------------------------------------------------------ helpers

    def _register_contact(self, normal: np.ndarray, motion_dir: np.ndarray):
        impact = abs(self.v) * max(0.0, -float(np.dot(motion_dir, normal)))
        self.impact_speed += impact
        self.contacts.append(float(np.arctan2(-normal[1], -normal[0])))

    def _observe(self) -> dict:
        if self.sensors is not None:
            return self.sensors.observe(self)
        return {"proprio": np.array([
            self.v / self.cfg.v_max, np.sin(self.heading),
            np.cos(self.heading), *self.last_action], dtype=np.float32)}

    def _info(self, food_eaten: int) -> dict:
        info = {
            "pos": self.pos.copy(), "heading": self.heading, "speed": self.v,
            "collision": bool(self.contacts),
            "impact_speed": self.impact_speed,
            "food_eaten": food_eaten,
            "ambient_temp": float(self.temperature(self.pos)),
            "step": self.step_count,
        }
        if self.homeostasis is not None:
            info.update(self.homeostasis.info())
        return info
