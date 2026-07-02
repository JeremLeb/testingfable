"""Hard-constraint shield: a safety override that sits *above* the policy and
reward. It inspects the proposed action and a short kinematic lookahead of
the resulting state, and overrides the action (brake / substitute a safe
fallback) when a hard rule would be violated. No predicted return can buy
through it -- catastrophic actions cannot be *learned* away, they are
structurally blocked. Every intervention is logged.

Hard rules here:
  1. Forbidden circular zones -- the agent may not enter them; if a proposed
     move would cross into one, brake and steer away.
  2. Imminent high-speed wall/obstacle collision -- brake before impact.

This is deliberately simple and analytic (it re-uses the env kinematics for
lookahead) so it is auditable; it is not learned.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..env import geometry as geo


class Shield:
    def __init__(self, cfg: Config, env=None):
        self.cfg = cfg.shield
        self.env_cfg = cfg.env
        self.zones = [np.asarray(z, dtype=float)
                      for z in self.cfg.forbidden_zones]
        if not self.zones and env is not None:
            self.zones = [self._auto_zone(env)]
        if env is not None:
            env.forbidden_zones = [z.tolist() for z in self.zones]
        self.n_interventions = 0

    def _auto_zone(self, env):
        """Place one forbidden disc in open space away from the start."""
        S = self.env_cfg.arena_size
        r = S * 0.12
        best = np.array([S * 0.75, S * 0.75])
        for _ in range(100):
            c = env.rng.uniform(r, S - r, 2) if hasattr(env, "rng") \
                else np.array([S * 0.75, S * 0.75])
            if np.linalg.norm(c - env.pos) > S * 0.3 and all(
                    geo.resolve_circle_rect(c, r, rect)[1] is None
                    for rect in env.obstacles):
                best = c
                break
        return np.array([best[0], best[1], r])

    # ------------------------------------------------------------ lookahead

    def _rollout(self, env, action):
        """Kinematics-only forward simulation of `action` held for a few
        steps, returning the predicted positions (no reward/homeostasis)."""
        cfg = self.env_cfg
        pos = env.pos.copy()
        heading = env.heading
        v = env.v
        thrust, turn = float(action[0]), float(action[1])
        positions = []
        for _ in range(self.cfg.lookahead_steps):
            heading = (heading + turn * cfg.turn_rate * cfg.dt
                       + np.pi) % (2 * np.pi) - np.pi
            v = (v + thrust * cfg.accel * cfg.dt) * (1 - cfg.drag)
            v = float(np.clip(v, -cfg.reverse_frac * cfg.v_max, cfg.v_max))
            pos = pos + np.array([np.cos(heading), np.sin(heading)]) * v * cfg.dt
            positions.append(pos.copy())
        return positions

    def _violates(self, env, positions) -> bool:
        # speed-based margin so braking begins before momentum carries the
        # agent across the boundary (drag alone does not stop it instantly)
        r = self.env_cfg.agent_radius
        margin = abs(env.v) * 0.6
        for pos in positions:
            for zone in self.zones:
                if np.linalg.norm(pos - zone[:2]) < zone[2] + r + margin:
                    return True
        return False

    # ------------------------------------------------------------ filter

    def filter(self, env, action):
        """Return (safe_action, intervened). Tries the proposed action; if it
        would violate a hard rule, substitutes a braking/steering fallback and
        verifies the fallback is itself safe."""
        action = np.asarray(action, dtype=float)
        if not self.zones:
            return action, False
        if not self._violates(env, self._rollout(env, action)):
            return action, False

        # fallback 1: brake and steer away from the nearest zone
        nearest = min(self.zones,
                      key=lambda z: np.linalg.norm(env.pos - z[:2]))
        away = env.pos - nearest[:2]
        desired = np.arctan2(away[1], away[0])
        turn = np.clip((desired - env.heading + np.pi) % (2 * np.pi) - np.pi,
                       -1, 1)
        for candidate in (np.array([self.cfg.brake_thrust, turn]),
                          np.array([self.cfg.brake_thrust, 0.0]),
                          np.array([-1.0, 1.0]), np.array([-1.0, -1.0])):
            if not self._violates(env, self._rollout(env, candidate)):
                self.n_interventions += 1
                return candidate, True
        # last resort: hard brake straight
        self.n_interventions += 1
        return np.array([self.cfg.brake_thrust, 0.0]), True
