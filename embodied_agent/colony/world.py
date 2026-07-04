"""The colony world: one shared arena, many bodies.

Reuses a single-agent ``PetriEnv`` purely as the *static world* (walls,
temperature field, food, smell), then simulates N creatures on top of it --
unicycle kinematics, wall/obstacle/creature collisions, contested food,
per-body homeostasis, reproduction into live offspring, and death. The learned
"species brain" that drives the creatures lives in the runner, not here.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..env.arena import PetriEnv
from ..env import geometry as geo
from ..evolution.genome import Genome
from .creature import Creature


class ColonyEnv:
    def __init__(self, cfg: Config, seed: int = 0):
        self.full_cfg = cfg
        self.cfg = cfg.env
        self.sensor_cfg = cfg.sensor
        self.reward_cfg = cfg.reward
        self.col = cfg.colony
        self.rng = np.random.default_rng(seed)
        self._seed = seed
        # a PetriEnv owns the static world; we ignore its single body
        self.world_env = PetriEnv(cfg.env, cfg.sensor, seed=seed)
        self.obstacles = self.world_env.obstacles
        self.foods = self.world_env.foods
        self.creatures: list[Creature] = []
        self.births = 0
        self.deaths = 0
        self.steps = 0

    # world-field proxies used by each creature's senses
    def temperature(self, xy):
        return self.world_env.temperature(xy)

    def smell_field(self, xy):
        return self.world_env.smell_field(xy)

    # ------------------------------------------------------ population
    @property
    def living(self):
        return [c for c in self.creatures if c.alive]

    def _spawn(self, genome, pos=None, generation=0) -> Creature:
        if pos is None:
            pos = self.world_env._free_position(self.cfg.agent_radius * 2)
        heading = self.rng.uniform(-np.pi, np.pi)
        c = Creature(self, genome, pos, heading,
                     seed=int(self.rng.integers(1 << 31)), generation=generation)
        c.birth_step = self.steps
        self.creatures.append(c)
        return c

    def reset(self) -> list[Creature]:
        Creature._next_id = 0
        self.world_env.reset(seed=self._seed)
        self.obstacles = self.world_env.obstacles
        self.foods = self.world_env.foods
        self.creatures = []
        self.births = self.deaths = self.steps = 0
        for _ in range(self.col.n_init):
            self._spawn(Genome.random(self.rng))
        return self.living

    # ------------------------------------------------------ dynamics
    def _integrate(self, c: Creature, action):
        cfg = self.cfg
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        thrust, turn = action
        c.heading = (c.heading + turn * cfg.turn_rate * cfg.dt
                     + np.pi) % (2 * np.pi) - np.pi
        c.v += thrust * cfg.accel * cfg.dt
        c.v *= (1.0 - cfg.drag)
        c.v = float(np.clip(c.v, -cfg.reverse_frac * cfg.v_max, cfg.v_max))
        d = np.array([np.cos(c.heading), np.sin(c.heading)])
        new = c.pos + d * c.v * cfg.dt
        c.contacts = []
        c.impact_speed = 0.0
        lo, hi = cfg.agent_radius, cfg.arena_size - cfg.agent_radius
        for ax, nrm in ((0, np.array([1.0, 0.0])), (1, np.array([0.0, 1.0]))):
            if new[ax] < lo:
                new[ax] = lo; self._contact(c, nrm, d)
            elif new[ax] > hi:
                new[ax] = hi; self._contact(c, -nrm, d)
        for rect in self.obstacles:
            new, normal = geo.resolve_circle_rect(new, cfg.agent_radius, rect)
            if normal is not None:
                self._contact(c, normal, d)
        if c.contacts:
            c.v *= 0.1
        c.pos = new
        c.last_action = action

    def _contact(self, c, normal, motion_dir):
        impact = abs(c.v) * max(0.0, -float(np.dot(motion_dir, normal)))
        c.impact_speed += impact
        c.contacts.append(float(np.arctan2(-normal[1], -normal[0])))

    def _creature_collisions(self, living):
        r = self.cfg.agent_radius
        for i in range(len(living)):
            for j in range(i + 1, len(living)):
                a, b = living[i], living[j]
                delta = a.pos - b.pos
                dist = float(np.linalg.norm(delta))
                if 1e-6 < dist < 2 * r:
                    push = (2 * r - dist) / 2 * delta / dist
                    a.pos = a.pos + push
                    b.pos = b.pos - push
                    rel = 0.5 * (abs(a.v) + abs(b.v))
                    dmg = self.col.creature_collision_damage * rel * 10
                    a.impact_speed += dmg
                    b.impact_speed += dmg
                    ang = float(np.arctan2(delta[1], delta[0]))
                    a.contacts.append(ang)
                    b.contacts.append(ang + np.pi)
                    a.v *= 0.5
                    b.v *= 0.5

    def _feed(self, living):
        cfg = self.cfg
        eaten = {c.id: 0 for c in living}
        reach = cfg.agent_radius + cfg.food_radius
        for f in self.foods:
            if not f.active:
                continue
            best, bd = None, reach
            for c in living:
                dd = float(np.linalg.norm(c.pos - f.pos))
                if dd < bd:
                    best, bd = c, dd
            if best is not None:
                f.active = False
                f.timer = cfg.food_respawn_steps
                eaten[best.id] += 1
        for f in self.foods:  # respawn on the shared clock
            if not f.active:
                f.timer -= 1
                if f.timer <= 0:
                    f.active = True
                    f.pos = self.world_env._free_position(reach)
        return eaten

    def step(self, actions: dict):
        """actions: {creature_id -> np.array([thrust, turn])} for living
        creatures. Returns (results, births, deaths) where results maps id ->
        (obs, reward, done, info)."""
        living = self.living
        for c in living:
            self._integrate(c, actions.get(c.id, np.zeros(2)))
        self._creature_collisions(living)
        eaten = self._feed(living)

        results, births, deaths = {}, [], []
        for c in living:
            reward, dead = c.homeostasis.update(c, food_eaten=eaten[c.id])
            c.age += 1
            if dead:
                c.alive = False
                self.deaths += 1
                deaths.append(c)
                results[c.id] = (c.observe(), reward, True, c.info())
                continue
            # reproduction: sustained energy surplus -> a live, mutated child
            child = self._maybe_reproduce(c)
            if child is not None:
                births.append(child)
            results[c.id] = (c.observe(), reward, False, c.info())

        # immigration: keep a minimum population so the colony never dies out
        while len(self.living) < self.col.n_min:
            g = (Genome.random(self.rng) if self.col.newcomer_random
                 else self.living[0].genome.clone())
            births.append(self._spawn(g))

        self.steps += 1
        return results, births, deaths

    def _maybe_reproduce(self, c: Creature):
        col = self.col
        h = c.homeostasis
        if h.energy >= col.repro_energy_threshold:
            c.repro_readiness += col.repro_rate
        if (c.repro_readiness >= 1.0 and h.energy > col.repro_cost
                and len(self.living) < col.n_max):
            c.repro_readiness -= 1.0
            h.energy = float(np.clip(h.energy - col.repro_cost, 0.0, 1.0))
            g = c.genome.mutate(self.rng, col.mutation_rate, prob=0.9)
            off = np.clip(c.pos + self.rng.uniform(-1.0, 1.0, 2),
                          self.cfg.agent_radius,
                          self.cfg.arena_size - self.cfg.agent_radius)
            self.births += 1
            child = self._spawn(g, pos=off, generation=c.generation + 1)
            child.parent = c          # so the runner can inherit the parent's mind
            child.founder = c.founder  # same bloodline as the parent
            return child
        return None

    # ------------------------------------------------------ reporting
    def stats(self) -> dict:
        living = self.living
        if not living:
            return {"population": 0, "births": self.births,
                    "deaths": self.deaths, "generation": 0,
                    "mean_energy": 0.0, "max_generation": 0}
        gens = [c.generation for c in living]
        return {
            "population": len(living),
            "births": self.births,
            "deaths": self.deaths,
            "generation": float(np.mean(gens)),
            "max_generation": int(np.max(gens)),
            "mean_energy": float(np.mean([c.homeostasis.energy for c in living])),
            "mean_temp": float(np.mean([c.homeostasis.temp for c in living])),
            "mean_integrity": float(np.mean([c.homeostasis.integrity
                                             for c in living])),
            "mean_age": float(np.mean([c.age for c in living])),
        }

    # ------------------------------------------------------ evolution / lineage
    def gene_means(self) -> dict:
        """Population-mean innate traits of the living -- these drift as
        selection acts, so the colony evolves in place."""
        living = self.living
        if not living:
            return {"temp_setpoint": 0.0, "action_bias_thrust": 0.0,
                    "w_energy": 0.0}
        def m(k):
            return float(np.mean([c.genome.genes[k] for c in living]))
        return {"temp_setpoint": m("temp_setpoint"),
                "action_bias_thrust": m("action_bias_thrust"),
                "w_energy": m("w_energy")}

    def lineages(self):
        """(founder_id, living_count) per surviving bloodline, biggest first."""
        from collections import Counter
        return Counter(c.founder for c in self.living).most_common()

    def tree_creatures(self):
        """Every creature that is alive or an ancestor of a living one -- the
        family tree of the current population (dead ancestors are retained)."""
        keep = {}
        for c in self.living:
            node = c
            while node is not None and id(node) not in keep:
                keep[id(node)] = node
                node = node.parent
        return list(keep.values())
