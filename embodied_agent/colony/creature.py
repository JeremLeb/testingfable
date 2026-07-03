"""One creature in the colony: a body with its own genome, metabolism and
senses that *shares* the world's static fields (walls, food, temperature).

A Creature doubles as the ``env`` view its own SensorSuite and Homeostasis
read from -- it exposes the body attributes (pos, heading, velocity, contacts,
last action, homeostasis) directly and proxies the world-level fields
(obstacles, foods, temperature, smell) to the shared world. That lets the
existing single-agent sensors and interoception work unchanged, one per body.
"""
from __future__ import annotations

import copy

import numpy as np

from ..env.homeostasis import Homeostasis
from ..env.sensors import SensorSuite


class Creature:
    _next_id = 0

    def __init__(self, world, genome, pos, heading, seed, generation=0):
        self.world = world
        self.genome = genome
        self.id = Creature._next_id
        Creature._next_id += 1
        self.generation = generation

        g = genome.genes
        # the body plan (morphology, kinematics, world) is shared; the genome
        # personalises the set-points and drive weights (its "temperament").
        self.cfg = copy.deepcopy(world.cfg)
        self.cfg.temp_setpoint = float(g["temp_setpoint"])
        reward_cfg = copy.deepcopy(world.reward_cfg)
        reward_cfg.w_energy = float(g["w_energy"])
        reward_cfg.w_thermal = float(g["w_thermal"])
        reward_cfg.w_integrity = float(g["w_integrity"])
        self.homeostasis = Homeostasis(self.cfg, reward_cfg=reward_cfg)
        self.sensors = SensorSuite(world.sensor_cfg, self.cfg,
                                   np.random.default_rng(seed))
        self.action_bias = np.array([g["action_bias_thrust"],
                                     g["action_bias_turn"]], dtype=np.float32)

        # body state
        self.pos = np.asarray(pos, dtype=np.float64)
        self.heading = float(heading)
        self.v = 0.0
        self.last_action = np.zeros(2)
        self.impact_speed = 0.0
        self.contacts = []
        self.age = 0
        self.repro_readiness = 0.0
        self.alive = True
        # attached by the runner: this creature's mind (its own world model +
        # actor + memory, unless the colony is set to a shared brain), a
        # DreamerAgent carrying its recurrent state, and a replay Episode.
        self.parent = None       # set at birth; the runner inherits its brain
        self.mind = None
        self.agent = None
        self.episode = None
        self.obs = self.sensors.observe(self)

    # ------------------------------------------------------ world proxy
    @property
    def obstacles(self):
        return self.world.obstacles

    @property
    def foods(self):
        return self.world.foods

    @property
    def sensor_cfg(self):
        return self.world.sensor_cfg

    def temperature(self, xy):
        return self.world.temperature(xy)

    def smell_field(self, xy):
        return self.world.smell_field(xy)

    # ------------------------------------------------------ senses
    def observe(self) -> dict:
        self.obs = self.sensors.observe(self)
        return self.obs

    def info(self) -> dict:
        h = self.homeostasis
        return {"id": self.id, "generation": self.generation, "age": self.age,
                "energy": h.energy, "temp": h.temp, "integrity": h.integrity,
                "pos": self.pos.copy(), "heading": self.heading}
