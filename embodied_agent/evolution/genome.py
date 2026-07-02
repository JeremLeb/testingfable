"""The genome: the heritable innate priors an individual is born with.

A genome is a small set of scalar genes that are *not* learned within a life --
they are set by evolution and stamped onto a Config before the individual is
built. They span the three things biology makes innate:

  * homeostatic set-points / drive arbitration (temp_setpoint, base drive
    weights) -- what the body treats as "good",
  * sensor morphology (n_rays, fov_deg) -- the shape of the receptor array,
  * a behavioural prior (action_bias) -- an instinct the newborn acts on before
    it has learned anything (the substrate for the Baldwin effect), and the
    early plasticity (young_gain) that development (B3) then anneals.

Lifetime learning (B1-B4) tunes whatever the genome leaves underspecified.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

# gene -> (low, high, is_integer)
GENE_BOUNDS = {
    "temp_setpoint": (0.2, 0.8, False),
    "w_energy": (0.3, 3.0, False),
    "w_thermal": (0.0, 2.0, False),
    "w_integrity": (0.3, 3.0, False),
    "n_rays": (6, 20, True),
    "fov_deg": (60.0, 200.0, False),
    "young_gain": (1.0, 4.0, False),
    "action_bias_thrust": (-1.0, 1.0, False),
    "action_bias_turn": (-1.0, 1.0, False),
}


@dataclass
class Genome:
    genes: dict = field(default_factory=dict)

    # ------------------------------------------------------------ construction

    @classmethod
    def random(cls, rng: np.random.Generator) -> "Genome":
        g = {}
        for name, (lo, hi, is_int) in GENE_BOUNDS.items():
            v = rng.uniform(lo, hi)
            g[name] = int(round(v)) if is_int else float(v)
        return cls(g)

    def clone(self) -> "Genome":
        return Genome(dict(self.genes))

    # ------------------------------------------------------------ operators

    def mutate(self, rng: np.random.Generator, rate: float,
               prob: float) -> "Genome":
        """Gaussian perturbation per gene (std = rate * gene range), clipped to
        bounds; integer genes are rounded."""
        g = dict(self.genes)
        for name, (lo, hi, is_int) in GENE_BOUNDS.items():
            if rng.random() > prob:
                continue
            v = g[name] + rng.normal(0.0, rate * (hi - lo))
            v = float(np.clip(v, lo, hi))
            g[name] = int(round(v)) if is_int else v
        return Genome(g)

    @staticmethod
    def crossover(a: "Genome", b: "Genome",
                  rng: np.random.Generator) -> "Genome":
        """Uniform per-gene inheritance from two parents."""
        return Genome({name: (a.genes[name] if rng.random() < 0.5
                              else b.genes[name]) for name in GENE_BOUNDS})

    # ------------------------------------------------------------ expression

    def to_config(self, base_cfg) -> "object":
        """Stamp the genome onto a *copy* of a base Config (phenotype)."""
        cfg = copy.deepcopy(base_cfg)
        g = self.genes
        cfg.env.temp_setpoint = g["temp_setpoint"]
        cfg.reward.w_energy = g["w_energy"]
        cfg.reward.w_thermal = g["w_thermal"]
        cfg.reward.w_integrity = g["w_integrity"]
        cfg.sensor.n_rays = int(g["n_rays"])
        cfg.sensor.fov_deg = g["fov_deg"]
        cfg.dev.young_gain = g["young_gain"]
        cfg.agent.action_bias = [g["action_bias_thrust"], g["action_bias_turn"]]
        return cfg

    def vector(self) -> np.ndarray:
        return np.array([self.genes[k] for k in GENE_BOUNDS], dtype=float)
