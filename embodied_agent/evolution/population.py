"""Population dynamics: evaluate genomes by fitness, select, reproduce.

Fitness is *endogenous* -- it is not a hand-designed objective but what the
body actually achieves in the world (how long it survives, how much it eats,
how many offspring it bears). Genomes that do better leave more (mutated)
descendants, so the innate priors drift toward whatever suits the arena.

``innate_fitness`` scores a genome cheaply by its *instinct alone* -- a
reactive newborn policy driven by the genome's ``action_bias``, no learning --
which isolates how well the innate priors fit the arena. The full-life fitness
(``train.live_one_life``) additionally lets B1-B4 lifetime learning run.
"""
from __future__ import annotations

import numpy as np

from .genome import GENE_BOUNDS, Genome
from ..env import make_env


def innate_fitness(cfg, seed: int, life_steps: int) -> float:
    """Lifespan (+ food + offspring) of a newborn acting on instinct only.

    The reactive policy is the genome's innate ``action_bias`` plus a little
    wander; there is no world model and no learning, so this measures the
    survival value of the *innate* priors in this arena. Death (energy or
    integrity hitting zero) ends the life."""
    env = make_env(cfg, seed=seed)
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed + 1)
    bias = np.asarray(cfg.agent.action_bias, dtype=float)
    a = np.zeros(2)
    steps = food = 0
    offspring = 0
    for _ in range(life_steps):
        a = 0.85 * a + 0.15 * (bias + rng.uniform(-0.4, 0.4, 2))
        obs, _, term, _, info = env.step(np.clip(a, -1.0, 1.0))
        steps += 1
        food += info["food_eaten"]
        offspring = info.get("offspring", 0)
        if term:  # death; truncation is ignored (fitness runs to life_steps)
            break
    return float(steps + 2.0 * food + 8.0 * offspring)


class Population:
    def __init__(self, evo_cfg, base_cfg, seed: int = 0):
        self.evo = evo_cfg
        self.base_cfg = base_cfg
        self.rng = np.random.default_rng(seed)
        self.members = [Genome.random(self.rng)
                        for _ in range(evo_cfg.population)]
        self.history: list[dict] = []

    # ------------------------------------------------------------ evolution

    def evolve(self, fitness_fn, generations: int | None = None) -> list[dict]:
        """Run the outer loop. `fitness_fn(genome, seed) -> float`."""
        gens = generations if generations is not None else self.evo.generations
        for gen in range(gens):
            fits = np.array([fitness_fn(m, 1000 * gen + i)
                             for i, m in enumerate(self.members)])
            self._record(gen, fits)
            self.members = self._next_generation(fits)
        return self.history

    def _next_generation(self, fits: np.ndarray) -> list:
        evo, n = self.evo, len(self.members)
        order = np.argsort(fits)[::-1]
        n_elite = max(1, int(evo.elite_frac * n))
        nxt = [self.members[i].clone() for i in order[:n_elite]]  # elitism
        while len(nxt) < n:
            p1 = self._tournament(fits)
            p2 = self._tournament(fits)
            child = Genome.crossover(p1, p2, self.rng).mutate(
                self.rng, evo.mutation_rate, evo.mutation_prob)
            nxt.append(child)
        return nxt

    def _tournament(self, fits: np.ndarray) -> Genome:
        idx = self.rng.choice(len(self.members),
                              size=min(self.evo.tournament, len(self.members)),
                              replace=False)
        return self.members[idx[int(np.argmax(fits[idx]))]]

    def _record(self, gen: int, fits: np.ndarray):
        genes = np.stack([m.vector() for m in self.members])
        self.history.append({
            "gen": gen,
            "fitness_mean": float(fits.mean()),
            "fitness_max": float(fits.max()),
            "gene_mean": {k: float(genes[:, i].mean())
                          for i, k in enumerate(GENE_BOUNDS)},
        })
