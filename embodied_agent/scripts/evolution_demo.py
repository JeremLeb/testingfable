"""B5 demo: evolutionary outer loop & reproduction.

  (1) Innate priors adapt to the arena. In a cold, food-scarce world a
      population's *innate* genome (evaluated by instinct alone -- a reactive
      newborn, no learning) evolves: mean fitness rises and the innate thermal
      set-point slides toward the cold arena's optimum over generations.
  (2) The Baldwin effect. When a trait can be both learned within a life and
      inherited, evolution moves the *innate* component toward the adaptive
      value, so successive generations need less learning to reach competence
      -- a learned behaviour becomes partly innate.

Both use the real Genome / Population / selection machinery; only the fitness
differs. (The full-life fitness that also runs B1-B4 learning lives in
`train.live_one_life`; it is wired and tested but too heavy for this figure.)

  python -m embodied_agent.scripts.evolution_demo --config cpu_small
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np

from ..config import load_config
from ..evolution import Population, innate_fitness


def _harsh_arena(config: str):
    """A cold, food-scarce world so genomes actually differ in survival."""
    cfg = load_config(config)
    e = cfg.env
    e.fixed_layout = True
    e.n_hot, e.n_cold, e.temp_amp, e.temp_sigma_frac = 0, 4, 0.5, 0.35
    e.temp_danger, e.temp_damage_rate = 0.2, 0.02   # thermal mismatch bites
    e.n_food, e.food_energy, e.base_metabolism = 4, 0.35, 0.0015
    e.max_episode_steps = 1_000_000                 # lifespan capped by life_steps
    return cfg


def _baldwin_trace(evo_cfg, base_cfg, target=0.6, learn=0.5, cost=0.4,
                   seed=0):
    """Surrogate showing a learned trait becoming innate. The trait is one gene
    (the innate forward instinct); an individual can *learn* to close half the
    gap to the arena optimum, but learning is costly, so evolution is selected
    to move the innate value toward the optimum -- shrinking the gap that must
    be learned each generation."""
    gene = "action_bias_thrust"

    def fitness(genome, _seed):
        x0 = genome.genes[gene]                     # innate value
        x_learned = x0 + learn * (target - x0)      # what it learns to
        return -((target - x_learned) ** 2) - cost * abs(target - x0)

    pop = Population(evo_cfg, base_cfg, seed=seed)
    hist = pop.evolve(fitness)
    innate = np.array([h["gene_mean"][gene] for h in hist])
    gap = np.abs(target - innate)                   # learning still required
    return innate, gap, target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cpu_small")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--population", type=int, default=10)
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--life-steps", type=int, default=1200)
    ap.add_argument("--out", default="runs/evolution_demo")
    args = ap.parse_args()

    # (1) innate priors adapt to a harsh arena (reactive fitness)
    cfg = _harsh_arena(args.config)
    cfg.evolution.population = args.population
    cfg.evolution.generations = args.generations
    pop = Population(cfg.evolution, cfg, seed=args.seed)
    hist = pop.evolve(
        lambda g, s: innate_fitness(g.to_config(cfg), s, args.life_steps))
    gens = [h["gen"] for h in hist]
    fit = [h["fitness_mean"] for h in hist]
    temp_sp = [h["gene_mean"]["temp_setpoint"] for h in hist]

    # (2) Baldwin effect (surrogate, same machinery)
    bcfg = load_config(args.config)
    bcfg.evolution.population = 24
    bcfg.evolution.generations = 20
    bcfg.evolution.mutation_rate = 0.1
    innate, gap, target = _baldwin_trace(bcfg.evolution, bcfg, seed=args.seed)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _plot(gens, fit, temp_sp, innate, gap, target, out / "evolution_demo.png")

    print("=== B5 evolutionary outer loop & reproduction ===")
    print(f"innate adaptation: fitness {fit[0]:.0f} -> {fit[-1]:.0f}; "
          f"innate temp_setpoint {temp_sp[0]:.2f} -> {temp_sp[-1]:.2f} "
          f"(cold arena favours a low set-point)")
    print(f"Baldwin: innate trait {innate[0]:.2f} -> {innate[-1]:.2f} "
          f"(optimum {target}); learning gap {gap[0]:.2f} -> {gap[-1]:.2f}")
    print(f"wrote {out / 'evolution_demo.png'}")


def _plot(gens, fit, temp_sp, innate, gap, target, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(gens, fit, "o-", color="tab:green", label="mean fitness")
    ax.set_xlabel("generation")
    ax.set_ylabel("mean fitness (lifespan)", color="tab:green")
    ax2 = ax.twinx()
    ax2.plot(gens, temp_sp, "s--", color="tab:blue",
             label="innate temp_setpoint")
    ax2.set_ylabel("innate temp_setpoint", color="tab:blue")
    ax.set_title("Innate priors adapt to the arena\n(cold world -> low set-point)")

    ax = axes[1]
    g = range(len(innate))
    ax.plot(g, innate, "o-", color="tab:purple", label="innate trait value")
    ax.axhline(target, color="k", ls="--", alpha=0.6, label="arena optimum")
    ax.plot(g, gap, "^-", color="tab:red", label="learning still required")
    ax.set_title("Baldwin effect:\nlearned behaviour becomes innate")
    ax.set_xlabel("generation")
    ax.set_ylabel("trait value / learning gap")
    ax.legend(fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
