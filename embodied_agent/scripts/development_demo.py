"""B3 demo: development & critical periods.

  (1) The age-dependent plasticity schedule actually used in training: high at
      birth, annealing to a mature floor (a critical period).
  (2) A critical-period *imprinting* probe. An organism lives one life whose
      world changes half-way through (regime A -> regime B). Under a critical
      period (high early plasticity, low late plasticity) the *early* regime A
      imprints and resists being overwritten; under constant plasticity --
      matched for total learning budget -- the late regime B wins and A is
      forgotten. Early experience has an outsized, sticky effect.

  python -m embodied_agent.scripts.development_demo --config cpu_small

The probe is a controlled online-regression toy so the effect is crisp and
CPU-cheap; it uses the *same* Development schedule the trainer applies to the
world-model learning rate.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np

from ..config import load_config
from ..agent.development import Development


def _imprint_life(schedule, base_lr, T, switch, d, rng):
    """One online life: fit a linear model whose target mapping flips from
    w_A to w_B at `switch`. Returns per-step error on held-out A and B sets.
    `schedule` maps age -> lr multiplier."""
    w_a = rng.standard_normal(d)
    w_b = rng.standard_normal(d)
    xa = rng.standard_normal((256, d)); ya = xa @ w_a
    xb = rng.standard_normal((256, d)); yb = xb @ w_b
    w = np.zeros(d)
    err_a, err_b = [], []
    for age in range(T):
        w_true = w_a if age < switch else w_b
        x = rng.standard_normal(d)
        y = x @ w_true
        lr = base_lr * schedule(age)
        w -= lr * (x @ w - y) * x
        err_a.append(float(np.mean((xa @ w - ya) ** 2)))
        err_b.append(float(np.mean((xb @ w - yb) ** 2)))
    return np.array(err_a), np.array(err_b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cpu_small")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--life", type=int, default=2000)
    ap.add_argument("--dim", type=int, default=8)
    ap.add_argument("--base-lr", type=float, default=0.01)
    ap.add_argument("--out", default="runs/development_demo")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.dev.enabled = True
    cfg.dev.critical_period_steps = args.life // 8
    # a pronounced critical period (low mature floor) so the imprinting effect
    # is legible; the training default floor is milder so the agent keeps
    # learning through adulthood.
    cfg.dev.floor_gain = 0.02
    dev = Development(cfg.dev)

    T, switch = args.life, args.life // 2
    ages = np.arange(T)
    cp_gains = np.array([dev.plasticity_gain(a) for a in ages])
    # mean-match the constant control so only the *timing* of plasticity differs
    const_gain = float(cp_gains.mean())

    rng = np.random.default_rng(args.seed)
    a_cp, b_cp = _imprint_life(dev.plasticity_gain, args.base_lr, T, switch,
                               args.dim, np.random.default_rng(args.seed))
    a_ct, b_ct = _imprint_life(lambda age: const_gain, args.base_lr, T, switch,
                               args.dim, np.random.default_rng(args.seed))

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _plot(ages, cp_gains, const_gain, switch, a_cp, a_ct, out / "development_demo.png")

    print("=== B3 development & critical periods ===")
    print(f"plasticity gain: birth {cp_gains[0]:.2f} -> mature "
          f"{cp_gains[-1]:.2f} (floor {cfg.dev.floor_gain}); "
          f"mean-matched constant control = {const_gain:.2f}")
    print(f"final error on EARLY regime A: critical-period {a_cp[-1]:.3f} "
          f"vs constant {a_ct[-1]:.3f}  "
          f"(lower under the critical period = early experience imprinted)")
    print(f"wrote {out / 'development_demo.png'}")


def _plot(ages, cp_gains, const_gain, switch, a_cp, a_ct, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(ages, cp_gains, color="tab:purple", label="critical period")
    ax.axhline(const_gain, color="gray", ls="--",
               label=f"constant (mean-matched)")
    ax.set_title("Plasticity schedule vs age")
    ax.set_xlabel("age (steps since birth)")
    ax.set_ylabel("learning-rate multiplier")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(a_cp, color="tab:purple", label="critical period")
    ax.plot(a_ct, color="gray", label="constant")
    ax.axvline(switch, color="k", ls=":", alpha=0.6, label="world changes (A->B)")
    ax.set_title("Error on EARLY regime A\n(lower = imprinted, not overwritten)")
    ax.set_xlabel("age (steps)")
    ax.set_ylabel("held-out MSE on regime A")
    ax.legend(fontsize=8)

    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
