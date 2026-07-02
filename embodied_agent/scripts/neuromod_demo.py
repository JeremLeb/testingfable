"""B1 demo: show that drive arbitration becomes state-dependent (allostatic)
instead of a hand-set constant, and (optionally) that it helps survival.

  python -m embodied_agent.scripts.neuromod_demo --config cpu_small
  python -m embodied_agent.scripts.neuromod_demo --config cpu_small --ablate

Panels: (1) the allostatic weighting function -- each drive's weight vs its own
deficit; (2) a real random-agent trace showing the energy weight tracking the
falling energy in situ; (3, with --ablate) survival time with allostasis on vs
off from two short trainings.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np

from ..config import load_config
from ..env import make_env
from ..env.neuromod import allostatic_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cpu_small")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trace-steps", type=int, default=1500)
    ap.add_argument("--ablate", action="store_true",
                    help="also run two short trainings (allostasis on/off)")
    ap.add_argument("--train-steps", type=int, default=4000)
    ap.add_argument("--out", default="runs/neuromod_demo")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.neuromod.enabled = True
    rc, nm = cfg.reward, cfg.neuromod

    # (1) the weighting function, swept analytically
    deficits = np.linspace(0, 1, 50)
    curve = {k: [] for k in ("energy", "thermal", "integrity")}
    for d in deficits:
        for k in curve:
            drives = {"energy": 0.0, "thermal": 0.0, "integrity": 0.0}
            drives[k] = d
            curve[k].append(allostatic_weights(drives, rc, nm)[k])

    # (2) a real trace: energy and its instantaneous weight over a life
    env = make_env(cfg, seed=args.seed)
    obs, _ = env.reset(seed=args.seed)
    rng = np.random.default_rng(args.seed)
    a = np.zeros(2)
    energy_hist, w_energy_hist, w_integ_hist = [], [], []
    for _ in range(args.trace_steps):
        a = 0.85 * a + 0.15 * rng.uniform(-1, 1, 2)
        obs, r, term, trunc, info = env.step(a)
        energy_hist.append(info["energy"])
        w_energy_hist.append(info.get("weight_energy", np.nan))
        w_integ_hist.append(info.get("weight_integrity", np.nan))
        if term or trunc:
            obs, _ = env.reset()

    ablation = None
    if args.ablate:
        ablation = _survival_ablation(args)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _plot(deficits, curve, energy_hist, w_energy_hist, w_integ_hist,
          ablation, out / "neuromod_demo.png")

    print("=== B1 neuromodulation & allostasis ===")
    print(f"weight_energy at full deficit: {curve['energy'][-1]:.2f} "
          f"(base {rc.w_energy}); integrity {curve['integrity'][-1]:.2f} "
          f"(base {rc.w_integrity})")
    if ablation:
        print(f"survival (mean ep length): allostatic "
              f"{ablation['on']:.0f} vs fixed {ablation['off']:.0f}")
    print(f"wrote {out / 'neuromod_demo.png'}")


def _survival_ablation(args) -> dict:
    from ..train import run
    results = {}
    for label, enabled in (("on", True), ("off", False)):
        cfg = load_config(args.config)
        cfg.neuromod.enabled = enabled
        cfg.train.total_steps = args.train_steps
        cfg.train.out_dir = f"{args.out}/train_{label}"
        cfg.train.seed = args.seed
        summary = run(cfg, verbose=False)
        results[label] = summary["final"]["eval_length"]
    return results


def _plot(deficits, curve, energy_hist, w_e, w_i, ablation, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = 3 if ablation else 2
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))

    ax = axes[0]
    for k, c in zip(("energy", "thermal", "integrity"),
                    ("tab:orange", "tab:cyan", "tab:red")):
        ax.plot(deficits, curve[k], label=k, color=c)
    ax.set_title("Allostatic weighting:\ndrive weight vs its own deficit")
    ax.set_xlabel("drive deficit (distance from setpoint)")
    ax.set_ylabel("effective weight")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(energy_hist, color="tab:green", label="energy level")
    ax2 = ax.twinx()
    ax2.plot(w_e, color="tab:orange", alpha=0.8, label="weight_energy")
    ax.set_title("In situ: energy weight rises\nas energy falls")
    ax.set_xlabel("step")
    ax.set_ylabel("energy", color="tab:green")
    ax2.set_ylabel("weight_energy", color="tab:orange")

    if ablation:
        ax = axes[2]
        ax.bar(["allostatic", "fixed"], [ablation["on"], ablation["off"]],
               color=["tab:blue", "gray"])
        ax.set_title("Survival (mean ep length)")
        ax.set_ylabel("steps alive")

    for ax in axes[:1]:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
