"""B4 demo: metabolic cost of cognition & sensorimotor realism.

  (1) Bounded planning: the imagination horizon the agent can afford shrinks as
      its body energy falls -- it literally "thinks less when starving".
  (2) Sparse cortical code: with k-winners-take-all on the latent, only a
      fraction of the assembly's groups fire (vs a dense code).
  (3, with --ablate) Sensorimotor robustness: end-to-end eval reward with no
      sensory latency vs a delayed/noisy closed loop.

  python -m embodied_agent.scripts.metabolism_demo --config cpu_small
  python -m embodied_agent.scripts.metabolism_demo --config cpu_small --ablate
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch

from ..config import load_config
from ..agent.collect import collect_random
from ..agent.metabolism import Metabolism
from ..agent.replay import ReplayBuffer
from ..env import make_env
from ..model.world_model import WorldModel


def _active_group_fraction(cfg, seed, steps, sparse):
    """Fraction of latent groups that fire, on real posterior states."""
    cfg = load_config(cfg) if isinstance(cfg, str) else cfg
    cfg.model.sparse_latent = sparse
    env = make_env(cfg, seed=seed)
    buffer = ReplayBuffer(cfg.train.buffer_capacity)
    collect_random(env, buffer, steps, np.random.default_rng(seed))
    wm = WorldModel(cfg, env.sensors.spaces)
    batch = buffer.sample(cfg.train.batch_size, cfg.train.seq_len,
                          np.random.default_rng(0))
    with torch.no_grad():
        post, _ = wm.rssm.observe(wm.encoder(batch["obs"]), batch["prev_action"])
        z = post.z.reshape(*post.z.shape[:-1], wm.rssm.groups, wm.rssm.classes)
        return float((z.abs().amax(-1) > 0).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cpu_small")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--collect", type=int, default=2000)
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--train-steps", type=int, default=6000)
    ap.add_argument("--out", default="runs/metabolism_demo")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg.metab.enabled = True
    metab = Metabolism(cfg.metab)

    # (1) bounded planning: horizon vs energy
    energies = np.linspace(0, 1, 50)
    horizons = [metab.planning_horizon(e, cfg.agent.horizon) for e in energies]

    # (2) sparse vs dense representation sparsity
    dense_frac = _active_group_fraction(args.config, args.seed, args.collect, False)
    sparse_frac = _active_group_fraction(args.config, args.seed, args.collect, True)

    ablation = _latency_ablation(args) if args.ablate else None

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _plot(energies, horizons, cfg.agent.horizon, cfg.metab.min_horizon,
          dense_frac, sparse_frac, ablation, out / "metabolism_demo.png")

    print("=== B4 metabolic cost of cognition & sensorimotor realism ===")
    print(f"bounded planning: horizon {cfg.metab.min_horizon} (starving) .. "
          f"{cfg.agent.horizon} (satiated)")
    print(f"active latent-group fraction: dense {dense_frac:.2f} vs "
          f"k-winners {sparse_frac:.2f} (target {cfg.model.sparse_frac})")
    if ablation:
        print(f"eval reward: no-latency {ablation['clean']:.1f} vs "
              f"delayed+noisy {ablation['delayed']:.1f}")
    print(f"wrote {out / 'metabolism_demo.png'}")


def _latency_ablation(args) -> dict:
    from ..train import run
    results = {}
    for label, delay, noise in (("clean", 0, 0.0), ("delayed", 4, 0.1)):
        cfg = load_config(args.config)
        cfg.metab.enabled = True
        cfg.metab.obs_delay = delay
        cfg.metab.action_delay = delay // 2
        cfg.metab.motor_noise = noise
        cfg.train.total_steps = args.train_steps
        cfg.train.out_dir = f"{args.out}/train_{label}"
        cfg.train.seed = args.seed
        summary = run(cfg, verbose=False)
        results[label] = summary["final"]["eval_reward"]
    return results


def _plot(energies, horizons, base_h, min_h, dense, sparse, ablation, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = 3 if ablation else 2
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))

    ax = axes[0]
    ax.plot(energies, horizons, color="tab:green")
    ax.axhline(base_h, color="gray", ls="--", alpha=0.6, label="satiated horizon")
    ax.axhline(min_h, color="tab:red", ls=":", alpha=0.7, label="starving floor")
    ax.set_title("Bounded planning:\nthink less when starving")
    ax.set_xlabel("body energy")
    ax.set_ylabel("affordable imagination horizon")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.bar(["dense", "k-winners"], [dense, sparse],
           color=["gray", "tab:purple"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Sparse cortical code\n(active latent-group fraction)")
    ax.set_ylabel("fraction of groups firing")

    if ablation:
        ax = axes[2]
        ax.bar(["no latency", "delayed+noisy"],
               [ablation["clean"], ablation["delayed"]],
               color=["tab:blue", "gray"])
        ax.set_title("Sensorimotor robustness")
        ax.set_ylabel("eval reward")

    for ax in axes[:1]:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
